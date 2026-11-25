"""Unit tests for strava_mcp.rate_limit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from strava_mcp.config import RateLimitError
from strava_mcp.rate_limit import (
    DEFAULT_OVERALL_LIMIT_15M,
    DEFAULT_OVERALL_LIMIT_DAILY,
    DEFAULT_READ_LIMIT_15M,
    DEFAULT_READ_LIMIT_DAILY,
    RateLimitTracker,
    format_utc_iso,
    next_15m_reset_utc,
    next_daily_reset_utc,
    parse_rate_limit_pair,
    retry_after_seconds,
)


def test_standard_tier_defaults_and_snapshot() -> None:
    now = datetime(2026, 11, 25, 4, 7, 30, tzinfo=UTC)
    tracker = RateLimitTracker(clock=lambda: now)
    snap = tracker.snapshot()

    assert snap["observed"] is False
    assert snap["last_updated_at_utc"] is None
    assert snap["is_limited"] is False
    assert snap["overall"] == {
        "observed": False,
        "usage_15m": 0,
        "limit_15m": DEFAULT_OVERALL_LIMIT_15M,
        "remaining_15m": 400,
        "usage_daily": 0,
        "limit_daily": DEFAULT_OVERALL_LIMIT_DAILY,
        "remaining_daily": 4000,
    }
    assert snap["read"] == {
        "observed": False,
        "usage_15m": 0,
        "limit_15m": DEFAULT_READ_LIMIT_15M,
        "remaining_15m": 200,
        "usage_daily": 0,
        "limit_daily": DEFAULT_READ_LIMIT_DAILY,
        "remaining_daily": 2000,
    }
    assert snap["resets"] == {
        "fifteen_min_reset_at_utc": "2026-11-25T04:15:00Z",
        "fifteen_min_retry_after_seconds": 450,
        "daily_reset_at_utc": "2026-11-26T00:00:00Z",
        "daily_retry_after_seconds": 71550,
    }


def test_utc_window_boundaries_and_year_rollover() -> None:
    for minute in (0, 15, 30, 45):
        boundary = datetime(2026, 11, 25, 4, minute, 0, tzinfo=UTC)
        reset_15m = next_15m_reset_utc(boundary)
        assert retry_after_seconds(reset_15m, boundary) == 900
        expected_min = (minute + 15) % 60
        expected_hr = 5 if minute == 45 else 4
        assert reset_15m == datetime(2026, 11, 25, expected_hr, expected_min, 0, tzinfo=UTC)

    sub_sec = datetime(2026, 11, 25, 4, 14, 59, 200_000, tzinfo=UTC)
    assert retry_after_seconds(next_15m_reset_utc(sub_sec), sub_sec) == 1

    midnight = datetime(2026, 11, 25, 0, 0, 0, tzinfo=UTC)
    daily_reset = next_daily_reset_utc(midnight)
    assert daily_reset == datetime(2026, 11, 26, 0, 0, 0, tzinfo=UTC)
    assert retry_after_seconds(daily_reset, midnight) == 86400

    nye = datetime(2026, 12, 31, 23, 50, 0, tzinfo=UTC)
    assert format_utc_iso(next_15m_reset_utc(nye)) == "2027-01-01T00:00:00Z"
    assert format_utc_iso(next_daily_reset_utc(nye)) == "2027-01-01T00:00:00Z"

    # Non-UTC offset is normalized cleanly
    pst = timezone(timedelta(hours=-8))
    local_dt = datetime(2026, 11, 24, 20, 10, 0, tzinfo=pst)
    assert format_utc_iso(next_15m_reset_utc(local_dt)) == "2026-11-25T04:15:00Z"


def test_parse_rate_limit_pair_and_header_resilience() -> None:
    assert parse_rate_limit_pair("12,340") == (12, 340)
    assert parse_rate_limit_pair(" 0 , 2000 ") == (0, 2000)
    for bad in (None, "", "abc", "1,2,3", "-5,10", "1.5,10", "²³,10", "10,"):
        assert parse_rate_limit_pair(bad) is None

    now = datetime(2026, 11, 25, 4, 5, 0, tzinfo=UTC)
    tracker = RateLimitTracker(clock=lambda: now)
    tracker.update_from_headers(
        {
            "x-ratelimit-limit": " 400 , 4000 ",
            "X-RateLimit-Usage": "15, 220",
            "X-READRATELIMIT-LIMIT": "200,2000",
            "x-readratelimit-usage": "14,210",
        }
    )
    assert tracker.overall.usage_15m == 15
    assert tracker.overall.usage_daily == 220
    assert tracker.read.usage_15m == 14
    assert tracker.read.usage_daily == 210

    # Malformed headers must not raise or clobber prior valid state
    tracker.update_from_headers(
        {
            "X-RateLimit-Limit": "bad",
            "X-RateLimit-Usage": "1,2,3",
            "X-ReadRateLimit-Limit": "-1,100",
            "X-ReadRateLimit-Usage": "",
        }
    )
    assert tracker.overall.observed is True
    assert tracker.overall.usage_15m == 15
    assert tracker.read.usage_15m == 14


def test_window_rollover_resets_15m_without_clearing_daily() -> None:
    current = datetime(2026, 11, 25, 4, 14, 50, tzinfo=UTC)
    tracker = RateLimitTracker(clock=lambda: current)
    tracker.update_from_headers(
        {
            "X-RateLimit-Limit": "400,4000",
            "X-RateLimit-Usage": "200,500",
            "X-ReadRateLimit-Limit": "200,2000",
            "X-ReadRateLimit-Usage": "200,500",
        }
    )

    current = datetime(2026, 11, 25, 4, 14, 55, tzinfo=UTC)
    with pytest.raises(RateLimitError) as exc_info:
        tracker.check_available(is_read=True)
    assert exc_info.value.limit_type == "read_15m"
    assert exc_info.value.reset_at_utc == "2026-11-25T04:15:00Z"
    assert exc_info.value.retry_after_seconds == 5

    # 15-min boundary crossed -> 15m usage resets to 0 while daily remains 500
    current = datetime(2026, 11, 25, 4, 15, 0, tzinfo=UTC)
    tracker.check_available(is_read=True)
    snap = tracker.snapshot()
    assert snap["is_limited"] is False
    assert snap["read"]["usage_15m"] == 0
    assert snap["read"]["usage_daily"] == 500
    assert snap["overall"]["usage_15m"] == 0
    assert snap["overall"]["usage_daily"] == 500

    # Midnight UTC rollover clears daily usage as well
    current = datetime(2026, 11, 26, 0, 0, 0, tzinfo=UTC)
    snap_next_day = tracker.snapshot()
    assert snap_next_day["read"]["usage_daily"] == 0
    assert snap_next_day["overall"]["usage_daily"] == 0


def test_preemptive_check_priority_and_is_read_flag() -> None:
    now = datetime(2026, 11, 25, 4, 10, 0, tzinfo=UTC)

    # Daily takes precedence over 15m; read takes precedence over overall
    tracker = RateLimitTracker(clock=lambda: now)
    tracker.update_from_headers(
        {
            "X-RateLimit-Limit": "400,4000",
            "X-RateLimit-Usage": "400,4000",
            "X-ReadRateLimit-Limit": "200,2000",
            "X-ReadRateLimit-Usage": "200,2000",
        }
    )
    with pytest.raises(RateLimitError) as exc_read_daily:
        tracker.check_can_request(is_read=True)
    assert exc_read_daily.value.code == "rate_limited"
    assert exc_read_daily.value.limit_type == "read_daily"
    assert exc_read_daily.value.usage == 2000
    assert exc_read_daily.value.limit == 2000
    assert exc_read_daily.value.reset_at_utc == "2026-11-26T00:00:00Z"

    # When only read is exhausted, is_read=False checks overall only and succeeds
    tracker_read_only = RateLimitTracker(clock=lambda: now)
    tracker_read_only.update_from_headers(
        {
            "X-RateLimit-Limit": "400,4000",
            "X-RateLimit-Usage": "200,500",
            "X-ReadRateLimit-Limit": "200,2000",
            "X-ReadRateLimit-Usage": "200,500",
        }
    )
    tracker_read_only.check_available(is_read=False)
    with pytest.raises(RateLimitError) as exc_read_15m:
        tracker_read_only.check_available(is_read=True)
    assert exc_read_15m.value.limit_type == "read_15m"

    # Overall daily before overall 15m when read is under limit
    tracker_overall = RateLimitTracker(clock=lambda: now)
    tracker_overall.update_from_headers(
        {
            "X-RateLimit-Limit": "400,4000",
            "X-RateLimit-Usage": "400,4000",
            "X-ReadRateLimit-Limit": "200,2000",
            "X-ReadRateLimit-Usage": "10,100",
        }
    )
    with pytest.raises(RateLimitError) as exc_ov_daily:
        tracker_overall.check_available(is_read=True)
    assert exc_ov_daily.value.limit_type == "overall_daily"


def test_record_response_429_with_and_without_headers() -> None:
    now = datetime(2026, 11, 25, 4, 12, 0, tzinfo=UTC)
    tracker = RateLimitTracker(clock=lambda: now)

    resp_429 = httpx.Response(
        429,
        headers={
            "X-RateLimit-Limit": "400,4000",
            "X-RateLimit-Usage": "201,1000",
            "X-ReadRateLimit-Limit": "200,2000",
            "X-ReadRateLimit-Usage": "201,1000",
        },
        json={"message": "Rate Limit Exceeded", "errors": []},
    )
    with pytest.raises(RateLimitError) as exc_info:
        tracker.record_response(resp_429, is_read=True)
    assert exc_info.value.limit_type == "read_15m"
    assert exc_info.value.usage == 201
    assert exc_info.value.limit == 200
    assert exc_info.value.reset_at_utc == "2026-11-25T04:15:00Z"
    assert exc_info.value.retry_after_seconds == 180
    assert exc_info.value.details["strava_message"] == "Rate Limit Exceeded"

    # Headerless / Retry-After fallback on fresh tracker
    fresh = RateLimitTracker(clock=lambda: now)
    headerless = httpx.Response(429, headers={"Retry-After": "45"}, text="Too Many Requests")
    with pytest.raises(RateLimitError) as exc_fallback:
        fresh.record_response(headerless, is_read=True)
    assert exc_fallback.value.limit_type == "read_15m"
    assert exc_fallback.value.usage is None
    assert exc_fallback.value.limit is None
    assert exc_fallback.value.retry_after_seconds == 45
    assert exc_fallback.value.reset_at_utc == "2026-11-25T04:12:45Z"
