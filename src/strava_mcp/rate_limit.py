"""Strava API v3 rate-limit header parsing, UTC window tracking, and 429 handling."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx

from strava_mcp.config import RateLimitError

DEFAULT_OVERALL_LIMIT_15M = 400
DEFAULT_OVERALL_LIMIT_DAILY = 4000
DEFAULT_READ_LIMIT_15M = 200
DEFAULT_READ_LIMIT_DAILY = 2000

HEADER_OVERALL_LIMIT = "x-ratelimit-limit"
HEADER_OVERALL_USAGE = "x-ratelimit-usage"
HEADER_READ_LIMIT = "x-readratelimit-limit"
HEADER_READ_USAGE = "x-readratelimit-usage"

LimitType = Literal["read_15m", "read_daily", "overall_15m", "overall_daily"]


@dataclass(frozen=True)
class BucketState:
    """Snapshot of a single rate-limit bucket (overall or read)."""

    usage_15m: int
    limit_15m: int
    usage_daily: int
    limit_daily: int
    observed: bool = False
    observed_at_utc: datetime | None = None
    reset_15m_at_utc: datetime | None = None
    reset_daily_at_utc: datetime | None = None


def _normalize_utc(
    dt: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> datetime:
    resolved = dt if dt is not None else (clock() if clock is not None else datetime.now(UTC))
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=UTC)
    return resolved.astimezone(UTC)


def next_15m_reset_utc(now: datetime) -> datetime:
    """Return the exact next UTC 15-minute boundary (:00, :15, :30, :45) strictly after now."""
    utc_now = _normalize_utc(now)
    bucket_minute = (utc_now.minute // 15) * 15
    bucket_start = utc_now.replace(minute=bucket_minute, second=0, microsecond=0)
    return bucket_start + timedelta(minutes=15)


def next_daily_reset_utc(now: datetime) -> datetime:
    """Return the exact next UTC midnight (00:00:00 UTC) strictly after now."""
    utc_now = _normalize_utc(now)
    return utc_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def retry_after_seconds(reset_at: datetime, now: datetime) -> int:
    """Return whole seconds until reset_at, clamped to at least 1s."""
    delta = (_normalize_utc(reset_at) - _normalize_utc(now)).total_seconds()
    return max(1, math.ceil(delta))


def format_utc_iso(dt: datetime) -> str:
    """Format a datetime as compact ISO-8601 UTC string (YYYY-MM-DDTHH:MM:SSZ)."""
    return _normalize_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_rate_limit_pair(raw: str | None) -> tuple[int, int] | None:
    """Parse a Strava '15m,daily' header pair into non-negative integers, or None if invalid."""
    if not isinstance(raw, str):
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    first = parts[0].strip()
    second = parts[1].strip()
    if not (first.isascii() and first.isdigit() and second.isascii() and second.isdigit()):
        return None
    return (int(first), int(second))


def _parse_retry_after_header(raw: str | None, utc_now: datetime) -> int | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.isascii() and cleaned.isdigit():
        return max(1, int(cleaned))
    try:
        parsed_dt = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        return None
    return retry_after_seconds(parsed_dt, utc_now)


def _extract_fault_message(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    return message.strip()[:200]


class RateLimitTracker:
    """Thread-safe tracker for Strava API 15-minute and daily rate-limit buckets."""

    def __init__(
        self,
        *,
        overall_limit_15m: int = DEFAULT_OVERALL_LIMIT_15M,
        overall_limit_daily: int = DEFAULT_OVERALL_LIMIT_DAILY,
        read_limit_15m: int = DEFAULT_READ_LIMIT_15M,
        read_limit_daily: int = DEFAULT_READ_LIMIT_DAILY,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._overall = BucketState(
            usage_15m=0,
            limit_15m=overall_limit_15m,
            usage_daily=0,
            limit_daily=overall_limit_daily,
        )
        self._read = BucketState(
            usage_15m=0,
            limit_15m=read_limit_15m,
            usage_daily=0,
            limit_daily=read_limit_daily,
        )

    def _utc_now(self, now: datetime | None = None) -> datetime:
        return _normalize_utc(now, self._clock)

    @staticmethod
    def _rollover_bucket(bucket: BucketState, utc_now: datetime) -> BucketState:
        usage_15m = bucket.usage_15m
        usage_daily = bucket.usage_daily
        reset_15m = bucket.reset_15m_at_utc
        reset_daily = bucket.reset_daily_at_utc

        if reset_15m is not None and utc_now >= reset_15m:
            usage_15m = 0
            reset_15m = next_15m_reset_utc(utc_now)
        if reset_daily is not None and utc_now >= reset_daily:
            usage_daily = 0
            reset_daily = next_daily_reset_utc(utc_now)

        if (
            usage_15m == bucket.usage_15m
            and usage_daily == bucket.usage_daily
            and reset_15m == bucket.reset_15m_at_utc
            and reset_daily == bucket.reset_daily_at_utc
        ):
            return bucket

        return BucketState(
            usage_15m=usage_15m,
            limit_15m=bucket.limit_15m,
            usage_daily=usage_daily,
            limit_daily=bucket.limit_daily,
            observed=bucket.observed,
            observed_at_utc=bucket.observed_at_utc,
            reset_15m_at_utc=reset_15m,
            reset_daily_at_utc=reset_daily,
        )

    @property
    def overall(self) -> BucketState:
        with self._lock:
            return self._rollover_bucket(self._overall, self._utc_now())

    @property
    def read(self) -> BucketState:
        with self._lock:
            return self._rollover_bucket(self._read, self._utc_now())

    def update_from_headers(
        self,
        headers: Mapping[str, str],
        *,
        now: datetime | None = None,
    ) -> None:
        """Update stored rate-limit buckets from Strava HTTP response headers."""
        normalized = {str(k).lower(): str(v) for k, v in headers.items()}
        overall_limit = parse_rate_limit_pair(normalized.get(HEADER_OVERALL_LIMIT))
        overall_usage = parse_rate_limit_pair(normalized.get(HEADER_OVERALL_USAGE))
        read_limit = parse_rate_limit_pair(normalized.get(HEADER_READ_LIMIT))
        read_usage = parse_rate_limit_pair(normalized.get(HEADER_READ_USAGE))

        utc_now = self._utc_now(now)
        reset_15m = next_15m_reset_utc(utc_now)
        reset_daily = next_daily_reset_utc(utc_now)

        with self._lock:
            cur_overall = self._rollover_bucket(self._overall, utc_now)
            if overall_limit is not None or overall_usage is not None:
                self._overall = BucketState(
                    usage_15m=(
                        overall_usage[0] if overall_usage is not None else cur_overall.usage_15m
                    ),
                    limit_15m=(
                        overall_limit[0] if overall_limit is not None else cur_overall.limit_15m
                    ),
                    usage_daily=(
                        overall_usage[1] if overall_usage is not None else cur_overall.usage_daily
                    ),
                    limit_daily=(
                        overall_limit[1] if overall_limit is not None else cur_overall.limit_daily
                    ),
                    observed=True,
                    observed_at_utc=utc_now,
                    reset_15m_at_utc=reset_15m,
                    reset_daily_at_utc=reset_daily,
                )
            else:
                self._overall = cur_overall

            cur_read = self._rollover_bucket(self._read, utc_now)
            if read_limit is not None or read_usage is not None:
                self._read = BucketState(
                    usage_15m=(
                        read_usage[0] if read_usage is not None else cur_read.usage_15m
                    ),
                    limit_15m=(
                        read_limit[0] if read_limit is not None else cur_read.limit_15m
                    ),
                    usage_daily=(
                        read_usage[1] if read_usage is not None else cur_read.usage_daily
                    ),
                    limit_daily=(
                        read_limit[1] if read_limit is not None else cur_read.limit_daily
                    ),
                    observed=True,
                    observed_at_utc=utc_now,
                    reset_15m_at_utc=reset_15m,
                    reset_daily_at_utc=reset_daily,
                )
            else:
                self._read = cur_read

    def _find_exhausted(
        self,
        overall_eff: BucketState,
        read_eff: BucketState,
        *,
        is_read: bool,
        reset_15m: datetime,
        reset_daily: datetime,
    ) -> tuple[LimitType, int, int, datetime] | None:
        candidates: list[tuple[LimitType, int, int, datetime]] = []
        if is_read:
            candidates.append(
                ("read_daily", read_eff.usage_daily, read_eff.limit_daily, reset_daily)
            )
        candidates.append(
            ("overall_daily", overall_eff.usage_daily, overall_eff.limit_daily, reset_daily)
        )
        if is_read:
            candidates.append(("read_15m", read_eff.usage_15m, read_eff.limit_15m, reset_15m))
        candidates.append(
            ("overall_15m", overall_eff.usage_15m, overall_eff.limit_15m, reset_15m)
        )
        for limit_type, usage, limit, reset_at in candidates:
            if usage >= limit:
                return (limit_type, usage, limit, reset_at)
        return None

    def check_available(
        self,
        *,
        is_read: bool = True,
        now: datetime | None = None,
    ) -> None:
        """Raise RateLimitError preemptively if the active window quota is exhausted."""
        utc_now = self._utc_now(now)
        reset_15m = next_15m_reset_utc(utc_now)
        reset_daily = next_daily_reset_utc(utc_now)

        with self._lock:
            overall_eff = self._rollover_bucket(self._overall, utc_now)
            read_eff = self._rollover_bucket(self._read, utc_now)
            exhausted = self._find_exhausted(
                overall_eff,
                read_eff,
                is_read=is_read,
                reset_15m=reset_15m,
                reset_daily=reset_daily,
            )

        if exhausted is not None:
            limit_type, usage, limit, reset_at = exhausted
            retry_sec = retry_after_seconds(reset_at, utc_now)
            reset_iso = format_utc_iso(reset_at)
            raise RateLimitError(
                f"Strava {limit_type} rate limit reached ({usage}/{limit}). "
                f"Resets at {reset_iso} (in {retry_sec}s).",
                limit_type=limit_type,
                usage=usage,
                limit=limit,
                reset_at_utc=reset_iso,
                retry_after_seconds=retry_sec,
                details={"preemptive": True},
            )

    def check_can_request(
        self,
        *,
        is_read: bool = True,
        now: datetime | None = None,
    ) -> None:
        """Alias for check_available."""
        self.check_available(is_read=is_read, now=now)

    def error_from_429(
        self,
        response: httpx.Response,
        *,
        is_read: bool = True,
        now: datetime | None = None,
    ) -> RateLimitError:
        """Build a structured RateLimitError from an HTTP 429 response."""
        utc_now = self._utc_now(now)
        reset_15m = next_15m_reset_utc(utc_now)
        reset_daily = next_daily_reset_utc(utc_now)

        with self._lock:
            overall_eff = self._rollover_bucket(self._overall, utc_now)
            read_eff = self._rollover_bucket(self._read, utc_now)
            exhausted = self._find_exhausted(
                overall_eff,
                read_eff,
                is_read=is_read,
                reset_15m=reset_15m,
                reset_daily=reset_daily,
            )

        details: dict[str, Any] = {"status_code": response.status_code}
        fault_msg = _extract_fault_message(response)
        if fault_msg is not None:
            details["strava_message"] = fault_msg

        if exhausted is not None:
            limit_type, usage, limit, reset_at = exhausted
            retry_sec = retry_after_seconds(reset_at, utc_now)
            reset_iso = format_utc_iso(reset_at)
            return RateLimitError(
                f"Strava API rate limit exceeded ({limit_type}: {usage}/{limit}). "
                f"Resets at {reset_iso} (in {retry_sec}s).",
                limit_type=limit_type,
                usage=usage,
                limit=limit,
                reset_at_utc=reset_iso,
                retry_after_seconds=retry_sec,
                details=details,
            )

        fallback_type: LimitType = "read_15m" if is_read else "overall_15m"
        header_retry = _parse_retry_after_header(response.headers.get("retry-after"), utc_now)
        if header_retry is not None:
            retry_sec = header_retry
            reset_at = utc_now + timedelta(seconds=retry_sec)
        else:
            reset_at = reset_15m
            retry_sec = retry_after_seconds(reset_at, utc_now)
        reset_iso = format_utc_iso(reset_at)
        return RateLimitError(
            f"Strava API returned HTTP 429 ({fallback_type}). "
            f"Retry after {retry_sec}s (at {reset_iso}).",
            limit_type=fallback_type,
            usage=None,
            limit=None,
            reset_at_utc=reset_iso,
            retry_after_seconds=retry_sec,
            details=details,
        )

    def record_response(
        self,
        response: httpx.Response,
        *,
        is_read: bool = True,
        now: datetime | None = None,
    ) -> None:
        """Record headers from an HTTP response and raise RateLimitError on HTTP 429."""
        self.update_from_headers(response.headers, now=now)
        if response.status_code == 429:
            raise self.error_from_429(response, is_read=is_read, now=now)

    def snapshot(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return a JSON-safe status snapshot of overall/read limits and UTC resets."""
        utc_now = self._utc_now(now)
        reset_15m = next_15m_reset_utc(utc_now)
        reset_daily = next_daily_reset_utc(utc_now)

        with self._lock:
            overall_eff = self._rollover_bucket(self._overall, utc_now)
            read_eff = self._rollover_bucket(self._read, utc_now)

        observed_times = [
            dt
            for dt in (overall_eff.observed_at_utc, read_eff.observed_at_utc)
            if dt is not None
        ]
        last_observed = max(observed_times) if observed_times else None
        is_limited = any(
            (
                overall_eff.usage_15m >= overall_eff.limit_15m,
                overall_eff.usage_daily >= overall_eff.limit_daily,
                read_eff.usage_15m >= read_eff.limit_15m,
                read_eff.usage_daily >= read_eff.limit_daily,
            )
        )

        return {
            "observed": overall_eff.observed or read_eff.observed,
            "last_updated_at_utc": (
                format_utc_iso(last_observed) if last_observed is not None else None
            ),
            "resets": {
                "fifteen_min_reset_at_utc": format_utc_iso(reset_15m),
                "fifteen_min_retry_after_seconds": retry_after_seconds(reset_15m, utc_now),
                "daily_reset_at_utc": format_utc_iso(reset_daily),
                "daily_retry_after_seconds": retry_after_seconds(reset_daily, utc_now),
            },
            "overall": {
                "observed": overall_eff.observed,
                "usage_15m": overall_eff.usage_15m,
                "limit_15m": overall_eff.limit_15m,
                "remaining_15m": max(0, overall_eff.limit_15m - overall_eff.usage_15m),
                "usage_daily": overall_eff.usage_daily,
                "limit_daily": overall_eff.limit_daily,
                "remaining_daily": max(0, overall_eff.limit_daily - overall_eff.usage_daily),
            },
            "read": {
                "observed": read_eff.observed,
                "usage_15m": read_eff.usage_15m,
                "limit_15m": read_eff.limit_15m,
                "remaining_15m": max(0, read_eff.limit_15m - read_eff.usage_15m),
                "usage_daily": read_eff.usage_daily,
                "limit_daily": read_eff.limit_daily,
                "remaining_daily": max(0, read_eff.limit_daily - read_eff.usage_daily),
            },
            "is_limited": is_limited,
        }
