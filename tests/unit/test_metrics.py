"""Unit tests for strava_mcp.metrics."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from strava_mcp.client import StravaClient
from strava_mcp.config import Config, TokenData, save_client, save_tokens
from strava_mcp.metrics import (
    METERS_PER_KM,
    METERS_PER_MILE,
    downsample_streams,
    fetch_activities_paginated,
    format_activity_detail,
    format_activity_summary,
    format_duration,
    format_pace,
    matches_sport_type,
    normalize_activity_units,
    normalize_athlete,
    normalize_athlete_stats,
    normalize_gear,
    normalize_lap,
    normalize_route,
    normalize_segment,
    normalize_sport_type,
    parse_date_filter,
    parse_date_iso,
    summarize_metrics,
    summarize_streams,
    validate_gear_id,
)


def test_parse_date_filter_valid_formats_and_timezones() -> None:
    fixed_now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)

    assert parse_date_filter(None) is None
    assert parse_date_filter(0) == 0
    assert parse_date_filter(1700000000) == 1700000000
    assert parse_date_filter(1700000000.9) == 1700000000
    assert parse_date_filter("1700000000") == 1700000000
    assert parse_date_filter("1700000000.9") == 1700000000

    assert parse_date_filter("2026-11-01", is_end=False) == int(
        datetime(2026, 11, 1, 0, 0, 0, tzinfo=UTC).timestamp()
    )
    assert parse_date_filter("2026-11-01", is_end=True) == int(
        datetime(2026, 11, 1, 23, 59, 59, tzinfo=UTC).timestamp()
    )
    assert parse_date_filter(date(2026, 11, 1), is_end=True) == int(
        datetime(2026, 11, 1, 23, 59, 59, tzinfo=UTC).timestamp()
    )

    assert parse_date_filter("2026-11-01T10:30:00Z") == int(
        datetime(2026, 11, 1, 10, 30, 0, tzinfo=UTC).timestamp()
    )
    assert parse_date_filter("2026-11-01T12:30:00+02:00") == int(
        datetime(2026, 11, 1, 10, 30, 0, tzinfo=UTC).timestamp()
    )
    assert parse_date_filter("2026-11-01T02:30:00-08:00") == int(
        datetime(2026, 11, 1, 10, 30, 0, tzinfo=UTC).timestamp()
    )
    assert parse_date_filter(datetime(2026, 11, 1, 10, 30, 0)) == int(
        datetime(2026, 11, 1, 10, 30, 0, tzinfo=UTC).timestamp()
    )
    pst = timezone(timedelta(hours=-8))
    assert parse_date_filter(datetime(2026, 11, 1, 2, 30, 0, tzinfo=pst)) == int(
        datetime(2026, 11, 1, 10, 30, 0, tzinfo=UTC).timestamp()
    )

    assert parse_date_filter("24h", now=fixed_now) == int(
        (fixed_now - timedelta(hours=24)).timestamp()
    )
    assert parse_date_filter("7D", clock=lambda: fixed_now) == int(
        (fixed_now - timedelta(days=7)).timestamp()
    )
    assert parse_date_filter("2w", now=fixed_now) == int(
        (fixed_now - timedelta(weeks=2)).timestamp()
    )
    assert parse_date_filter("3m", now=fixed_now) == int(
        (fixed_now - timedelta(days=90)).timestamp()
    )
    assert parse_date_filter("1y", now=fixed_now) == int(
        (fixed_now - timedelta(days=365)).timestamp()
    )
    assert parse_date_filter("0d", now=fixed_now) == int(fixed_now.timestamp())
    assert parse_date_iso("2026-11-01", is_end=False) == "2026-11-01T00:00:00Z"
    assert parse_date_iso(None) is None


def test_parse_date_filter_invalid_inputs() -> None:
    for bad in (
        True,
        False,
        "",
        "   ",
        -1,
        "-5",
        math.nan,
        math.inf,
        "not-a-date",
        "2026-02-30",
        "-7d",
        "7x",
        "999999999999d",
        ["2026-11-01"],
        10**400,
        "9999-12-31T23:59:59-12:00",
        "0001-01-01T00:00:00+12:00",
    ):
        with pytest.raises(ValueError):
            parse_date_filter(bad)  # type: ignore[arg-type]

    for bad_iso in ("100000000000000000000", 10**400):
        with pytest.raises(ValueError):
            parse_date_iso(bad_iso)


def test_unit_normalization_pace_and_zero_guards() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(305) == "5:05"
    assert format_duration(3665) == "1:01:05"
    assert format_duration(-1) is None
    assert format_duration(None) is None

    # Rounding before divmod avoids "4:60"
    speed_near_5min_km = METERS_PER_KM / 299.8
    assert format_pace(speed_near_5min_km, meters_per_unit=METERS_PER_KM) == "5:00"
    assert format_pace(0.0, meters_per_unit=METERS_PER_KM) is None
    assert format_pace(-2.0, meters_per_unit=METERS_PER_MILE) is None

    run_units = normalize_activity_units(
        {
            "distance": 10000.0,
            "moving_time": 2500,
            "elapsed_time": 2600,
            "total_elevation_gain": 100.0,
            "average_speed": 4.0,
        }
    )
    assert run_units["distance_km"] == 10.0
    assert run_units["distance_mi"] == round(10000.0 / METERS_PER_MILE, 2)
    assert run_units["elevation_gain_m"] == 100.0
    assert run_units["elevation_gain_ft"] == 328.1
    assert run_units["moving_time_formatted"] == "41:40"
    assert run_units["avg_speed_kph"] == 14.4
    assert run_units["pace_min_per_km"] == "4:10"

    # Zero-speed and zero-distance activities never raise ZeroDivisionError
    stationary = normalize_activity_units(
        {"distance": 0.0, "moving_time": 1800, "elapsed_time": 1800, "average_speed": 0.0}
    )
    assert stationary["pace_min_per_km"] is None
    assert stationary["pace_min_per_mi"] is None
    assert stationary["avg_speed_kph"] == 0.0

    derived = normalize_activity_units({"distance": 5000.0, "moving_time": 1250})
    assert derived["pace_min_per_km"] == "4:10"

    missing = normalize_activity_units({"distance": 0.0, "moving_time": 0})
    assert missing["pace_min_per_km"] is None
    assert missing["avg_speed_kph"] is None


def test_sport_type_preference_and_type_alias() -> None:
    trail = {"sport_type": "TrailRun", "type": "Run"}
    legacy = {"type": "Ride"}
    empty: dict[str, Any] = {}

    assert normalize_sport_type(trail) == "TrailRun"
    assert normalize_sport_type(legacy) == "Ride"
    assert normalize_sport_type(empty) == "Unknown"

    assert matches_sport_type(trail, "trail_run") is True
    assert matches_sport_type(trail, "Run") is True
    assert matches_sport_type({"sport_type": "GravelRide", "type": "Ride"}, "GravelRide") is True
    assert matches_sport_type(legacy, "GravelRide") is False
    assert matches_sport_type(trail, None) is True


def test_format_activity_summary_and_detail_compactness() -> None:
    raw = {
        "id": 101,
        "name": "Morning Gravel",
        "sport_type": "GravelRide",
        "type": "Ride",
        "start_date": "2026-11-20T14:00:00Z",
        "start_date_local": "2026-11-20T06:00:00Z",
        "distance": 42000.0,
        "moving_time": 5400,
        "elapsed_time": 5600,
        "total_elevation_gain": 620.0,
        "average_speed": 7.777,
        "map": {"id": "a101", "polyline": "abcdef", "summary_polyline": "abc"},
        "segment_efforts": [{"id": 1}, {"id": 2}, {"id": 3}],
        "laps": [{"id": 10, "lap_index": 1, "distance": 5000.0, "moving_time": 600}],
        "gear": {"id": "b1", "name": "Gravel Bike", "distance": 1500000.0},
    }

    summary = format_activity_summary(raw)
    assert "map" not in summary
    assert "polyline" not in summary
    assert "summary_polyline" not in summary
    assert summary["sport_type"] == "GravelRide"

    summary_with_poly = format_activity_summary(raw, include_polyline=True)
    assert summary_with_poly["map"]["summary_polyline"] == "abc"

    detail = format_activity_detail(raw)
    assert detail["segment_efforts_count"] == 3
    assert "segment_efforts" not in detail
    assert "map" not in detail
    assert detail["gear"]["distance_km"] == 1500.0
    assert len(detail["laps"]) == 1

    detail_full = format_activity_detail(raw, include_all_efforts=True, include_polyline=True)
    assert len(detail_full["segment_efforts"]) == 3
    assert detail_full["map"]["polyline"] == "abcdef"


def test_summarize_and_downsample_streams() -> None:
    n = 1000
    streams = {
        "time": {"data": list(range(n)), "original_size": n, "resolution": "high"},
        "heartrate": {
            "data": [100 + (i % 81) for i in range(n)],
            "original_size": n,
            "resolution": "high",
        },
        "moving": {
            "data": [i % 5 != 0 for i in range(n)],
            "original_size": n,
            "resolution": "high",
        },
        "latlng": {
            "data": [[45.0 + i * 0.0001, -122.0 - i * 0.0001] for i in range(n)],
            "original_size": n,
            "resolution": "high",
        },
    }

    summaries = summarize_streams(streams)
    assert summaries["time"]["stream_type"] == "numeric"
    assert summaries["time"]["min"] == 0.0
    assert summaries["time"]["max"] == 999.0
    assert summaries["time"]["p95"] == pytest.approx(949.05, abs=0.01)

    # Boolean stream must NOT be coerced to numeric stats
    assert summaries["moving"]["stream_type"] == "boolean"
    assert summaries["moving"]["min"] is None
    assert summaries["moving"]["true_count"] == 800
    assert summaries["moving"]["false_count"] == 200
    assert summaries["moving"]["true_fraction"] == 0.8

    assert summaries["latlng"]["stream_type"] == "latlng"
    assert summaries["latlng"]["min"] is None
    assert summaries["latlng"]["start_latlng"] == [45.0, -122.0]
    assert summaries["latlng"]["end_latlng"] == [45.0999, -122.0999]
    assert summaries["latlng"]["bounds"]["max_lat"] == pytest.approx(45.0999)

    down = downsample_streams(streams, max_points=200)
    assert down["streams"]["time"]["sampled_count"] == 200
    assert down["streams"]["time"]["data"][0] == 0
    assert down["streams"]["time"]["data"][-1] == 999
    assert len(down["streams"]["heartrate"]["data"]) == 200
    assert len(down["streams"]["latlng"]["data"]) == 200

    # When already <= max_points, return untouched
    small = downsample_streams({"watts": {"data": [100, 200, 300]}}, max_points=50)
    assert small["streams"]["watts"]["data"] == [100, 200, 300]

    one = downsample_streams(streams, max_points=1)
    assert one["streams"]["time"]["data"] == [0]

    summary_only = downsample_streams(streams, summary_only=True)
    assert summary_only["streams"]["time"]["sampled_count"] == 0
    assert "data" not in summary_only["streams"]["time"]

    assert summarize_streams({}) == {}
    for bad_max in (0, -5, True, False, 2.5, "2", None):
        with pytest.raises(ValueError):
            downsample_streams(streams, max_points=bad_max)  # type: ignore[arg-type]


def test_summarize_metrics_groupings_and_visibility() -> None:
    activities = [
        {
            "id": 1,
            "sport_type": "GravelRide",
            "start_date": "2026-11-16T15:00:00Z",
            "start_date_local": "2026-11-16T07:00:00Z",
            "distance": 40000.0,
            "moving_time": 3600,
            "elapsed_time": 3800,
            "total_elevation_gain": 500.0,
            "average_heartrate": 150.0,
            "max_heartrate": 175.0,
            "average_watts": 220.0,
            "weighted_average_watts": 240.0,
            "kilojoules": 792.0,
            "suffer_score": 80.0,
            "private": True,
            "visibility": "only_me",
        },
        {
            "id": 2,
            "sport_type": "TrailRun",
            "start_date": "2026-11-17T15:00:00Z",
            "start_date_local": "2026-11-17T07:00:00Z",
            "distance": 10000.0,
            "moving_time": 1800,
            "elapsed_time": 1900,
            "total_elevation_gain": 300.0,
            "average_heartrate": 165.0,
            "max_heartrate": 182.0,
            "suffer_score": 60.0,
            "private": False,
            "visibility": "followers_only",
        },
        {
            "id": 3,
            "sport_type": "Ride",
            "start_date": "2026-10-01T15:00:00Z",
            "start_date_local": "2026-10-01T08:00:00Z",
            "distance": 20000.0,
            "moving_time": 2400,
            "elapsed_time": 2500,
            "total_elevation_gain": 150.0,
            "private": False,
            "visibility": "everyone",
        },
    ]

    weekly = summarize_metrics(activities, after="2026-11-01", group_by="week")
    assert weekly["activity_count"] == 2
    assert weekly["totals"]["private_count"] == 2
    assert weekly["totals"]["distance_km"] == 50.0
    # Time-weighted HR: (150*3600 + 165*1800) / 5400 = 155.0
    assert weekly["totals"]["avg_heartrate"] == 155.0
    assert weekly["totals"]["max_heartrate"] == 182.0
    assert weekly["totals"]["kilojoules"] == 792.0
    assert weekly["totals"]["suffer_score"] == 140.0
    assert "2026-W47" in weekly["groups"]
    assert set(weekly["sport_breakdown"].keys()) == {"GravelRide", "TrailRun"}
    assert "Everyone" in weekly["visibility_note"]

    monthly = summarize_metrics(activities, group_by="month")
    assert list(monthly["groups"].keys()) == ["2026-10", "2026-11"]

    by_sport = summarize_metrics(activities, group_by="sport")
    assert set(by_sport["groups"].keys()) == {"GravelRide", "TrailRun", "Ride"}

    total = summarize_metrics(activities, group_by="total")
    assert list(total["groups"].keys()) == ["total"]
    assert total["groups"]["total"]["activity_count"] == 3

    with pytest.raises(ValueError):
        summarize_metrics(activities, group_by="day")  # type: ignore[arg-type]


async def test_validate_gear_id_normalizers_and_pagination(tmp_path: Any) -> None:
    assert validate_gear_id(" b12345 ") == "b12345"
    for bad_gear in ("", " ", "../athlete", "a/b", "a\\b", "b1?x=1", "b1#f", "b%2f", "b\x00", "."):
        with pytest.raises(ValueError):
            validate_gear_id(bad_gear)

    ath = normalize_athlete(
        {
            "id": 7,
            "weight": 70.0,
            "bikes": [{"id": "b1", "name": "Road", "distance": 2000000.0}],
            "shoes": [{"id": "g1", "name": "Trail", "distance": 500000.0}],
            "clubs": [{"id": 9, "name": "Club", "sport_type": "cycling"}],
        }
    )
    assert ath["weight_lbs"] == 154.3
    assert ath["bikes"][0]["distance_km"] == 2000.0

    stats = normalize_athlete_stats(
        {"biggest_ride_distance": 100000.0, "recent_run_totals": {"count": 4, "distance": 40000.0}}
    )
    assert stats["biggest_ride_distance_km"] == 100.0
    assert stats["recent_run_totals"]["distance_km"] == 40.0

    lap_norm = normalize_lap({"id": 1, "distance": 1000.0, "moving_time": 250})
    assert lap_norm["pace_min_per_km"] == "4:10"
    assert normalize_gear({"id": "b1", "distance": 16093.44})["distance_mi"] == 10.0
    route_norm = normalize_route({"id": 1, "distance": 10000.0, "elevation_gain": 200.0})
    assert route_norm["distance_km"] == 10.0
    seg_norm = normalize_segment({"id": 2, "distance": 1000.0, "total_elevation_gain": 50.0})
    assert seg_norm["distance_km"] == 1.0

    # Test pagination helper with StravaClient
    cfg = Config(
        config_dir=tmp_path / "cfg",
        cache_dir=tmp_path / "cache",
        api_base_url="https://mock.strava.test/api/v3",
        oauth_base_url="https://mock.strava.test/oauth",
    )
    from strava_mcp.config import ClientConfig

    save_client(cfg, ClientConfig(client_id="1", client_secret="sec"))
    save_tokens(
        cfg,
        TokenData(
            access_token="tok",
            refresh_token="ref",
            expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
            scope="read,activity:read_all",
        ),
    )

    pages_data = {
        1: [
            {"id": 1, "sport_type": "GravelRide", "type": "Ride"},
            {"id": 2, "sport_type": "RoadRide", "type": "Ride"},
            {"id": 3, "sport_type": "GravelRide", "type": "Ride"},
        ],
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        return httpx.Response(200, json=pages_data.get(page, []))

    async with StravaClient(cfg, transport=httpx.MockTransport(_handler)) as client:
        items, pages_fetched, has_more = await fetch_activities_paginated(
            client, sport_type="GravelRide", limit=1, max_pages=5
        )
        assert [a["id"] for a in items] == [1]
        assert pages_fetched == 1
        assert has_more is True
