"""Tier 1 opaque-box E2E happy-path tests for F6-F10 domain tools and CLI."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from conftest import (
    SeededConfig,
    SyncMCPSession,
    open_stdio_session,
    parse_tool_result,
    resolve_strava_mcp_cmd,
)
from mock_strava import DEFAULT_ATHLETE_ID, MockStravaServer

_STREAM_KEYS = frozenset(
    {
        "time",
        "distance",
        "latlng",
        "altitude",
        "velocity_smooth",
        "heartrate",
        "cadence",
        "watts",
        "temp",
        "moving",
        "grade_smooth",
    }
)


def _call_ok(
    session: SyncMCPSession,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    res = session.call_tool(tool, arguments or {})
    assert not getattr(res, "is_error", False), (
        f"Tool {tool!r} returned unexpected error: {res!r}"
    )
    assert getattr(res, "structured_content", None) is not None, (
        f"Tool {tool!r} missing structured_content: {res!r}"
    )
    assert res.content, f"Tool {tool!r} missing text content: {res!r}"
    parsed_text = json.loads(res.content[0].text)
    assert isinstance(parsed_text, dict), (
        f"Tool {tool!r} text content is not a JSON object: {type(parsed_text)}"
    )
    return parse_tool_result(res)


def _extract_list(payload: dict[str, Any], *candidate_keys: str) -> list[dict[str, Any]]:
    for key in (*candidate_keys, "items", "results", "data"):
        val = payload.get(key)
        if isinstance(val, list):
            return [item for item in val if isinstance(item, dict)]
    raise AssertionError(
        f"Expected list under {(*candidate_keys, 'items', 'results', 'data')} in {payload!r}"
    )


def _all_numbers(obj: Any) -> list[float]:
    nums: list[float] = []
    if isinstance(obj, bool):
        return nums
    if isinstance(obj, (int, float)):
        nums.append(float(obj))
    elif isinstance(obj, dict):
        for val in obj.values():
            nums.extend(_all_numbers(val))
    elif isinstance(obj, list):
        for val in obj:
            nums.extend(_all_numbers(val))
    return nums


def _has_approx_number(obj: Any, target: float, rel_tol: float = 0.02) -> bool:
    return any(
        math.isclose(val, target, rel_tol=rel_tol, abs_tol=1e-6)
        for val in _all_numbers(obj)
    )


def _assert_no_raw_secrets(text: str, seeded: SeededConfig) -> None:
    for secret in (
        seeded.access_token,
        seeded.refresh_token,
        seeded.client_secret,
        seeded.mcp_token,
    ):
        assert secret not in text, "Raw secret leaked in output"


def _stream_keys_arg(session: SyncMCPSession, keys: list[str]) -> list[str] | str:
    tools_res = session.list_tools()
    for tool in getattr(tools_res, "tools", []):
        if getattr(tool, "name", None) == "strava_get_activity_streams":
            schema = getattr(tool, "inputSchema", {}) or {}
            prop = (schema.get("properties") or {}).get("keys") or {}
            if prop.get("type") == "string":
                return ",".join(keys)
            for option in prop.get("anyOf", []):
                if isinstance(option, dict) and option.get("type") == "string":
                    return ",".join(keys)
            return keys
    return keys


_SCALAR_METRIC_KEYS = frozenset(
    {
        "count",
        "activity_count",
        "activities_count",
        "activities",
        "distance",
        "distance_m",
        "distance_meters",
        "total_distance",
        "distance_km",
        "moving_time",
        "moving_time_s",
        "moving_time_seconds",
        "total_moving_time",
        "elevation_gain",
        "total_elevation_gain",
        "elevation_m",
        "elevation",
    }
)


def _has_scalar_metric_keys(obj: dict[str, Any]) -> bool:
    return any(
        isinstance(obj.get(k), (int, float)) and not isinstance(obj.get(k), bool)
        for k in _SCALAR_METRIC_KEYS
    )


def _extract_metric_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    list_keys = (
        "groups",
        "buckets",
        "periods",
        "rollups",
        "weeks",
        "months",
        "sports",
        "by_sport",
        "summary",
        "items",
        "results",
        "data",
        "breakdown",
    )
    for key in list_keys:
        val = payload.get(key)
        if isinstance(val, list) and val:
            sub_items = [item for item in val if isinstance(item, dict)]
            if sub_items:
                return sub_items
    for key in list_keys:
        val = payload.get(key)
        if isinstance(val, dict):
            if _has_scalar_metric_keys(val):
                return [val]
            sub_groups = [
                {**item, "_group_key": gkey}
                for gkey, item in val.items()
                if isinstance(item, dict)
            ]
            if sub_groups:
                return sub_groups
    for key in ("totals", "total"):
        val = payload.get(key)
        if isinstance(val, dict):
            if _has_scalar_metric_keys(val):
                return [val]
            sub_groups = [
                {**item, "_group_key": gkey}
                for gkey, item in val.items()
                if isinstance(item, dict)
            ]
            if sub_groups:
                return sub_groups
            return [val]
    return [payload]


def _extract_streams_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    container: Any = payload
    for key in ("streams", "summaries", "summary"):
        if isinstance(payload.get(key), (dict, list)):
            container = payload[key]
            break
    if isinstance(container, dict):
        found = {
            k: v
            for k, v in container.items()
            if isinstance(v, dict) and k in _STREAM_KEYS
        }
        if found:
            return found
    if isinstance(container, list):
        out: dict[str, dict[str, Any]] = {}
        for item in container:
            if isinstance(item, dict):
                k = item.get("type") or item.get("key") or item.get("name")
                if isinstance(k, str) and k in _STREAM_KEYS:
                    out[k] = item
        if out:
            return out
    raise AssertionError(f"Could not extract streams map from {payload!r}")


def _metric_field(bucket: dict[str, Any], *keys: str) -> float:
    sources = [bucket]
    for sub in ("metrics", "totals", "total", "stats"):
        if isinstance(bucket.get(sub), dict):
            sources.append(bucket[sub])
    for src in sources:
        for key in keys:
            val = src.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    raise AssertionError(f"None of {keys} found in metric bucket {bucket!r}")


# --- Feature F6: Athlete, Stats & Zones ---


def test_f6_get_athlete_detailed_with_profile_read_all(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    data = _call_ok(stdio_session, "strava_get_athlete")
    athlete = data.get("athlete") if isinstance(data.get("athlete"), dict) else data
    assert athlete["id"] == DEFAULT_ATHLETE_ID
    assert athlete["firstname"] == "Alex"
    assert athlete["lastname"] == "Rider"
    assert athlete["ftp"] == 285
    assert athlete["weight"] == 72.5
    assert athlete["measurement_preference"] == "meters"
    bike_ids = {b["id"] for b in athlete.get("bikes", [])}
    shoe_ids = {s["id"] for s in athlete.get("shoes", [])}
    club_ids = {c["id"] for c in athlete.get("clubs", [])}
    assert "b1001" in bike_ids
    assert "g2001" in shoe_ids
    assert {5001, 5002}.issubset(club_ids)
    reqs = [
        r for r in mock_strava.get_requests("/api/v3/athlete", "GET") if r.path == "/api/v3/athlete"
    ]
    assert len(reqs) >= 1


def test_f6_get_athlete_summary_with_read_only_token(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seed_config(scope="read")
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]
    with open_stdio_session(cmd, isolated_env, tmp_path / "stdio_stderr.log") as session:
        data = _call_ok(session, "strava_get_athlete")
    athlete = data.get("athlete") if isinstance(data.get("athlete"), dict) else data
    assert athlete["id"] == DEFAULT_ATHLETE_ID
    assert athlete["firstname"] == "Alex"
    assert athlete["username"] == "alex_rider"
    for detailed_key in ("ftp", "weight", "bikes", "shoes"):
        assert not athlete.get(detailed_key)
    reqs = [
        r for r in mock_strava.get_requests("/api/v3/athlete", "GET") if r.path == "/api/v3/athlete"
    ]
    assert any(r.status_code == 200 for r in reqs)


def test_f6_get_athlete_stats_totals_and_everyone_caveat(
    mcp_http_client: SyncMCPSession,
) -> None:
    data = _call_ok(mcp_http_client, "strava_get_athlete_stats")
    stats = data.get("stats") if isinstance(data.get("stats"), dict) else data
    recent_ride = stats["recent_ride_totals"]
    assert recent_ride["count"] == 1
    assert math.isclose(recent_ride["distance"], 64373.8, rel_tol=0.01) or _has_approx_number(
        recent_ride, 64.37, rel_tol=0.02
    )
    assert recent_ride["moving_time"] == 8100
    all_ride = stats["all_ride_totals"]
    assert all_ride["count"] == 2
    assert math.isclose(all_ride["distance"], 114373.8, rel_tol=0.01) or _has_approx_number(
        all_ride, 114.37, rel_tol=0.02
    )
    assert all_ride["moving_time"] == 13500
    assert stats["all_run_totals"]["count"] == 0
    assert stats["all_swim_totals"]["count"] == 1
    assert _has_approx_number(stats, 64373.8, rel_tol=0.01) or _has_approx_number(
        stats, 64.37, rel_tol=0.02
    )
    serialized = json.dumps(data).lower()
    assert any(kw in serialized for kw in ("everyone", "public", "visibility", "caveat", "private"))


def test_f6_get_athlete_zones_hr_and_power(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    data = _call_ok(stdio_session, "strava_get_athlete_zones")
    zones_obj = data.get("zones") if isinstance(data.get("zones"), dict) else data
    hr = zones_obj["heart_rate"]
    assert hr["custom_zones"] is False
    assert len(hr["zones"]) == 5
    assert hr["zones"][0] == {"min": 0, "max": 122}
    assert hr["zones"][-1] == {"min": 176, "max": -1}
    power = zones_obj["power"]
    assert len(power["zones"]) == 7
    assert power["zones"][0] == {"min": 0, "max": 156}
    assert power["zones"][-1] == {"min": 428, "max": -1}
    reqs = mock_strava.get_requests("/api/v3/athlete/zones", "GET")
    assert len(reqs) >= 1


def test_f6_cli_stats_command(
    seed_config: Callable[..., SeededConfig],
    run_cli: Callable[..., Any],
) -> None:
    seeded = seed_config()
    proc = run_cli("stats", check=True)
    combined = proc.stdout + proc.stderr
    assert any(tok in combined for tok in ("114.37", "114.4", "114373", "114,373", "64.37", "40.0"))
    assert "2" in combined
    _assert_no_raw_secrets(combined, seeded)


# --- Feature F7: Activities List & Detail ---


def test_f7_list_activities_default_all_visibilities(
    stdio_session: SyncMCPSession,
) -> None:
    data = _call_ok(stdio_session, "strava_list_activities", {"limit": 20})
    items = _extract_list(data, "activities")
    ids = [a["id"] for a in items]
    assert ids == [9001, 9002, 9003, 9004, 9005, 9006, 9007]
    sport_types = [a["sport_type"] for a in items]
    assert sport_types == [
        "GravelRide",
        "TrailRun",
        "MountainBikeRide",
        "WeightTraining",
        "Run",
        "Ride",
        "Swim",
    ]
    serialized = json.dumps(data)
    assert "mock_full_poly_" not in serialized


def test_f7_list_activities_relative_date_filters(
    mcp_http_client: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    windows = [
        ({"after": "7d", "refresh": True}, {9001, 9002, 9003}),
        ({"after": "30d", "refresh": True}, {9001, 9002, 9003, 9004, 9005}),
        ({"after": "3m", "refresh": True}, {9001, 9002, 9003, 9004, 9005, 9006}),
        ({"after": "1y", "refresh": True}, {9001, 9002, 9003, 9004, 9005, 9006, 9007}),
        ({"before": "7d", "refresh": True}, {9004, 9005, 9006, 9007}),
    ]
    for args, expected_ids in windows:
        data = _call_ok(mcp_http_client, "strava_list_activities", args)
        got_ids = {a["id"] for a in _extract_list(data, "activities")}
        assert got_ids == expected_ids, f"Window {args} expected {expected_ids}, got {got_ids}"
    reqs = mock_strava.get_requests("/api/v3/athlete/activities", "GET")
    assert any("after" in r.query for r in reqs)
    assert any("before" in r.query for r in reqs)
    for r in reqs:
        for param in ("after", "before"):
            if param in r.query:
                int(r.query[param][0])


def test_f7_list_activities_iso_dates_and_sport_type(
    stdio_session: SyncMCPSession,
) -> None:
    for sport, expected_id in (
        ("GravelRide", 9001),
        ("TrailRun", 9002),
        ("MountainBikeRide", 9003),
    ):
        res = _call_ok(
            stdio_session,
            "strava_list_activities",
            {"sport_type": sport, "refresh": True},
        )
        ids = {a["id"] for a in _extract_list(res, "activities")}
        assert ids == {expected_id}

    today = datetime.now(UTC).date()
    after_iso = (today - timedelta(days=48)).isoformat()
    before_iso = (today - timedelta(days=40)).isoformat()
    iso_res = _call_ok(
        stdio_session,
        "strava_list_activities",
        {"after": after_iso, "before": before_iso, "refresh": True},
    )
    iso_items = _extract_list(iso_res, "activities")
    assert {a["id"] for a in iso_items} == {9006}
    assert iso_items[0]["name"] == "Willamette Valley 50K"


def test_f7_get_activity_compact_summary_default(
    stdio_session: SyncMCPSession,
) -> None:
    data = _call_ok(stdio_session, "strava_get_activity", {"activity_id": 9001})
    act = data.get("activity") if isinstance(data.get("activity"), dict) else data
    assert act["id"] == 9001
    assert act["name"] == "Morning Gravel Grind"
    assert act["sport_type"] == "GravelRide"
    assert _has_approx_number(act, 64373.8, rel_tol=0.01) or _has_approx_number(
        act, 64.37, rel_tol=0.02
    )
    assert _has_approx_number(act, 40.0, rel_tol=0.02)
    assert act["average_watts"] == 225.4
    assert act["average_heartrate"] == 152.0
    assert act["device_name"] == "Garmin Edge 840"
    serialized = json.dumps(data)
    assert "mock_full_poly_9001" not in serialized
    assert "Saltzman Road Gravel Climb (Lap 2)" not in serialized


def test_f7_get_activity_include_all_efforts_and_polyline(
    mcp_http_client: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    data = _call_ok(
        mcp_http_client,
        "strava_get_activity",
        {"activity_id": 9001, "include_all_efforts": True, "include_polyline": True},
    )
    serialized = json.dumps(data)
    assert "mock_full_poly_9001" in serialized
    assert "Saltzman Road Gravel Climb" in serialized
    assert "Saltzman Road Gravel Climb (Lap 2)" in serialized
    reqs = [
        r
        for r in mock_strava.get_requests("/api/v3/activities/9001", "GET")
        if r.path == "/api/v3/activities/9001"
    ]
    assert any(
        [v.lower() for v in r.query.get("include_all_efforts", [])] == ["true"] for r in reqs
    )


def test_f7_cli_activities_and_activity_detail(
    seed_config: Callable[..., SeededConfig],
    run_cli: Callable[..., Any],
) -> None:
    seeded = seed_config()
    list_proc = run_cli(
        "activities",
        "--after",
        "30d",
        "--sport-type",
        "GravelRide",
        check=True,
    )
    list_out = list_proc.stdout + list_proc.stderr
    assert "9001" in list_out or "Morning Gravel Grind" in list_out
    assert "9002" not in list_out
    assert "9006" not in list_out
    _assert_no_raw_secrets(list_out, seeded)

    detail_proc = run_cli("activity", "9001", check=True)
    detail_out = detail_proc.stdout + detail_proc.stderr
    assert "Morning Gravel Grind" in detail_out
    assert "GravelRide" in detail_out
    assert any(tok in detail_out for tok in ("64.37", "64.4", "64373", "40.0"))
    _assert_no_raw_secrets(detail_out, seeded)


# --- Feature F8: Activity Laps, Zones & Streams ---


def test_f8_get_activity_laps(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    data = _call_ok(stdio_session, "strava_get_activity_laps", {"activity_id": 9001})
    laps = _extract_list(data, "laps")
    assert len(laps) == 3
    assert [lap["id"] for lap in laps] == [90011, 90012, 90013]
    for lap in laps:
        assert lap["moving_time"] == 2700
        assert _has_approx_number(lap, 21457.9, rel_tol=0.02) or _has_approx_number(
            lap, 21.46, rel_tol=0.02
        )
        assert lap["average_watts"] == 225.4
        assert lap["average_heartrate"] == 152.0
    assert len(mock_strava.get_requests("/api/v3/activities/9001/laps", "GET")) >= 1


def test_f8_get_activity_zones(
    mcp_http_client: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    data = _call_ok(mcp_http_client, "strava_get_activity_zones", {"activity_id": 9001})
    zones = _extract_list(data, "zones")
    by_type = {z["type"]: z for z in zones}
    assert {"heartrate", "power"}.issubset(by_type)
    hr_zone = by_type["heartrate"]
    assert hr_zone["score"] == 82
    assert len(hr_zone["distribution_buckets"]) == 5
    assert sum(b["time"] for b in hr_zone["distribution_buckets"]) == 8100
    pwr_zone = by_type["power"]
    assert len(pwr_zone["distribution_buckets"]) == 5
    assert sum(b["time"] for b in pwr_zone["distribution_buckets"]) == 8100
    assert len(mock_strava.get_requests("/api/v3/activities/9001/zones", "GET")) >= 1


def test_f8_get_activity_streams_keys_and_max_points_downsampling(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    keys = ["time", "heartrate", "watts", "altitude"]
    data = _call_ok(
        stdio_session,
        "strava_get_activity_streams",
        {
            "activity_id": 9001,
            "keys": _stream_keys_arg(stdio_session, keys),
            "max_points": 50,
        },
    )
    streams = _extract_streams_map(data)
    assert set(streams.keys()) == set(keys)
    for k in keys:
        series = streams[k]
        pts = series["data"]
        assert 2 <= len(pts) <= 50
        assert series.get("original_size") == 1200
    assert streams["time"]["data"][0] == 0
    assert streams["time"]["data"][-1] >= 7900
    for unrequested in ("latlng", "temp", "grade_smooth"):
        assert unrequested not in streams
    reqs = mock_strava.get_requests("/api/v3/activities/9001/streams", "GET")
    assert any(r.query.get("key_by_type") == ["true"] for r in reqs)


def test_f8_get_activity_streams_summary_only(
    mcp_http_client: SyncMCPSession,
) -> None:
    keys = ["heartrate", "watts", "altitude"]
    data = _call_ok(
        mcp_http_client,
        "strava_get_activity_streams",
        {
            "activity_id": 9001,
            "keys": _stream_keys_arg(mcp_http_client, keys),
            "summary_only": True,
        },
    )
    serialized = json.dumps(data)
    assert len(serialized) < 8000
    streams = _extract_streams_map(data)
    assert set(streams.keys()) == set(keys)
    for key, exp_min, exp_max, exp_avg in (
        ("heartrate", 134.0, 162.0, 148.0),
        ("watts", 180.0, 270.0, 225.0),
        ("altitude", 40.0, 200.0, 125.0),
    ):
        s = streams[key]
        assert not isinstance(s.get("data"), list) or len(s["data"]) < 50
        assert _has_approx_number(s, exp_min, rel_tol=0.05)
        assert _has_approx_number(s, exp_max, rel_tol=0.05)
        assert _has_approx_number(s, exp_avg, rel_tol=0.05)


def test_f8_get_activity_streams_metadata_and_units(
    stdio_session: SyncMCPSession,
) -> None:
    keys = ["time", "distance", "altitude", "velocity_smooth"]
    data = _call_ok(
        stdio_session,
        "strava_get_activity_streams",
        {
            "activity_id": 9002,
            "keys": _stream_keys_arg(stdio_session, keys),
            "max_points": 30,
        },
    )
    streams = _extract_streams_map(data)
    assert set(streams.keys()) == set(keys)
    for k in keys:
        s = streams[k]
        assert len(s["data"]) <= 30
        assert s.get("original_size") == 300
        assert s.get("resolution") == "high"
    assert streams["time"].get("series_type") == "time"
    assert streams["distance"].get("series_type") == "distance"
    assert _has_approx_number(streams["distance"], 16093.4, rel_tol=0.04) or _has_approx_number(
        data, 16.09, rel_tol=0.04
    )


# --- Feature F9: Derived Training Metrics ---


def test_f9_summarize_metrics_group_by_total(
    stdio_session: SyncMCPSession,
) -> None:
    data = _call_ok(stdio_session, "strava_summarize_metrics", {"group_by": "total"})
    groups = _extract_metric_groups(data)
    total_count = sum(
        _metric_field(g, "count", "activity_count", "activities_count", "activities")
        for g in groups
    )
    assert total_count == 7
    assert _has_approx_number(data, 171467.2, rel_tol=0.01) or _has_approx_number(
        data, 171.47, rel_tol=0.02
    )
    assert _has_approx_number(data, 106.54, rel_tol=0.02)
    assert _has_approx_number(data, 32700, rel_tol=0.01)
    assert _has_approx_number(data, 2635.0, rel_tol=0.01) or _has_approx_number(
        data, 8645.0, rel_tol=0.02
    )


def test_f9_summarize_metrics_group_by_week(
    mcp_http_client: SyncMCPSession,
) -> None:
    data = _call_ok(mcp_http_client, "strava_summarize_metrics", {"group_by": "week"})
    groups = _extract_metric_groups(data)
    assert len(groups) >= 2
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in groups
        )
        == 7
    )
    assert (
        sum(
            _metric_field(
                g, "moving_time", "moving_time_s", "moving_time_seconds", "total_moving_time"
            )
            for g in groups
        )
        == 32700
    )
    dist_sum = sum(
        _metric_field(
            g, "distance", "distance_m", "distance_meters", "total_distance", "distance_km"
        )
        for g in groups
    )
    assert math.isclose(dist_sum, 171467.2, rel_tol=0.01) or math.isclose(
        dist_sum, 171.47, rel_tol=0.02
    )
    elev_sum = sum(
        _metric_field(g, "elevation_gain", "total_elevation_gain", "elevation_m", "elevation")
        for g in groups
    )
    assert math.isclose(elev_sum, 2635.0, rel_tol=0.01) or math.isclose(
        elev_sum, 8645.0, rel_tol=0.02
    )


def test_f9_summarize_metrics_group_by_month(
    stdio_session: SyncMCPSession,
) -> None:
    data = _call_ok(stdio_session, "strava_summarize_metrics", {"group_by": "month"})
    groups = _extract_metric_groups(data)
    assert len(groups) >= 2
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in groups
        )
        == 7
    )
    assert (
        sum(
            _metric_field(
                g, "moving_time", "moving_time_s", "moving_time_seconds", "total_moving_time"
            )
            for g in groups
        )
        == 32700
    )
    dist_sum = sum(
        _metric_field(
            g, "distance", "distance_m", "distance_meters", "total_distance", "distance_km"
        )
        for g in groups
    )
    assert math.isclose(dist_sum, 171467.2, rel_tol=0.01) or math.isclose(
        dist_sum, 171.47, rel_tol=0.02
    )
    elev_sum = sum(
        _metric_field(g, "elevation_gain", "total_elevation_gain", "elevation_m", "elevation")
        for g in groups
    )
    assert math.isclose(elev_sum, 2635.0, rel_tol=0.01) or math.isclose(
        elev_sum, 8645.0, rel_tol=0.02
    )


def _bucket_has_sport(bucket: dict[str, Any], sport_name: str) -> bool:
    def _has_exact(obj: Any) -> bool:
        if isinstance(obj, str):
            return obj == sport_name
        if isinstance(obj, dict):
            return any(k == sport_name or _has_exact(v) for k, v in obj.items())
        if isinstance(obj, list):
            return any(_has_exact(v) for v in obj)
        return False

    return _has_exact(bucket) or sport_name in json.dumps(bucket)


def test_f9_summarize_metrics_group_by_sport(
    mcp_http_client: SyncMCPSession,
) -> None:
    data = _call_ok(mcp_http_client, "strava_summarize_metrics", {"group_by": "sport"})
    groups = _extract_metric_groups(data)
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in groups
        )
        == 7
    )
    for sport_name, expected_m, expected_km in (
        ("GravelRide", 64373.8, 64.37),
        ("TrailRun", 16093.4, 16.09),
        ("MountainBikeRide", 28500.0, 28.5),
        ("WeightTraining", 0.0, 0.0),
        ("Run", 10000.0, 10.0),
        ("Ride", 50000.0, 50.0),
        ("Swim", 2500.0, 2.5),
    ):
        exact = [
            g
            for g in groups
            if any(
                v == sport_name
                for v in g.values()
                if isinstance(v, str)
            )
            or (
                isinstance(g.get("_group_key"), str)
                and g["_group_key"] == sport_name
            )
        ]
        matching = exact if exact else [g for g in groups if _bucket_has_sport(g, sport_name)]
        assert len(matching) >= 1, f"Missing group for sport {sport_name!r} in {groups!r}"
        bucket = matching[0]
        assert _has_approx_number(bucket, expected_m, rel_tol=0.01) or _has_approx_number(
            bucket, expected_km, rel_tol=0.02
        ), (
            f"Sport {sport_name!r} bucket {bucket!r} missing expected "
            f"distance {expected_m}m/{expected_km}km"
        )
        if sport_name == "WeightTraining":
            dist = _metric_field(
                bucket,
                "distance",
                "distance_m",
                "distance_meters",
                "total_distance",
                "distance_km",
            )
            assert dist == 0.0, f"WeightTraining distance expected 0.0, got {dist}"


def test_f9_summarize_metrics_filtered_dates_sport_and_private(
    stdio_session: SyncMCPSession,
) -> None:
    recent = _call_ok(
        stdio_session,
        "strava_summarize_metrics",
        {"after": "30d", "group_by": "total"},
    )
    recent_groups = _extract_metric_groups(recent)
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in recent_groups
        )
        == 5
    )
    assert _has_approx_number(recent, 118967.2, rel_tol=0.01) or _has_approx_number(
        recent, 118.97, rel_tol=0.02
    )
    assert _has_approx_number(recent, 73.92, rel_tol=0.02)
    assert _has_approx_number(recent, 24600, rel_tol=0.01)
    assert _has_approx_number(recent, 2235.0, rel_tol=0.01)

    mtb = _call_ok(
        stdio_session,
        "strava_summarize_metrics",
        {
            "after": "10d",
            "before": "3d",
            "sport_type": "MountainBikeRide",
            "group_by": "total",
            "refresh": True,
        },
    )
    mtb_groups = _extract_metric_groups(mtb)
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in mtb_groups
        )
        == 1
    )
    assert _has_approx_number(mtb, 28500.0, rel_tol=0.01) or _has_approx_number(
        mtb, 28.5, rel_tol=0.02
    )
    assert _has_approx_number(mtb, 17.71, rel_tol=0.02)
    assert _has_approx_number(mtb, 5400, rel_tol=0.01)
    assert _has_approx_number(mtb, 780.0, rel_tol=0.01)


# --- Feature F10: Gear, Routes, Segments & Clubs ---


def test_f10_get_gear_bike_and_shoe(
    stdio_session: SyncMCPSession,
) -> None:
    bike_data = _call_ok(stdio_session, "strava_get_gear", {"gear_id": "b1001"})
    bike = bike_data.get("gear") if isinstance(bike_data.get("gear"), dict) else bike_data
    assert bike["id"] == "b1001"
    assert bike["name"] == "Canyon Grail CF SL"
    assert bike["brand_name"] == "Canyon"
    assert bike["model_name"] == "Grail CF SL 8"
    assert bike["primary"] is True
    assert _has_approx_number(bike_data, 4250000.0, rel_tol=0.01) or _has_approx_number(
        bike_data, 4250.0, rel_tol=0.01
    )
    assert _has_approx_number(bike_data, 2640.83, rel_tol=0.02)

    shoe_data = _call_ok(stdio_session, "strava_get_gear", {"gear_id": "g2001"})
    shoe = shoe_data.get("gear") if isinstance(shoe_data.get("gear"), dict) else shoe_data
    assert shoe["id"] == "g2001"
    assert shoe["name"] == "Hoka Speedgoat 5"
    assert shoe["brand_name"] == "Hoka"
    assert _has_approx_number(shoe_data, 620000.0, rel_tol=0.01) or _has_approx_number(
        shoe_data, 620.0, rel_tol=0.01
    )
    assert _has_approx_number(shoe_data, 385.25, rel_tol=0.02)


def _find_xml_text(obj: Any, marker: str) -> str:
    if isinstance(obj, str) and marker in obj:
        return obj
    if isinstance(obj, dict):
        for val in obj.values():
            found = _find_xml_text(val, marker)
            if found:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = _find_xml_text(val, marker)
            if found:
                return found
    return ""


def test_f10_list_routes_and_get_route_gpx(
    mcp_http_client: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    listed = _call_ok(mcp_http_client, "strava_list_routes")
    routes = _extract_list(listed, "routes")
    by_id = {int(r["id"]): r for r in routes}
    assert {7001, 7002}.issubset(by_id)
    assert by_id[7001]["name"] == "Forest Park Gravel Loop"
    assert by_id[7002]["name"] == "Private Backcountry Trail"

    detail = _call_ok(
        mcp_http_client,
        "strava_get_route",
        {"route_id": 7001, "include_export": "gpx"},
    )
    assert _has_approx_number(detail, 45200.0, rel_tol=0.01) or _has_approx_number(
        detail, 45.2, rel_tol=0.02
    )
    assert _has_approx_number(detail, 810.0, rel_tol=0.01)
    gpx_xml = _find_xml_text(detail, "<gpx")
    assert gpx_xml, f"Expected GPX XML in {detail!r}"
    root = ET.fromstring(gpx_xml)
    assert root.tag.endswith("gpx")
    assert "Forest Park Gravel Loop" in gpx_xml
    assert len(mock_strava.get_requests("/api/v3/routes/7001/export_gpx", "GET")) >= 1


def test_f10_get_route_none_and_tcx_export(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    plain = _call_ok(
        stdio_session,
        "strava_get_route",
        {"route_id": 7002, "include_export": "none"},
    )
    route = plain.get("route") if isinstance(plain.get("route"), dict) else plain
    assert route["private"] is True
    assert _has_approx_number(plain, 18500.0, rel_tol=0.01) or _has_approx_number(
        plain, 18.5, rel_tol=0.02
    )
    assert _has_approx_number(plain, 540.0, rel_tol=0.01)
    assert mock_strava.get_requests("/api/v3/routes/7002/export_tcx", "GET") == []
    assert mock_strava.get_requests("/api/v3/routes/7002/export_gpx", "GET") == []

    exported = _call_ok(
        stdio_session,
        "strava_get_route",
        {"route_id": 7002, "include_export": "tcx", "refresh": True},
    )
    tcx_xml = _find_xml_text(exported, "<TrainingCenterDatabase")
    assert tcx_xml, f"Expected TCX XML in {exported!r}"
    root = ET.fromstring(tcx_xml)
    assert root.tag.endswith("TrainingCenterDatabase")
    assert "Private Backcountry Trail" in tcx_xml
    assert len(mock_strava.get_requests("/api/v3/routes/7002/export_tcx", "GET")) >= 1


def test_f10_list_starred_segments_and_get_segment(
    stdio_session: SyncMCPSession,
) -> None:
    starred = _call_ok(stdio_session, "strava_list_starred_segments")
    segs = _extract_list(starred, "segments")
    by_id = {int(s["id"]): s for s in segs}
    assert {8001, 8002}.issubset(by_id)
    assert by_id[8001]["name"] == "Saltzman Road Gravel Climb"
    assert by_id[8002]["name"] == "Wildwood Secret Sprint"

    seg_data = _call_ok(stdio_session, "strava_get_segment", {"segment_id": 8001})
    seg = seg_data.get("segment") if isinstance(seg_data.get("segment"), dict) else seg_data
    assert seg["distance"] == 4200.0
    assert seg["average_grade"] == 6.4
    assert seg["maximum_grade"] == 12.1
    assert seg["total_elevation_gain"] == 269.0
    stats = seg["athlete_segment_stats"]
    assert stats["pr_elapsed_time"] == 980
    assert stats["effort_count"] == 8


def test_f10_list_segment_efforts(
    mcp_http_client: SyncMCPSession,
) -> None:
    eff1 = _call_ok(mcp_http_client, "strava_list_segment_efforts", {"segment_id": 8001})
    list1 = _extract_list(eff1, "efforts", "segment_efforts")
    by_id1 = {int(e["id"]): e for e in list1}
    assert {6001, 6003}.issubset(by_id1)
    assert by_id1[6001]["elapsed_time"] == 985
    assert by_id1[6001]["average_watts"] == 265.0
    assert by_id1[6001]["pr_rank"] == 2
    assert by_id1[6003]["elapsed_time"] == 1030
    assert by_id1[6003]["average_watts"] == 245.0

    eff2 = _call_ok(mcp_http_client, "strava_list_segment_efforts", {"segment_id": 8002})
    list2 = _extract_list(eff2, "efforts", "segment_efforts")
    by_id2 = {int(e["id"]): e for e in list2}
    assert 6002 in by_id2
    assert by_id2[6002]["elapsed_time"] == 430
    assert by_id2[6002]["pr_rank"] == 1


def test_f10_list_clubs(
    stdio_session: SyncMCPSession,
) -> None:
    data = _call_ok(stdio_session, "strava_list_clubs")
    clubs = _extract_list(data, "clubs")
    by_id = {int(c["id"]): c for c in clubs}
    assert set(by_id.keys()) == {5001, 5002}
    assert by_id[5001]["name"] == "Cascade Gravel Club"
    assert by_id[5001]["sport_type"] == "cycling"
    assert by_id[5001]["member_count"] == 240
    assert by_id[5001]["private"] is False
    assert by_id[5002]["name"] == "PDX Trail Runners"
    assert by_id[5002]["sport_type"] == "running"
    assert by_id[5002]["member_count"] == 85
    assert by_id[5002]["private"] is True
