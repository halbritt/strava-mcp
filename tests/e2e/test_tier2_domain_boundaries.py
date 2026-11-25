"""Tier 2 opaque-box E2E domain boundary & error tests for F6-F10."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conftest import (
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


def _open_session(
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> AbstractContextManager[SyncMCPSession]:
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]
    return open_stdio_session(cmd, isolated_env, tmp_path / "stdio_stderr.log")


def _extract_error_code_and_text(
    result: Any,
) -> tuple[dict[str, Any] | None, str | None, str]:
    assert getattr(result, "is_error", False) is True, (
        f"Expected is_error=True, got {result!r}"
    )
    raw_text = ""
    content = getattr(result, "content", None) or []
    if content:
        raw_text = getattr(content[0], "text", "") or ""
    parsed: dict[str, Any] | None = None
    sc = getattr(result, "structured_content", None)
    if isinstance(sc, dict):
        parsed = sc
    elif raw_text.strip():
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded
    code: str | None = None
    if parsed is not None:
        if isinstance(parsed.get("code"), str):
            code = parsed["code"]
        elif isinstance(parsed.get("error"), dict) and isinstance(
            parsed["error"].get("code"), str
        ):
            code = parsed["error"]["code"]
        combined_text = json.dumps(parsed)
    else:
        combined_text = raw_text
    return parsed, code, combined_text


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


def _streams_keys_arg(session: SyncMCPSession, keys: list[str]) -> list[str] | str:
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
    return {}


def _synthetic_summary_activity(aid: int, start_epoch: int) -> dict[str, Any]:
    iso_utc = datetime.fromtimestamp(start_epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    iso_loc = datetime.fromtimestamp(start_epoch - 28800, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": aid,
        "resource_state": 2,
        "athlete": {"id": DEFAULT_ATHLETE_ID, "resource_state": 1},
        "name": f"Synthetic Ride {aid}",
        "distance": 15000.0,
        "moving_time": 1800,
        "elapsed_time": 1900,
        "total_elevation_gain": 120.0,
        "type": "Ride",
        "sport_type": "Ride",
        "start_date": iso_utc,
        "start_date_local": iso_loc,
        "timezone": "(GMT-08:00) America/Los_Angeles",
        "utc_offset": -28800.0,
        "start_latlng": [45.552, -122.742],
        "end_latlng": [45.568, -122.771],
        "achievement_count": 1,
        "kudos_count": 0,
        "comment_count": 0,
        "athlete_count": 1,
        "photo_count": 0,
        "map": {
            "id": f"a{aid}",
            "summary_polyline": f"mock_sum_poly_{aid}",
            "polyline": f"mock_full_poly_{aid}",
            "resource_state": 2,
        },
        "trainer": False,
        "commute": False,
        "manual": False,
        "private": False,
        "visibility": "everyone",
        "gear_id": "b1001",
        "average_speed": 8.33,
        "max_speed": 11.5,
        "average_cadence": 84.0,
        "average_watts": 220.0,
        "weighted_average_watts": 235.0,
        "device_watts": True,
        "kilojoules": 396.0,
        "has_heartrate": True,
        "average_heartrate": 150.0,
        "max_heartrate": 172.0,
        "elev_high": 220.0,
        "elev_low": 100.0,
        "pr_count": 0,
    }


# --- Feature F6: Athlete, Stats & Zones ---


def test_f6_athlete_zones_read_only_insufficient_scope(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_get_athlete_zones", {})
    _, code, text = _extract_error_code_and_text(res)
    assert code == "insufficient_scope"
    assert "profile:read_all" in text
    assert "strava-mcp auth" in text


def test_f6_athlete_stats_everyone_only_vs_total_visibility(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    stats_res = stdio_session.call_tool("strava_get_athlete_stats", {})
    assert not getattr(stats_res, "is_error", False)
    stats_data = parse_tool_result(stats_res)
    stats = stats_data.get("stats") if isinstance(stats_data.get("stats"), dict) else stats_data
    assert stats["all_ride_totals"]["count"] == 2
    assert stats["all_run_totals"]["count"] == 0
    assert stats["all_swim_totals"]["count"] == 1

    acts_res = stdio_session.call_tool("strava_list_activities", {"limit": 20})
    assert not getattr(acts_res, "is_error", False)
    acts = _extract_list(parse_tool_result(acts_res), "activities")
    assert len(acts) == 7
    ride_count = sum(
        1
        for a in acts
        if a.get("type") == "Ride"
        or a.get("sport_type") in {"Ride", "GravelRide", "MountainBikeRide"}
    )
    run_count = sum(
        1 for a in acts if a.get("type") == "Run" or a.get("sport_type") in {"Run", "TrailRun"}
    )
    assert ride_count == 3
    assert run_count == 2

    reqs = mock_strava.get_requests(f"/api/v3/athletes/{DEFAULT_ATHLETE_ID}/stats", "GET")
    assert any(r.status_code == 200 for r in reqs)


def test_f6_athlete_401_triggers_transparent_refresh(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seed_config()
    mock_strava.expire_access_token()
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_get_athlete", {})
    assert not getattr(res, "is_error", False)
    data = parse_tool_result(res)
    athlete = data.get("athlete") if isinstance(data.get("athlete"), dict) else data
    assert athlete["id"] == DEFAULT_ATHLETE_ID
    refresh_reqs = mock_strava.get_requests("/oauth/token", "POST")
    assert len([r for r in refresh_reqs if r.status_code == 200]) == 1
    athlete_reqs = [
        r for r in mock_strava.get_requests("/api/v3/athlete", "GET") if r.path == "/api/v3/athlete"
    ]
    assert [r.status_code for r in athlete_reqs] == [401, 200]


def test_f6_athlete_invalid_token_dead_refresh_returns_auth_error(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config(
        access_token="invalid-tok",
        refresh_token="dead-ref",
        expires_at=int(time.time()) + 7200,
        sync_mock=False,
    )
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_get_athlete", {})
        stderr_out = session.stderr_text
    _, code, text = _extract_error_code_and_text(res)
    assert code in {"auth_expired", "refresh_failed", "auth_missing"}
    athlete_reqs = [
        r for r in mock_strava.get_requests("/api/v3/athlete", "GET") if r.path == "/api/v3/athlete"
    ]
    assert any(r.status_code == 401 for r in athlete_reqs)
    token_reqs = mock_strava.get_requests("/oauth/token", "POST")
    assert any(r.status_code == 400 for r in token_reqs)
    for secret in ("invalid-tok", "dead-ref", seeded.client_secret, seeded.mcp_token):
        assert secret not in text
        assert secret not in stderr_out


def test_f6_athlete_zones_cloudfront_html_403_returns_upstream_error(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    mock_strava.inject_fault(
        fault_type="cloudfront_403",
        path_pattern=r"/api/v3/athlete/zones$",
        times=1,
    )
    res = stdio_session.call_tool("strava_get_athlete_zones", {})
    parsed, code, _ = _extract_error_code_and_text(res)
    assert parsed is not None
    assert code == "upstream_error"
    assert _has_approx_number(parsed, 403, rel_tol=0.0)
    reqs = mock_strava.get_requests("/api/v3/athlete/zones", "GET")
    assert len(reqs) == 1
    assert reqs[0].status_code == 403


def test_f6_athlete_empty_bikes_and_shoes_handled_cleanly(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    empty_athlete = {
        "id": DEFAULT_ATHLETE_ID,
        "username": "alex_rider",
        "resource_state": 3,
        "firstname": "Alex",
        "lastname": "Rider",
        "city": "Portland",
        "state": "Oregon",
        "country": "United States",
        "sex": "M",
        "premium": True,
        "summit": True,
        "created_at": "2020-01-15T10:00:00Z",
        "updated_at": "2026-11-25T00:00:00Z",
        "follower_count": 42,
        "friend_count": 38,
        "measurement_preference": "meters",
        "ftp": 285,
        "weight": 72.5,
        "clubs": [],
        "bikes": [],
        "shoes": [],
    }
    mock_strava.inject_fault(
        fault_type="status_error",
        path_pattern=r"/api/v3/athlete$",
        times=1,
        status_code=200,
        body=empty_athlete,
    )
    res = stdio_session.call_tool("strava_get_athlete", {})
    assert not getattr(res, "is_error", False)
    data = parse_tool_result(res)
    athlete = data.get("athlete") if isinstance(data.get("athlete"), dict) else data
    assert athlete["id"] == DEFAULT_ATHLETE_ID
    assert athlete.get("bikes", []) == []
    assert athlete.get("shoes", []) == []
    reqs = [
        r for r in mock_strava.get_requests("/api/v3/athlete", "GET") if r.path == "/api/v3/athlete"
    ]
    assert len(reqs) == 1
    assert reqs[0].status_code == 200


# --- Feature F7: Activities List & Detail ---


def test_f7_list_activities_read_only_insufficient_scope(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_list_activities", {})
    _, code, text = _extract_error_code_and_text(res)
    assert code == "insufficient_scope"
    assert "activity:read" in text
    assert "strava-mcp auth" in text


def test_f7_list_activities_per_page_200_and_multipage_limit(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    res_a = stdio_session.call_tool("strava_list_activities", {"per_page": 2, "limit": 5})
    assert not getattr(res_a, "is_error", False)
    items_a = _extract_list(parse_tool_result(res_a), "activities")
    assert [a["id"] for a in items_a] == [9001, 9002, 9003, 9004, 9005]
    reqs_a = mock_strava.get_requests("/api/v3/athlete/activities", "GET")
    assert len(reqs_a) == 3
    assert [r.query.get("page") for r in reqs_a] == [["1"], ["2"], ["3"]]
    assert all(r.query.get("per_page") == ["2"] for r in reqs_a)

    mock_strava.clear_requests()
    now = int(time.time())
    page1 = [_synthetic_summary_activity(10001 + i, now - (i + 1) * 3600) for i in range(200)]
    page2 = [
        _synthetic_summary_activity(10201 + i, now - (201 + i) * 3600) for i in range(15)
    ]
    mock_strava.inject_fault(
        fault_type="status_error",
        path_pattern=r"/api/v3/athlete/activities$",
        times=1,
        status_code=200,
        body=page1,  # type: ignore[arg-type]
    )
    mock_strava.inject_fault(
        fault_type="status_error",
        path_pattern=r"/api/v3/athlete/activities$",
        times=1,
        status_code=200,
        body=page2,  # type: ignore[arg-type]
    )
    res_b = stdio_session.call_tool(
        "strava_list_activities",
        {"per_page": 200, "limit": 210, "refresh": True},
    )
    assert not getattr(res_b, "is_error", False)
    items_b = _extract_list(parse_tool_result(res_b), "activities")
    assert len(items_b) == 210
    ids_b = [a["id"] for a in items_b]
    assert 10001 in ids_b
    assert 10200 in ids_b
    assert 10201 in ids_b
    assert 10210 in ids_b
    assert set(ids_b) == set(range(10001, 10211))
    reqs_b = mock_strava.get_requests("/api/v3/athlete/activities", "GET")
    assert len(reqs_b) == 2
    assert [r.query.get("page") for r in reqs_b] == [["1"], ["2"]]
    assert all(r.query.get("per_page") == ["200"] for r in reqs_b)


def test_f7_list_activities_empty_result_sets(
    stdio_session: SyncMCPSession,
) -> None:
    future_res = stdio_session.call_tool("strava_list_activities", {"after": "2099-01-01"})
    assert not getattr(future_res, "is_error", False)
    assert _extract_list(parse_tool_result(future_res), "activities") == []

    rowing_res = stdio_session.call_tool(
        "strava_list_activities",
        {"sport_type": "Rowing", "refresh": True},
    )
    assert not getattr(rowing_res, "is_error", False)
    assert _extract_list(parse_tool_result(rowing_res), "activities") == []


def test_f7_list_activities_legacy_type_ride_vs_sport_type_mountain_bike(
    stdio_session: SyncMCPSession,
) -> None:
    mtb_res = stdio_session.call_tool("strava_list_activities", {"sport_type": "MountainBikeRide"})
    assert not getattr(mtb_res, "is_error", False)
    mtb_items = _extract_list(parse_tool_result(mtb_res), "activities")
    assert [a["id"] for a in mtb_items] == [9003]
    assert mtb_items[0]["sport_type"] == "MountainBikeRide"

    ride_res = stdio_session.call_tool(
        "strava_list_activities",
        {"sport_type": "Ride", "refresh": True},
    )
    assert not getattr(ride_res, "is_error", False)
    ride_ids = {a["id"] for a in _extract_list(parse_tool_result(ride_res), "activities")}
    assert 9006 in ride_ids
    assert not ({9002, 9004, 9005, 9007} & ride_ids)


def test_f7_get_activity_unknown_id_returns_not_found(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    res = stdio_session.call_tool("strava_get_activity", {"activity_id": 9999999})
    _, code, _ = _extract_error_code_and_text(res)
    assert code == "not_found"
    reqs = mock_strava.get_requests("/api/v3/activities/9999999", "GET")
    assert any(r.status_code == 404 for r in reqs)


def test_f7_list_activities_invalid_date_string_validation_error(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    res = stdio_session.call_tool("strava_list_activities", {"after": "not-a-valid-date"})
    _, _, text = _extract_error_code_and_text(res)
    lowered = text.lower()
    assert any(kw in lowered for kw in ("after", "date", "invalid", "format", "parse"))
    assert mock_strava.get_requests("/api/v3/athlete/activities", "GET") == []


# --- Feature F8: Activity Laps, Zones & Streams ---


def test_f8_get_activity_streams_manual_9004_404_graceful(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    keys = _streams_keys_arg(stdio_session, ["time", "distance"])
    res = stdio_session.call_tool(
        "strava_get_activity_streams",
        {"activity_id": 9004, "keys": keys},
    )
    reqs = mock_strava.get_requests("/api/v3/activities/9004/streams", "GET")
    assert any(r.status_code == 404 for r in reqs)
    if getattr(res, "is_error", False):
        parsed, code, _ = _extract_error_code_and_text(res)
        assert parsed is not None
        assert code == "not_found"
    else:
        data = parse_tool_result(res)
        assert _extract_streams_map(data) == {}


def test_f8_get_activity_streams_max_points_10_bounds_1200_pts(
    stdio_session: SyncMCPSession,
) -> None:
    keys = ["time", "distance", "heartrate", "watts"]
    res = stdio_session.call_tool(
        "strava_get_activity_streams",
        {
            "activity_id": 9001,
            "keys": _streams_keys_arg(stdio_session, keys),
            "max_points": 10,
        },
    )
    assert not getattr(res, "is_error", False)
    streams = _extract_streams_map(parse_tool_result(res))
    assert set(streams.keys()) == set(keys)
    for k in keys:
        assert 1 <= len(streams[k]["data"]) <= 10


def test_f8_get_activity_zones_non_subscriber_402_and_403(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    mock_strava.set_subscription(False, error_status=402)
    res_402 = stdio_session.call_tool("strava_get_activity_zones", {"activity_id": 9001})
    _, code_402, _ = _extract_error_code_and_text(res_402)
    assert code_402 == "subscription_required"

    mock_strava.set_subscription(False, error_status=403)
    res_403 = stdio_session.call_tool(
        "strava_get_activity_zones",
        {"activity_id": 9001, "refresh": True},
    )
    _, code_403, _ = _extract_error_code_and_text(res_403)
    assert code_403 == "subscription_required"

    reqs = mock_strava.get_requests("/api/v3/activities/9001/zones", "GET")
    assert [r.status_code for r in reqs] == [402, 403]


def test_f8_get_activity_laps_cloudfront_403_upstream_error(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    mock_strava.inject_fault(
        fault_type="cloudfront_403",
        path_pattern=r"/api/v3/activities/9001/laps$",
        times=1,
    )
    res = stdio_session.call_tool("strava_get_activity_laps", {"activity_id": 9001})
    parsed, code, _ = _extract_error_code_and_text(res)
    assert parsed is not None
    assert code == "upstream_error"
    assert _has_approx_number(parsed, 403, rel_tol=0.0)
    reqs = mock_strava.get_requests("/api/v3/activities/9001/laps", "GET")
    assert len(reqs) == 1
    assert reqs[0].status_code == 403


def test_f8_get_activity_streams_read_only_insufficient_scope(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool(
            "strava_get_activity_streams",
            {"activity_id": 9001, "keys": _streams_keys_arg(session, ["time", "heartrate"])},
        )
    _, code, text = _extract_error_code_and_text(res)
    assert code == "insufficient_scope"
    assert "activity:read" in text
    assert "strava-mcp auth" in text


def test_f8_get_activity_streams_only_me_with_activity_read_insufficient_scope(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read,activity:read")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool(
            "strava_get_activity_streams",
            {"activity_id": 9003, "keys": _streams_keys_arg(session, ["time", "heartrate"])},
        )
    _, code, text = _extract_error_code_and_text(res)
    assert code == "insufficient_scope"
    assert "activity:read_all" in text


# --- Feature F9: Derived Training Metrics ---


def test_f9_summarize_metrics_empty_date_window_zeroed_totals(
    stdio_session: SyncMCPSession,
) -> None:
    res = stdio_session.call_tool(
        "strava_summarize_metrics",
        {"after": "2099-01-01", "before": "2099-02-01", "group_by": "total"},
    )
    assert not getattr(res, "is_error", False)
    data = parse_tool_result(res)
    groups = _extract_metric_groups(data)
    assert len(groups) >= 1, f"Expected non-empty metric groups/totals in {data!r}"
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in groups
        )
        == 0
    )
    assert (
        sum(
            _metric_field(
                g, "distance", "distance_m", "distance_meters", "total_distance", "distance_km"
            )
            for g in groups
        )
        == 0
    )
    assert (
        sum(
            _metric_field(
                g, "moving_time", "moving_time_s", "moving_time_seconds", "total_moving_time"
            )
            for g in groups
        )
        == 0
    )
    assert (
        sum(
            _metric_field(g, "elevation_gain", "total_elevation_gain", "elevation_m", "elevation")
            for g in groups
        )
        == 0
    )
    if isinstance(data.get("totals"), dict):
        totals = data["totals"]
        assert (
            _metric_field(
                totals, "count", "activity_count", "activities_count", "activities"
            )
            == 0
        )
        assert (
            _metric_field(
                totals,
                "distance",
                "distance_m",
                "distance_meters",
                "total_distance",
                "distance_km",
            )
            == 0
        )
        assert (
            _metric_field(
                totals,
                "moving_time",
                "moving_time_s",
                "moving_time_seconds",
                "total_moving_time",
            )
            == 0
        )
        assert (
            _metric_field(
                totals,
                "elevation_gain",
                "total_elevation_gain",
                "elevation_m",
                "elevation",
            )
            == 0
        )


def test_f9_summarize_metrics_includes_only_me_and_followers(
    stdio_session: SyncMCPSession,
) -> None:
    res = stdio_session.call_tool(
        "strava_summarize_metrics",
        {"after": "30d", "group_by": "total"},
    )
    assert not getattr(res, "is_error", False)
    data = parse_tool_result(res)
    groups = _extract_metric_groups(data)
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in groups
        )
        == 5
    )
    assert _has_approx_number(data, 118967.2, rel_tol=0.01) or _has_approx_number(
        data, 118.97, rel_tol=0.02
    )
    assert _has_approx_number(data, 24600, rel_tol=0.01)
    assert _has_approx_number(data, 2235.0, rel_tol=0.01)


def test_f9_summarize_metrics_invalid_group_by_validation_error(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    res = stdio_session.call_tool("strava_summarize_metrics", {"group_by": "day_invalid"})
    _, _, text = _extract_error_code_and_text(res)
    lowered = text.lower()
    assert "group_by" in lowered or "invalid" in lowered or "week" in lowered
    assert mock_strava.get_requests("/api/v3/athlete/activities", "GET") == []


def test_f9_summarize_metrics_invalid_date_format_validation_error(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    res = stdio_session.call_tool("strava_summarize_metrics", {"after": "not-a-date"})
    _, _, text = _extract_error_code_and_text(res)
    lowered = text.lower()
    assert any(kw in lowered for kw in ("after", "date", "invalid", "format", "parse"))
    assert mock_strava.get_requests("/api/v3/athlete/activities", "GET") == []


def test_f9_summarize_metrics_read_only_insufficient_scope(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_summarize_metrics", {"after": "30d"})
    _, code, text = _extract_error_code_and_text(res)
    assert code == "insufficient_scope"
    assert "activity:read" in text
    assert "strava-mcp auth" in text


def test_f9_summarize_metrics_single_activity_vs_multimonth_unit_consistency(
    stdio_session: SyncMCPSession,
) -> None:
    res_3d = stdio_session.call_tool(
        "strava_summarize_metrics",
        {"after": "3d", "group_by": "total"},
    )
    assert not getattr(res_3d, "is_error", False)
    d3 = parse_tool_result(res_3d)
    assert _has_approx_number(d3, 64.37, rel_tol=0.02) or _has_approx_number(
        d3, 64373.8, rel_tol=0.01
    )
    assert _has_approx_number(d3, 40.0, rel_tol=0.02)
    assert _has_approx_number(d3, 920.0, rel_tol=0.01)
    assert _has_approx_number(d3, 3018.4, rel_tol=0.02)

    res_150d = stdio_session.call_tool(
        "strava_summarize_metrics",
        {"after": "150d", "group_by": "month", "refresh": True},
    )
    assert not getattr(res_150d, "is_error", False)
    d150 = parse_tool_result(res_150d)
    groups = _extract_metric_groups(d150)
    assert (
        sum(
            _metric_field(g, "count", "activity_count", "activities_count", "activities")
            for g in groups
        )
        == 7
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
    assert _has_approx_number(d150, 106.54, rel_tol=0.02)
    assert _has_approx_number(d150, 8645.0, rel_tol=0.02)


# --- Feature F10: Gear, Routes, Segments & Clubs ---


def test_f10_get_route_private_7002_without_read_all_insufficient_scope(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read,activity:read_all,profile:read_all")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_get_route", {"route_id": 7002})
    _, code, text = _extract_error_code_and_text(res)
    assert code == "insufficient_scope"
    assert "read_all" in text
    assert "strava-mcp auth" in text


def test_f10_list_segment_efforts_non_subscriber_subscription_required(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    mock_strava.set_subscription(False, error_status=402)
    res_402 = stdio_session.call_tool("strava_list_segment_efforts", {"segment_id": 8001})
    _, code_402, _ = _extract_error_code_and_text(res_402)
    assert code_402 == "subscription_required"

    mock_strava.set_subscription(False, error_status=403)
    res_403 = stdio_session.call_tool(
        "strava_list_segment_efforts",
        {"segment_id": 8001, "refresh": True},
    )
    _, code_403, _ = _extract_error_code_and_text(res_403)
    assert code_403 == "subscription_required"


def test_f10_unknown_gear_route_segment_return_not_found(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    gear_res = stdio_session.call_tool("strava_get_gear", {"gear_id": "b999999"})
    _, gear_code, _ = _extract_error_code_and_text(gear_res)
    assert gear_code == "not_found"

    route_res = stdio_session.call_tool("strava_get_route", {"route_id": 9999999})
    _, route_code, _ = _extract_error_code_and_text(route_res)
    assert route_code == "not_found"

    seg_res = stdio_session.call_tool("strava_get_segment", {"segment_id": 9999999})
    _, seg_code, _ = _extract_error_code_and_text(seg_res)
    assert seg_code == "not_found"

    assert any(r.status_code == 404 for r in mock_strava.get_requests("/api/v3/gear/b999999"))
    assert any(r.status_code == 404 for r in mock_strava.get_requests("/api/v3/routes/9999999"))
    assert any(r.status_code == 404 for r in mock_strava.get_requests("/api/v3/segments/9999999"))


def test_f10_get_route_include_export_none_vs_gpx_vs_tcx(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    res_none = stdio_session.call_tool(
        "strava_get_route",
        {"route_id": 7001, "include_export": "none"},
    )
    assert not getattr(res_none, "is_error", False)
    ser_none = json.dumps(parse_tool_result(res_none))
    assert "<gpx" not in ser_none
    assert "<TrainingCenterDatabase" not in ser_none
    assert mock_strava.get_requests("/api/v3/routes/7001/export_gpx") == []
    assert mock_strava.get_requests("/api/v3/routes/7001/export_tcx") == []

    res_gpx = stdio_session.call_tool(
        "strava_get_route",
        {"route_id": 7001, "include_export": "gpx", "refresh": True},
    )
    assert not getattr(res_gpx, "is_error", False)
    ser_gpx = json.dumps(parse_tool_result(res_gpx))
    assert "<gpx" in ser_gpx
    gpx_reqs = mock_strava.get_requests("/api/v3/routes/7001/export_gpx")
    assert any(r.status_code == 200 for r in gpx_reqs)

    res_tcx = stdio_session.call_tool(
        "strava_get_route",
        {"route_id": 7001, "include_export": "tcx", "refresh": True},
    )
    assert not getattr(res_tcx, "is_error", False)
    ser_tcx = json.dumps(parse_tool_result(res_tcx))
    assert "<TrainingCenterDatabase" in ser_tcx
    tcx_reqs = mock_strava.get_requests("/api/v3/routes/7001/export_tcx")
    assert any(r.status_code == 200 for r in tcx_reqs)


def test_f10_list_clubs_works_with_read_only_token(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_list_clubs", {})
    assert not getattr(res, "is_error", False)
    clubs = _extract_list(parse_tool_result(res), "clubs")
    by_id = {int(c["id"]): c for c in clubs}
    assert set(by_id.keys()) == {5001, 5002}
    assert by_id[5001]["name"] == "Cascade Gravel Club"
    assert by_id[5002]["name"] == "PDX Trail Runners"


def test_f10_list_routes_without_read_all_filters_private_route(
    isolated_env: dict[str, str],
    seed_config: Callable[..., Any],
    tmp_path: Path,
) -> None:
    seed_config(scope="read,activity:read_all,profile:read_all")
    with _open_session(isolated_env, tmp_path) as session:
        res = session.call_tool("strava_list_routes", {})
    assert not getattr(res, "is_error", False)
    routes = _extract_list(parse_tool_result(res), "routes")
    ids = {int(r["id"]) for r in routes}
    assert ids == {7001}
