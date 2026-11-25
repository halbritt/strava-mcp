"""Tier 4 opaque-box E2E full application scenario tests (Scenarios 1-6)."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import socket
import stat
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import (
    DEFAULT_TEST_ATHLETE_ID,
    DEFAULT_TEST_CLIENT_ID,
    DEFAULT_TEST_CLIENT_SECRET,
    REPO_ROOT,
    SeededConfig,
    SyncMCPSession,
    open_mcp_http_session,
    open_stdio_session,
    parse_tool_result,
    resolve_strava_mcp_cmd,
    start_http_server,
)
from mock_strava import MockStravaServer, RecordedRequest

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


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_mode(path: Path, expected: int) -> None:
    assert path.exists(), f"Expected path to exist: {path}"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == expected, f"{path} has mode {oct(mode)}, expected {oct(expected)}"
    assert (mode & 0o077) == 0, f"{path} exposes group/other permissions: {oct(mode)}"


def _parse_scopes_any(raw: Any) -> set[str]:
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw]
    elif isinstance(raw, str):
        items = re.split(r"[,\s]+", raw.strip())
    else:
        return set()
    return {item for item in items if item}


def _read_tokens_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auth_header(req: RecordedRequest) -> str:
    return req.headers.get("Authorization") or req.headers.get("authorization") or ""


def _assert_no_secrets_leaked(
    text: str,
    mock_strava: MockStravaServer | None = None,
    *extra_secrets: str,
) -> None:
    candidates = [s for s in extra_secrets if s]
    if mock_strava is not None:
        active = mock_strava.get_active_tokens()
        candidates.extend(active.get("active_access_tokens", {}).keys())
        candidates.extend(active.get("active_refresh_tokens", {}).keys())
        candidates.extend(active.get("invalidated_refresh_tokens", []))
    for secret in dict.fromkeys(candidates):
        assert secret not in text, "Sensitive credential value leaked in output"


def _require_headers_script() -> Path:
    script = REPO_ROOT / "client" / "strava-mcp-headers"
    if not script.is_file() or not os.access(script, os.X_OK):
        pytest.fail(
            f"client/strava-mcp-headers missing or not executable: {script}",
            pytrace=False,
        )
    return script


def _run_cli_detached(
    isolated_env: dict[str, str],
    *args: str,
    input_text: str | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    cmd = [*resolve_strava_mcp_cmd(), *args]
    merged_env = dict(isolated_env)
    if env_overrides:
        merged_env.update(env_overrides)
    return subprocess.run(
        cmd,
        input=input_text,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        start_new_session=True,
        check=False,
    )


def _spawn_auth_and_capture_url(
    isolated_env: dict[str, str],
    extra_args: list[str],
    *,
    timeout: float = 10.0,
) -> tuple[subprocess.Popen[str], str, Callable[[float], tuple[int, str, str]]]:
    cmd = [*resolve_strava_mcp_cmd(), "auth", *extra_args]
    env = dict(isolated_env)
    env["BROWSER"] = "/bin/true"
    env["DISPLAY"] = ""
    env["WAYLAND_DISPLAY"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    out_lines: list[str] = []
    err_lines: list[str] = []
    found_url: list[str] = []
    found_event = threading.Event()
    url_re = re.compile(r"https?://[^\s\"'<>]+/oauth/authorize\?[^\s\"'<>]+")

    def _reader(stream: Any, bucket: list[str]) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            bucket.append(line)
            match = url_re.search(line)
            if match and not found_event.is_set():
                found_url.append(match.group(0))
                found_event.set()
        stream.close()

    t_out = threading.Thread(target=_reader, args=(proc.stdout, out_lines), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, err_lines), daemon=True)
    t_out.start()
    t_err.start()

    def _finish(wait_timeout: float = 10.0) -> tuple[int, str, str]:
        try:
            rc = proc.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=5.0)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        return rc, "".join(out_lines), "".join(err_lines)

    if not found_event.wait(timeout=timeout):
        rc, out_txt, err_txt = _finish(1.0)
        raise AssertionError(
            f"strava-mcp auth did not print authorize URL within {timeout}s (rc={rc}).\n"
            f"stdout:\n{out_txt}\nstderr:\n{err_txt}"
        )
    return proc, found_url[0], _finish


def _drive_oauth_authorize_to_callback(
    auth_url: str,
) -> tuple[httpx.Response, httpx.Response]:
    auth_resp = httpx.get(auth_url, follow_redirects=False, timeout=5.0)
    assert auth_resp.status_code == 302, (
        f"Expected 302 from /oauth/authorize, got {auth_resp.status_code}: {auth_resp.text}"
    )
    location = auth_resp.headers["Location"]
    cb_resp = httpx.get(location, follow_redirects=False, timeout=5.0)
    return auth_resp, cb_resp


def _call_ok(
    session: SyncMCPSession,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    res = session.call_tool(tool, arguments or {})
    assert not bool(getattr(res, "is_error", False)), (
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


def _extract_error_dict(result: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    sc = getattr(result, "structured_content", None)
    if isinstance(sc, dict):
        raw = dict(sc)
    else:
        content = getattr(result, "content", None) or []
        if content and isinstance(getattr(content[0], "text", None), str):
            text = content[0].text.strip()
            try:
                parsed = json.loads(text)
                raw = parsed if isinstance(parsed, dict) else {"message": text}
            except json.JSONDecodeError:
                raw = {"message": text}
    merged: dict[str, Any] = dict(raw)
    for nested_key in ("error", "details"):
        nested = raw.get(nested_key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                merged.setdefault(k, v)
            sub_details = nested.get("details")
            if isinstance(sub_details, dict):
                for k, v in sub_details.items():
                    merged.setdefault(k, v)
    return merged


def _extract_list(payload: dict[str, Any], *candidate_keys: str) -> list[dict[str, Any]]:
    for key in (*candidate_keys, "items", "results", "data"):
        val = payload.get(key)
        if isinstance(val, list):
            return [item for item in val if isinstance(item, dict)]
    raise AssertionError(
        f"Expected list under {(*candidate_keys, 'items', 'results', 'data')} in {payload!r}"
    )


def _collect_ints(obj: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, int):
        found.add(obj)
    elif isinstance(obj, dict):
        for val in obj.values():
            found.update(_collect_ints(val))
    elif isinstance(obj, list):
        for val in obj:
            found.update(_collect_ints(val))
    return found


def _collect_iso_datetimes(obj: Any) -> list[datetime]:
    out: list[datetime] = []
    if isinstance(obj, str):
        candidate = obj.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", candidate):
            normalized = candidate.replace("Z", "+00:00")
            try:
                out.append(datetime.fromisoformat(normalized))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for val in obj.values():
            out.extend(_collect_iso_datetimes(val))
    elif isinstance(obj, list):
        for val in obj:
            out.extend(_collect_iso_datetimes(val))
    return out


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


def _stream_keys_arg(session: SyncMCPSession, keys: list[str]) -> list[str] | str:
    tools_res = session.list_tools()
    for tool in getattr(tools_res, "tools", []):
        if getattr(tool, "name", None) == "strava_get_activity_streams":
            schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )
            prop = (schema.get("properties") or {}).get("keys") or {}
            prop_type = prop.get("type")
            if prop_type == "string" or (isinstance(prop_type, list) and "string" in prop_type):
                return ",".join(keys)
            for option in prop.get("anyOf", []):
                if isinstance(option, dict) and option.get("type") == "string":
                    return ",".join(keys)
            return keys
    return keys


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


def _has_scalar_metric_keys(obj: dict[str, Any]) -> bool:
    return any(
        isinstance(obj.get(k), (int, float)) and not isinstance(obj.get(k), bool)
        for k in _SCALAR_METRIC_KEYS
    )


def _extract_metric_groups(
    payload: dict[str, Any],
    *,
    group_by: str | None = None,
) -> list[dict[str, Any]]:
    if group_by == "total":
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
        if _has_scalar_metric_keys(payload):
            return [payload]

    if group_by == "sport":
        ordered_keys: tuple[str, ...] = (
            "sports",
            "by_sport",
            "groups",
            "buckets",
            "rollups",
            "summary",
            "items",
            "results",
            "data",
            "breakdown",
        )
    elif group_by == "week":
        ordered_keys = (
            "weeks",
            "groups",
            "buckets",
            "periods",
            "rollups",
            "summary",
            "items",
            "results",
            "data",
            "breakdown",
        )
    else:
        ordered_keys = (
            "groups",
            "buckets",
            "periods",
            "rollups",
            "weeks",
            "months",
            "summary",
            "items",
            "results",
            "data",
        )

    for key in ordered_keys:
        val = payload.get(key)
        if isinstance(val, list) and val:
            sub_items = [item for item in val if isinstance(item, dict)]
            if sub_items:
                return sub_items
    for key in ordered_keys:
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
    if group_by is None:
        for key in ("sports", "by_sport", "breakdown"):
            val = payload.get(key)
            if isinstance(val, list) and val:
                sub_items = [item for item in val if isinstance(item, dict)]
                if sub_items:
                    return sub_items
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


# =============================================================================
# Scenario 1: Onboarding & Scope Upgrade Journey (init -> status -> stdio -> auth -> stdio)
# =============================================================================


def test_t4_s01_onboarding_and_scope_upgrade_journey(
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    old_access = "synthetic-s01-read-access"
    old_refresh = "synthetic-s01-read-refresh"
    exp_at = int(time.time()) + 18000

    seed_file = tmp_path / "static_read_tokens.json"
    seed_file.write_text(
        json.dumps(
            {
                "access_token": old_access,
                "refresh_token": old_refresh,
                "expires_at": exp_at,
                "expires_in": 18000,
                "token_type": "Bearer",
                "scope": "read",
                "athlete_id": DEFAULT_TEST_ATHLETE_ID,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mock_strava.set_client_credentials(DEFAULT_TEST_CLIENT_ID, DEFAULT_TEST_CLIENT_SECRET)
    mock_strava.set_token_state(
        access_token=old_access,
        refresh_token=old_refresh,
        scopes="read",
        expires_at=exp_at,
        athlete_id=DEFAULT_TEST_ATHLETE_ID,
    )

    init_proc = _run_cli_detached(
        isolated_env,
        "init",
        "--client-id",
        DEFAULT_TEST_CLIENT_ID,
        "--import-tokens-file",
        str(seed_file),
        input_text=f"{DEFAULT_TEST_CLIENT_SECRET}\n",
    )
    assert init_proc.returncode == 0, f"init failed: {init_proc.stdout}\n{init_proc.stderr}"

    cfg_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    _assert_mode(cfg_dir, 0o700)
    for fname in ("client.json", "tokens.json", "mcp-token"):
        _assert_mode(cfg_dir / fname, 0o600)
    mcp_token = (cfg_dir / "mcp-token").read_text(encoding="utf-8").strip()

    mock_strava.clear_requests()
    st_proc = run_cli("status", check=True)
    live_proc = run_cli("status", "--check-live", check=True)
    assert len(mock_strava.get_requests("/api/v3/athlete", "GET")) == 1
    live_low = (live_proc.stdout + "\n" + live_proc.stderr).lower()
    assert "read" in live_low
    assert "activity:read_all" in live_low or "profile:read_all" in live_low
    assert "strava-mcp auth" in live_low

    cmd = [*resolve_strava_mcp_cmd(), "stdio"]
    with open_stdio_session(cmd, isolated_env, tmp_path / "s01_before.err") as s1:
        sum_data = _call_ok(s1, "strava_get_athlete", {})
        sum_ath = sum_data.get("athlete") if isinstance(sum_data.get("athlete"), dict) else sum_data
        assert sum_ath["id"] == DEFAULT_TEST_ATHLETE_ID
        assert not sum_ath.get("ftp")

        res_acts = s1.call_tool("strava_list_activities", {})
        err_acts = _extract_error_dict(res_acts)
        assert err_acts.get("code") == "insufficient_scope"
        assert "activity:read" in json.dumps(err_acts)
        assert "strava-mcp auth" in json.dumps(err_acts)

        res_zones = s1.call_tool("strava_get_athlete_zones", {})
        err_zones = _extract_error_dict(res_zones)
        assert err_zones.get("code") == "insufficient_scope"
        assert "profile:read_all" in json.dumps(err_zones)
        assert "strava-mcp auth" in json.dumps(err_zones)
        err1_txt = s1.stderr_text

    mock_strava.clear_requests()
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(isolated_env, ["--port", str(port)])
    try:
        auth_resp, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert auth_resp.status_code == 302
        assert cb_resp.status_code == 200
        rc, auth_out, auth_err = finish(10.0)
        assert rc == 0, f"auth failed (rc={rc}):\n{auth_out}\n{auth_err}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    auth_reqs = mock_strava.get_requests("/oauth/authorize", "GET")
    assert len(auth_reqs) == 1
    assert auth_reqs[0].query.get("approval_prompt") == ["force"]
    assert "activity:read_all" in (auth_reqs[0].query.get("scope") or [""])[0]
    assert any(r.status_code == 200 for r in mock_strava.get_requests("/oauth/token", "POST"))

    _assert_mode(cfg_dir / "tokens.json", 0o600)
    upgraded = _read_tokens_file(cfg_dir / "tokens.json")
    assert {
        "read",
        "read_all",
        "profile:read_all",
        "activity:read_all",
    } <= _parse_scopes_any(upgraded.get("scope") or upgraded.get("scopes"))
    st_after = run_cli("status", check=True)
    assert "activity:read_all" in st_after.stdout

    with open_stdio_session(cmd, isolated_env, tmp_path / "s01_after.err") as s2:
        det_data = _call_ok(s2, "strava_get_athlete", {"refresh": True})
        det_ath = det_data.get("athlete") if isinstance(det_data.get("athlete"), dict) else det_data
        assert det_ath["ftp"] == 285
        assert "b1001" in {b["id"] for b in det_ath.get("bikes", [])}
        assert "g2001" in {s["id"] for s in det_ath.get("shoes", [])}

        z_data = _call_ok(s2, "strava_get_athlete_zones", {})
        zones_obj = z_data.get("zones") if isinstance(z_data.get("zones"), dict) else z_data
        assert len(zones_obj["heart_rate"]["zones"]) == 5
        assert len(zones_obj["power"]["zones"]) == 7

        all_acts = _call_ok(s2, "strava_list_activities", {"limit": 20})
        assert [a["id"] for a in _extract_list(all_acts, "activities")] == [
            9001, 9002, 9003, 9004, 9005, 9006, 9007,
        ]
        act_9003 = _call_ok(s2, "strava_get_activity", {"activity_id": 9003})
        a3 = act_9003.get("activity") if isinstance(act_9003.get("activity"), dict) else act_9003
        assert a3["name"] == "Private Singletrack MTB"
        assert a3["private"] is True
        err2_txt = s2.stderr_text

    _assert_no_secrets_leaked(
        "\n".join([
            init_proc.stdout, init_proc.stderr, st_proc.stdout, live_proc.stdout,
            auth_out, auth_err, st_after.stdout, err1_txt, err2_txt,
            json.dumps(det_data), json.dumps(all_acts),
        ]),
        mock_strava,
        DEFAULT_TEST_CLIENT_SECRET,
        old_access, old_refresh,
        upgraded["access_token"], upgraded["refresh_token"],
        mcp_token,
    )


# =============================================================================
# Scenario 2: AI Coaching Weekly Review over stdio (compact payloads + framing)
# =============================================================================


def test_t4_s02_weekly_endurance_coach_briefing_stdio(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config()
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]

    with open_stdio_session(cmd, isolated_env, tmp_path / "s02_stdio.err") as session:
        prompt_res = session.get_prompt("weekly_training_review", {})
        assert prompt_res.messages

        status_data = _call_ok(session, "strava_auth_status", {})
        status_txt = json.dumps(status_data)
        assert "activity:read_all" in status_txt
        assert "profile:read_all" in status_txt
        rl_section = (
            status_data.get("rate_limits") or status_data.get("rate_limit") or status_data
        )
        assert {200, 2000, 400, 4000} & _collect_ints(rl_section)

        metrics_data = _call_ok(
            session,
            "strava_summarize_metrics",
            {"after": "30d", "group_by": "week"},
        )
        groups = _extract_metric_groups(metrics_data, group_by="week")
        assert len(groups) >= 2
        assert (
            sum(
                _metric_field(g, "count", "activity_count", "activities_count", "activities")
                for g in groups
            )
            == 5
        )
        assert (
            sum(
                _metric_field(
                    g, "moving_time", "moving_time_s", "moving_time_seconds", "total_moving_time"
                )
                for g in groups
            )
            == 24600
        )
        assert _has_approx_number(metrics_data, 118967.2, rel_tol=0.01) or _has_approx_number(
            metrics_data, 118.97, rel_tol=0.02
        )
        assert _has_approx_number(metrics_data, 73.92, rel_tol=0.02)
        assert _has_approx_number(metrics_data, 2235.0, rel_tol=0.01)

        zones_data = _call_ok(session, "strava_get_athlete_zones", {})
        zones_obj = (
            zones_data.get("zones") if isinstance(zones_data.get("zones"), dict) else zones_data
        )
        assert len(zones_obj["heart_rate"]["zones"]) == 5
        assert len(zones_obj["power"]["zones"]) == 7

        gravel_list = _call_ok(
            session,
            "strava_list_activities",
            {"sport_type": "GravelRide"},
        )
        gravel_items = _extract_list(gravel_list, "activities")
        assert [a["id"] for a in gravel_items] == [9001]
        assert gravel_items[0]["name"] == "Morning Gravel Grind"
        assert "mock_full_poly_9001" not in json.dumps(gravel_list)

        act_data = _call_ok(session, "strava_get_activity", {"activity_id": 9001})
        act = act_data.get("activity") if isinstance(act_data.get("activity"), dict) else act_data
        assert act["id"] == 9001
        assert act["average_watts"] == 225.4
        assert act["average_heartrate"] == 152.0
        act_ser = json.dumps(act_data)
        assert "mock_full_poly_9001" not in act_ser
        assert "Saltzman Road Gravel Climb (Lap 2)" not in act_ser

        laps_data = _call_ok(session, "strava_get_activity_laps", {"activity_id": 9001})
        laps = _extract_list(laps_data, "laps")
        assert [lap["id"] for lap in laps] == [90011, 90012, 90013]
        assert all(lap["moving_time"] == 2700 for lap in laps)

        keys = ["time", "heartrate", "watts", "altitude"]
        streams_keys_arg = _stream_keys_arg(session, keys)
        streams_data = _call_ok(
            session,
            "strava_get_activity_streams",
            {
                "activity_id": 9001,
                "keys": streams_keys_arg,
                "max_points": 50,
            },
        )
        streams = _extract_streams_map(streams_data)
        assert set(streams.keys()) == set(keys)
        for k in keys:
            assert 2 <= len(streams[k]["data"]) <= 50
            assert streams[k].get("original_size") == 1200
        assert streams["time"]["data"][0] == 0
        assert streams["time"]["data"][-1] >= 7900
        assert len(json.dumps(streams_data)) < 25000
        session_stderr = session.stderr_text
    assert "Traceback (most recent call last)" not in session_stderr

    frames = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-framing-probe", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "strava_summarize_metrics",
                "arguments": {"after": "30d", "group_by": "week"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "strava_get_activity_streams",
                "arguments": {
                    "activity_id": 9001,
                    "keys": streams_keys_arg,
                    "max_points": 50,
                },
            },
        },
    ]
    raw_proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=isolated_env,
        text=True,
    )
    assert raw_proc.stdin is not None
    assert raw_proc.stdout is not None
    assert raw_proc.stderr is not None

    stdout_queue: queue.Queue[str | None] = queue.Queue()
    stderr_chunks: list[str] = []

    def _drain_stdout() -> None:
        assert raw_proc.stdout is not None
        try:
            for line in raw_proc.stdout:
                stdout_queue.put(line)
        finally:
            stdout_queue.put(None)

    def _drain_stderr() -> None:
        assert raw_proc.stderr is not None
        stderr_chunks.append(raw_proc.stderr.read())

    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    def _next_frame(timeout: float = 15.0) -> tuple[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"Timed out waiting for stdio frame:\n{''.join(stderr_chunks)}"
                )
            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise AssertionError(
                    f"Timed out waiting for stdio frame:\n{''.join(stderr_chunks)}"
                ) from exc
            if line is None:
                raise AssertionError(f"Unexpected stdio EOF:\n{''.join(stderr_chunks)}")
            if not line.strip():
                continue
            parsed = json.loads(line)
            assert isinstance(parsed, dict) and parsed.get("jsonrpc") == "2.0"
            return line, parsed

    raw_stdout_lines: list[str] = []
    parsed_frames: list[dict[str, Any]] = []
    try:
        raw_proc.stdin.write(json.dumps(frames[0]) + "\n")
        raw_proc.stdin.flush()
        ln1, msg1 = _next_frame()
        raw_stdout_lines.append(ln1)
        parsed_frames.append(msg1)
        assert msg1.get("id") == 1 and "result" in msg1 and "error" not in msg1

        raw_proc.stdin.write("\n".join(json.dumps(f) for f in frames[1:]) + "\n")
        raw_proc.stdin.flush()
        for _ in range(2):
            ln_n, msg_n = _next_frame()
            raw_stdout_lines.append(ln_n)
            parsed_frames.append(msg_n)

        raw_proc.stdin.close()
        while (extra := stdout_queue.get(timeout=5.0)) is not None:
            if extra.strip():
                raw_stdout_lines.append(extra)
                parsed_extra = json.loads(extra)
                assert isinstance(parsed_extra, dict) and parsed_extra.get("jsonrpc") == "2.0"
                parsed_frames.append(parsed_extra)
        returncode = raw_proc.wait(timeout=10.0)
    finally:
        if raw_proc.poll() is None:
            raw_proc.kill()
            raw_proc.wait(timeout=5.0)
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

    raw_stdout_txt = "".join(raw_stdout_lines)
    raw_stderr_txt = "".join(stderr_chunks)
    assert returncode == 0, f"raw stdio failed:\n{raw_stdout_txt}\n{raw_stderr_txt}"
    assert len(parsed_frames) >= 3
    by_id = {f["id"]: f for f in parsed_frames if "id" in f}
    assert {1, 2, 3} <= set(by_id)
    for req_id in (1, 2, 3):
        msg = by_id[req_id]
        assert "result" in msg and "error" not in msg, f"id={req_id} failed: {msg}"
        if req_id in (2, 3):
            assert not msg["result"].get("isError"), f"id={req_id} tool error: {msg}"
    assert "Traceback (most recent call last)" not in raw_stderr_txt

    _assert_no_secrets_leaked(
        "\n".join([
            session_stderr, status_txt, json.dumps(metrics_data), act_ser,
            json.dumps(streams_data), raw_stdout_txt, raw_stderr_txt,
        ]),
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


# =============================================================================
# Scenario 3: Multi-Agent Concurrent Streamable HTTP + Rotation + Cache
# =============================================================================


def test_t4_s03_multi_agent_concurrent_http_rotation_and_cache(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 120)
    mock_strava.set_near_expiry(120, seeded.access_token)
    mock_strava.configure_oauth(include_scope_on_refresh=False, include_athlete_on_refresh=False)
    mock_strava.clear_requests()

    headers_script = _require_headers_script()
    hdr_proc = subprocess.run(
        [str(headers_script)],
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert hdr_proc.returncode == 0, f"headers helper failed: {hdr_proc.stdout}\n{hdr_proc.stderr}"
    token = json.loads(hdr_proc.stdout)["Authorization"].removeprefix("Bearer ").strip()
    assert token == seeded.mcp_token

    after_iso = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    cmd = resolve_strava_mcp_cmd()
    barrier = threading.Barrier(5)

    def _run_three_tools(sess: SyncMCPSession) -> tuple[int, list[int], str]:
        ath = _call_ok(sess, "strava_get_athlete", {})
        ath_obj = ath.get("athlete") if isinstance(ath.get("athlete"), dict) else ath
        acts = _call_ok(sess, "strava_list_activities", {"after": after_iso})
        act_ids = [a["id"] for a in _extract_list(acts, "activities")]
        act = _call_ok(sess, "strava_get_activity", {"activity_id": 9001})
        act_obj = act.get("activity") if isinstance(act.get("activity"), dict) else act
        return int(ath_obj["id"]), act_ids, str(act_obj["name"])

    with start_http_server(cmd, isolated_env, token, tmp_path) as info:
        def _pass1_worker() -> tuple[int, list[int], str]:
            with open_mcp_http_session(info.mcp_url, token, timeout=25.0) as sess:
                barrier.wait(timeout=20.0)
                return _run_three_tools(sess)

        with ThreadPoolExecutor(max_workers=5) as pool:
            pass1_results = [
                fut.result(timeout=30.0) for fut in [pool.submit(_pass1_worker) for _ in range(5)]
            ]

        for aid, act_ids, act_name in pass1_results:
            assert aid == DEFAULT_TEST_ATHLETE_ID
            assert act_ids == [9001, 9002, 9003, 9004, 9005]
            assert act_name == "Morning Gravel Grind"

        tok_reqs = mock_strava.get_requests("/oauth/token", "POST")
        assert [r.status_code for r in tok_reqs] == [200]
        assert seeded.refresh_token in mock_strava.get_active_tokens()["invalidated_refresh_tokens"]

        _assert_mode(seeded.tokens_file, 0o600)
        updated = _read_tokens_file(seeded.tokens_file)
        new_access = updated["access_token"]
        new_refresh = updated["refresh_token"]
        assert new_access != seeded.access_token
        assert new_refresh != seeded.refresh_token
        assert {
            "read", "read_all", "profile:read_all", "activity:read_all",
        } <= _parse_scopes_any(updated.get("scope") or updated.get("scopes"))

        api_reqs = mock_strava.get_requests("/api/v3/")
        assert api_reqs
        assert all(_auth_header(r) == f"Bearer {new_access}" for r in api_reqs)

        mock_strava.clear_requests()
        for _ in range(2):
            with open_mcp_http_session(info.mcp_url, token) as http_sess:
                assert _run_three_tools(http_sess) == pass1_results[0]
        with open_stdio_session([*cmd, "stdio"], isolated_env, tmp_path / "s03_p2.err") as st_sess:
            assert _run_three_tools(st_sess) == pass1_results[0]
            st_err = st_sess.stderr_text
        assert mock_strava.requests == []
        srv_err = info.stderr_text

    cache_db = Path(isolated_env["STRAVA_MCP_CACHE_DIR"]) / "cache.sqlite"
    _assert_mode(cache_db, 0o600)
    stats_proc = run_cli("cache", "stats", check=True)
    assert any(int(n) >= 3 for n in re.findall(r"\b\d+\b", stats_proc.stdout))

    _assert_no_secrets_leaked(
        "\n".join([srv_err, st_err, stats_proc.stdout]),
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        new_access,
        new_refresh,
        token,
    )


# =============================================================================
# Scenario 4: Quota Exhaustion (429) & Graceful Cached Degradation
# =============================================================================


def test_t4_s04_quota_exhaustion_and_graceful_degradation(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seed_config()
    mock_strava.set_rate_limits(
        read_limit_15m=200,
        read_usage_15m=197,
        overall_limit_15m=400,
        overall_usage_15m=197,
    )
    mock_strava.clear_requests()

    after_iso = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]

    with open_stdio_session(cmd, isolated_env, tmp_path / "s04_1.err") as s1:
        acts = _call_ok(s1, "strava_list_activities", {"after": after_iso})
        assert [a["id"] for a in _extract_list(acts, "activities")] == [
            9001, 9002, 9003, 9004, 9005,
        ]
        act = _call_ok(s1, "strava_get_activity", {"activity_id": 9001})
        act_obj = act.get("activity") if isinstance(act.get("activity"), dict) else act
        assert act_obj["id"] == 9001
        ath = _call_ok(s1, "strava_get_athlete", {})
        ath_obj = ath.get("athlete") if isinstance(ath.get("athlete"), dict) else ath
        assert ath_obj["id"] == DEFAULT_TEST_ATHLETE_ID
        assert len(mock_strava.get_requests("/api/v3/")) == 3

        rl_status = _call_ok(s1, "strava_auth_status", {})
        rl_section = (
            rl_status.get("rate_limits") or rl_status.get("rate_limit") or rl_status
        )
        assert 200 in _collect_ints(rl_section)

        mock_strava.clear_requests()
        res_streams = s1.call_tool(
            "strava_get_activity_streams",
            {
                "activity_id": 9001,
                "keys": _stream_keys_arg(s1, ["time", "heartrate"]),
                "max_points": 50,
            },
        )
        err = _extract_error_dict(res_streams)
        assert bool(getattr(res_streams, "is_error", False)) or err.get("code") == "rate_limited"
        assert err.get("code") == "rate_limited"
        assert "read" in str(err.get("limit_type") or "").lower()
        assert 200 in _collect_ints(err)
        dts = _collect_iso_datetimes(err.get("reset_at_utc") or err.get("reset_at") or err)
        assert dts and any(dt.minute in {0, 15, 30, 45} and dt.second == 0 for dt in dts)
        assert float(err.get("retry_after_seconds", 0)) > 0
        assert len(mock_strava.get_requests("/api/v3/activities/9001/streams")) <= 1

        mock_strava.clear_requests()
        assert [
            a["id"]
            for a in _extract_list(
                _call_ok(s1, "strava_list_activities", {"after": after_iso}),
                "activities",
            )
        ] == [9001, 9002, 9003, 9004, 9005]
        assert _call_ok(s1, "strava_get_activity", {"activity_id": 9001})
        assert _call_ok(s1, "strava_get_athlete", {})
        assert mock_strava.get_requests("/api/v3/") == []

        res_laps = s1.call_tool("strava_get_activity_laps", {"activity_id": 9001})
        laps_err = _extract_error_dict(res_laps)
        assert laps_err.get("code") == "rate_limited"
        assert len(mock_strava.get_requests("/api/v3/activities/9001/laps")) <= 1

    mock_strava.clear_requests()
    with open_stdio_session(cmd, isolated_env, tmp_path / "s04_2.err") as s2:
        acts2 = _call_ok(s2, "strava_list_activities", {"after": after_iso})
        assert [a["id"] for a in _extract_list(acts2, "activities")] == [
            9001, 9002, 9003, 9004, 9005,
        ]
        act2 = _call_ok(s2, "strava_get_activity", {"activity_id": 9001})
        act2_obj = act2.get("activity") if isinstance(act2.get("activity"), dict) else act2
        assert act2_obj["id"] == 9001
    assert mock_strava.get_requests("/api/v3/") == []


# =============================================================================
# Scenario 5: Post-Workout Gear Wear, Routes GPX/TCX & Subscription 402 Survival
# =============================================================================


def test_t4_s05_post_workout_routes_gear_wear_and_402_segments(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config()
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]

    with open_stdio_session(cmd, isolated_env, tmp_path / "s05_stdio.err") as session:
        ath_data = _call_ok(session, "strava_get_athlete", {})
        athlete = (
            ath_data.get("athlete") if isinstance(ath_data.get("athlete"), dict) else ath_data
        )
        bike_id = athlete["bikes"][0]["id"]
        shoe_id = athlete["shoes"][0]["id"]
        assert bike_id == "b1001"
        assert shoe_id == "g2001"

        bike_data = _call_ok(session, "strava_get_gear", {"gear_id": bike_id})
        bike = bike_data.get("gear") if isinstance(bike_data.get("gear"), dict) else bike_data
        assert bike["name"] == "Canyon Grail CF SL"
        assert _has_approx_number(bike_data, 4250000.0, 0.01) or _has_approx_number(
            bike_data, 4250.0, 0.01
        )
        assert _has_approx_number(bike_data, 2640.83, 0.02)

        shoe_data = _call_ok(session, "strava_get_gear", {"gear_id": shoe_id})
        shoe = shoe_data.get("gear") if isinstance(shoe_data.get("gear"), dict) else shoe_data
        assert shoe["name"] == "Hoka Speedgoat 5"
        assert _has_approx_number(shoe_data, 620000.0, 0.01) or _has_approx_number(
            shoe_data, 620.0, 0.01
        )
        assert _has_approx_number(shoe_data, 385.25, 0.02)

        act_data = _call_ok(session, "strava_get_activity", {"activity_id": 9001})
        assert "mock_full_poly_9001" not in json.dumps(act_data)
        laps_data = _call_ok(session, "strava_get_activity_laps", {"activity_id": 9001})
        assert len(_extract_list(laps_data, "laps")) == 3

        keys = ["time", "distance", "heartrate", "watts"]
        streams_data = _call_ok(
            session,
            "strava_get_activity_streams",
            {
                "activity_id": 9001,
                "keys": _stream_keys_arg(session, keys),
                "max_points": 40,
            },
        )
        streams = _extract_streams_map(streams_data)
        assert set(streams.keys()) == set(keys)
        for k in keys:
            assert 2 <= len(streams[k]["data"]) <= 40
            assert streams[k].get("original_size") == 1200

        routes_data = _call_ok(session, "strava_list_routes", {})
        assert {int(r["id"]) for r in _extract_list(routes_data, "routes")} == {7001, 7002}

        gpx_data = _call_ok(
            session,
            "strava_get_route",
            {"route_id": 7001, "include_export": "gpx"},
        )
        gpx_xml = _find_xml_text(gpx_data, "<gpx")
        assert gpx_xml and ET.fromstring(gpx_xml).tag.endswith("gpx")
        assert "Forest Park Gravel Loop" in gpx_xml

        tcx_data = _call_ok(
            session,
            "strava_get_route",
            {"route_id": 7002, "include_export": "tcx"},
        )
        tcx_xml = _find_xml_text(tcx_data, "<TrainingCenterDatabase")
        assert tcx_xml and ET.fromstring(tcx_xml).tag.endswith("TrainingCenterDatabase")
        assert "Private Backcountry Trail" in tcx_xml

        starred = _call_ok(session, "strava_list_starred_segments", {})
        assert {int(s["id"]) for s in _extract_list(starred, "segments")} >= {8001, 8002}

        seg1_data = _call_ok(session, "strava_get_segment", {"segment_id": 8001})
        seg1 = seg1_data.get("segment") if isinstance(seg1_data.get("segment"), dict) else seg1_data
        assert seg1["name"] == "Saltzman Road Gravel Climb"
        assert seg1["distance"] == 4200.0

        clubs1 = _call_ok(session, "strava_list_clubs", {})
        assert {int(c["id"]) for c in _extract_list(clubs1, "clubs")} == {5001, 5002}

        mock_strava.set_subscription(False, error_status=402)
        eff_res = session.call_tool("strava_list_segment_efforts", {"segment_id": 8001})
        eff_err = _extract_error_dict(eff_res)
        assert getattr(eff_res, "is_error", False) is True
        assert eff_err.get("code") == "subscription_required"
        effort_reqs = mock_strava.get_requests("/api/v3/segment_efforts")
        assert any(r.status_code == 402 for r in effort_reqs)

        clubs2 = _call_ok(session, "strava_list_clubs", {"refresh": True})
        assert {int(c["id"]) for c in _extract_list(clubs2, "clubs")} == {5001, 5002}
        seg2_data = _call_ok(session, "strava_get_segment", {"segment_id": 8002})
        seg2 = seg2_data.get("segment") if isinstance(seg2_data.get("segment"), dict) else seg2_data
        assert seg2["name"] == "Wildwood Secret Sprint"
        s5_stderr = session.stderr_text

    _assert_no_secrets_leaked(
        "\n".join([s5_stderr, json.dumps(ath_data), json.dumps(gpx_data), json.dumps(tcx_data)]),
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


# =============================================================================
# Scenario 6: Token Revocation & Re-Authorization Recovery
# =============================================================================


def test_t4_s06_token_revocation_and_reauthorization_recovery(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seeded = seed_config()
    old_access = seeded.access_token
    old_refresh = seeded.refresh_token
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]

    with open_stdio_session(cmd, isolated_env, tmp_path / "s06_baseline.err") as s0:
        ath0 = _call_ok(s0, "strava_get_athlete", {})
        ath0_obj = ath0.get("athlete") if isinstance(ath0.get("athlete"), dict) else ath0
        assert ath0_obj["id"] == DEFAULT_TEST_ATHLETE_ID
        err0_txt = s0.stderr_text

    mock_strava.clear_requests()
    rev_proc = run_cli("revoke", check=True)
    assert len(mock_strava.get_requests("/oauth/revoke", "POST")) == 1
    active = mock_strava.get_active_tokens()
    assert old_access not in active["active_access_tokens"]
    assert old_refresh in active["invalidated_refresh_tokens"]

    if seeded.tokens_file.exists():
        raw_tok = seeded.tokens_file.read_text(encoding="utf-8")
        assert old_access not in raw_tok
        assert old_refresh not in raw_tok
        parsed_tok = json.loads(raw_tok) if raw_tok.strip() else {}
        assert (
            (not parsed_tok.get("access_token") and not parsed_tok.get("refresh_token"))
            or parsed_tok.get("revoked") is True
        )

    rej_resp = httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": seeded.client_id,
            "client_secret": seeded.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": old_refresh,
        },
        timeout=5.0,
    )
    assert rej_resp.status_code == 400

    st_rev = run_cli("status", "--check-live")
    assert st_rev.returncode != 0
    st_rev_low = (st_rev.stdout + "\n" + st_rev.stderr).lower()
    assert not re.search(r'"?authenticated"?\s*[:=]\s*true\b', st_rev_low)
    has_explicit_false = bool(re.search(r'"?authenticated"?\s*[:=]\s*false\b', st_rev_low))
    cleaned_low = re.sub(
        r'(?:not\s+authenticated|"?authenticated"?\s*[:=]\s*false\b)',
        "",
        st_rev_low,
    )
    assert not re.search(r"\bauthenticated\b", cleaned_low)
    assert has_explicit_false or any(
        kw in st_rev_low
        for kw in (
            "unauthenticated",
            "not authenticated",
            "revoked",
            "auth_missing",
            "missing",
            "no token",
        )
    )

    ref_rev = run_cli("refresh")
    assert ref_rev.returncode != 0

    with open_stdio_session(cmd, isolated_env, tmp_path / "s06_revoked.err") as s_rev:
        err_ath = _extract_error_dict(s_rev.call_tool("strava_get_athlete", {"refresh": True}))
        err_acts = _extract_error_dict(s_rev.call_tool("strava_list_activities", {"refresh": True}))
        assert err_ath.get("code") in {"auth_missing", "auth_expired", "refresh_failed"}
        assert err_acts.get("code") in {"auth_missing", "auth_expired", "refresh_failed"}
        err_rev_txt = s_rev.stderr_text

    mock_strava.clear_requests()
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(isolated_env, ["--port", str(port)])
    try:
        auth_resp, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert auth_resp.status_code == 302
        assert cb_resp.status_code == 200
        rc, auth_out, auth_err = finish(10.0)
        assert rc == 0, f"re-auth failed (rc={rc}):\n{auth_out}\n{auth_err}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    _assert_mode(seeded.tokens_file, 0o600)
    reauthed = _read_tokens_file(seeded.tokens_file)
    new_access = reauthed["access_token"]
    new_refresh = reauthed["refresh_token"]
    assert new_access and new_access != old_access
    assert new_refresh and new_refresh != old_refresh
    assert {
        "read", "read_all", "profile:read_all", "activity:read_all",
    } <= _parse_scopes_any(reauthed.get("scope") or reauthed.get("scopes"))

    mock_strava.clear_requests()
    st_ok = run_cli("status", "--check-live", check=True)
    ath_reqs = mock_strava.get_requests("/api/v3/athlete", "GET")
    assert len(ath_reqs) == 1
    assert _auth_header(ath_reqs[0]) == f"Bearer {new_access}"

    with open_stdio_session(cmd, isolated_env, tmp_path / "s06_recovered.err") as s_rec:
        rec_ath = _call_ok(s_rec, "strava_get_athlete", {"refresh": True})
        rec_ath_obj = (
            rec_ath.get("athlete") if isinstance(rec_ath.get("athlete"), dict) else rec_ath
        )
        assert rec_ath_obj["ftp"] == 285

        rec_acts = _call_ok(
            s_rec,
            "strava_list_activities",
            {"after": "30d", "refresh": True},
        )
        assert [a["id"] for a in _extract_list(rec_acts, "activities")] == [
            9001, 9002, 9003, 9004, 9005,
        ]
        rec_zones = _call_ok(s_rec, "strava_get_athlete_zones", {})
        zones_obj = (
            rec_zones.get("zones") if isinstance(rec_zones.get("zones"), dict) else rec_zones
        )
        assert len(zones_obj["heart_rate"]["zones"]) == 5
        assert len(zones_obj["power"]["zones"]) == 7
        err_rec_txt = s_rec.stderr_text

    _assert_no_secrets_leaked(
        "\n".join([
            err0_txt, rev_proc.stdout, rev_proc.stderr, st_rev.stdout, st_rev.stderr,
            ref_rev.stdout, ref_rev.stderr, err_rev_txt, auth_out, auth_err,
            st_ok.stdout, st_ok.stderr, err_rec_txt,
            json.dumps(rec_ath), json.dumps(rec_acts), json.dumps(rec_zones),
        ]),
        mock_strava,
        seeded.client_secret,
        old_access,
        old_refresh,
        new_access,
        new_refresh,
        seeded.mcp_token,
    )
