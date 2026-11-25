"""Tier 3 opaque-box E2E pairwise cross-feature integration tests (F1-F11)."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import stat
import subprocess
import threading
import time
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


def _bucket_has_sport(bucket: dict[str, Any], sport_name: str) -> bool:
    def _has_exact(obj: Any) -> bool:
        if isinstance(obj, str):
            return obj == sport_name
        if isinstance(obj, dict):
            return any(k == sport_name or _has_exact(v) for k, v in obj.items())
        if isinstance(obj, list):
            return any(_has_exact(v) for v in obj)
        return False

    return _has_exact(bucket)


# =============================================================================
# Pair 1 (F1 + F2): init -> refresh rotates token -> status shows new expiry
# =============================================================================


def test_t3_p01_f1_f2_init_refresh_rotates_and_status_clean(
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    old_access = "synthetic-p01-access-init"
    old_refresh = "synthetic-p01-refresh-init"
    old_exp = int(time.time()) + 120
    full_scope = "read,read_all,profile:read_all,activity:read_all"

    seed_file = tmp_path / "seed_tokens.json"
    seed_file.write_text(
        json.dumps(
            {
                "access_token": old_access,
                "refresh_token": old_refresh,
                "expires_at": old_exp,
                "expires_in": 120,
                "token_type": "Bearer",
                "scope": full_scope,
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
        scopes=full_scope,
        expires_at=old_exp,
        athlete_id=DEFAULT_TEST_ATHLETE_ID,
    )
    mock_strava.configure_oauth(include_scope_on_refresh=False, include_athlete_on_refresh=False)

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
    refresh_proc = run_cli("refresh", check=True)

    tok_reqs = mock_strava.get_requests("/oauth/token", "POST")
    assert len(tok_reqs) == 1
    assert tok_reqs[0].status_code == 200

    updated = _read_tokens_file(cfg_dir / "tokens.json")
    _assert_mode(cfg_dir / "tokens.json", 0o600)
    new_access = updated["access_token"]
    new_refresh = updated["refresh_token"]
    new_exp = int(updated["expires_at"])
    assert new_access and new_access != old_access
    assert new_refresh and new_refresh != old_refresh
    assert new_exp > old_exp
    assert _parse_scopes_any(updated.get("scope") or updated.get("scopes")) == {
        "read",
        "read_all",
        "profile:read_all",
        "activity:read_all",
    }
    athlete_val = updated.get("athlete_id") or (updated.get("athlete") or {}).get("id")
    assert int(athlete_val) == DEFAULT_TEST_ATHLETE_ID
    assert old_refresh in mock_strava.get_active_tokens()["invalidated_refresh_tokens"]

    st_proc = run_cli("status", check=True)
    live_proc = run_cli("status", "--check-live", check=True)
    for proc in (st_proc, live_proc):
        out = proc.stdout
        assert str(DEFAULT_TEST_ATHLETE_ID) in out
        assert "activity:read_all" in out
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            exp_date = datetime.fromtimestamp(new_exp, tz=UTC).strftime("%Y-%m-%d")
            assert exp_date in out or str(new_exp) in out
        else:
            assert new_exp in _collect_ints(parsed) or (
                datetime.fromtimestamp(new_exp, tz=UTC).strftime("%Y-%m-%d") in out
            )

    ath_reqs = mock_strava.get_requests("/api/v3/athlete", "GET")
    assert len(ath_reqs) == 1
    assert _auth_header(ath_reqs[0]) == f"Bearer {new_access}"

    combined = "\n".join(
        p.stdout + p.stderr for p in (init_proc, refresh_proc, st_proc, live_proc)
    )
    _assert_no_secrets_leaked(
        combined,
        mock_strava,
        DEFAULT_TEST_CLIENT_SECRET,
        old_access,
        old_refresh,
        new_access,
        new_refresh,
        mcp_token,
    )


# =============================================================================
# Pair 2 (F3 + F1 + F6): auth upgrades read -> DetailedAthlete & athlete zones
# =============================================================================


def test_t3_p02_f3_f1_f6_auth_upgrades_read_to_detailed_athlete_and_zones(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config(
        scope="read",
        access_token="synthetic-p02-read-access",
        refresh_token="synthetic-p02-read-refresh",
    )
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]

    with open_stdio_session(cmd, isolated_env, tmp_path / "p02_before.err") as s1:
        sum_data = _call_ok(s1, "strava_get_athlete", {})
        sum_ath = sum_data.get("athlete") if isinstance(sum_data.get("athlete"), dict) else sum_data
        assert sum_ath["id"] == DEFAULT_TEST_ATHLETE_ID
        assert not sum_ath.get("ftp")
        assert not sum_ath.get("bikes")

        res_zones = s1.call_tool("strava_get_athlete_zones", {})
        err = _extract_error_dict(res_zones)
        is_err = bool(getattr(res_zones, "is_error", False))
        assert is_err or err.get("code") == "insufficient_scope"
        assert err.get("code") == "insufficient_scope"
        err_text = json.dumps(err)
        assert "profile:read_all" in err_text
        assert "strava-mcp auth" in err_text
        err1_txt = s1.stderr_text

    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(isolated_env, ["--port", str(port)])
    try:
        auth_resp, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert auth_resp.status_code == 302
        assert cb_resp.status_code == 200
        assert "no-store" in cb_resp.headers.get("cache-control", "").lower()
        rc, auth_out, auth_err = finish(10.0)
        assert rc == 0, f"auth failed (rc={rc}):\n{auth_out}\n{auth_err}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    _assert_mode(seeded.tokens_file, 0o600)
    upgraded = _read_tokens_file(seeded.tokens_file)
    assert upgraded["access_token"] != seeded.access_token
    assert upgraded["refresh_token"] != seeded.refresh_token
    assert {
        "read",
        "read_all",
        "profile:read_all",
        "activity:read_all",
    } <= _parse_scopes_any(upgraded.get("scope") or upgraded.get("scopes"))
    athlete_val = upgraded.get("athlete_id") or (upgraded.get("athlete") or {}).get("id")
    assert int(athlete_val) == DEFAULT_TEST_ATHLETE_ID

    with open_stdio_session(cmd, isolated_env, tmp_path / "p02_after.err") as s2:
        det_data = _call_ok(s2, "strava_get_athlete", {"refresh": True})
        det_ath = det_data.get("athlete") if isinstance(det_data.get("athlete"), dict) else det_data
        assert det_ath["id"] == DEFAULT_TEST_ATHLETE_ID
        assert det_ath["ftp"] == 285
        assert det_ath["weight"] == 72.5
        assert "b1001" in {b["id"] for b in det_ath.get("bikes", [])}
        assert "g2001" in {s["id"] for s in det_ath.get("shoes", [])}

        z_data = _call_ok(s2, "strava_get_athlete_zones", {})
        zones_obj = z_data.get("zones") if isinstance(z_data.get("zones"), dict) else z_data
        assert len(zones_obj["heart_rate"]["zones"]) == 5
        assert len(zones_obj["power"]["zones"]) == 7
        err2_txt = s2.stderr_text

    _assert_no_secrets_leaked(
        "\n".join(
            [
                auth_out,
                auth_err,
                err1_txt,
                err2_txt,
                json.dumps(det_data),
                json.dumps(z_data),
            ]
        ),
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        upgraded["access_token"],
        upgraded["refresh_token"],
        seeded.mcp_token,
    )


# =============================================================================
# Pair 3 (F2 + F7 + F11): Expired token in stdio triggers 401 refresh + retry
# =============================================================================


def test_t3_p03_f2_f7_f11_stdio_expired_token_401_refresh_and_list_activities(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 7200)
    mock_strava.expire_access_token(seeded.access_token)
    mock_strava.clear_requests()

    cmd = [*resolve_strava_mcp_cmd(), "stdio"]
    with open_stdio_session(cmd, isolated_env, tmp_path / "p03_stdio.err") as session:
        assert session.init_result.protocol_version
        tools_by_name = {t.name: t for t in session.list_tools().tools}
        list_tool = tools_by_name["strava_list_activities"]
        assert list_tool.annotations is not None
        assert list_tool.annotations.read_only_hint is True

        res = session.call_tool("strava_list_activities", {"after": "30d", "limit": 10})
        assert res.is_error is not True
        assert isinstance(res.structured_content, dict)
        assert res.content and isinstance(res.content[0].text, str)
        assert json.loads(res.content[0].text) == res.structured_content
        payload = parse_tool_result(res)
        items = _extract_list(payload, "activities")
        assert [a["id"] for a in items] == [9001, 9002, 9003, 9004, 9005]

        act_data = _call_ok(session, "strava_get_activity", {"activity_id": 9003})
        act = act_data.get("activity") if isinstance(act_data.get("activity"), dict) else act_data
        assert act["id"] == 9003
        assert act["name"] == "Private Singletrack MTB"
        stderr_txt = session.stderr_text

    trace = [(r.method, r.path, r.status_code) for r in mock_strava.requests]
    assert trace == [
        ("GET", "/api/v3/athlete/activities", 401),
        ("POST", "/oauth/token", 200),
        ("GET", "/api/v3/athlete/activities", 200),
        ("GET", "/api/v3/activities/9003", 200),
    ]

    _assert_mode(seeded.tokens_file, 0o600)
    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] != seeded.access_token
    assert updated["refresh_token"] != seeded.refresh_token
    assert {
        "read",
        "read_all",
        "profile:read_all",
        "activity:read_all",
    } <= _parse_scopes_any(updated.get("scope") or updated.get("scopes"))
    athlete_val = updated.get("athlete_id") or (updated.get("athlete") or {}).get("id")
    assert int(athlete_val) == DEFAULT_TEST_ATHLETE_ID
    assert "Traceback (most recent call last)" not in stderr_txt

    _assert_no_secrets_leaked(
        "\n".join([stderr_txt, json.dumps(payload), json.dumps(act_data)]),
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        updated["access_token"],
        updated["refresh_token"],
        seeded.mcp_token,
    )


# =============================================================================
# Pair 4 (F2 + F5 + F7): Cross-process flock refresh + shared SQLite cache
# =============================================================================


def test_t3_p04_f2_f5_f7_cross_process_flock_refresh_and_cache_write(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 7200)
    mock_strava.expire_access_token(seeded.access_token)
    mock_strava.clear_requests()

    after_iso = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    cmd = resolve_strava_mcp_cmd()
    barrier = threading.Barrier(6)

    with start_http_server(cmd, isolated_env, seeded.mcp_token, tmp_path) as info:

        def _stdio_worker(idx: int) -> list[int]:
            with open_stdio_session(
                [*cmd, "stdio"],
                isolated_env,
                tmp_path / f"p04_stdio_{idx}.err",
                timeout=25.0,
            ) as sess:
                barrier.wait(timeout=20.0)
                data = _call_ok(sess, "strava_list_activities", {"after": after_iso})
                return [a["id"] for a in _extract_list(data, "activities")]

        def _http_worker() -> list[int]:
            with open_mcp_http_session(info.mcp_url, info.token, timeout=25.0) as sess:
                barrier.wait(timeout=20.0)
                data = _call_ok(sess, "strava_list_activities", {"after": after_iso})
                return [a["id"] for a in _extract_list(data, "activities")]

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_stdio_worker, i) for i in range(4)]
            futures.extend(pool.submit(_http_worker) for _ in range(2))
            results = [fut.result(timeout=35.0) for fut in futures]

    for ids in results:
        assert ids == [9001, 9002, 9003, 9004, 9005]

    tok_reqs = mock_strava.get_requests("/oauth/token", "POST")
    codes = [r.status_code for r in tok_reqs]
    assert codes.count(200) == 1, f"Expected single flock-serialized 200 refresh, got {codes}"
    assert codes.count(400) == 0, f"Unexpected rotation collision 400 in {codes}"

    _assert_mode(seeded.tokens_file, 0o600)
    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] != seeded.access_token
    assert updated["refresh_token"] != seeded.refresh_token

    cache_db = Path(isolated_env["STRAVA_MCP_CACHE_DIR"]) / "cache.sqlite"
    _assert_mode(cache_db, 0o600)

    mock_strava.clear_requests()
    with open_stdio_session(
        [*cmd, "stdio"],
        isolated_env,
        tmp_path / "p04_post_wave.err",
    ) as post_sess:
        post_data = _call_ok(post_sess, "strava_list_activities", {"after": after_iso})
    post_ids = [a["id"] for a in _extract_list(post_data, "activities")]
    assert post_ids == [9001, 9002, 9003, 9004, 9005]
    assert mock_strava.requests == []

    stats_proc = run_cli("cache", "stats", check=True)
    assert any(int(n) >= 1 for n in re.findall(r"\b\d+\b", stats_proc.stdout))


def test_t3_p04b_f2_f5_f7_cold_start_concurrent_cli_and_cache_clear(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    after_iso = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    cmd = resolve_strava_mcp_cmd()
    last_env = dict(isolated_env)
    seeded = seed_config(expires_at=int(time.time()) + 7200)

    for trial in range(4):
        trial_cache_dir = tmp_path / f"p04b_cold_{trial}"
        trial_cache_dir.mkdir(mode=0o700)
        trial_env = dict(isolated_env)
        trial_env["STRAVA_MCP_CACHE_DIR"] = str(trial_cache_dir)
        last_env = trial_env

        seeded = seed_config(
            access_token=f"tok-p04b-acc-{trial}",
            refresh_token=f"tok-p04b-ref-{trial}",
            expires_at=int(time.time()) + 7200,
        )
        mock_strava.expire_access_token(seeded.access_token)
        mock_strava.clear_requests()

        barrier = threading.Barrier(6)

        def _cli_worker(cli_args: tuple[str, ...]) -> tuple[tuple[str, ...], int, str, str]:
            barrier.wait(timeout=20.0)
            proc = _run_cli_detached(trial_env, *cli_args, timeout=25.0)
            return cli_args, proc.returncode, proc.stdout, proc.stderr

        tasks = [
            ("activities", "--after", after_iso, "--json"),
            ("activities", "--after", after_iso, "--json"),
            ("activities", "--after", after_iso, "--json"),
            ("activities", "--after", after_iso, "--json"),
            ("cache", "stats", "--json"),
            ("cache", "clear", "--json"),
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(_cli_worker, tasks))

        for cli_args, rc, out_txt, err_txt in outcomes:
            assert rc == 0, f"trial={trial} {cli_args} failed rc={rc}\nstderr:\n{err_txt}"
            assert "Traceback (most recent call last)" not in err_txt
            assert "database is locked" not in err_txt.lower()
            parsed = json.loads(out_txt)
            if cli_args[0] == "activities":
                ids = [a["id"] for a in _extract_list(parsed, "activities")]
                assert ids == [9001, 9002, 9003, 9004, 9005]
            _assert_no_secrets_leaked(out_txt + "\n" + err_txt, mock_strava, seeded.mcp_token)

        tok_codes = [r.status_code for r in mock_strava.get_requests("/oauth/token", "POST")]
        assert tok_codes.count(200) == 1, (
            f"trial={trial} expected single 200 refresh, got {tok_codes}"
        )
        assert tok_codes.count(400) == 0, f"trial={trial} got 400 rotation collision: {tok_codes}"
        _assert_mode(trial_cache_dir / "cache.sqlite", 0o600)
        _assert_mode(trial_cache_dir / "cache.lock", 0o600)

    with start_http_server(cmd, last_env, seeded.mcp_token, tmp_path / "p04b_http") as info:
        mix_barrier = threading.Barrier(6)

        def _http_reader() -> list[int]:
            with open_mcp_http_session(info.mcp_url, info.token, timeout=25.0) as sess:
                mix_barrier.wait(timeout=20.0)
                data = _call_ok(
                    sess,
                    "strava_list_activities",
                    {"after": after_iso, "refresh": True},
                )
                return [a["id"] for a in _extract_list(data, "activities")]

        def _cli_mix(cli_args: tuple[str, ...]) -> dict[str, Any]:
            mix_barrier.wait(timeout=20.0)
            proc = _run_cli_detached(last_env, *cli_args, timeout=25.0)
            assert proc.returncode == 0, f"{cli_args} failed:\n{proc.stderr}"
            assert "Traceback (most recent call last)" not in proc.stderr
            return json.loads(proc.stdout)

        with ThreadPoolExecutor(max_workers=6) as pool:
            http_futs = [pool.submit(_http_reader) for _ in range(2)]
            act_futs = [
                pool.submit(_cli_mix, ("activities", "--after", after_iso, "--json"))
                for _ in range(2)
            ]
            clr_futs = [pool.submit(_cli_mix, ("cache", "clear", "--json")) for _ in range(2)]
            for hf in http_futs:
                assert hf.result(timeout=30.0) == [9001, 9002, 9003, 9004, 9005]
            for af in act_futs:
                assert [a["id"] for a in _extract_list(af.result(timeout=30.0), "activities")] == [
                    9001,
                    9002,
                    9003,
                    9004,
                    9005,
                ]
            for cf in clr_futs:
                assert "cleared" in cf.result(timeout=30.0) or isinstance(
                    cf.result(timeout=30.0), dict
                )

    final_stats = _run_cli_detached(last_env, "cache", "stats", "--json")
    assert final_stats.returncode == 0, final_stats.stderr
    assert isinstance(json.loads(final_stats.stdout)["entry_count"], int)


# =============================================================================
# Pair 5 (F4 + F5 + F7): Cache hit leaves rate limits unchanged; refresh updates
# =============================================================================


def test_t3_p05_f4_f5_f7_cache_hit_preserves_rate_limits_and_refresh_updates_status(
    mock_strava: MockStravaServer,
    stdio_session: SyncMCPSession,
) -> None:
    mock_strava.set_rate_limits(
        overall_usage_15m=15,
        overall_usage_day=150,
        read_usage_15m=10,
        read_usage_day=100,
    )
    mock_strava.clear_requests()

    after_iso = (datetime.now(UTC).date() - timedelta(days=30)).isoformat()
    first = _call_ok(stdio_session, "strava_list_activities", {"after": after_iso})
    assert [a["id"] for a in _extract_list(first, "activities")] == [
        9001,
        9002,
        9003,
        9004,
        9005,
    ]
    assert len(mock_strava.get_requests("/api/v3/athlete/activities")) == 1

    st1 = _call_ok(stdio_session, "strava_auth_status", {})
    rl1 = st1.get("rate_limits") or st1.get("rate_limit") or st1
    assert {11, 101, 16, 151} <= _collect_ints(rl1)

    second = _call_ok(stdio_session, "strava_list_activities", {"after": after_iso})
    third = _call_ok(stdio_session, "strava_list_activities", {"after": after_iso})
    assert second == first
    assert third == first
    assert len(mock_strava.get_requests("/api/v3/")) == 1

    st2 = _call_ok(stdio_session, "strava_auth_status", {})
    rl2 = st2.get("rate_limits") or st2.get("rate_limit") or st2
    nums2 = _collect_ints(rl2)
    assert {11, 101, 16, 151} <= nums2
    assert 12 not in nums2 and 102 not in nums2

    refreshed = _call_ok(
        stdio_session,
        "strava_list_activities",
        {"after": after_iso, "refresh": True},
    )
    assert [a["id"] for a in _extract_list(refreshed, "activities")] == [
        9001,
        9002,
        9003,
        9004,
        9005,
    ]
    assert len(mock_strava.get_requests("/api/v3/athlete/activities")) == 2

    st3 = _call_ok(stdio_session, "strava_auth_status", {})
    rl3 = st3.get("rate_limits") or st3.get("rate_limit") or st3
    assert {12, 102, 17, 152} <= _collect_ints(rl3)


# =============================================================================
# Pair 6 (F4 + F8): 429 on streams -> rate_limited + UTC reset + no retry loop
# =============================================================================


def test_t3_p06_f4_f8_429_on_activity_streams_returns_rate_limited_no_retry(
    mock_strava: MockStravaServer,
    stdio_session: SyncMCPSession,
) -> None:
    mock_strava.inject_fault(
        fault_type="rate_limit_429",
        path_pattern=r"/api/v3/activities/9001/streams$",
        times=1,
    )
    mock_strava.clear_requests()

    keys_arg = _stream_keys_arg(stdio_session, ["time", "heartrate", "watts"])
    res = stdio_session.call_tool(
        "strava_get_activity_streams",
        {"activity_id": 9001, "keys": keys_arg, "max_points": 50},
    )
    err = _extract_error_dict(res)
    assert bool(getattr(res, "is_error", False)) or err.get("code") == "rate_limited"
    assert err.get("code") == "rate_limited"
    dts = _collect_iso_datetimes(err.get("reset_at_utc") or err.get("reset_at") or err)
    assert dts, f"Expected UTC reset timestamp in rate_limited error: {err!r}"
    assert any(dt.minute in {0, 15, 30, 45} and dt.second == 0 for dt in dts)
    assert float(err.get("retry_after_seconds", 0)) > 0
    assert len(mock_strava.get_requests("/api/v3/activities/9001/streams")) == 1

    ok_data = _call_ok(
        stdio_session,
        "strava_get_activity_streams",
        {"activity_id": 9001, "keys": keys_arg, "max_points": 50},
    )
    streams = _extract_streams_map(ok_data)
    assert set(streams.keys()) == {"time", "heartrate", "watts"}
    for key in ("time", "heartrate", "watts"):
        assert 2 <= len(streams[key]["data"]) <= 50
    assert len(mock_strava.get_requests("/api/v3/activities/9001/streams")) == 2


# =============================================================================
# Pair 7 (F7 + F9): summarize_metrics includes private/followers beyond /stats
# =============================================================================


def test_t3_p07_f7_f9_summarize_metrics_includes_private_beyond_athlete_stats(
    stdio_session: SyncMCPSession,
) -> None:
    stats_data = _call_ok(stdio_session, "strava_get_athlete_stats", {})
    stats = stats_data.get("stats") if isinstance(stats_data.get("stats"), dict) else stats_data
    recent_ride = stats["recent_ride_totals"]
    recent_run = stats["recent_run_totals"]
    assert recent_ride["count"] == 1
    assert math.isclose(recent_ride["distance"], 64373.8, rel_tol=0.01) or _has_approx_number(
        recent_ride, 64.37, rel_tol=0.02
    )
    assert recent_run["count"] == 0
    assert stats["all_run_totals"]["count"] == 0

    acts_data = _call_ok(stdio_session, "strava_list_activities", {"after": "30d", "limit": 20})
    items = _extract_list(acts_data, "activities")
    assert {a["id"] for a in items} == {9001, 9002, 9003, 9004, 9005}
    ride_family = [
        a
        for a in items
        if a.get("sport_type") in {"Ride", "GravelRide", "MountainBikeRide"}
        or a.get("type") == "Ride"
    ]
    run_family = [
        a
        for a in items
        if a.get("sport_type") in {"Run", "TrailRun"} or a.get("type") == "Run"
    ]
    assert len(ride_family) == 2
    assert len(run_family) == 2

    tot_data = _call_ok(
        stdio_session,
        "strava_summarize_metrics",
        {"after": "30d", "group_by": "total"},
    )
    tot_groups = _extract_metric_groups(tot_data, group_by="total")
    tot_count = sum(
        _metric_field(g, "count", "activity_count", "activities_count", "activities")
        for g in tot_groups
    )
    assert tot_count == 5
    assert tot_count > recent_ride["count"] + recent_run["count"]
    assert _has_approx_number(tot_data, 118967.2, rel_tol=0.01) or _has_approx_number(
        tot_data, 118.97, rel_tol=0.02
    )
    assert _has_approx_number(tot_data, 73.92, rel_tol=0.02)
    assert _has_approx_number(tot_data, 24600, rel_tol=0.01)
    assert _has_approx_number(tot_data, 2235.0, rel_tol=0.01)
    tot_dist_m = sum(
        _metric_field(
            g, "distance", "distance_m", "distance_meters", "total_distance", "distance_km"
        )
        for g in tot_groups
    )
    if tot_dist_m < 1000:
        tot_dist_m *= 1000.0
    assert tot_dist_m > float(recent_ride["distance"]) + float(recent_run["distance"])

    sport_data = _call_ok(
        stdio_session,
        "strava_summarize_metrics",
        {"after": "30d", "group_by": "sport", "refresh": True},
    )
    sport_groups = _extract_metric_groups(sport_data, group_by="sport")
    for sport_name, expected_m, expected_km in (
        ("GravelRide", 64373.8, 64.37),
        ("MountainBikeRide", 28500.0, 28.5),
        ("TrailRun", 16093.4, 16.09),
        ("Run", 10000.0, 10.0),
        ("WeightTraining", 0.0, 0.0),
    ):
        exact = [
            g
            for g in sport_groups
            if any(v == sport_name for v in g.values() if isinstance(v, str))
            or g.get("_group_key") == sport_name
        ]
        matching = exact if exact else [g for g in sport_groups if _bucket_has_sport(g, sport_name)]
        assert len(matching) >= 1, f"Missing sport group {sport_name!r} in {sport_groups!r}"
        bucket = matching[0]
        assert _has_approx_number(bucket, expected_m, rel_tol=0.01) or _has_approx_number(
            bucket, expected_km, rel_tol=0.02
        )
        if sport_name == "WeightTraining":
            assert (
                _metric_field(
                    bucket,
                    "distance",
                    "distance_m",
                    "distance_meters",
                    "total_distance",
                    "distance_km",
                )
                == 0.0
            )


# =============================================================================
# Pair 8 (F6 + F10): Athlete bike/shoe IDs feed directly into strava_get_gear
# =============================================================================


def test_t3_p08_f6_f10_athlete_gear_ids_feed_get_gear(
    stdio_session: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    mock_strava.clear_requests()
    ath_data = _call_ok(stdio_session, "strava_get_athlete", {})
    athlete = ath_data.get("athlete") if isinstance(ath_data.get("athlete"), dict) else ath_data
    bike_ids = [b["id"] for b in athlete.get("bikes", []) if isinstance(b, dict)]
    shoe_ids = [s["id"] for s in athlete.get("shoes", []) if isinstance(s, dict)]
    assert "b1001" in bike_ids
    assert "g2001" in shoe_ids

    bike_id = next(bid for bid in bike_ids if bid == "b1001")
    shoe_id = next(sid for sid in shoe_ids if sid == "g2001")

    bike_data = _call_ok(stdio_session, "strava_get_gear", {"gear_id": bike_id})
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

    shoe_data = _call_ok(stdio_session, "strava_get_gear", {"gear_id": shoe_id})
    shoe = shoe_data.get("gear") if isinstance(shoe_data.get("gear"), dict) else shoe_data
    assert shoe["id"] == "g2001"
    assert shoe["name"] == "Hoka Speedgoat 5"
    assert shoe["brand_name"] == "Hoka"
    assert _has_approx_number(shoe_data, 620000.0, rel_tol=0.01) or _has_approx_number(
        shoe_data, 620.0, rel_tol=0.01
    )
    assert _has_approx_number(shoe_data, 385.25, rel_tol=0.02)

    assert any(r.status_code == 200 for r in mock_strava.get_requests("/api/v3/athlete", "GET"))
    assert any(r.status_code == 200 for r in mock_strava.get_requests("/api/v3/gear/b1001", "GET"))
    assert any(r.status_code == 200 for r in mock_strava.get_requests("/api/v3/gear/g2001", "GET"))


# =============================================================================
# Pair 9 (F7 + F8): list_activities feeds manual 404 & downsampled GPS streams
# =============================================================================


def test_t3_p09_f7_f8_list_activities_feeds_manual_404_and_downsampled_streams(
    mcp_http_client: SyncMCPSession,
    mock_strava: MockStravaServer,
) -> None:
    mock_strava.clear_requests()
    listed = _call_ok(mcp_http_client, "strava_list_activities", {"limit": 20})
    items = _extract_list(listed, "activities")

    manual_act = next(a for a in items if a.get("manual") is True)
    gravel_act = next(a for a in items if a.get("sport_type") == "GravelRide")
    manual_id = int(manual_act["id"])
    gravel_id = int(gravel_act["id"])
    assert manual_id == 9004
    assert manual_act.get("sport_type") == "WeightTraining"
    assert gravel_id == 9001

    res_manual = mcp_http_client.call_tool(
        "strava_get_activity_streams",
        {
            "activity_id": manual_id,
            "keys": _stream_keys_arg(mcp_http_client, ["time", "distance"]),
        },
    )
    manual_reqs = mock_strava.get_requests(f"/api/v3/activities/{manual_id}/streams", "GET")
    assert any(r.status_code == 404 for r in manual_reqs)
    if bool(getattr(res_manual, "is_error", False)):
        assert _extract_error_dict(res_manual).get("code") == "not_found"
    else:
        manual_parsed = parse_tool_result(res_manual)
        assert manual_parsed.get("code") in (None, "not_found")
        assert _extract_streams_map(manual_parsed) == {}

    keys = ["time", "heartrate", "watts", "altitude"]
    gravel_data = _call_ok(
        mcp_http_client,
        "strava_get_activity_streams",
        {
            "activity_id": gravel_id,
            "keys": _stream_keys_arg(mcp_http_client, keys),
            "max_points": 40,
        },
    )
    streams = _extract_streams_map(gravel_data)
    assert set(streams.keys()) == set(keys)
    for k in keys:
        pts = streams[k]["data"]
        assert 2 <= len(pts) <= 40
        assert streams[k].get("original_size") == 1200
    assert streams["time"]["data"][0] == 0
    assert streams["time"]["data"][-1] >= 7900
    gravel_reqs = mock_strava.get_requests(f"/api/v3/activities/{gravel_id}/streams", "GET")
    assert any(r.query.get("key_by_type") == ["true"] for r in gravel_reqs)


# =============================================================================
# Pair 10 (F3 + F10): OAuth consent opt-out of read_all filters/blocks routes
# =============================================================================


def test_t3_p10_f3_f10_oauth_opt_out_read_all_filters_and_blocks_private_route(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config(include_tokens=False)
    mock_strava.configure_oauth(
        consent_scopes=["read", "profile:read_all", "activity:read", "activity:read_all"]
    )

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

    _assert_mode(seeded.tokens_file, 0o600)
    toks = _read_tokens_file(seeded.tokens_file)
    scopes = _parse_scopes_any(toks.get("scope") or toks.get("scopes"))
    assert "read_all" not in scopes
    assert "activity:read_all" in scopes

    cmd = [*resolve_strava_mcp_cmd(), "stdio"]
    with open_stdio_session(cmd, isolated_env, tmp_path / "p10_stdio.err") as session:
        listed = _call_ok(session, "strava_list_routes", {})
        routes = _extract_list(listed, "routes")
        assert {int(r["id"]) for r in routes} == {7001}

        pub_data = _call_ok(
            session,
            "strava_get_route",
            {"route_id": 7001, "include_export": "none"},
        )
        pub_route = (
            pub_data.get("route") if isinstance(pub_data.get("route"), dict) else pub_data
        )
        assert pub_route["name"] == "Forest Park Gravel Loop"

        priv_res = session.call_tool(
            "strava_get_route",
            {"route_id": 7002, "include_export": "none"},
        )
        err = _extract_error_dict(priv_res)
        assert (
            bool(getattr(priv_res, "is_error", False))
            or err.get("code") == "insufficient_scope"
        )
        assert err.get("code") == "insufficient_scope"
        err_text = json.dumps(err)
        assert "read_all" in err_text
        assert "strava-mcp auth" in err_text


# =============================================================================
# Pair 11 (F11 + F5 + F6): strava-mcp-headers + HTTP resource + tool cache hit
# =============================================================================


def test_t3_p11_f11_f5_f6_headers_helper_http_resource_and_tool_cache_hit(
    http_server: Any,
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
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
    hdr_payload = json.loads(hdr_proc.stdout)
    assert hdr_payload.get("Authorization") == f"Bearer {http_server.token}"
    token = hdr_payload["Authorization"].removeprefix("Bearer ").strip()

    mock_strava.clear_requests()
    with open_mcp_http_session(http_server.mcp_url, token) as session:
        res_uri = session.read_resource("strava://athlete/profile")
        assert res_uri.contents
        res_text = getattr(res_uri.contents[0], "text", "")
        res_json = json.loads(res_text)
        res_ath = (
            res_json.get("athlete") if isinstance(res_json.get("athlete"), dict) else res_json
        )
        assert res_ath["id"] == DEFAULT_TEST_ATHLETE_ID
        assert res_ath["firstname"] == "Alex"
        assert res_ath["ftp"] == 285
        assert len(mock_strava.get_requests("/api/v3/athlete", "GET")) == 1

        res_tool = session.call_tool("strava_get_athlete", {})
        assert res_tool.is_error is not True
        assert isinstance(res_tool.structured_content, dict)
        assert res_tool.content and isinstance(res_tool.content[0].text, str)
        assert json.loads(res_tool.content[0].text) == res_tool.structured_content
        tool_data = parse_tool_result(res_tool)
        tool_ath = (
            tool_data.get("athlete") if isinstance(tool_data.get("athlete"), dict) else tool_data
        )
        assert tool_ath["id"] == DEFAULT_TEST_ATHLETE_ID
        assert tool_ath["ftp"] == 285

        res_tool2 = session.call_tool("strava_get_athlete", {})
        assert parse_tool_result(res_tool2) == tool_data
        assert len(mock_strava.get_requests("/api/v3/athlete", "GET")) == 1

    _assert_no_secrets_leaked(
        "\n".join([res_text, json.dumps(tool_data), http_server.stderr_text]),
        mock_strava,
        DEFAULT_TEST_CLIENT_SECRET,
        http_server.token,
    )


# =============================================================================
# Pair 12 (F1 + F2): revoke clears local tokens & status fails strictly
# =============================================================================


def test_t3_p12_f1_f2_revoke_clears_local_tokens_and_status_fails_strictly(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()
    mock_strava.clear_requests()

    rev_proc = run_cli("revoke", check=True)
    rev_reqs = mock_strava.get_requests("/oauth/revoke", "POST")
    assert len(rev_reqs) == 1
    assert rev_reqs[0].status_code == 200
    assert _auth_header(rev_reqs[0]).startswith("Basic ")

    active = mock_strava.get_active_tokens()
    assert seeded.access_token not in active["active_access_tokens"]
    assert seeded.refresh_token in active["invalidated_refresh_tokens"]

    if seeded.tokens_file.exists():
        raw = seeded.tokens_file.read_text(encoding="utf-8")
        assert seeded.access_token not in raw
        assert seeded.refresh_token not in raw
        parsed_toks = json.loads(raw) if raw.strip() else {}
        assert (
            (not parsed_toks.get("access_token") and not parsed_toks.get("refresh_token"))
            or parsed_toks.get("revoked") is True
        )

    status_proc = run_cli("status", "--check-live")
    assert status_proc.returncode != 0
    combined_low = (status_proc.stdout + "\n" + status_proc.stderr).lower()
    assert not re.search(r'"?authenticated"?\s*[:=]\s*true\b', combined_low)
    has_explicit_false = bool(re.search(r'"?authenticated"?\s*[:=]\s*false\b', combined_low))
    cleaned_low = re.sub(
        r'(?:not\s+authenticated|"?authenticated"?\s*[:=]\s*false\b)',
        "",
        combined_low,
    )
    assert not re.search(r"\bauthenticated\b", cleaned_low)
    assert has_explicit_false or any(
        kw in combined_low
        for kw in (
            "unauthenticated",
            "not authenticated",
            "revoked",
            "auth_missing",
            "missing",
            "no token",
        )
    )

    ref_proc = run_cli("refresh")
    assert ref_proc.returncode != 0

    _assert_no_secrets_leaked(
        "\n".join(
            p.stdout + p.stderr for p in (rev_proc, status_proc, ref_proc)
        ),
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )
