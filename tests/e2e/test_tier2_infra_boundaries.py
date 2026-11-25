"""Tier 2 opaque-box E2E boundary & error tests for F1-F5 and F11."""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from conftest import (
    DEFAULT_TEST_ATHLETE_ID,
    DEFAULT_TEST_CLIENT_ID,
    DEFAULT_TEST_CLIENT_SECRET,
    REPO_ROOT,
    SeededConfig,
    open_mcp_http_session,
    open_stdio_session,
    parse_tool_result,
    resolve_strava_mcp_cmd,
    start_http_server,
)
from mcp.shared.exceptions import MCPError
from mcp.types import LATEST_PROTOCOL_VERSION
from mock_strava import MockStravaServer


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _assert_cli_error(
    proc: subprocess.CompletedProcess[str],
    expected_codes: tuple[str, ...],
    keywords: tuple[str, ...],
) -> None:
    assert proc.returncode != 0, (
        f"Expected non-zero exit code, got 0.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    combined = proc.stdout + "\n" + proc.stderr
    assert "Traceback (most recent call last)" not in combined, (
        f"Unhandled traceback in CLI output:\n{combined}"
    )
    for text in (proc.stdout, proc.stderr):
        stripped = text.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            code = parsed.get("code") or (
                parsed.get("error", {}).get("code")
                if isinstance(parsed.get("error"), dict)
                else None
            )
            if isinstance(code, str):
                assert code in expected_codes, f"Unexpected error code {code!r} in {parsed!r}"
                return
    low = combined.lower()
    assert any(c.lower() in low for c in expected_codes) or any(
        kw.lower() in low for kw in keywords
    ), f"Expected error code/keywords {expected_codes}/{keywords} in:\n{combined}"


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


# =============================================================================
# F1: Config & Secret Store Boundaries (6 test functions)
# =============================================================================


def test_f1_tier2_missing_config_dir_and_files_returns_config_missing(
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "does_not_exist"
    assert not missing_dir.exists()

    p_status = run_cli("status", env_overrides={"STRAVA_MCP_CONFIG_DIR": str(missing_dir)})
    _assert_cli_error(p_status, ("config_missing",), ("missing", "config", "init"))

    p_refresh = run_cli("refresh", env_overrides={"STRAVA_MCP_CONFIG_DIR": str(missing_dir)})
    _assert_cli_error(p_refresh, ("config_missing",), ("missing", "config", "init"))

    # Empty existing 0700 config dir (no client.json / tokens.json)
    p_empty = run_cli("status")
    _assert_cli_error(p_empty, ("config_missing", "auth_missing"), ("missing", "config", "init"))


@pytest.mark.parametrize("target", ["tokens.json", "client.json", "mcp-token"])
def test_f1_tier2_insecure_permissions_0644_refused(
    target: str,
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(dir_mode=0o700, file_mode=0o600)
    os.chmod(seeded.config_dir / target, 0o644)

    if target == "mcp-token":
        proc = run_cli("serve", "--listen", f"127.0.0.1:{_allocate_free_port()}")
    else:
        proc = run_cli("status")

    _assert_cli_error(
        proc,
        ("insecure_permissions",),
        ("permission", "0600", "0700", "insecure"),
    )
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


def test_f1_tier2_insecure_group_readable_0640_refused(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(file_mode=0o640)
    proc = run_cli("status")
    _assert_cli_error(
        proc,
        ("insecure_permissions",),
        ("permission", "0600", "insecure"),
    )
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


def test_f1_tier2_malformed_json_returns_invalid_config(
    seed_config: Callable[..., SeededConfig],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()
    valid_client = seeded.client_file.read_text(encoding="utf-8")

    seeded.client_file.write_text("{bad json", encoding="utf-8")
    os.chmod(seeded.client_file, 0o600)
    p_client = run_cli("status")
    _assert_cli_error(p_client, ("invalid_config",), ("invalid", "json", "parse", "config"))

    seeded.client_file.write_text(valid_client, encoding="utf-8")
    os.chmod(seeded.client_file, 0o600)
    seeded.tokens_file.write_text("{bad json", encoding="utf-8")
    os.chmod(seeded.tokens_file, 0o600)
    p_tokens = run_cli("status")
    _assert_cli_error(p_tokens, ("invalid_config",), ("invalid", "json", "parse", "config"))


def test_f1_tier2_env_vars_never_authenticate_or_leak_secrets(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    env_secrets = {
        "STRAVA_CLIENT_SECRET": "env-secret-leak-001",
        "STRAVA_ACCESS_TOKEN": "env-access-leak-001",
        "STRAVA_REFRESH_TOKEN": "env-refresh-leak-001",
        "STRAVA_MCP_TOKEN": "env-mcp-leak-001",
    }
    p_unseeded = run_cli("status", "--check-live", env_overrides=env_secrets)
    _assert_cli_error(p_unseeded, ("config_missing", "auth_missing"), ("missing", "config", "init"))
    _assert_no_secrets_leaked(
        p_unseeded.stdout + p_unseeded.stderr,
        mock_strava,
        *env_secrets.values(),
    )

    seeded = seed_config(client_secret="wrong-secret-for-fail", sync_mock=False)
    mock_strava.set_client_credentials(DEFAULT_TEST_CLIENT_ID, DEFAULT_TEST_CLIENT_SECRET)
    p_status = run_cli("status", env_overrides=env_secrets, check=True)
    p_fail = run_cli("refresh", env_overrides=env_secrets)
    assert p_fail.returncode != 0
    _assert_no_secrets_leaked(
        p_status.stdout + p_status.stderr + p_fail.stdout + p_fail.stderr,
        mock_strava,
        *env_secrets.values(),
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


def test_f1_tier2_headers_helper_refuses_missing_or_insecure_mcp_token(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
) -> None:
    headers_script = REPO_ROOT / "client" / "strava-mcp-headers"
    if not headers_script.is_file() or not os.access(headers_script, os.X_OK):
        pytest.fail(
            f"client/strava-mcp-headers missing or not executable: {headers_script}",
            pytrace=False,
        )

    seed_config(include_mcp_token=False)
    p_missing = subprocess.run(
        [str(headers_script)],
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert p_missing.returncode != 0
    assert p_missing.stdout.strip() == ""
    assert p_missing.stderr.strip()

    seeded = seed_config(dir_mode=0o700, file_mode=0o600)
    os.chmod(seeded.mcp_token_file, 0o644)
    p_insecure = subprocess.run(
        [str(headers_script)],
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert p_insecure.returncode != 0
    assert p_insecure.stdout.strip() == ""
    assert any(w in p_insecure.stderr.lower() for w in ("perm", "0600", "insecure", "mode"))
    _assert_no_secrets_leaked(
        p_insecure.stdout + p_insecure.stderr,
        mock_strava,
        seeded.mcp_token,
    )


# =============================================================================
# F2: OAuth2 Lifecycle & Concurrency Boundaries (5 tests)
# =============================================================================


def test_f2_tier2_concurrent_refresh_rotation_race_serialized(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 7200)
    mock_strava.expire_access_token(seeded.access_token)
    mock_strava.clear_requests()

    barrier = threading.Barrier(6)

    with start_http_server(
        resolve_strava_mcp_cmd(), isolated_env, seeded.mcp_token, tmp_path
    ) as info:

        def _worker(_: int) -> int:
            with open_mcp_http_session(info.mcp_url, info.token) as sess:
                barrier.wait(timeout=10.0)
                res = sess.call_tool("strava_get_athlete", {})
                assert res.is_error is not True
                return int(parse_tool_result(res)["id"])

        with ThreadPoolExecutor(max_workers=6) as pool:
            ids = list(pool.map(_worker, range(6)))

    assert ids == [DEFAULT_TEST_ATHLETE_ID] * 6
    tok_reqs = mock_strava.get_requests("/oauth/token", "POST")
    codes = [r.status_code for r in tok_reqs]
    assert codes.count(200) == 1, f"Expected single serialized refresh, got {codes}"
    assert codes.count(400) == 0, f"Detected refresh rotation race 400s: {codes}"

    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] != seeded.access_token
    assert updated["refresh_token"] != seeded.refresh_token


def test_f2_tier2_refresh_omitting_scope_and_athlete_preserves_stored(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(
        expires_at=int(time.time()) + 60,
        scope="read,activity:read",
        athlete_id=DEFAULT_TEST_ATHLETE_ID,
    )
    mock_strava.configure_oauth(include_scope_on_refresh=False, include_athlete_on_refresh=False)

    run_cli("refresh", check=True)
    updated = _read_tokens_file(seeded.tokens_file)
    assert _parse_scopes_any(updated.get("scope")) == {"read", "activity:read"}
    athlete = updated.get("athlete_id") or (updated.get("athlete") or {}).get("id")
    assert athlete == DEFAULT_TEST_ATHLETE_ID


def test_f2_tier2_refresh_within_gt_1h_window_handles_same_access_token(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 18000)
    mock_strava.configure_oauth(enforce_one_hour_window=True, rotate_within_hour=False)

    run_cli("refresh", check=True)
    mode = stat.S_IMODE(seeded.tokens_file.stat().st_mode)
    assert mode == 0o600
    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] == seeded.access_token
    assert _parse_scopes_any(updated.get("scope")) == _parse_scopes_any(seeded.scope)
    athlete = updated.get("athlete_id") or (updated.get("athlete") or {}).get("id")
    assert athlete == DEFAULT_TEST_ATHLETE_ID

    live_proc = run_cli("status", "--check-live", check=True)
    assert live_proc.returncode == 0


def test_f2_tier2_stale_or_revoked_refresh_token_returns_refresh_failed(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(
        expires_at=int(time.time()) - 60,
        refresh_token="stale-revoked-refresh-token",
        sync_mock=False,
    )
    mock_strava.set_client_credentials(seeded.client_id, seeded.client_secret)

    proc = run_cli("refresh")
    _assert_cli_error(
        proc,
        ("refresh_failed", "auth_expired"),
        ("refresh", "failed", "invalid", "re-authorize", "auth"),
    )
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
    )


def test_f2_tier2_revoke_token_type_access_vs_refresh_token(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded_a = seed_config(
        access_token="tok-revoke-access-1",
        refresh_token="ref-revoke-access-1",
    )
    mock_strava.clear_requests()
    run_cli("revoke", "--token-type", "access_token", check=True)
    reqs_a = mock_strava.get_requests("/oauth/revoke", "POST")
    assert len(reqs_a) == 1
    body_a = parse_qs(reqs_a[0].body)
    assert body_a.get("token") == [seeded_a.access_token]
    assert body_a.get("token_type_hint") == ["access_token"]
    assert seeded_a.access_token not in mock_strava.get_active_tokens()["active_access_tokens"]

    mock_strava.reset()
    seeded_b = seed_config(
        access_token="tok-revoke-refresh-2",
        refresh_token="ref-revoke-refresh-2",
    )
    mock_strava.clear_requests()
    run_cli("revoke", "--token-type", "refresh_token", check=True)
    reqs_b = mock_strava.get_requests("/oauth/revoke", "POST")
    assert len(reqs_b) == 1
    body_b = parse_qs(reqs_b[0].body)
    assert body_b.get("token") == [seeded_b.refresh_token]
    assert body_b.get("token_type_hint") == ["refresh_token"]
    assert seeded_b.refresh_token in mock_strava.get_active_tokens()["invalidated_refresh_tokens"]


# =============================================================================
# F3: OAuth2 Auth Helper Boundaries (5 tests)
# =============================================================================


def test_f3_tier2_auth_csrf_state_mismatch_preserves_existing_tokens(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seeded = seed_config(
        access_token="keep-me-access-token",
        refresh_token="keep-me-refresh-token",
    )
    mock_strava.clear_requests()
    port = _allocate_free_port()
    proc, _auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port)]
    )
    try:
        resp = httpx.get(
            f"http://127.0.0.1:{port}/callback",
            params={"code": "forged-code", "state": "wrong-csrf-state"},
            timeout=5.0,
        )
        assert resp.status_code in (400, 403), (
            f"Expected 400/403 on CSRF mismatch, got {resp.status_code}: {resp.text}"
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
        finish(2.0)

    assert mock_strava.get_requests("/oauth/token") == []
    saved = _read_tokens_file(seeded.tokens_file)
    assert saved["access_token"] == "keep-me-access-token"
    assert saved["refresh_token"] == "keep-me-refresh-token"


def test_f3_tier2_auth_access_denied_callback_fails_cleanly(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seeded = seed_config(include_tokens=False)
    mock_strava.configure_oauth(deny_access=True)
    mock_strava.clear_requests()
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port)]
    )
    try:
        auth_resp, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert "error=access_denied" in auth_resp.headers["Location"]
        assert cb_resp.status_code in (200, 400, 403)
        rc, out_txt, err_txt = finish(10.0)
        assert rc != 0
        combined = (out_txt + "\n" + err_txt).lower()
        assert "access_denied" in combined or "denied" in combined
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    assert mock_strava.get_requests("/oauth/token") == []
    assert not seeded.tokens_file.exists()


def test_f3_tier2_auth_callback_and_code_replay_rejected(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seed_config(include_tokens=False)
    mock_strava.clear_requests()
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port)]
    )
    try:
        auth_resp = httpx.get(auth_url, follow_redirects=False, timeout=5.0)
        assert auth_resp.status_code == 302
        location = auth_resp.headers["Location"]
        cb_resp = httpx.get(location, follow_redirects=False, timeout=5.0)
        assert cb_resp.status_code == 200
        rc, out_txt, err_txt = finish(10.0)
        assert rc == 0, f"auth failed: {out_txt}\n{err_txt}"

        try:
            replay_cb = httpx.get(location, follow_redirects=False, timeout=2.0)
            assert replay_cb.status_code >= 400
        except httpx.HTTPError:
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    assert len(mock_strava.get_requests("/oauth/token")) == 1


def test_f3_tier2_auth_callback_response_cache_control_no_store(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
) -> None:
    seed_config(include_tokens=False)
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port)]
    )
    try:
        _, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert cb_resp.status_code == 200
        assert "no-store" in cb_resp.headers.get("cache-control", "").lower()
        rc, out_txt, err_txt = finish(10.0)
        assert rc == 0, f"auth failed: {out_txt}\n{err_txt}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_f3_tier2_auth_scope_opt_out_persists_subset_and_warns(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(include_tokens=False)
    mock_strava.configure_oauth(consent_scopes=["read", "activity:read"])
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port)]
    )
    try:
        _, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert cb_resp.status_code == 200
        rc, out_txt, err_txt = finish(10.0)
        assert rc == 0, f"auth failed: {out_txt}\n{err_txt}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    saved = _read_tokens_file(seeded.tokens_file)
    scopes = _parse_scopes_any(saved.get("scope"))
    assert {"read", "activity:read"} <= scopes
    assert "activity:read_all" not in scopes
    assert "profile:read_all" not in scopes

    status_proc = run_cli("status", check=True)
    combined = (out_txt + "\n" + err_txt + "\n" + status_proc.stdout + "\n" + status_proc.stderr)
    assert "activity:read_all" in combined or "profile:read_all" in combined


# =============================================================================
# F4: Rate-Limit Boundaries (6 tests)
# =============================================================================


def test_f4_tier2_429_read_limit_structured_details_and_no_retry(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.set_rate_limits(
        read_limit_15m=200,
        read_usage_15m=200,
        overall_limit_15m=400,
        overall_usage_15m=50,
    )
    mock_strava.clear_requests()

    res = stdio_session.call_tool("strava_get_athlete", {})
    err = _extract_error_dict(res)
    assert err.get("code") == "rate_limited"
    limit_type = str(err.get("limit_type", "")).lower()
    assert "read" in limit_type
    assert err.get("usage") in (200, 201)
    assert err.get("limit") == 200
    assert err.get("reset_at_utc") or err.get("reset_at")
    assert float(err.get("retry_after_seconds", 0)) > 0
    assert len(mock_strava.get_requests("/api/v3/athlete")) == 1


def test_f4_tier2_429_overall_limit_distinguished_from_read_limit(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.set_rate_limits(
        overall_limit_15m=400,
        overall_usage_15m=400,
        read_limit_15m=200,
        read_usage_15m=25,
    )
    mock_strava.clear_requests()

    res = stdio_session.call_tool("strava_get_athlete", {})
    err = _extract_error_dict(res)
    assert err.get("code") == "rate_limited"
    limit_type = str(err.get("limit_type", "")).lower()
    assert "overall" in limit_type and "read" not in limit_type
    assert err.get("usage") in (400, 401)
    assert err.get("limit") == 400
    assert len(mock_strava.get_requests("/api/v3/athlete")) == 1


def test_f4_tier2_boundary_199_then_200_then_429(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.set_rate_limits(read_limit_15m=200, read_usage_15m=198)
    mock_strava.clear_requests()

    r1 = stdio_session.call_tool("strava_get_athlete", {})
    assert r1.is_error is not True

    r2 = stdio_session.call_tool("strava_get_athlete_stats", {})
    assert r2.is_error is not True

    r3 = stdio_session.call_tool("strava_list_clubs", {})
    err3 = _extract_error_dict(r3)
    assert r3.is_error is True or err3.get("code") == "rate_limited"
    assert err3.get("code") == "rate_limited"
    assert len(mock_strava.get_requests("/api/v3/athlete/clubs")) <= 1


def test_f4_tier2_utc_reset_distinction_15m_vs_daily(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config()

    mock_strava.set_rate_limits(
        read_limit_15m=200,
        read_usage_15m=200,
        read_limit_day=2000,
        read_usage_day=50,
    )
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_reset_15m.err",
    ) as s15:
        err_15 = _extract_error_dict(s15.call_tool("strava_get_athlete", {}))
    assert err_15.get("code") == "rate_limited"
    dts_15 = _collect_iso_datetimes(err_15.get("reset_at_utc") or err_15)
    assert dts_15
    assert dts_15[0].minute in {0, 15, 30, 45}
    assert dts_15[0].second == 0
    retry_15 = float(err_15.get("retry_after_seconds", 0))
    assert 0 < retry_15 <= 900

    mock_strava.set_rate_limits(
        read_limit_15m=200,
        read_usage_15m=5,
        read_limit_day=2000,
        read_usage_day=2000,
    )
    now_utc = datetime.now(tz=UTC)
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_reset_day.err",
    ) as sday:
        err_day = _extract_error_dict(sday.call_tool("strava_get_athlete_stats", {}))
    assert err_day.get("code") == "rate_limited"
    dts_day = _collect_iso_datetimes(err_day.get("reset_at_utc") or err_day)
    assert dts_day
    assert dts_day[0].hour == 0 and dts_day[0].minute == 0 and dts_day[0].second == 0
    assert dts_day[0].date() == (now_utc + timedelta(days=1)).date()
    retry_day = float(err_day.get("retry_after_seconds", 0))
    assert 0 < retry_day <= 86400


def test_f4_tier2_malformed_rate_limit_headers_graceful_fallback(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.inject_fault(
        fault_type="status_error",
        path_pattern=r"/athlete$",
        times=1,
        status_code=200,
        body={"id": DEFAULT_TEST_ATHLETE_ID, "firstname": "Alex", "lastname": "Rider"},
        headers={
            "X-RateLimit-Limit": "bad",
            "X-RateLimit-Usage": "nan,x",
            "X-ReadRateLimit-Limit": "",
            "X-ReadRateLimit-Usage": "onlyone",
        },
    )
    res = stdio_session.call_tool("strava_get_athlete", {})
    assert res.is_error is not True
    assert parse_tool_result(res)["id"] == DEFAULT_TEST_ATHLETE_ID

    status_res = stdio_session.call_tool("strava_auth_status", {})
    assert status_res.is_error is not True


def test_f4_tier2_missing_rate_limit_headers_graceful_fallback(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config(sync_mock=False)
    payload = json.dumps(
        {
            "id": DEFAULT_TEST_ATHLETE_ID,
            "username": "alex_rider",
            "firstname": "Alex",
            "lastname": "Rider",
            "resource_state": 3,
        }
    ).encode("utf-8")

    class _NoRateHeadersHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/api/v3/athlete"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _NoRateHeadersHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(isolated_env)
        env["STRAVA_API_BASE_URL"] = f"http://127.0.0.1:{port}/api/v3"
        with open_stdio_session(
            [*resolve_strava_mcp_cmd(), "stdio"],
            env,
            tmp_path / "stdio_no_rl_headers.err",
        ) as session:
            ath = session.call_tool("strava_get_athlete", {})
            assert ath.is_error is not True
            assert parse_tool_result(ath)["id"] == DEFAULT_TEST_ATHLETE_ID
            status = session.call_tool("strava_auth_status", {})
            assert status.is_error is not True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# =============================================================================
# F5: Cache Boundaries (6 tests)
# =============================================================================


def _extract_activity_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("activities", "items", "results", "data"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    for val in payload.values():
        if isinstance(val, list):
            return val
    raise AssertionError(f"Could not locate activity list in payload: {payload!r}")


def test_f5_tier2_cache_clear_when_empty_or_missing_db(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seed_config()
    cache_db = Path(isolated_env["STRAVA_MCP_CACHE_DIR"]) / "cache.sqlite"
    assert not cache_db.exists()

    p1 = run_cli("cache", "clear", check=True)
    p2 = run_cli("cache", "clear", check=True)
    assert "Traceback (most recent call last)" not in (p1.stdout + p1.stderr)
    assert "Traceback (most recent call last)" not in (p2.stdout + p2.stderr)


def test_f5_tier2_distinct_cache_keys_for_filter_params(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.clear_requests()

    gravel = _extract_activity_list(
        parse_tool_result(
            stdio_session.call_tool("strava_list_activities", {"sport_type": "GravelRide"})
        )
    )
    trail = _extract_activity_list(
        parse_tool_result(
            stdio_session.call_tool("strava_list_activities", {"sport_type": "TrailRun"})
        )
    )
    week = _extract_activity_list(
        parse_tool_result(stdio_session.call_tool("strava_list_activities", {"after": "7d"}))
    )
    month = _extract_activity_list(
        parse_tool_result(stdio_session.call_tool("strava_list_activities", {"after": "30d"}))
    )

    assert gravel and all(a.get("sport_type") == "GravelRide" for a in gravel)
    assert trail and all(a.get("sport_type") == "TrailRun" for a in trail)
    assert {a["id"] for a in gravel} != {a["id"] for a in trail}
    assert len(month) > len(week)

    reqs_after_first = len(mock_strava.get_requests("/api/v3/athlete/activities"))
    assert reqs_after_first >= 1

    gravel_2 = _extract_activity_list(
        parse_tool_result(
            stdio_session.call_tool("strava_list_activities", {"sport_type": "GravelRide"})
        )
    )
    trail_2 = _extract_activity_list(
        parse_tool_result(
            stdio_session.call_tool("strava_list_activities", {"sport_type": "TrailRun"})
        )
    )
    week_2 = _extract_activity_list(
        parse_tool_result(stdio_session.call_tool("strava_list_activities", {"after": "7d"}))
    )
    month_2 = _extract_activity_list(
        parse_tool_result(stdio_session.call_tool("strava_list_activities", {"after": "30d"}))
    )

    assert gravel_2 == gravel
    assert trail_2 == trail
    assert week_2 == week
    assert month_2 == month
    assert len(mock_strava.get_requests("/api/v3/athlete/activities")) == reqs_after_first


def test_f5_tier2_refresh_true_updates_stale_cache_entry(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config(scope="read,activity:read_all")
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_stale_1.err",
    ) as s1:
        summary = parse_tool_result(s1.call_tool("strava_get_athlete", {}))
        assert "ftp" not in summary or summary.get("ftp") is None
    assert len(mock_strava.get_requests("/api/v3/athlete")) == 1

    seed_config(scope="read,read_all,profile:read_all,activity:read_all")
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_stale_2.err",
    ) as s2:
        cached_summary = parse_tool_result(s2.call_tool("strava_get_athlete", {}))
        assert "ftp" not in cached_summary or cached_summary.get("ftp") is None
        assert len(mock_strava.get_requests("/api/v3/athlete")) == 1

        refreshed = parse_tool_result(s2.call_tool("strava_get_athlete", {"refresh": True}))
        assert refreshed.get("ftp") == 285
        assert len(mock_strava.get_requests("/api/v3/athlete")) == 2

        cached_detailed = parse_tool_result(s2.call_tool("strava_get_athlete", {}))
        assert cached_detailed.get("ftp") == 285
        assert len(mock_strava.get_requests("/api/v3/athlete")) == 2


def test_f5_tier2_immutable_historical_entries_persist_across_sessions(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seed_config()
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_imm_1.err",
    ) as s1:
        assert s1.call_tool("strava_get_activity", {"activity_id": 9006}).is_error is not True
        assert s1.call_tool("strava_get_activity_laps", {"activity_id": 9001}).is_error is not True
        assert (
            s1.call_tool(
                "strava_get_activity_streams", {"activity_id": 9001, "max_points": 50}
            ).is_error
            is not True
        )

    assert len(mock_strava.get_requests("/api/v3/")) == 3

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_imm_2.err",
    ) as s2:
        assert s2.call_tool("strava_get_activity", {"activity_id": 9006}).is_error is not True
        assert s2.call_tool("strava_get_activity_laps", {"activity_id": 9001}).is_error is not True
        assert (
            s2.call_tool(
                "strava_get_activity_streams", {"activity_id": 9001, "max_points": 50}
            ).is_error
            is not True
        )
        assert len(mock_strava.get_requests("/api/v3/")) == 3

        assert (
            s2.call_tool("strava_get_activity", {"activity_id": 9006, "refresh": True}).is_error
            is not True
        )
        assert s2.call_tool("strava_get_activity_laps", {"activity_id": 9001}).is_error is not True
        assert len(mock_strava.get_requests("/api/v3/")) == 4

    stats_proc = run_cli("cache", "stats", check=True)
    nums = [int(n) for n in re.findall(r"\b\d+\b", stats_proc.stdout)]
    assert any(n >= 3 for n in nums), f"Expected >= 3 cached entries: {stats_proc.stdout!r}"


def test_f5_tier2_corrupted_cache_sqlite_handled_cleanly(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seed_config()
    cache_db = Path(isolated_env["STRAVA_MCP_CACHE_DIR"]) / "cache.sqlite"
    cache_db.write_bytes(b"not-a-valid-sqlite-db")
    os.chmod(cache_db, 0o600)

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_corrupt_cache.err",
    ) as session:
        res = session.call_tool("strava_get_athlete", {})
        assert "Traceback (most recent call last)" not in session.stderr_text
        if res.is_error is not True:
            assert parse_tool_result(res)["id"] == DEFAULT_TEST_ATHLETE_ID

    clear_proc = run_cli("cache", "clear", check=True)
    assert "Traceback (most recent call last)" not in (clear_proc.stdout + clear_proc.stderr)
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_corrupt_recovered.err",
    ) as s2:
        res2 = s2.call_tool("strava_get_athlete", {})
        assert res2.is_error is not True
        assert parse_tool_result(res2)["id"] == DEFAULT_TEST_ATHLETE_ID


def test_f5_tier2_non_writable_cache_dir_handled_cleanly(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seeded = seed_config()
    cache_dir = Path(isolated_env["STRAVA_MCP_CACHE_DIR"])
    os.chmod(cache_dir, 0o500)
    try:
        stats_proc = run_cli("cache", "stats")
        combined = stats_proc.stdout + "\n" + stats_proc.stderr
        assert "Traceback (most recent call last)" not in combined
        _assert_no_secrets_leaked(
            combined,
            mock_strava,
            seeded.client_secret,
            seeded.access_token,
            seeded.refresh_token,
            seeded.mcp_token,
        )

        with open_stdio_session(
            [*resolve_strava_mcp_cmd(), "stdio"],
            isolated_env,
            tmp_path / "stdio_ro_cache.err",
        ) as session:
            res = session.call_tool("strava_get_athlete", {})
            assert "Traceback (most recent call last)" not in session.stderr_text
            if res.is_error is not True:
                assert parse_tool_result(res)["id"] == DEFAULT_TEST_ATHLETE_ID
    finally:
        os.chmod(cache_dir, 0o700)

    file_as_cache_dir = tmp_path / "regular_file_not_dir"
    file_as_cache_dir.write_text("not a dir", encoding="utf-8")
    p_file = run_cli(
        "cache",
        "stats",
        env_overrides={"STRAVA_MCP_CACHE_DIR": str(file_as_cache_dir)},
    )
    assert "Traceback (most recent call last)" not in (p_file.stdout + p_file.stderr)


# =============================================================================
# F11: CLI & Transport Boundaries (6 tests)
# =============================================================================


def test_f11_tier2_stdio_stdout_clean_only_jsonrpc_logs_to_stderr(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
) -> None:
    seed_config()
    frame_objs: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "e2e-raw-stdio", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "strava_get_athlete", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "strava_get_activity", "arguments": {"activity_id": 999999}},
        },
    ]

    proc = subprocess.Popen(
        [*resolve_strava_mcp_cmd(), "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=isolated_env,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    stdout_queue: queue.Queue[str | None] = queue.Queue()
    stderr_chunks: list[str] = []

    def _drain_stdout() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                stdout_queue.put(line)
        finally:
            stdout_queue.put(None)

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        stderr_chunks.append(proc.stderr.read())

    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    def _next_frame(timeout: float = 15.0) -> dict[str, Any]:
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
            return parsed

    parsed_frames: list[dict[str, Any]] = []
    try:
        proc.stdin.write(json.dumps(frame_objs[0]) + "\n")
        proc.stdin.flush()
        msg1 = _next_frame()
        parsed_frames.append(msg1)
        assert msg1.get("id") == 1 and "result" in msg1 and "error" not in msg1

        proc.stdin.write("\n".join(json.dumps(f) for f in frame_objs[1:]) + "\n")
        proc.stdin.flush()
        for _ in range(3):
            parsed_frames.append(_next_frame())

        proc.stdin.close()
        while (extra := stdout_queue.get(timeout=5.0)) is not None:
            if extra.strip():
                parsed_extra = json.loads(extra)
                assert isinstance(parsed_extra, dict) and parsed_extra.get("jsonrpc") == "2.0"
                parsed_frames.append(parsed_extra)
        returncode = proc.wait(timeout=10.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

    stderr_txt = "".join(stderr_chunks)
    assert returncode == 0, f"stdio exited {returncode}.\nstderr:\n{stderr_txt}"
    assert "Traceback (most recent call last)" not in stderr_txt

    assert len(parsed_frames) >= 4, f"Expected >= 4 JSON-RPC response lines, got:\n{parsed_frames}"
    by_id: dict[int, dict[str, Any]] = {}
    for msg in parsed_frames:
        if isinstance(msg.get("id"), int):
            by_id[msg["id"]] = msg
    assert {1, 2, 3, 4} <= set(by_id)
    for req_id in (1, 2, 3, 4):
        assert "result" in by_id[req_id] and "error" not in by_id[req_id], (
            f"id={req_id} failed: {by_id[req_id]}"
        )


def test_f11_tier2_http_missing_and_invalid_bearer_returns_401(
    http_server: Any,
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "e2e-http-auth", "version": "1.0"},
        },
    }
    base_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    r_missing = httpx.post(http_server.mcp_url, headers=base_headers, json=payload, timeout=5.0)
    assert r_missing.status_code == 401
    assert "bearer" in r_missing.headers.get("www-authenticate", "").lower()

    r_wrong = httpx.post(
        http_server.mcp_url,
        headers={**base_headers, "Authorization": "Bearer wrong-token"},
        json=payload,
        timeout=5.0,
    )
    assert r_wrong.status_code == 401
    assert "bearer" in r_wrong.headers.get("www-authenticate", "").lower()

    r_basic = httpx.post(
        http_server.mcp_url,
        headers={**base_headers, "Authorization": "Basic dXNlcjpwYXNz"},
        json=payload,
        timeout=5.0,
    )
    assert r_basic.status_code == 401

    with open_mcp_http_session(http_server.mcp_url, http_server.token) as session:
        res = session.call_tool("strava_get_athlete", {})
        assert res.is_error is not True
        assert parse_tool_result(res)["id"] == DEFAULT_TEST_ATHLETE_ID


def test_f11_tier2_http_dns_rebinding_spoofed_host_and_origin_rejected(
    http_server: Any,
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "e2e-dns-rebind", "version": "1.0"},
        },
    }
    auth_headers = {
        "Authorization": f"Bearer {http_server.token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    r_host = httpx.post(
        http_server.mcp_url,
        headers={**auth_headers, "Host": "evil.example:8772"},
        json=payload,
        timeout=5.0,
    )
    assert r_host.status_code in (403, 421)

    r_origin = httpx.post(
        http_server.mcp_url,
        headers={**auth_headers, "Origin": "https://evil.example"},
        json=payload,
        timeout=5.0,
    )
    assert r_origin.status_code == 403


def test_f11_tier2_serve_rejects_non_loopback_and_invalid_listen(
    seed_config: Callable[..., SeededConfig],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seed_config()

    p_non_loopback = run_cli("serve", "--listen", "0.0.0.0:8772", timeout=10.0)
    _assert_cli_error(
        p_non_loopback,
        ("invalid_config", "insecure_listen"),
        ("loopback", "127.0.0.1", "localhost", "bind", "listen"),
    )

    p_bad_port = run_cli("serve", "--listen", "127.0.0.1:badport", timeout=10.0)
    _assert_cli_error(
        p_bad_port,
        ("invalid_config",),
        ("port", "listen", "invalid"),
    )


def test_f11_tier2_unknown_tool_invalid_args_unknown_resource_no_crash(
    stdio_session: Any,
) -> None:
    try:
        r_unknown = stdio_session.call_tool("nonexistent_tool", {})
        assert r_unknown.is_error is True
    except MCPError:
        pass

    try:
        r_bad_args = stdio_session.call_tool("strava_get_activity", {"activity_id": "bad"})
        assert r_bad_args.is_error is True
    except MCPError:
        pass

    with pytest.raises(MCPError):
        stdio_session.read_resource("strava://unknown/missing")

    with pytest.raises(MCPError):
        stdio_session.get_prompt("nonexistent_prompt", {})

    r_ok = stdio_session.call_tool("strava_auth_status", {})
    assert r_ok.is_error is not True


def test_f11_tier2_headers_helper_missing_token_fails_cleanly(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
) -> None:
    headers_script = REPO_ROOT / "client" / "strava-mcp-headers"
    if not headers_script.is_file() or not os.access(headers_script, os.X_OK):
        pytest.fail(
            f"client/strava-mcp-headers missing or not executable: {headers_script}",
            pytrace=False,
        )

    seeded = seed_config(include_mcp_token=False)
    proc = subprocess.run(
        [str(headers_script)],
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
    )
