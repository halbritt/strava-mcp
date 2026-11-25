"""Tier 1 opaque-box E2E tests for F1-F5: config, OAuth2, auth helper, rate limits, cache."""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from conftest import (
    DEFAULT_TEST_ACCESS_TOKEN,
    DEFAULT_TEST_ATHLETE_ID,
    DEFAULT_TEST_CLIENT_ID,
    DEFAULT_TEST_CLIENT_SECRET,
    DEFAULT_TEST_SCOPE,
    SeededConfig,
    open_stdio_session,
    parse_tool_result,
    resolve_strava_mcp_cmd,
)
from mock_strava import MockStravaServer


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
                if isinstance(parsed, dict):
                    raw = parsed
                else:
                    raw = {"message": text}
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


# =============================================================================
# F1: Secure Config & Secret Store (6 Tier 1 tests)
# =============================================================================


def test_f1_tier1_config_dir_isolation_overrides_xdg_and_home(
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    home_cfg = Path(isolated_env["HOME"]) / ".config" / "strava-mcp"
    xdg_cfg = Path(isolated_env["XDG_CONFIG_HOME"]) / "strava-mcp"
    home_cfg.mkdir(parents=True, exist_ok=True)
    xdg_cfg.mkdir(parents=True, exist_ok=True)

    proc = _run_cli_detached(
        isolated_env,
        "init",
        "--client-id",
        DEFAULT_TEST_CLIENT_ID,
        input_text=f"{DEFAULT_TEST_CLIENT_SECRET}\n",
    )
    assert proc.returncode == 0, f"init failed: {proc.stdout}\n{proc.stderr}"

    cfg_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    assert (cfg_dir / "client.json").is_file()
    assert list(home_cfg.iterdir()) == []
    assert list(xdg_cfg.iterdir()) == []
    _assert_no_secrets_leaked(proc.stdout + proc.stderr, mock_strava, DEFAULT_TEST_CLIENT_SECRET)


def test_f1_tier1_init_creates_0700_dir_and_0600_client_json(
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    parent_755 = tmp_path / "parent_755"
    parent_755.mkdir(mode=0o755)
    os.chmod(parent_755, 0o755)
    fresh_cfg = parent_755 / "fresh_cfg"
    assert not fresh_cfg.exists()

    proc = _run_cli_detached(
        isolated_env,
        "init",
        "--client-id",
        DEFAULT_TEST_CLIENT_ID,
        input_text=f"{DEFAULT_TEST_CLIENT_SECRET}\n",
        env_overrides={"STRAVA_MCP_CONFIG_DIR": str(fresh_cfg)},
    )
    assert proc.returncode == 0, f"init failed: {proc.stdout}\n{proc.stderr}"

    _assert_mode(fresh_cfg, 0o700)
    client_file = fresh_cfg / "client.json"
    _assert_mode(client_file, 0o600)
    client_data = json.loads(client_file.read_text(encoding="utf-8"))
    assert client_data["client_id"] == DEFAULT_TEST_CLIENT_ID
    assert client_data["client_secret"] == DEFAULT_TEST_CLIENT_SECRET
    _assert_no_secrets_leaked(proc.stdout + proc.stderr, mock_strava, DEFAULT_TEST_CLIENT_SECRET)


def test_f1_tier1_init_import_tokens_file_writes_0600_tokens(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    tmp_path: Path,
) -> None:
    seeded = seed_config(include_tokens=False, include_mcp_token=False)
    ext_tokens = tmp_path / "external_tokens.json"
    exp = int(time.time()) + 7200
    ext_tokens.write_text(
        json.dumps(
            {
                "access_token": "synthetic-imported-access-001",
                "refresh_token": "synthetic-imported-refresh-001",
                "expires_at": exp,
                "scope": "read,activity:read_all",
                "athlete_id": DEFAULT_TEST_ATHLETE_ID,
            }
        ),
        encoding="utf-8",
    )

    proc = _run_cli_detached(
        isolated_env,
        "init",
        "--client-id",
        DEFAULT_TEST_CLIENT_ID,
        "--import-tokens-file",
        str(ext_tokens),
        input_text=f"{DEFAULT_TEST_CLIENT_SECRET}\n",
    )
    assert proc.returncode == 0, f"init --import-tokens-file failed: {proc.stdout}\n{proc.stderr}"

    _assert_mode(seeded.tokens_file, 0o600)
    saved = _read_tokens_file(seeded.tokens_file)
    assert saved["access_token"] == "synthetic-imported-access-001"
    assert saved["refresh_token"] == "synthetic-imported-refresh-001"
    assert int(saved["expires_at"]) == exp
    assert {"read", "activity:read_all"} <= _parse_scopes_any(saved.get("scope"))
    athlete = saved.get("athlete_id") or (saved.get("athlete") or {}).get("id")
    assert athlete == DEFAULT_TEST_ATHLETE_ID
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        DEFAULT_TEST_CLIENT_SECRET,
        "synthetic-imported-access-001",
        "synthetic-imported-refresh-001",
    )


def test_f1_tier1_status_non_secret_output(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()
    proc = run_cli("status", check=True)
    out = proc.stdout.strip()
    assert out, "status produced empty stdout"
    try:
        data = json.loads(out)
        serialized = json.dumps(data).lower()
        assert str(DEFAULT_TEST_ATHLETE_ID) in serialized
        assert "activity:read_all" in serialized
        assert any(k in serialized for k in ("expires", "expiry", "valid", "authenticated"))
    except json.JSONDecodeError:
        low = out.lower()
        assert str(DEFAULT_TEST_ATHLETE_ID) in out
        assert "activity:read_all" in low
        assert any(k in low for k in ("expir", "valid", "auth"))

    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


def test_f1_tier1_status_check_live_queries_mock_strava(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()
    mock_strava.clear_requests()
    proc = run_cli("status", "--check-live", check=True)

    reqs = mock_strava.get_requests("/api/v3/athlete", "GET")
    assert len(reqs) == 1
    auth_hdr = reqs[0].headers.get("Authorization") or reqs[0].headers.get("authorization")
    assert auth_hdr == f"Bearer {DEFAULT_TEST_ACCESS_TOKEN}"
    out_low = proc.stdout.lower()
    assert any(w in out_low for w in ("alex", "100001", "authenticated", "live", "ok", "valid"))
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


def test_f1_tier1_mcp_token_auto_generated_0600_and_stable(
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    cfg_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    mcp_token_file = cfg_dir / "mcp-token"
    assert not mcp_token_file.exists()

    proc1 = _run_cli_detached(
        isolated_env,
        "init",
        "--client-id",
        DEFAULT_TEST_CLIENT_ID,
        input_text=f"{DEFAULT_TEST_CLIENT_SECRET}\n",
    )
    assert proc1.returncode == 0, f"first init failed: {proc1.stdout}\n{proc1.stderr}"
    _assert_mode(mcp_token_file, 0o600)
    tok1 = mcp_token_file.read_text(encoding="utf-8").strip()
    assert len(tok1) >= 24
    assert "\n" not in tok1

    proc2 = _run_cli_detached(
        isolated_env,
        "init",
        "--client-id",
        DEFAULT_TEST_CLIENT_ID,
        input_text=f"{DEFAULT_TEST_CLIENT_SECRET}\n",
    )
    assert proc2.returncode == 0, f"second init failed: {proc2.stdout}\n{proc2.stderr}"
    _assert_mode(mcp_token_file, 0o600)
    tok2 = mcp_token_file.read_text(encoding="utf-8").strip()
    assert tok2 == tok1

    status_proc = run_cli("status", check=True)
    _assert_no_secrets_leaked(
        proc1.stdout + proc1.stderr + proc2.stdout + proc2.stderr + status_proc.stdout,
        mock_strava,
        DEFAULT_TEST_CLIENT_SECRET,
        tok1,
    )


# =============================================================================
# F2: OAuth2 Token Lifecycle & Locking (5 Tier 1 tests)
# =============================================================================


def test_f2_tier1_cli_refresh_rotates_and_preserves_scope_and_athlete(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config(
        expires_at=int(time.time()) + 120,
        scope=DEFAULT_TEST_SCOPE,
        athlete_id=DEFAULT_TEST_ATHLETE_ID,
    )
    mock_strava.configure_oauth(include_scope_on_refresh=False, include_athlete_on_refresh=False)
    mock_strava.clear_requests()

    proc = run_cli("refresh", check=True)
    tok_reqs = mock_strava.get_requests("/oauth/token", "POST")
    assert len(tok_reqs) == 1
    assert parse_qs(tok_reqs[0].body).get("grant_type") == ["refresh_token"]

    _assert_mode(seeded.tokens_file, 0o600)
    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] != seeded.access_token
    assert updated["refresh_token"] != seeded.refresh_token
    athlete = updated.get("athlete_id") or (updated.get("athlete") or {}).get("id")
    assert athlete == DEFAULT_TEST_ATHLETE_ID
    assert _parse_scopes_any(updated.get("scope")) == {
        "read",
        "read_all",
        "profile:read_all",
        "activity:read_all",
    }
    assert seeded.refresh_token in mock_strava.get_active_tokens()["invalidated_refresh_tokens"]
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
    )


def test_f2_tier1_proactive_refresh_near_expiry_under_300s(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 120)
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_proactive.err",
    ) as session:
        res = session.call_tool("strava_get_athlete", {})
        assert res.is_error is not True
        athlete = parse_tool_result(res)
        assert athlete["id"] == DEFAULT_TEST_ATHLETE_ID

    reqs = mock_strava.requests
    oauth_idx = [
        i for i, r in enumerate(reqs) if r.path.endswith("/oauth/token") and r.status_code == 200
    ]
    api_idx = [
        i for i, r in enumerate(reqs) if r.path == "/api/v3/athlete" and r.status_code == 200
    ]
    assert len(oauth_idx) == 1
    assert len(api_idx) == 1
    assert oauth_idx[0] < api_idx[0]

    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] != seeded.access_token
    assert updated["refresh_token"] != seeded.refresh_token
    api_auth = reqs[api_idx[0]].headers.get("Authorization") or reqs[api_idx[0]].headers.get(
        "authorization"
    )
    assert api_auth == f"Bearer {updated['access_token']}"


def test_f2_tier1_transparent_401_refresh_and_retry(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 7200)
    mock_strava.expire_access_token(seeded.access_token)
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_401.err",
    ) as session:
        res = session.call_tool("strava_get_athlete", {})
        assert res.is_error is not True
        athlete = parse_tool_result(res)
        assert athlete["id"] == DEFAULT_TEST_ATHLETE_ID

    seq = [(r.method, r.path, r.status_code) for r in mock_strava.requests]
    assert seq == [
        ("GET", "/api/v3/athlete", 401),
        ("POST", "/oauth/token", 200),
        ("GET", "/api/v3/athlete", 200),
    ]
    updated = _read_tokens_file(seeded.tokens_file)
    assert updated["access_token"] != seeded.access_token
    assert updated["refresh_token"] != seeded.refresh_token
    assert _parse_scopes_any(updated.get("scope")) == _parse_scopes_any(DEFAULT_TEST_SCOPE)
    athlete_id = updated.get("athlete_id") or (updated.get("athlete") or {}).get("id")
    assert athlete_id == DEFAULT_TEST_ATHLETE_ID


def test_f2_tier1_rotated_token_persists_across_subsequent_calls(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seeded = seed_config(expires_at=int(time.time()) + 90)
    run_cli("refresh", check=True)
    rotated = _read_tokens_file(seeded.tokens_file)
    new_access = rotated["access_token"]
    assert new_access != seeded.access_token

    mock_strava.clear_requests()
    run_cli("status", "--check-live", check=True)
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_persisted.err",
    ) as session:
        res = session.call_tool("strava_get_athlete", {})
        assert res.is_error is not True
        assert parse_tool_result(res)["id"] == DEFAULT_TEST_ATHLETE_ID

    assert mock_strava.get_requests("/oauth/token") == []
    api_reqs = mock_strava.get_requests("/api/v3/athlete", "GET")
    assert len(api_reqs) >= 1
    for req in api_reqs:
        auth = req.headers.get("Authorization") or req.headers.get("authorization")
        assert auth == f"Bearer {new_access}"


def test_f2_tier1_cli_revoke_calls_oauth_revoke_and_invalidates_local(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()
    mock_strava.clear_requests()

    proc = run_cli("revoke", check=True)
    revoke_reqs = mock_strava.get_requests("/oauth/revoke", "POST")
    assert len(revoke_reqs) == 1
    assert revoke_reqs[0].status_code == 200
    auth_hdr = revoke_reqs[0].headers.get("Authorization") or revoke_reqs[0].headers.get(
        "authorization", ""
    )
    assert auth_hdr.startswith("Basic ")

    active = mock_strava.get_active_tokens()
    assert seeded.access_token not in active["active_access_tokens"]
    assert seeded.refresh_token in active["invalidated_refresh_tokens"]

    status_proc = run_cli("status", "--check-live")
    combined = (status_proc.stdout + "\n" + status_proc.stderr).lower()
    assert status_proc.returncode != 0 or any(
        w in combined for w in ("unauthenticated", "revoked", "missing", "expired", "auth")
    )
    _assert_no_secrets_leaked(
        proc.stdout + proc.stderr + status_proc.stdout + status_proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
    )


# =============================================================================
# F3: OAuth2 Scope-Upgrade Auth Helper (5 Tier 1 tests)
# =============================================================================


def test_f3_tier1_auth_loopback_url_construction_and_params(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seed_config(include_tokens=False)
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port)]
    )
    try:
        parsed = urlsplit(auth_url)
        assert parsed.scheme == "http"
        assert parsed.netloc == f"127.0.0.1:{mock_strava.port}"
        assert parsed.path.endswith("/oauth/authorize")
        qs = parse_qs(parsed.query)
        assert qs.get("client_id") == [DEFAULT_TEST_CLIENT_ID]
        assert qs.get("response_type") == ["code"]
        assert qs.get("approval_prompt") == ["force"]
        assert qs.get("redirect_uri") == [f"http://127.0.0.1:{port}/callback"]
        state_val = (qs.get("state") or [""])[0]
        assert len(state_val) >= 16
        req_scopes = _parse_scopes_any((qs.get("scope") or [""])[0])
        assert {"read", "read_all", "profile:read_all", "activity:read_all"} <= req_scopes
        _, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert cb_resp.status_code == 200
        rc, out_txt, err_txt = finish(10.0)
        assert rc == 0, f"auth failed: {out_txt}\n{err_txt}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_f3_tier1_auth_full_scope_exchange_writes_0600_tokens(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seeded = seed_config(include_tokens=False)
    mock_strava.clear_requests()
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

    _assert_mode(seeded.tokens_file, 0o600)
    saved = _read_tokens_file(seeded.tokens_file)
    assert saved["access_token"]
    assert saved["refresh_token"]
    assert int(saved["expires_at"]) > int(time.time())
    athlete = saved.get("athlete_id") or (saved.get("athlete") or {}).get("id")
    assert athlete == DEFAULT_TEST_ATHLETE_ID
    assert {"read", "read_all", "profile:read_all", "activity:read_all"} <= _parse_scopes_any(
        saved.get("scope")
    )
    tok_reqs = mock_strava.get_requests("/oauth/token", "POST")
    assert len(tok_reqs) == 1
    assert parse_qs(tok_reqs[0].body).get("grant_type") == ["authorization_code"]
    _assert_no_secrets_leaked(
        out_txt + err_txt,
        mock_strava,
        seeded.client_secret,
        saved["access_token"],
        saved["refresh_token"],
    )


def test_f3_tier1_auth_custom_redirect_uri_option(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seeded = seed_config(include_tokens=False)
    port = _allocate_free_port()
    custom_uri = f"http://127.0.0.1:{port}/custom/oauth-cb"
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--redirect-uri", custom_uri]
    )
    try:
        qs = parse_qs(urlsplit(auth_url).query)
        assert qs.get("redirect_uri") == [custom_uri]
        auth_resp, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert urlsplit(auth_resp.headers["Location"]).path == "/custom/oauth-cb"
        assert cb_resp.status_code == 200
        rc, out_txt, err_txt = finish(10.0)
        assert rc == 0, f"auth failed: {out_txt}\n{err_txt}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
    _assert_mode(seeded.tokens_file, 0o600)


def test_f3_tier1_auth_custom_scopes_option(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seeded = seed_config(include_tokens=False)
    port = _allocate_free_port()
    proc, auth_url, finish = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port), "--scopes", "read,activity:read"]
    )
    try:
        req_scopes = _parse_scopes_any((parse_qs(urlsplit(auth_url).query).get("scope") or [""])[0])
        assert req_scopes == {"read", "activity:read"}
        _, cb_resp = _drive_oauth_authorize_to_callback(auth_url)
        assert cb_resp.status_code == 200
        rc, out_txt, err_txt = finish(10.0)
        assert rc == 0, f"auth failed: {out_txt}\n{err_txt}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    saved = _read_tokens_file(seeded.tokens_file)
    assert _parse_scopes_any(saved.get("scope")) == {"read", "activity:read"}


def test_f3_tier1_auth_single_use_state_enforcement(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> None:
    seed_config(include_tokens=False)
    mock_strava.clear_requests()
    port1 = _allocate_free_port()
    proc1, auth_url1, finish1 = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port1)]
    )
    try:
        state1 = (parse_qs(urlsplit(auth_url1).query).get("state") or [""])[0]
        auth_resp1 = httpx.get(auth_url1, follow_redirects=False, timeout=5.0)
        assert auth_resp1.status_code == 302
        location1 = auth_resp1.headers["Location"]
        cb1 = httpx.get(location1, follow_redirects=False, timeout=5.0)
        assert cb1.status_code == 200
        rc1, _, _ = finish1(10.0)
        assert rc1 == 0
        try:
            replay_resp = httpx.get(location1, follow_redirects=False, timeout=2.0)
            assert replay_resp.status_code >= 400
        except httpx.HTTPError:
            pass
    finally:
        if proc1.poll() is None:
            proc1.kill()
            proc1.wait(timeout=5.0)

    assert len(mock_strava.get_requests("/oauth/token")) == 1

    port2 = _allocate_free_port()
    proc2, auth_url2, finish2 = _spawn_auth_and_capture_url(
        isolated_env, ["--port", str(port2)]
    )
    try:
        state2 = (parse_qs(urlsplit(auth_url2).query).get("state") or [""])[0]
        assert state2 and state2 != state1
        _, cb2 = _drive_oauth_authorize_to_callback(auth_url2)
        assert cb2.status_code == 200
        rc2, _, _ = finish2(10.0)
        assert rc2 == 0
    finally:
        if proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=5.0)


# =============================================================================
# F4: Rate-Limit Tracker & 429 Handling (6 Tier 1 tests)
# =============================================================================


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


def test_f4_tier1_rate_limit_headers_reflected_in_auth_status(
    seed_config: Callable[..., SeededConfig],
    stdio_session: Any,
) -> None:
    seed_config()
    ath = stdio_session.call_tool("strava_get_athlete", {})
    assert ath.is_error is not True

    status = parse_tool_result(stdio_session.call_tool("strava_auth_status", {}))
    rl_section = status.get("rate_limits") or status.get("rate_limit") or status
    nums = _collect_ints(rl_section)
    assert {400, 4000, 200, 2000, 1} <= nums
    assert 199 in nums or 399 in nums
    assert 1999 in nums or 3999 in nums


def test_f4_tier1_cumulative_rate_limit_updates_across_requests(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config()
    mock_strava.set_rate_limits(
        overall_usage_15m=10,
        overall_usage_day=20,
        read_usage_15m=5,
        read_usage_day=15,
    )
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_rl_cum.err",
    ) as session:
        assert session.call_tool("strava_get_athlete", {}).is_error is not True
        s1 = parse_tool_result(session.call_tool("strava_auth_status", {}))
        nums1 = _collect_ints(s1.get("rate_limits") or s1.get("rate_limit") or s1)
        assert {6, 16, 11, 21} <= nums1

        assert session.call_tool("strava_get_athlete_stats", {}).is_error is not True
        s2 = parse_tool_result(session.call_tool("strava_auth_status", {}))
        nums2 = _collect_ints(s2.get("rate_limits") or s2.get("rate_limit") or s2)
        assert {7, 17, 12, 22} <= nums2

    assert len(mock_strava.get_requests("/api/v3/")) == 2


def test_f4_tier1_oauth_refresh_does_not_exhaust_rest_counters(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config(expires_at=int(time.time()) + 60)
    mock_strava.set_near_expiry(60)
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_rl_oauth.err",
    ) as session:
        assert session.call_tool("strava_get_athlete", {}).is_error is not True
        status = parse_tool_result(session.call_tool("strava_auth_status", {}))

    assert len(mock_strava.get_requests("/oauth/token")) == 1
    assert len(mock_strava.get_requests("/api/v3/")) == 1
    rl_section = status.get("rate_limits") or status.get("rate_limit") or status
    assert 1 in _collect_ints(rl_section)
    assert 2 not in _collect_ints(rl_section)


def test_f4_tier1_utc_reset_window_metadata_exposed(
    stdio_session: Any,
) -> None:
    assert stdio_session.call_tool("strava_get_athlete", {}).is_error is not True
    status = parse_tool_result(stdio_session.call_tool("strava_auth_status", {}))
    rl_section = status.get("rate_limits") or status.get("rate_limit") or status
    dts = _collect_iso_datetimes(rl_section)
    assert dts, f"Expected UTC reset timestamps in rate-limit status: {rl_section!r}"
    assert any(dt.minute in {0, 15, 30, 45} and dt.second == 0 for dt in dts)
    assert any(dt.hour == 0 and dt.minute == 0 and dt.second == 0 for dt in dts)


def test_f4_tier1_429_returns_structured_error_without_retry_loop(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.inject_fault(fault_type="rate_limit_429", times=1)
    mock_strava.clear_requests()

    res = stdio_session.call_tool("strava_get_athlete", {})
    err = _extract_error_dict(res)
    assert res.is_error is True or err.get("code") == "rate_limited"
    assert err.get("code") == "rate_limited"
    assert "reset_at_utc" in err or "reset_at" in err
    assert "retry_after_seconds" in err
    assert len(mock_strava.get_requests("/api/v3/athlete")) == 1

    stats_res = stdio_session.call_tool("strava_get_athlete_stats", {})
    assert stats_res.is_error is not True


def test_f4_tier1_cli_status_check_live_reports_rate_limits(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()
    mock_strava.set_rate_limits(read_usage_15m=12, read_usage_day=42)
    proc = run_cli("status", "--check-live", check=True)
    out = proc.stdout
    assert "13" in out and "200" in out
    assert "43" in out and "2000" in out
    assert "400" in out and "4000" in out
    _assert_no_secrets_leaked(
        out + proc.stderr,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )


# =============================================================================
# F5: Local SQLite Cache (6 Tier 1 tests)
# =============================================================================


def test_f5_tier1_cache_dir_and_sqlite_permissions_0700_0600(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config()
    fresh_cache = tmp_path / "fresh_cache"
    assert not fresh_cache.exists()
    env = dict(isolated_env)
    env["STRAVA_MCP_CACHE_DIR"] = str(fresh_cache)

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        env,
        tmp_path / "stdio_cache_perms.err",
    ) as session:
        res = session.call_tool("strava_get_activity", {"activity_id": 9001})
        assert res.is_error is not True

    _assert_mode(fresh_cache, 0o700)
    _assert_mode(fresh_cache / "cache.sqlite", 0o600)


def test_f5_tier1_cache_hit_on_repeated_activity_and_athlete_fetch(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.clear_requests()

    a1 = parse_tool_result(stdio_session.call_tool("strava_get_activity", {"activity_id": 9001}))
    a2 = parse_tool_result(stdio_session.call_tool("strava_get_activity", {"activity_id": 9001}))
    ath1 = parse_tool_result(stdio_session.call_tool("strava_get_athlete", {}))
    ath2 = parse_tool_result(stdio_session.call_tool("strava_get_athlete", {}))

    assert a1 == a2
    assert ath1 == ath2
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 1
    assert len(mock_strava.get_requests("/api/v3/athlete")) == 1


def test_f5_tier1_cache_persists_across_process_restarts(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config()
    mock_strava.clear_requests()

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_cache_p1.err",
    ) as s1:
        r1 = parse_tool_result(s1.call_tool("strava_get_activity", {"activity_id": 9001}))

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_cache_p2.err",
    ) as s2:
        r2 = parse_tool_result(s2.call_tool("strava_get_activity", {"activity_id": 9001}))

    assert r1 == r2
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 1


def test_f5_tier1_refresh_true_bypasses_cache(
    mock_strava: MockStravaServer,
    stdio_session: Any,
) -> None:
    mock_strava.clear_requests()

    stdio_session.call_tool("strava_get_activity", {"activity_id": 9001})
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 1

    stdio_session.call_tool("strava_get_activity", {"activity_id": 9001, "refresh": True})
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 2

    stdio_session.call_tool("strava_get_activity", {"activity_id": 9001})
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 2


def test_f5_tier1_cli_cache_stats_reports_entries(
    seed_config: Callable[..., SeededConfig],
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seed_config()
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_cache_stats.err",
    ) as session:
        assert session.call_tool("strava_get_athlete", {}).is_error is not True
        assert (
            session.call_tool("strava_get_activity", {"activity_id": 9001}).is_error is not True
        )

    proc = run_cli("cache", "stats", check=True)
    out = proc.stdout
    assert "cache.sqlite" in out or isolated_env["STRAVA_MCP_CACHE_DIR"] in out
    nums = [int(n) for n in re.findall(r"\b\d+\b", out)]
    assert any(n >= 2 for n in nums), f"Expected >= 2 entries in cache stats: {out!r}"


def test_f5_tier1_cli_cache_clear_empties_cache(
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
        tmp_path / "stdio_cache_clear1.err",
    ) as s1:
        assert s1.call_tool("strava_get_activity", {"activity_id": 9001}).is_error is not True
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 1

    run_cli("cache", "clear", check=True)
    stats_proc = run_cli("cache", "stats", check=True)
    stats_text = stats_proc.stdout
    try:
        stats_json = json.loads(stats_text)
        nums = _collect_ints(stats_json)
        assert 0 in nums
    except json.JSONDecodeError:
        assert re.search(r"\b0\b", stats_text), f"Expected 0 entries after clear: {stats_text!r}"

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_cache_clear2.err",
    ) as s2:
        assert s2.call_tool("strava_get_activity", {"activity_id": 9001}).is_error is not True
    assert len(mock_strava.get_requests("/api/v3/activities/9001")) == 2
