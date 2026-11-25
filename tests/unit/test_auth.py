"""Unit tests for strava_mcp.auth."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import json
import logging
import os
import socket
import stat
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from strava_mcp.auth import (
    DEFAULT_SCOPE_STRING,
    LoopbackAuthServer,
    TokenManager,
    _acquire_file_lock,
    build_authorize_url,
    format_scopes,
    has_scope,
    missing_scopes,
    parse_scopes,
    require_scope,
    revoke_token,
    run_loopback_auth,
)
from strava_mcp.config import (
    AuthError,
    ClientConfig,
    ConfigError,
    InsufficientScopeError,
    TokenData,
    load_tokens,
    save_client,
    save_tokens,
)


def _seed_auth_dir(
    config_dir: Path,
    *,
    access_token: str = "access-initial",
    refresh_token: str = "refresh-initial",
    expires_at: int = 2_000,
    scope: str = "read,activity:read_all",
    athlete_id: int | None = 4242,
) -> Path:
    save_client(
        config_dir,
        ClientConfig(client_id="999001", client_secret="test-client-secret-999"),
    )
    save_tokens(
        config_dir,
        TokenData(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            expires_in=21600,
            scope=scope,
            athlete_id=athlete_id,
        ),
    )
    return config_dir


def test_scope_utilities() -> None:
    assert parse_scopes("read, read_all activity:read_all,read") == [
        "read",
        "read_all",
        "activity:read_all",
    ]
    assert parse_scopes(["read,profile:read_all", "profile:read_all"]) == [
        "read",
        "profile:read_all",
    ]
    assert parse_scopes(None) == []
    assert format_scopes(["read", "activity:read_all", "read"]) == "read,activity:read_all"

    assert has_scope("activity:read_all", "activity:read") is True
    assert has_scope("activity:read_all", "activity:read_all") is True
    assert has_scope("activity:read", "activity:read_all") is False
    assert has_scope("read_all", "read") is False
    assert missing_scopes("read", ["read", "activity:read_all", "profile:read_all"]) == [
        "activity:read_all",
        "profile:read_all",
    ]

    require_scope("read,activity:read_all", "activity:read")
    with pytest.raises(InsufficientScopeError) as exc_info:
        require_scope("read", ["activity:read_all"], context="strava_list_activities")
    assert exc_info.value.code == "insufficient_scope"
    assert exc_info.value.required_scopes == ["activity:read_all"]
    assert exc_info.value.granted_scopes == ["read"]
    assert "strava-mcp auth" in exc_info.value.message


async def test_proactive_refresh_threshold_and_force(tmp_path: Path) -> None:
    now = 1_000
    calls: list[dict[str, list[str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(parse_qs(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "access_token": "access-rotated-1",
                "refresh_token": "refresh-rotated-1",
                "expires_at": now + 21600,
                "expires_in": 21600,
                "token_type": "Bearer",
            },
        )

    config_dir = _seed_auth_dir(tmp_path / "cfg1", expires_at=now + 301)
    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(handler),
        clock=lambda: float(now),
    )

    # 301s remaining -> no refresh
    token = await tm.get_valid_token()
    assert token == "access-initial"
    assert len(calls) == 0

    # Exactly 300s remaining -> proactive refresh triggers
    save_tokens(
        config_dir,
        TokenData(
            access_token="access-initial",
            refresh_token="refresh-initial",
            expires_at=now + 300,
            scope="read,activity:read_all",
            athlete_id=4242,
        ),
    )
    rotated = await tm.get_valid_tokens()
    assert rotated.access_token == "access-rotated-1"
    assert rotated.refresh_token == "refresh-rotated-1"
    assert rotated.scope == "read,activity:read_all"
    assert rotated.athlete_id == 4242
    assert len(calls) == 1
    assert calls[0]["grant_type"] == ["refresh_token"]
    assert stat.S_IMODE((config_dir / "tokens.json").stat().st_mode) == 0o600

    # 7200s remaining + force=True -> refreshes anyway
    save_tokens(
        config_dir,
        TokenData(
            access_token="access-fresh",
            refresh_token="refresh-fresh",
            expires_at=now + 7200,
            scope="read",
            athlete_id=4242,
        ),
    )
    forced = await tm.get_valid_tokens(force=True)
    assert forced.access_token == "access-rotated-1"
    assert len(calls) == 2


async def test_concurrent_refresh_and_401_dedup_single_http_call(tmp_path: Path) -> None:
    now = 1_000
    request_count = 0
    active_refresh = "refresh-initial"

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count, active_refresh
        request_count += 1
        form = parse_qs(request.content.decode("utf-8"))
        sent_refresh = form["refresh_token"][0]
        await asyncio.sleep(0.05)
        if sent_refresh != active_refresh:
            return httpx.Response(400, json={"message": "Invalid refresh token"})
        active_refresh = "refresh-rotated-2"
        return httpx.Response(
            200,
            json={
                "access_token": "access-rotated-2",
                "refresh_token": active_refresh,
                "expires_at": now + 21600,
                "expires_in": 21600,
            },
        )

    config_dir = _seed_auth_dir(tmp_path / "cfg", expires_at=now + 100)
    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(handler),
        clock=lambda: float(now),
    )

    t1, t2 = await asyncio.gather(tm.get_valid_token(), tm.get_valid_token())
    assert t1 == "access-rotated-2"
    assert t2 == "access-rotated-2"
    assert request_count == 1

    # Concurrent 401 retry race using if_access_token_matches
    request_count = 0
    active_refresh = "refresh-rotated-2"
    save_tokens(
        config_dir,
        TokenData(
            access_token="access-stale-401",
            refresh_token=active_refresh,
            expires_at=now + 10_000,
            scope="read",
        ),
    )

    async def handler_401(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        await asyncio.sleep(0.04)
        return httpx.Response(
            200,
            json={
                "access_token": "access-after-401",
                "refresh_token": "refresh-after-401",
                "expires_at": now + 21600,
                "expires_in": 21600,
            },
        )

    tm_401 = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(handler_401),
        clock=lambda: float(now),
    )
    r1, r2 = await asyncio.gather(
        tm_401.get_valid_token(if_access_token_matches="access-stale-401"),
        tm_401.get_valid_token(if_access_token_matches="access-stale-401"),
    )
    assert r1 == "access-after-401"
    assert r2 == "access-after-401"
    assert request_count == 1


async def test_cross_process_flock_waits_and_rereads_and_times_out(tmp_path: Path) -> None:
    now = 5_000
    config_dir = _seed_auth_dir(tmp_path / "cfg", expires_at=now + 50)
    lock_path = config_dir / "tokens.lock"
    http_calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(500)

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _external_writer() -> None:
        try:
            time.sleep(0.06)
            save_tokens(
                config_dir,
                TokenData(
                    access_token="access-written-by-peer",
                    refresh_token="refresh-written-by-peer",
                    expires_at=now + 7200,
                    scope="read",
                ),
            )
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    thread = threading.Thread(target=_external_writer)
    thread.start()
    try:
        tm = TokenManager(
            config_dir,
            oauth_base_url="https://mock.strava.test/oauth",
            transport=httpx.MockTransport(handler),
            clock=lambda: float(now),
            lock_timeout_seconds=2.0,
        )
        token = await tm.get_valid_token()
        assert token == "access-written-by-peer"
        assert http_calls == 0
    finally:
        thread.join(timeout=2.0)

    # Lock timeout case
    fd2 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        tm_short = TokenManager(
            config_dir,
            oauth_base_url="https://mock.strava.test/oauth",
            transport=httpx.MockTransport(handler),
            clock=lambda: float(now),
            lock_timeout_seconds=0.05,
        )
        with pytest.raises(AuthError) as exc_info:
            await tm_short.get_valid_token(force=True)
        assert exc_info.value.code == "refresh_failed"
        assert lock_path.exists()
    finally:
        fcntl.flock(fd2, fcntl.LOCK_UN)
        os.close(fd2)


async def test_exchange_code_athlete_and_error_sanitization(tmp_path: Path) -> None:
    now = 2_000
    config_dir = _seed_auth_dir(tmp_path / "cfg", expires_at=now + 10)

    async def exchange_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/token")
        return httpx.Response(
            200,
            json={
                "access_token": "exchanged-access",
                "refresh_token": "exchanged-refresh",
                "expires_at": now + 21600,
                "expires_in": 21600,
                "token_type": "Bearer",
                "athlete": {"id": 884422},
            },
        )

    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(exchange_handler),
        clock=lambda: float(now),
    )
    tokens = await tm.exchange_code("code-123", callback_scope="read,activity:read_all")
    assert tokens.athlete_id == 884422
    assert tokens.scope == "read,activity:read_all"
    assert load_tokens(config_dir).athlete_id == 884422

    # Error responses never leak secrets
    for response in (
        httpx.Response(400, json={"message": "Bad Request test-client-secret-999"}),
        httpx.Response(500, text="<html>upstream error</html>"),
    ):

        async def err_handler(
            _req: httpx.Request,
            resp: httpx.Response = response,
        ) -> httpx.Response:
            return resp

        tm_err = TokenManager(
            config_dir,
            oauth_base_url="https://mock.strava.test/oauth",
            transport=httpx.MockTransport(err_handler),
            clock=lambda: float(now),
        )
        with pytest.raises(AuthError) as exc_info:
            await tm_err.get_valid_token(force=True)
        assert exc_info.value.code == "refresh_failed"
        assert "test-client-secret-999" not in str(exc_info.value)
        assert "test-client-secret-999" not in json.dumps(exc_info.value.to_dict())
        assert "exchanged-refresh" not in json.dumps(exc_info.value.to_dict())

    async def connect_err_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    tm_conn = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(connect_err_handler),
    )
    with pytest.raises(AuthError) as exc_conn:
        await tm_conn.get_valid_token(force=True)
    assert exc_conn.value.code == "refresh_failed"


def test_build_authorize_url_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRAVA_OAUTH_BASE_URL", "http://127.0.0.1:9999/oauth/")
    url = build_authorize_url(
        "293001",
        redirect_uri="http://127.0.0.1:38762/callback",
        state="state-token-xyz",
    )
    parsed = urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "http://127.0.0.1:9999/oauth/authorize"
    qs = parse_qs(parsed.query)
    assert qs["client_id"] == ["293001"]
    assert qs["redirect_uri"] == ["http://127.0.0.1:38762/callback"]
    assert qs["response_type"] == ["code"]
    assert qs["approval_prompt"] == ["force"]
    assert qs["scope"] == [DEFAULT_SCOPE_STRING]
    assert qs["state"] == ["state-token-xyz"]


async def test_run_loopback_auth_happy_path_and_silenced_logs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    config_dir = tmp_path / "cfg"
    save_client(
        config_dir,
        ClientConfig(client_id="555111", client_secret="secret-loopback"),
    )

    async def token_handler(request: httpx.Request) -> httpx.Response:
        form = parse_qs(request.content.decode("utf-8"))
        assert form["code"] == ["secret_code_xyz"]
        return httpx.Response(
            200,
            json={
                "access_token": "loopback-access",
                "refresh_token": "loopback-refresh",
                "expires_at": 1_900_000_000,
                "expires_in": 21600,
                "athlete": {"id": 31337},
            },
        )

    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(token_handler),
    )

    callback_response_headers: dict[str, str] = {}
    callback_status = 0

    async def simulate_browser(authorize_url: str) -> None:
        nonlocal callback_status
        import urllib.error
        import urllib.request

        qs = parse_qs(urlsplit(authorize_url).query)
        redirect_uri = qs["redirect_uri"][0]
        state = qs["state"][0]
        parsed_redir = urlsplit(redirect_uri)
        fav_url = f"{parsed_redir.scheme}://{parsed_redir.netloc}/favicon.ico"

        def _do_browser_requests() -> tuple[int, int, dict[str, str]]:
            fav_status = 0
            try:
                with urllib.request.urlopen(fav_url, timeout=3.0) as fav_resp:
                    fav_status = fav_resp.status
            except urllib.error.HTTPError as err:
                fav_status = err.code

            from urllib.parse import urlencode

            query = urlencode(
                {
                    "code": "secret_code_xyz",
                    "state": state,
                    "scope": DEFAULT_SCOPE_STRING,
                }
            )
            cb_url = f"{redirect_uri}?{query}"
            with urllib.request.urlopen(cb_url, timeout=3.0) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return fav_status, resp.status, headers

        fav_code, cb_code, headers = await asyncio.to_thread(_do_browser_requests)
        assert fav_code == 404
        callback_status = cb_code
        callback_response_headers.update(headers)

    result = await run_loopback_auth(
        tm,
        port=0,
        timeout_seconds=5.0,
        on_authorize_url=simulate_browser,
    )

    assert callback_status == 200
    assert callback_response_headers["cache-control"] == "no-store"
    assert callback_response_headers["referrer-policy"] == "no-referrer"
    assert result.tokens.access_token == "loopback-access"
    assert result.tokens.athlete_id == 31337
    assert result.warnings == []
    assert result.missing_recommended_scopes == []
    assert stat.S_IMODE((config_dir / "tokens.json").stat().st_mode) == 0o600

    captured = capsys.readouterr()
    assert "secret_code_xyz" not in captured.out
    assert "secret_code_xyz" not in captured.err
    assert "secret_code_xyz" not in caplog.text


async def test_run_loopback_auth_state_mismatch_denied_optout_and_replay(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    save_client(config_dir, ClientConfig(client_id="555111", client_secret="secret-loopback"))
    token_calls = 0

    async def token_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        token_calls += 1
        return httpx.Response(
            200,
            json={
                "access_token": "optout-access",
                "refresh_token": "optout-refresh",
                "expires_at": 1_900_000_000,
            },
        )

    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(token_handler),
    )

    # 1. State mismatch
    async def bad_state_browser(authorize_url: str) -> None:
        redirect_uri = parse_qs(urlsplit(authorize_url).query)["redirect_uri"][0]
        async with httpx.AsyncClient() as local_http:
            resp = await local_http.get(redirect_uri, params={"code": "c1", "state": "wrong"})
            assert resp.status_code == 400
            assert resp.headers["cache-control"] == "no-store"
            assert resp.headers["referrer-policy"] == "no-referrer"

    with pytest.raises(AuthError) as exc_state:
        await run_loopback_auth(tm, port=0, timeout_seconds=5.0, on_authorize_url=bad_state_browser)
    assert exc_state.value.code == "oauth_state_mismatch"
    assert token_calls == 0

    # 2. Access denied
    async def denied_browser(authorize_url: str) -> None:
        qs = parse_qs(urlsplit(authorize_url).query)
        async with httpx.AsyncClient() as local_http:
            resp = await local_http.get(
                qs["redirect_uri"][0],
                params={"error": "access_denied", "state": qs["state"][0]},
            )
            assert resp.status_code == 400
            assert resp.headers["cache-control"] == "no-store"

    with pytest.raises(AuthError) as exc_denied:
        await run_loopback_auth(tm, port=0, timeout_seconds=5.0, on_authorize_url=denied_browser)
    assert exc_denied.value.code == "access_denied"
    assert token_calls == 0

    # 3. Scope opt-out warning
    async def opt_out_browser(authorize_url: str) -> None:
        qs = parse_qs(urlsplit(authorize_url).query)
        async with httpx.AsyncClient() as local_http:
            resp = await local_http.get(
                qs["redirect_uri"][0],
                params={"code": "c-opt", "state": qs["state"][0], "scope": "read"},
            )
            assert resp.status_code == 200

    opt_result = await run_loopback_auth(
        tm,
        port=0,
        timeout_seconds=5.0,
        on_authorize_url=opt_out_browser,
    )
    assert opt_result.tokens.scope == "read"
    assert opt_result.missing_recommended_scopes == ["activity:read_all", "profile:read_all"]
    assert len(opt_result.warnings) == 2
    assert token_calls == 1

    # 4. Direct LoopbackAuthServer replay rejection
    with LoopbackAuthServer(port=0, expected_state="one-time-state") as srv:
        async with httpx.AsyncClient() as local_http:
            first = await local_http.get(
                srv.redirect_uri,
                params={"code": "c1", "state": "one-time-state"},
            )
            second = await local_http.get(
                srv.redirect_uri,
                params={"code": "c1", "state": "one-time-state"},
            )
        assert first.status_code == 200
        assert second.status_code == 400
        assert second.headers["cache-control"] == "no-store"
        assert second.headers["referrer-policy"] == "no-referrer"

    # 5. Timeout
    with pytest.raises(AuthError) as exc_timeout:
        await run_loopback_auth(tm, port=0, timeout_seconds=0.05)
    assert exc_timeout.value.code == "auth_expired"


async def test_revoke_token_basic_auth_and_failure_preserves_tokens(tmp_path: Path) -> None:
    config_dir = _seed_auth_dir(tmp_path / "cfg")
    seen_requests: list[tuple[str, dict[str, list[str]]]] = []

    async def fail_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "Service Unavailable"})

    with pytest.raises(AuthError) as exc_fail:
        await revoke_token(
            config_dir,
            oauth_base_url="https://mock.strava.test/oauth",
            transport=httpx.MockTransport(fail_handler),
        )
    assert exc_fail.value.code == "revoke_failed"
    assert (config_dir / "tokens.json").exists()

    async def ok_handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(
            (
                request.headers["authorization"],
                parse_qs(request.content.decode("utf-8")),
            )
        )
        return httpx.Response(200, content=b"")

    res = await revoke_token(
        config_dir,
        token_type_hint="refresh_token",
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(ok_handler),
    )
    assert res == {"revoked": True, "token_type_hint": "refresh_token", "local_cleared": True}
    expected_basic = base64.b64encode(b"999001:test-client-secret-999").decode("ascii")
    assert seen_requests[0][0] == f"Basic {expected_basic}"
    assert seen_requests[0][1] == {
        "token": ["refresh-initial"],
        "token_type_hint": ["refresh_token"],
    }
    assert not (config_dir / "tokens.json").exists()
    assert (config_dir / "client.json").exists()


async def test_token_manager_rejects_insecure_config_dir_without_downgrading_mode(
    tmp_path: Path,
) -> None:
    config_dir = _seed_auth_dir(tmp_path / "exposed_cfg", expires_at=1_000)

    async def fail_if_called(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP should not be called when config_dir permissions are insecure")

    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(fail_if_called),
    )

    for bad_mode in (0o755, 0o750):
        os.chmod(config_dir, bad_mode)
        with pytest.raises(ConfigError) as exc_get:
            await tm.get_valid_tokens()
        assert exc_get.value.code == "insecure_permissions"
        assert stat.S_IMODE(config_dir.stat().st_mode) == bad_mode

        with pytest.raises(ConfigError) as exc_rev:
            await tm.revoke()
        assert exc_rev.value.code == "insecure_permissions"
        assert stat.S_IMODE(config_dir.stat().st_mode) == bad_mode

        with pytest.raises(ConfigError) as exc_ex:
            await tm.exchange_code("code-123")
        assert exc_ex.value.code == "insecure_permissions"
        assert stat.S_IMODE(config_dir.stat().st_mode) == bad_mode

        with pytest.raises(ConfigError) as exc_lock:
            async with _acquire_file_lock(config_dir / "tokens.lock"):
                pass
        assert exc_lock.value.code == "insecure_permissions"
        assert stat.S_IMODE(config_dir.stat().st_mode) == bad_mode

    os.chmod(config_dir, 0o700)

    # Symlinked config_dir and symlinked tokens.lock are rejected
    sym_dir = tmp_path / "sym_cfg"
    sym_dir.symlink_to(config_dir)
    tm_sym = TokenManager(sym_dir, oauth_base_url="https://mock.strava.test/oauth")
    for op in (
        tm_sym.get_valid_tokens(),
        tm_sym.revoke(),
        tm_sym.exchange_code("code-123"),
    ):
        with pytest.raises(ConfigError) as exc_sym:
            await op
        assert exc_sym.value.code == "insecure_permissions"

    lock_link = config_dir / "tokens.lock"
    lock_target = tmp_path / "outside.lock"
    lock_target.write_text("", encoding="utf-8")
    lock_link.symlink_to(lock_target)
    with pytest.raises(ConfigError) as exc_lock_sym:
        await tm.get_valid_tokens()
    assert exc_lock_sym.value.code == "insecure_permissions"


async def test_token_manager_and_revoke_coerce_str_config_dir(tmp_path: Path) -> None:
    config_dir = _seed_auth_dir(tmp_path / "str_cfg", expires_at=9_999_999_999)

    async def ok_revoke(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    tm = TokenManager(str(config_dir), clock=lambda: 1_000.0)
    assert tm.config_dir == config_dir
    assert await tm.get_valid_token() == "access-initial"

    res = await revoke_token(
        str(config_dir),
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(ok_revoke),
    )
    assert res["revoked"] is True


async def test_run_loopback_auth_redirect_uri_port_and_path_consistency(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    save_client(config_dir, ClientConfig(client_id="555111", client_secret="secret-loopback"))

    async def token_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "redir-access",
                "refresh_token": "redir-refresh",
                "expires_at": 1_900_000_000,
                "expires_in": 21600,
            },
        )

    tm = TokenManager(
        config_dir,
        oauth_base_url="https://mock.strava.test/oauth",
        transport=httpx.MockTransport(token_handler),
    )

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    async def _hit_authorize_redirect(authorize_url: str) -> str:
        qs = parse_qs(urlsplit(authorize_url).query)
        effective_redir = qs["redirect_uri"][0]
        state = qs["state"][0]
        async with httpx.AsyncClient() as local_http:
            resp = await local_http.get(
                effective_redir,
                params={"code": "code-redir", "state": state, "scope": DEFAULT_SCOPE_STRING},
            )
            assert resp.status_code == 200
        return effective_redir

    # 1. Explicit :0 port in redirect_uri (default port arg) binds ephemeral port and emits it
    seen_redirects: list[str] = []

    async def on_url_zero(url: str) -> None:
        seen_redirects.append(await _hit_authorize_redirect(url))

    res_zero = await run_loopback_auth(
        tm,
        redirect_uri="http://127.0.0.1:0/cb",
        timeout_seconds=5.0,
        on_authorize_url=on_url_zero,
    )
    parsed_zero = urlsplit(seen_redirects[-1])
    assert parsed_zero.port is not None and parsed_zero.port != 0
    assert parsed_zero.port != 38762
    assert parsed_zero.path == "/cb"
    assert res_zero.tokens.access_token == "redir-access"

    # 2. Missing port in redirect_uri uses bound port in authorize_url
    free_port = _find_free_port()

    async def on_url_no_port(url: str) -> None:
        seen_redirects.append(await _hit_authorize_redirect(url))

    res_no_port = await run_loopback_auth(
        tm,
        port=free_port,
        redirect_uri="http://127.0.0.1/custom/oauth",
        timeout_seconds=5.0,
        on_authorize_url=on_url_no_port,
    )
    parsed_no_port = urlsplit(seen_redirects[-1])
    assert parsed_no_port.port == free_port
    assert parsed_no_port.path == "/custom/oauth"
    assert res_no_port.tokens.access_token == "redir-access"

    # 3. Missing path in redirect_uri normalizes to '/' on both server and authorize_url
    async def on_url_no_path(url: str) -> None:
        seen_redirects.append(await _hit_authorize_redirect(url))

    res_no_path = await run_loopback_auth(
        tm,
        redirect_uri="http://127.0.0.1:0",
        timeout_seconds=5.0,
        on_authorize_url=on_url_no_path,
    )
    parsed_no_path = urlsplit(seen_redirects[-1])
    assert parsed_no_path.port is not None and parsed_no_path.port != 0
    assert parsed_no_path.path == "/"
    assert res_no_path.tokens.access_token == "redir-access"


async def test_loopback_server_resists_stalled_tcp_client_and_closes_promptly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Case 1: stalled TCP client held open while valid callback completes and server closes promptly
    srv = LoopbackAuthServer(port=0, expected_state="stalled-state", request_timeout=0.5)
    stalled = socket.create_connection(("127.0.0.1", srv.port), timeout=2.0)
    try:
        # Send incomplete HTTP request and hold socket open to simulate stalled peer
        stalled.sendall(b"GET /callback")
        await asyncio.sleep(0.05)

        async with httpx.AsyncClient(timeout=1.0) as local_http:
            resp = await local_http.get(
                srv.redirect_uri,
                params={"code": "valid-code", "state": "stalled-state"},
            )
        assert resp.status_code == 200
        assert srv.state_box.code == "valid-code"

        t0 = time.monotonic()
        srv.close()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5
    finally:
        stalled.close()
        srv.close()

    # Case 2: stalled connection hits request_timeout cleanly with no stderr tracebacks
    with LoopbackAuthServer(port=0, expected_state="after-timeout", request_timeout=0.1) as srv2:
        stalled2 = socket.create_connection(("127.0.0.1", srv2.port), timeout=2.0)
        try:
            stalled2.sendall(b"GET /callback")
            await asyncio.sleep(0.2)
            async with httpx.AsyncClient(timeout=1.0) as local_http:
                resp2 = await local_http.get(
                    srv2.redirect_uri,
                    params={"code": "after-timeout-code", "state": "after-timeout"},
                )
            assert resp2.status_code == 200
            assert srv2.state_box.code == "after-timeout-code"
        finally:
            stalled2.close()

    captured = capsys.readouterr()
    assert captured.err == ""
