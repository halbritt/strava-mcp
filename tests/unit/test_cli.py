"""Unit tests for strava_mcp.cli."""

from __future__ import annotations

import io
import json
import os
import stat
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from strava_mcp import __version__, cli, mcp_server
from strava_mcp.cache import ResponseCache
from strava_mcp.config import (
    ENV_API_BASE_URL,
    ENV_CACHE_DIR,
    ENV_CONFIG_DIR,
    ENV_OAUTH_BASE_URL,
    ClientConfig,
    Config,
    TokenData,
    load_mcp_token,
    load_tokens,
    save_client,
    save_tokens,
    write_secret_text,
)


def _set_isolated_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    api_url: str = "https://mock.strava.test/api/v3",
    oauth_url: str = "https://mock.strava.test/oauth",
) -> Config:
    cfg_dir = tmp_path / "cfg"
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv(ENV_CONFIG_DIR, str(cfg_dir))
    monkeypatch.setenv(ENV_CACHE_DIR, str(cache_dir))
    monkeypatch.setenv(ENV_API_BASE_URL, api_url)
    monkeypatch.setenv(ENV_OAUTH_BASE_URL, oauth_url)
    return Config(
        config_dir=cfg_dir,
        cache_dir=cache_dir,
        api_base_url=api_url,
        oauth_base_url=oauth_url,
    )


def test_cli_help_version_and_no_secret_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ver_exit:
        cli.main(["--version"])
    assert ver_exit.value.code == 0
    assert __version__ in capsys.readouterr().out

    with pytest.raises(SystemExit) as help_exit:
        cli.main(["--help"])
    assert help_exit.value.code == 0
    help_out = capsys.readouterr().out
    for cmd in (
        "init",
        "auth",
        "status",
        "refresh",
        "revoke",
        "stdio",
        "serve",
        "cache",
        "activities",
        "activity",
        "stats",
    ):
        assert cmd in help_out

    for secret_flag in ("--client-secret", "--access-token", "--refresh-token"):
        with pytest.raises(SystemExit) as bad_exit:
            cli.main(["init", "--client-id", "123", secret_flag, "nope"])
        assert bad_exit.value.code != 0


def test_cli_init_stdin_import_tokens_and_permissions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    import_file = tmp_path / "seed_tokens.json"
    import_file.write_text(
        json.dumps(
            {
                "access_token": "cli_seed_access_secret",
                "refresh_token": "cli_seed_refresh_secret",
                "expires_at": int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
                "scope": "read",
                "athlete": {"id": 555},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("cli_client_secret_val\n"))
    rc = cli.main(
        [
            "init",
            "--client-id",
            "123456",
            "--client-secret-stdin",
            "--import-tokens-file",
            str(import_file),
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["initialized"] is True
    assert data["client_id"] == "123456"
    assert data["tokens_imported"] is True
    assert data["mcp_token_configured"] is True

    for secret in ("cli_client_secret_val", "cli_seed_access_secret", "cli_seed_refresh_secret"):
        assert secret not in captured.out
        assert secret not in captured.err

    assert stat.S_IMODE(cfg.config_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(cfg.client_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg.tokens_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg.mcp_token_file.stat().st_mode) == 0o600

    first_mcp_tok = load_mcp_token(cfg)
    assert load_tokens(cfg).athlete_id == 555

    # Re-running init preserves existing mcp-token
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    rc2 = cli.main(["init", "--json"])
    assert rc2 == 0
    assert load_mcp_token(cfg) == first_mcp_tok


def test_cli_auth_loopback_and_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _MockOAuthHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            body = json.dumps(
                {
                    "access_token": "oauth_new_access_tok",
                    "refresh_token": "oauth_new_refresh_tok",
                    "expires_at": int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
                    "scope": "read",
                    "athlete": {"id": 99},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    oauth_srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockOAuthHandler)
    oauth_port = oauth_srv.server_address[1]
    oauth_thread = Thread(target=oauth_srv.serve_forever, daemon=True)
    oauth_thread.start()

    try:
        cfg = _set_isolated_env(
            monkeypatch,
            tmp_path,
            oauth_url=f"http://127.0.0.1:{oauth_port}/oauth",
        )
        save_client(cfg, ClientConfig(client_id="123456", client_secret="oauth_client_sec"))

        orig_run = cli.run_loopback_auth

        async def _auto_callback_auth(*args: Any, **kwargs: Any) -> Any:
            orig_on_url = kwargs.get("on_authorize_url")

            def _trigger(url: str) -> None:
                if orig_on_url is not None:
                    orig_on_url(url)
                qs = parse_qs(urlsplit(url).query)
                redir = qs["redirect_uri"][0]
                state = qs["state"][0]
                Thread(
                    target=lambda: httpx.get(
                        redir,
                        params={"code": "auth_code_123", "state": state, "scope": "read"},
                        timeout=5.0,
                    ),
                    daemon=True,
                ).start()

            kwargs["on_authorize_url"] = _trigger
            return await orig_run(*args, **kwargs)

        monkeypatch.setattr(cli, "run_loopback_auth", _auto_callback_auth)
        rc = cli.main(["auth", "--port", "0", "--no-browser", "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["authenticated"] is True
        assert out["athlete_id"] == 99
        assert out["missing_recommended_scopes"] == ["activity:read_all", "profile:read_all"]
        assert len(out["warnings"]) == 2
        assert "oauth_new_access_tok" not in json.dumps(out)
    finally:
        oauth_srv.shutdown()
        oauth_srv.server_close()


def test_cli_status_offline_live_and_redaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    rc_empty = cli.main(["status", "--json"])
    assert rc_empty == 0
    empty_status = json.loads(capsys.readouterr().out)
    assert empty_status["client_configured"] is False
    assert empty_status["authenticated"] is False
    assert not cfg.cache_db_file.exists()

    save_client(cfg, ClientConfig(client_id="123456", client_secret="sec_status_secret"))
    save_tokens(
        cfg,
        TokenData(
            access_token="tok_status_access",
            refresh_token="tok_status_refresh",
            expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
            scope="read,activity:read_all,profile:read_all",
            athlete_id=42,
        ),
    )
    write_secret_text(cfg.mcp_token_file, "tok_status_mcp\n")

    def _live_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "X-RateLimit-Limit": "400,4000",
                "X-RateLimit-Usage": "5,20",
                "X-ReadRateLimit-Limit": "200,2000",
                "X-ReadRateLimit-Usage": "5,20",
            },
            json={"id": 42, "username": "live_rider"},
        )

    orig_client = cli.StravaClient
    monkeypatch.setattr(
        cli,
        "StravaClient",
        lambda c: orig_client(c, transport=httpx.MockTransport(_live_handler)),
    )

    rc_live = cli.main(["status", "--check-live", "--json"])
    assert rc_live == 0
    captured = capsys.readouterr()
    live_status = json.loads(captured.out)
    assert live_status["live_check"]["ok"] is True
    assert live_status["rate_limits"]["read"]["usage_15m"] == 5
    for secret in (
        "sec_status_secret",
        "tok_status_access",
        "tok_status_refresh",
        "tok_status_mcp",
    ):
        assert secret not in captured.out
        assert secret not in captured.err


def test_cli_refresh_and_revoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    save_client(cfg, ClientConfig(client_id="123456", client_secret="sec_ref"))
    save_tokens(
        cfg,
        TokenData(
            access_token="old_acc",
            refresh_token="old_ref",
            expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
            scope="read,activity:read_all",
            athlete_id=42,
        ),
    )

    def _oauth_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oauth/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "rotated_acc",
                    "refresh_token": "rotated_ref",
                    "expires_at": int(datetime(2099, 2, 1, tzinfo=UTC).timestamp()),
                },
            )
        if request.url.path == "/oauth/revoke":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    orig_tm = cli.TokenManager
    monkeypatch.setattr(
        cli,
        "TokenManager",
        lambda c: orig_tm(c, transport=httpx.MockTransport(_oauth_handler)),
    )
    rc_ref = cli.main(["refresh", "--json"])
    assert rc_ref == 0
    ref_out = json.loads(capsys.readouterr().out)
    assert ref_out["refreshed"] is True
    assert load_tokens(cfg).access_token == "rotated_acc"

    orig_revoke = cli.revoke_token

    async def _mock_revoke(c: Any, **kwargs: Any) -> dict[str, Any]:
        return await orig_revoke(c, transport=httpx.MockTransport(_oauth_handler), **kwargs)

    monkeypatch.setattr(cli, "revoke_token", _mock_revoke)
    rc_rev = cli.main(["revoke", "--token-type", "access_token", "--json"])
    assert rc_rev == 0
    rev_out = json.loads(capsys.readouterr().out)
    assert rev_out["revoked"] is True
    assert rev_out["local_cleared"] is True
    assert not cfg.tokens_file.exists()


def test_cli_cache_stats_and_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    with ResponseCache(cfg) as cache:
        cache.set("k1", {"ok": True}, ttl_seconds=600)

    assert cli.main(["cache", "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["entry_count"] == 1

    assert cli.main(["cache", "clear", "--json"]) == 0
    cleared = json.loads(capsys.readouterr().out)
    assert cleared["cleared"] is True
    assert cleared["removed_entries"] == 1


def test_cli_activities_activity_and_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    save_client(cfg, ClientConfig(client_id="123456", client_secret="sec"))
    save_tokens(
        cfg,
        TokenData(
            access_token="acc",
            refresh_token="ref",
            expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
            scope="read,activity:read_all",
            athlete_id=42,
        ),
    )

    def _api_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/athlete/activities":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 1,
                        "name": "Gravel 1",
                        "sport_type": "GravelRide",
                        "type": "Ride",
                        "start_date": "2026-11-20T16:00:00Z",
                        "start_date_local": "2026-11-20T08:00:00Z",
                        "distance": 30000.0,
                        "moving_time": 3600,
                        "map": {"summary_polyline": "p1"},
                    },
                    {
                        "id": 2,
                        "name": "Run 2",
                        "sport_type": "TrailRun",
                        "type": "Run",
                        "start_date": "2026-11-21T16:00:00Z",
                        "start_date_local": "2026-11-21T08:00:00Z",
                        "distance": 10000.0,
                        "moving_time": 3000,
                    },
                ],
            )
        if request.url.path == "/api/v3/activities/1":
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "name": "Gravel 1",
                    "sport_type": "GravelRide",
                    "distance": 30000.0,
                    "moving_time": 3600,
                    "map": {"polyline": "fullp"},
                    "segment_efforts": [{"id": 11}],
                },
            )
        if request.url.path == "/api/v3/athletes/42/stats":
            return httpx.Response(200, json={"biggest_ride_distance": 100000.0})
        return httpx.Response(404)

    orig_client = cli.StravaClient
    monkeypatch.setattr(
        cli,
        "StravaClient",
        lambda c: orig_client(c, transport=httpx.MockTransport(_api_handler)),
    )

    rc_acts = cli.main(
        [
            "activities",
            "--after",
            "30d",
            "--sport-type",
            "GravelRide",
            "--limit",
            "1",
            "--json",
        ]
    )
    assert rc_acts == 0
    acts_out = json.loads(capsys.readouterr().out)
    assert acts_out["count"] == 1
    assert "map" not in acts_out["activities"][0]

    assert cli.main(["activity", "1", "--json"]) == 0
    act_compact = json.loads(capsys.readouterr().out)
    assert "map" not in act_compact
    assert "segment_efforts" not in act_compact

    assert cli.main(["activity", "1", "--include-all-efforts", "--include-polyline", "--json"]) == 0
    act_full = json.loads(capsys.readouterr().out)
    assert act_full["map"]["polyline"] == "fullp"
    assert len(act_full["segment_efforts"]) == 1

    assert cli.main(["stats", "--json"]) == 0
    stats_out = json.loads(capsys.readouterr().out)
    assert "Everyone" in stats_out["visibility_caveat"]


def test_cli_stdio_and_serve_dispatch_and_loopback_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_isolated_env(monkeypatch, tmp_path)
    called: dict[str, Any] = {}

    monkeypatch.setattr(mcp_server, "run_stdio", lambda *, config=None: called.update(stdio=config))
    monkeypatch.setattr(
        mcp_server,
        "serve_http",
        lambda **kwargs: called.update(serve=kwargs),
    )

    assert cli.main(["stdio"]) == 0
    assert "stdio" in called

    assert cli.main(["serve", "--listen", "127.0.0.1:8772", "--allowed-host", "mybox.ts.net"]) == 0
    assert called["serve"]["host"] == "127.0.0.1"
    assert called["serve"]["port"] == 8772
    assert called["serve"]["allowed_hosts"] == ["mybox.ts.net"]

    assert cli.main(["serve", "--listen", "0.0.0.0:8772"]) == 1
    err = capsys.readouterr().err
    assert "invalid_config" in err


def test_cli_error_formatting_never_leaks_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    save_client(cfg, ClientConfig(client_id="123456", client_secret="super_secret_cli_val"))
    save_tokens(
        cfg,
        TokenData(
            access_token="access_secret_cli_val",
            refresh_token="refresh_secret_cli_val",
            expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
            scope="read",
            athlete_id=42,
        ),
    )

    # InsufficientScopeError path
    assert cli.main(["activities", "--json"]) == 1
    err_scope = capsys.readouterr().err
    parsed_scope = json.loads(err_scope)
    assert parsed_scope["error"]["code"] == "insufficient_scope"
    for secret in ("super_secret_cli_val", "access_secret_cli_val", "refresh_secret_cli_val"):
        assert secret not in err_scope

    # Insecure permissions path
    os.chmod(cfg.client_file, 0o644)
    assert cli.main(["status", "--json"]) == 1
    err_perm = capsys.readouterr().err
    assert json.loads(err_perm)["error"]["code"] == "insecure_permissions"


def test_cli_init_import_tokens_rejects_malformed_numeric_and_bool_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    save_client(cfg, ClientConfig(client_id="123456", client_secret="sec_import_test"))

    bad_payloads: list[dict[str, Any]] = [
        {"access_token": "tok_secret_a", "refresh_token": "tok_secret_r", "expires_in": [123]},
        {"access_token": "tok_secret_a", "refresh_token": "tok_secret_r", "expires_in": True},
        {
            "access_token": "tok_secret_a",
            "refresh_token": "tok_secret_r",
            "expires_in": "not-a-num",
        },
        {
            "access_token": "tok_secret_a",
            "refresh_token": "tok_secret_r",
            "expires_in": 1e400,
        },
        {
            "access_token": "tok_secret_a",
            "refresh_token": "tok_secret_r",
            "expires_at": 2000000000,
            "athlete": {"id": "abc"},
        },
        {
            "access_token": "tok_secret_a",
            "refresh_token": "tok_secret_r",
            "expires_at": 2000000000,
            "athlete": {"id": [123]},
        },
        {
            "access_token": "tok_secret_a",
            "refresh_token": "tok_secret_r",
            "expires_at": 2000000000,
            "athlete": {"id": True},
        },
        {
            "access_token": "tok_secret_a",
            "refresh_token": "tok_secret_r",
            "expires_at": 2000000000,
            "athlete_id": False,
        },
    ]

    for idx, payload in enumerate(bad_payloads):
        bad_file = tmp_path / f"bad_tokens_{idx}.json"
        bad_file.write_text(json.dumps(payload), encoding="utf-8")
        rc = cli.main(["init", "--import-tokens-file", str(bad_file), "--json"])
        assert rc == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        parsed = json.loads(captured.err)
        assert parsed["error"]["code"] == "invalid_config"
        assert "tok_secret_a" not in captured.err
        assert "tok_secret_r" not in captured.err

    # Valid expires_in integer without expires_at still succeeds
    ok_file = tmp_path / "ok_tokens.json"
    ok_file.write_text(
        json.dumps(
            {
                "access_token": "tok_ok_a",
                "refresh_token": "tok_ok_r",
                "expires_in": 3600,
                "athlete": {"id": "777"},
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["init", "--import-tokens-file", str(ok_file), "--json"]) == 0
    ok_out = json.loads(capsys.readouterr().out)
    assert ok_out["tokens_imported"] is True
    assert load_tokens(cfg).athlete_id == 777


def test_cli_out_of_range_token_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _set_isolated_env(monkeypatch, tmp_path)
    save_client(cfg, ClientConfig(client_id="123456", client_secret="sec_overflow_test"))

    overflow_payloads: list[dict[str, Any]] = [
        {
            "access_token": "tok_overflow_a",
            "refresh_token": "tok_overflow_r",
            "expires_at": 99999999999999999999,
        },
        {
            "access_token": "tok_overflow_a",
            "refresh_token": "tok_overflow_r",
            "expires_in": 99999999999999999999,
        },
        {
            "access_token": "tok_overflow_a",
            "refresh_token": "tok_overflow_r",
            "expires_at": 2000000000,
            "updated_at": 99999999999999999999,
        },
    ]
    for idx, payload in enumerate(overflow_payloads):
        bad_file = tmp_path / f"overflow_tokens_{idx}.json"
        bad_file.write_text(json.dumps(payload), encoding="utf-8")
        rc = cli.main(["init", "--import-tokens-file", str(bad_file), "--json"])
        assert rc == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        parsed = json.loads(captured.err)
        assert parsed["error"]["code"] == "invalid_config"
        assert "tok_overflow_a" not in captured.err
        assert "tok_overflow_r" not in captured.err

    # Pre-existing tokens.json with an out-of-range expires_at must not crash status
    save_tokens(
        cfg,
        TokenData(
            access_token="tok_extreme_a",
            refresh_token="tok_extreme_r",
            expires_at=99999999999999999999,
            scope="read,activity:read_all,profile:read_all",
            athlete_id=42,
        ),
    )
    rc_status = cli.main(["status", "--json"])
    assert rc_status == 0
    captured_status = capsys.readouterr()
    assert captured_status.err == ""
    status_out = json.loads(captured_status.out)
    assert status_out["authenticated"] is True
    assert status_out["expires_at"] == 99999999999999999999
    assert status_out["expires_at_utc"] is None
    assert status_out["is_expired"] is False

    rc_human = cli.main(["status"])
    assert rc_human == 0
    captured_human = capsys.readouterr()
    assert "(out-of-range)" in captured_human.out
