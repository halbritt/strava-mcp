"""Tier 5 opaque-box adversarial regression E2E tests for strava-mcp."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from conftest import (
    DEFAULT_TEST_ATHLETE_ID,
    REPO_ROOT,
    HttpServerInfo,
    SeededConfig,
    SyncMCPSession,
    open_stdio_session,
    parse_tool_result,
    resolve_strava_mcp_cmd,
)
from mcp.types import LATEST_PROTOCOL_VERSION
from mock_strava import MockStravaServer

HEADERS_SCRIPT = REPO_ROOT / "client" / "strava-mcp-headers"


def _extract_error(res: Any) -> dict[str, Any]:
    data = parse_tool_result(res)
    err = data.get("error")
    assert isinstance(err, dict), f"Expected structured error dict, got {data!r}"
    return err


def _assert_no_secrets(
    text: str,
    mock_strava: MockStravaServer,
    *secrets: str,
) -> None:
    for secret in secrets:
        if secret:
            assert secret not in text
    for req in mock_strava.requests:
        for secret in secrets:
            if secret:
                assert secret not in req.body


def test_tier5_cli_init_closed_stdin_and_bad_import_leaves_no_partial_client(
    isolated_env: dict[str, str],
) -> None:
    config_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    cmd = resolve_strava_mcp_cmd()

    # 1. Closed stdin (/dev/null) with --client-id fails cleanly without traceback or client.json
    proc_null = subprocess.run(
        [*cmd, "init", "--client-id", "999001", "--json"],
        stdin=subprocess.DEVNULL,
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )
    assert proc_null.returncode == 1
    assert "Traceback (most recent call last)" not in (proc_null.stdout + proc_null.stderr)
    err_null = json.loads(proc_null.stderr)
    assert err_null["error"]["code"] == "invalid_config"
    assert not (config_dir / "client.json").exists()

    # 2. Bad --import-tokens-file with valid stdin secret must NOT persist partial client.json
    bad_tokens = config_dir.parent / "bad_tokens.json"
    bad_tokens.write_text("{not-valid-json", encoding="utf-8")
    proc_bad = subprocess.run(
        [
            *cmd,
            "init",
            "--client-id",
            "999001",
            "--client-secret-stdin",
            "--import-tokens-file",
            str(bad_tokens),
            "--json",
        ],
        input="mock-secret-never-persisted\n",
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )
    assert proc_bad.returncode == 1
    assert "Traceback (most recent call last)" not in (proc_bad.stdout + proc_bad.stderr)
    assert "mock-secret-never-persisted" not in (proc_bad.stdout + proc_bad.stderr)
    err_bad = json.loads(proc_bad.stderr)
    assert err_bad["error"]["code"] == "invalid_config"
    assert not (config_dir / "client.json").exists()
    assert not (config_dir / "tokens.json").exists()
    assert not (config_dir / "mcp-token").exists()


def test_tier5_cli_stats_invalid_athlete_id_busy_port_auth_and_non_writable_parent(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    seeded = seed_config()
    mock_strava.clear_requests()

    # 1. stats --athlete-id 0 / -5 rejected cleanly with zero upstream requests
    for bad_id in ("0", "-5"):
        proc = run_cli("stats", "--athlete-id", bad_id, "--json")
        assert proc.returncode == 1
        assert "Traceback (most recent call last)" not in (proc.stdout + proc.stderr)
        err = json.loads(proc.stderr)
        assert err["error"]["code"] == "invalid_argument"
    assert mock_strava.get_requests("/api/v3/") == []

    # 2. auth on busy loopback port exits cleanly with structured error and zero traceback
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        proc_busy = run_cli(
            "auth",
            "--port",
            str(busy_port),
            "--no-browser",
            "--timeout",
            "2",
            "--json",
        )
        assert proc_busy.returncode == 1
        combined_busy = proc_busy.stdout + "\n" + proc_busy.stderr
        assert "Traceback (most recent call last)" not in combined_busy
        err_busy = json.loads(proc_busy.stderr)
        assert err_busy["error"]["code"] == "io_error"

    # 3. cache stats with non-existent subdir under 0500 parent exits cleanly without secret leak
    ro_parent = tmp_path / "ro_parent_dir"
    ro_parent.mkdir(mode=0o700)
    os.chmod(ro_parent, 0o500)
    try:
        proc_ro = run_cli(
            "cache",
            "stats",
            "--json",
            env_overrides={"STRAVA_MCP_CACHE_DIR": str(ro_parent / "sub")},
        )
        assert proc_ro.returncode == 1
        combined_ro = proc_ro.stdout + "\n" + proc_ro.stderr
        assert "Traceback (most recent call last)" not in combined_ro
        err_ro = json.loads(proc_ro.stderr)
        assert err_ro["error"]["code"] == "insecure_permissions"
        _assert_no_secrets(
            combined_ro,
            mock_strava,
            seeded.client_secret,
            seeded.access_token,
            seeded.refresh_token,
            seeded.mcp_token,
        )
    finally:
        os.chmod(ro_parent, 0o700)


def test_tier5_headers_helper_whitespace_multiline_and_json_escaping(
    isolated_env: dict[str, str],
) -> None:
    assert HEADERS_SCRIPT.is_file() and os.access(HEADERS_SCRIPT, os.X_OK)
    config_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    token_file = config_dir / "mcp-token"

    # Surrounding whitespace is trimmed cleanly
    token_file.write_text("  tok-trimmed-ok  \n", encoding="utf-8")
    os.chmod(token_file, 0o600)
    ok_proc = subprocess.run(
        [str(HEADERS_SCRIPT)],
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert ok_proc.returncode == 0
    assert json.loads(ok_proc.stdout) == {"Authorization": "Bearer tok-trimmed-ok"}

    # Whitespace-only, tab-only, multiline, internal spaces, quotes/backslashes, NUL must fail
    for bad_raw in (
        b"   \n",
        b"\t\n",
        b"a\nb\n",
        b'tok"\\x\n',
        b"tok internal space\n",
        b"tok\x00nul\n",
    ):
        token_file.write_bytes(bad_raw)
        os.chmod(token_file, 0o600)
        bad_proc = subprocess.run(
            [str(HEADERS_SCRIPT)],
            env=isolated_env,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
        assert bad_proc.returncode != 0, f"Expected failure for {bad_raw!r}"
        assert bad_proc.stdout == b""


def _seed_multipage_sqlite_with_corrupt_page2(cache_db: Path) -> None:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_db.parent, 0o700)
    conn = sqlite3.connect(cache_db, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                payload_type TEXT NOT NULL DEFAULT 'json',
                created_at REAL NOT NULL,
                expires_at REAL,
                etag TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at "
            "ON cache_entries(expires_at)"
        )
        for idx in range(40):
            conn.execute(
                "INSERT INTO cache_entries VALUES (?, ?, 'json', 1700000000.0, NULL, NULL, NULL)",
                (f"seed-key-{idx}", json.dumps({"idx": idx, "pad": "p" * 512})),
            )
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    raw = bytearray(cache_db.read_bytes())
    assert len(raw) >= 8192, f"Expected >=2 SQLite pages, got {len(raw)} bytes"
    raw[4096:8192] = b"\xff" * 4096
    cache_db.write_bytes(bytes(raw))
    os.chmod(cache_db, 0o600)


def test_tier5_stdio_page2_and_post_init_corrupt_cache_recovery_and_symlink_journal(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    seed_config()
    cache_dir = Path(isolated_env["STRAVA_MCP_CACHE_DIR"])
    cache_db = cache_dir / "cache.sqlite"

    # 1. Page-2+ corrupt cache.sqlite at startup auto-recovers across live stdio session
    _seed_multipage_sqlite_with_corrupt_page2(cache_db)
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_page2_init.err",
    ) as session:
        r1 = session.call_tool("strava_get_athlete", {})
        assert r1.is_error is not True
        assert parse_tool_result(r1)["id"] == DEFAULT_TEST_ATHLETE_ID

        # 2. Post-init corruption while stdio server is already running also auto-recovers
        # Ensure DB has >=2 pages on disk and corrupt page 2
        conn = sqlite3.connect(cache_db, isolation_level=None)
        try:
            for idx in range(40):
                conn.execute(
                    "INSERT OR REPLACE INTO cache_entries "
                    "VALUES (?, ?, 'json', 1700000000.0, NULL, NULL, NULL)",
                    (f"live-key-{idx}", json.dumps({"idx": idx, "pad": "q" * 512})),
                )
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
        raw = bytearray(cache_db.read_bytes())
        assert len(raw) >= 8192
        raw[4096:8192] = b"\xaa" * 4096
        cache_db.write_bytes(bytes(raw))

        r2 = session.call_tool("strava_get_activity", {"activity_id": 9001})
        assert r2.is_error is not True
        assert parse_tool_result(r2)["id"] == 9001
        r3 = session.call_tool("strava_auth_status", {})
        assert r3.is_error is not True
        assert "Traceback (most recent call last)" not in session.stderr_text

    # 3. Symlinked -journal sidecar is rejected cleanly without touching outside target
    outside = tmp_path / "outside.journal"
    outside.write_text("protected", encoding="utf-8")
    journal_link = cache_dir / "cache.sqlite-journal"
    journal_link.symlink_to(outside)
    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_sym_journal.err",
    ) as sym_session:
        r_sym = sym_session.call_tool("strava_get_athlete", {})
        assert r_sym.is_error is True
        assert _extract_error(r_sym)["code"] == "insecure_permissions"
    assert outside.read_text(encoding="utf-8") == "protected"


def test_tier5_http_and_stdio_bool_injection_auth_methods_and_segment_efforts_limit_200(
    http_server: HttpServerInfo,
    mcp_http_client: SyncMCPSession,
    mock_strava: MockStravaServer,
    isolated_env: dict[str, str],
    tmp_path: Path,
) -> None:
    mock_strava.clear_requests()

    # 1. Boolean injection rejected on both Streamable HTTP and stdio with zero upstream calls
    for bad_tool, bad_args in (
        ("strava_get_activity", {"activity_id": True}),
        ("strava_list_activities", {"limit": True}),
    ):
        http_res = mcp_http_client.call_tool(bad_tool, bad_args)
        assert http_res.is_error is True

    with open_stdio_session(
        [*resolve_strava_mcp_cmd(), "stdio"],
        isolated_env,
        tmp_path / "stdio_bool.err",
    ) as stdio:
        for bad_tool, bad_args in (
            ("strava_get_activity", {"activity_id": True}),
            ("strava_list_activities", {"limit": True}),
        ):
            stdio_res = stdio.call_tool(bad_tool, bad_args)
            assert stdio_res.is_error is True

    assert mock_strava.get_requests("/api/v3/") == []

    # 2. HTTP GET /mcp, DELETE /mcp, and lowercase 'bearer' rejected with 401
    init_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "tier5-http", "version": "1.0"},
        },
    }
    with httpx.Client(timeout=5.0) as raw_client:
        r_get = raw_client.get(http_server.mcp_url)
        assert r_get.status_code == 401
        assert r_get.headers.get("www-authenticate") == "Bearer"

        r_del = raw_client.delete(http_server.mcp_url)
        assert r_del.status_code == 401
        assert r_del.headers.get("www-authenticate") == "Bearer"

        r_lower = raw_client.post(
            http_server.mcp_url,
            headers={
                "authorization": f"bearer {http_server.token}",
                "accept": "application/json, text/event-stream",
            },
            json=init_body,
        )
        assert r_lower.status_code == 401
        assert r_lower.headers.get("www-authenticate") == "Bearer"

    # 3. strava_list_segment_efforts(limit=200) with 200 mock efforts returns has_more=True
    mock_strava.seed_segment_efforts(8001, 200)
    eff_res = mcp_http_client.call_tool(
        "strava_list_segment_efforts",
        {"segment_id": 8001, "limit": 200, "refresh": True},
    )
    assert eff_res.is_error is not True
    eff_data = parse_tool_result(eff_res)
    assert eff_data["count"] == 200
    assert len(eff_data["efforts"]) == 200
    assert eff_data["has_more"] is True
