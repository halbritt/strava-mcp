"""Standalone self-tests verifying MockStravaServer and conftest E2E harness."""

from __future__ import annotations

import asyncio
import base64
import json
import stat
import subprocess
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import (
    open_mcp_http_session,
    open_stdio_session,
    parse_tool_result,
    resolve_strava_mcp_cmd,
    start_http_server,
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mock_strava import (
    DEFAULT_ACCESS_TOKEN,
    DEFAULT_CLIENT_ID,
    DEFAULT_CLIENT_SECRET,
    DEFAULT_REFRESH_TOKEN,
    MockStravaServer,
)


def test_isolated_env_vars_and_no_host_home_touch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    mock_strava: MockStravaServer,
) -> None:
    monkeypatch.setenv("STRAVA_LEAK_CHECK", "should_be_stripped")
    isolated_env: dict[str, str] = request.getfixturevalue("isolated_env")
    assert isolated_env["STRAVA_API_BASE_URL"] == mock_strava.api_base_url
    assert isolated_env["STRAVA_OAUTH_BASE_URL"] == mock_strava.oauth_base_url
    assert Path(isolated_env["STRAVA_MCP_CONFIG_DIR"]).parent == tmp_path
    assert Path(isolated_env["STRAVA_MCP_CACHE_DIR"]).parent == tmp_path
    assert Path(isolated_env["HOME"]).parent == tmp_path
    assert Path(isolated_env["XDG_CONFIG_HOME"]).parent == tmp_path
    assert Path(isolated_env["XDG_DATA_HOME"]).parent == tmp_path
    assert isolated_env["NO_COLOR"] == "1"
    for key in ("STRAVA_MCP_CONFIG_DIR", "STRAVA_MCP_CACHE_DIR", "HOME"):
        mode = stat.S_IMODE(Path(isolated_env[key]).stat().st_mode)
        assert mode == 0o700
    assert "STRAVA_LEAK_CHECK" not in isolated_env


def test_seed_config_permissions_and_contents(
    isolated_env: dict[str, str],
    seed_config: Any,
    mock_strava: MockStravaServer,
) -> None:
    seeded = seed_config(scope="read")
    assert stat.S_IMODE(seeded.config_dir.stat().st_mode) == 0o700
    for fpath in (seeded.client_file, seeded.tokens_file, seeded.mcp_token_file):
        mode = stat.S_IMODE(fpath.stat().st_mode)
        assert mode == 0o600
        assert (mode & 0o077) == 0

    client_data = json.loads(seeded.client_file.read_text(encoding="utf-8"))
    assert client_data == {
        "client_id": DEFAULT_CLIENT_ID,
        "client_secret": DEFAULT_CLIENT_SECRET,
    }
    tokens_data = json.loads(seeded.tokens_file.read_text(encoding="utf-8"))
    assert tokens_data["access_token"] == DEFAULT_ACCESS_TOKEN
    assert tokens_data["refresh_token"] == DEFAULT_REFRESH_TOKEN
    assert tokens_data["scope"] == "read"
    assert seeded.mcp_token_file.read_text(encoding="utf-8").strip() == seeded.mcp_token

    active = mock_strava.get_active_tokens()
    assert active["active_access_tokens"][DEFAULT_ACCESS_TOKEN]["scopes"] == ["read"]

    no_tok = seed_config(include_tokens=False, include_mcp_token=False, file_mode=0o644)
    assert not no_tok.tokens_file.exists()
    assert not no_tok.mcp_token_file.exists()
    assert stat.S_IMODE(no_tok.client_file.stat().st_mode) == 0o644


def test_oauth_authorize_redirect_state_and_opt_out(mock_strava: MockStravaServer) -> None:
    mock_strava.configure_oauth(consent_scopes=["read", "activity:read"])
    resp = httpx.get(
        f"{mock_strava.oauth_base_url}/authorize",
        params={
            "client_id": DEFAULT_CLIENT_ID,
            "redirect_uri": "http://127.0.0.1:38762/callback",
            "response_type": "code",
            "scope": "read,read_all,profile:read_all,activity:read_all",
            "approval_prompt": "force",
            "state": "csrf-xyz-123",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302
    loc = urlsplit(resp.headers["Location"])
    qs = parse_qs(loc.query)
    assert qs["state"] == ["csrf-xyz-123"]
    assert qs["scope"] == ["read,activity:read"]
    assert len(qs["code"][0]) >= 10

    mock_strava.configure_oauth(deny_access=True)
    denied = httpx.get(
        f"{mock_strava.oauth_base_url}/authorize",
        params={
            "client_id": DEFAULT_CLIENT_ID,
            "redirect_uri": "http://localhost:38762/callback",
            "response_type": "code",
            "state": "csrf-deny",
        },
        follow_redirects=False,
    )
    assert denied.status_code == 302
    dqs = parse_qs(urlsplit(denied.headers["Location"]).query)
    assert dqs["error"] == ["access_denied"]
    assert dqs["state"] == ["csrf-deny"]

    bad_redir = httpx.get(
        f"{mock_strava.oauth_base_url}/authorize",
        params={
            "client_id": DEFAULT_CLIENT_ID,
            "redirect_uri": "https://evil.example.com/callback",
            "response_type": "code",
        },
    )
    assert bad_redir.status_code == 400


def test_oauth_token_exchange_and_rotation(mock_strava: MockStravaServer) -> None:
    auth_resp = httpx.get(
        f"{mock_strava.oauth_base_url}/authorize",
        params={
            "client_id": DEFAULT_CLIENT_ID,
            "redirect_uri": "http://127.0.0.1:38762/callback",
            "response_type": "code",
            "scope": "read,profile:read_all,activity:read_all",
            "state": "st1",
        },
        follow_redirects=False,
    )
    code = parse_qs(urlsplit(auth_resp.headers["Location"]).query)["code"][0]

    tok_resp = httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
    )
    assert tok_resp.status_code == 200
    tok = tok_resp.json()
    assert tok["token_type"] == "Bearer"
    assert tok["athlete"]["id"] == 100001
    assert "activity:read_all" in tok["scope"]

    reused = httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
    )
    assert reused.status_code == 400

    # Within >1h window, refresh returns existing access token when enforce_one_hour_window=True
    same_resp = httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"],
        },
    )
    assert same_resp.status_code == 200
    assert same_resp.json()["access_token"] == tok["access_token"]

    # Near expiry (<= 3600s) triggers rotation & omits scope/athlete by default
    mock_strava.set_near_expiry(seconds=120, access_token=tok["access_token"])
    rot_resp = httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"],
        },
    )
    assert rot_resp.status_code == 200
    rot = rot_resp.json()
    assert rot["access_token"] != tok["access_token"]
    assert rot["refresh_token"] != tok["refresh_token"]
    assert "scope" not in rot
    assert "athlete" not in rot

    # Old refresh token is invalidated immediately
    stale = httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"],
        },
    )
    assert stale.status_code == 400


def test_oauth_revoke_and_deauthorize(mock_strava: MockStravaServer) -> None:
    basic = base64.b64encode(f"{DEFAULT_CLIENT_ID}:{DEFAULT_CLIENT_SECRET}".encode()).decode()
    bad_auth = httpx.post(
        f"{mock_strava.oauth_base_url}/revoke",
        headers={"Authorization": "Basic invalid"},
        data={"token": DEFAULT_ACCESS_TOKEN},
    )
    assert bad_auth.status_code == 401

    missing_tok = httpx.post(
        f"{mock_strava.oauth_base_url}/revoke",
        headers={"Authorization": f"Basic {basic}"},
        data={},
    )
    assert missing_tok.status_code == 400

    rev = httpx.post(
        f"{mock_strava.oauth_base_url}/revoke",
        headers={"Authorization": f"Basic {basic}"},
        data={"token": DEFAULT_ACCESS_TOKEN, "token_type_hint": "access_token"},
    )
    assert rev.status_code == 200
    assert rev.content == b""

    me = httpx.get(
        f"{mock_strava.api_base_url}/athlete",
        headers={"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"},
    )
    assert me.status_code == 401

    mock_strava.set_token_state(
        access_token="tok-deauth-1",
        refresh_token="ref-deauth-1",
        scopes=["read"],
    )
    deauth = httpx.post(
        f"{mock_strava.oauth_base_url}/deauthorize",
        data={"access_token": "tok-deauth-1"},
    )
    assert deauth.status_code == 200
    assert deauth.json() == {"access_token": "tok-deauth-1"}
    deauth_again = httpx.post(
        f"{mock_strava.oauth_base_url}/deauthorize",
        data={"access_token": "tok-deauth-1"},
    )
    assert deauth_again.status_code == 401


def test_concurrent_refresh_rotation_thread_safety(mock_strava: MockStravaServer) -> None:
    mock_strava.set_near_expiry(seconds=60)
    url = f"{mock_strava.oauth_base_url}/token"

    def _do_refresh(_: int) -> int:
        resp = httpx.post(
            url,
            data={
                "client_id": DEFAULT_CLIENT_ID,
                "client_secret": DEFAULT_CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": DEFAULT_REFRESH_TOKEN,
            },
        )
        return resp.status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(_do_refresh, range(8)))

    assert codes.count(200) == 1
    assert codes.count(400) == 7
    assert DEFAULT_REFRESH_TOKEN in mock_strava.get_active_tokens()["invalidated_refresh_tokens"]


def test_scope_enforcement_read_vs_profile_vs_activity(mock_strava: MockStravaServer) -> None:
    hdrs = {"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"}

    mock_strava.set_token_state(scopes=["read"])
    summary = httpx.get(f"{mock_strava.api_base_url}/athlete", headers=hdrs)
    assert summary.status_code == 200
    assert "bikes" not in summary.json()
    assert httpx.get(f"{mock_strava.api_base_url}/athlete/zones", headers=hdrs).status_code == 401
    assert (
        httpx.get(f"{mock_strava.api_base_url}/athlete/activities", headers=hdrs).status_code == 401
    )

    stats = httpx.get(f"{mock_strava.api_base_url}/athletes/100001/stats", headers=hdrs)
    assert stats.status_code == 200
    # Only Everyone rides (9001, 9006) are aggregated; Only Me 9003 & Followers 9002 are excluded
    assert stats.json()["all_ride_totals"]["count"] == 2
    assert stats.json()["all_run_totals"]["count"] == 0

    mock_strava.set_token_state(scopes=["read", "profile:read_all"])
    detailed = httpx.get(f"{mock_strava.api_base_url}/athlete", headers=hdrs)
    assert detailed.status_code == 200
    assert detailed.json()["ftp"] == 285
    assert httpx.get(f"{mock_strava.api_base_url}/athlete/zones", headers=hdrs).status_code == 200

    mock_strava.set_token_state(scopes=["read", "activity:read"])
    acts_limited = httpx.get(f"{mock_strava.api_base_url}/athlete/activities", headers=hdrs).json()
    ids_limited = {a["id"] for a in acts_limited}
    assert ids_limited == {9001, 9002, 9004, 9006, 9007}
    assert httpx.get(f"{mock_strava.api_base_url}/activities/9003", headers=hdrs).status_code == 401
    routes_pub = httpx.get(
        f"{mock_strava.api_base_url}/athletes/100001/routes", headers=hdrs
    ).json()
    assert [r["id"] for r in routes_pub] == [7001]
    assert httpx.get(f"{mock_strava.api_base_url}/routes/7002", headers=hdrs).status_code == 401
    assert (
        httpx.get(
            f"{mock_strava.api_base_url}/segment_efforts/6001/streams",
            headers=hdrs,
            params={"keys": "distance", "key_by_type": "true"},
        ).status_code
        == 401
    )

    mock_strava.set_token_state(scopes=["read", "read_all", "activity:read_all"])
    acts_all = httpx.get(f"{mock_strava.api_base_url}/athlete/activities", headers=hdrs).json()
    assert {a["id"] for a in acts_all} == {9001, 9002, 9003, 9004, 9005, 9006, 9007}
    assert httpx.get(f"{mock_strava.api_base_url}/routes/7002", headers=hdrs).status_code == 200


def test_rate_limit_headers_and_oauth_exclusion(mock_strava: MockStravaServer) -> None:
    hdrs = {"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"}
    r1 = httpx.get(f"{mock_strava.api_base_url}/athlete", headers=hdrs)
    assert r1.status_code == 200
    assert r1.headers["X-RateLimit-Limit"] == "400,4000"
    assert r1.headers["X-RateLimit-Usage"] == "1,1"
    assert r1.headers["X-ReadRateLimit-Limit"] == "200,2000"
    assert r1.headers["X-ReadRateLimit-Usage"] == "1,1"

    # OAuth endpoint calls do not increment REST rate-limit counters
    mock_strava.set_near_expiry(seconds=60)
    httpx.post(
        f"{mock_strava.oauth_base_url}/token",
        data={
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": DEFAULT_REFRESH_TOKEN,
        },
    )
    mock_strava.set_token_state(access_token="tok-rl-check", refresh_token="ref-rl-check")
    r2 = httpx.get(
        f"{mock_strava.api_base_url}/athlete",
        headers={"Authorization": "Bearer tok-rl-check"},
    )
    assert r2.headers["X-RateLimit-Usage"] == "2,2"

    mock_strava.inject_fault(fault_type="rate_limit_429", times=1)
    r429 = httpx.get(
        f"{mock_strava.api_base_url}/athlete",
        headers={"Authorization": "Bearer tok-rl-check"},
    )
    assert r429.status_code == 429

    mock_strava.set_rate_limits(read_limit_15m=5, read_usage_15m=5, read_usage_day=10)
    auto_429 = httpx.get(
        f"{mock_strava.api_base_url}/athlete",
        headers={"Authorization": "Bearer tok-rl-check"},
    )
    assert auto_429.status_code == 429
    assert auto_429.headers["X-ReadRateLimit-Usage"] == "6,11"


def test_fault_injections_and_request_log(mock_strava: MockStravaServer) -> None:
    hdrs = {"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"}

    mock_strava.inject_fault(fault_type="cloudfront_403", path_pattern=r"/athlete$", times=1)
    cf = httpx.get(f"{mock_strava.api_base_url}/athlete", headers=hdrs)
    assert cf.status_code == 403
    assert "text/html" in cf.headers["Content-Type"]
    assert "CloudFront" in cf.text

    mock_strava.set_require_user_agent(True)
    ua_blocked = httpx.get(
        f"{mock_strava.api_base_url}/athlete",
        headers={**hdrs, "User-Agent": "Python-urllib/3.12"},
    )
    assert ua_blocked.status_code == 403
    mock_strava.set_require_user_agent(False)

    mock_strava.set_subscription(False, error_status=402)
    assert (
        httpx.get(f"{mock_strava.api_base_url}/activities/9001/zones", headers=hdrs).status_code
        == 402
    )
    assert (
        httpx.get(
            f"{mock_strava.api_base_url}/segment_efforts",
            headers=hdrs,
            params={"segment_id": "8001"},
        ).status_code
        == 402
    )
    mock_strava.set_subscription(False, error_status=403)
    assert (
        httpx.get(f"{mock_strava.api_base_url}/segment_efforts/6001", headers=hdrs).status_code
        == 403
    )

    assert (
        httpx.get(
            f"{mock_strava.api_base_url}/segments/explore",
            headers=hdrs,
            params={"bounds": "45.5,-122.8,45.6,-122.6"},
        ).status_code
        == 403
    )
    mock_strava.set_extended_tier(True)
    assert (
        httpx.get(
            f"{mock_strava.api_base_url}/segments/explore",
            headers=hdrs,
            params={"bounds": "45.5,-122.8,45.6,-122.6"},
        ).status_code
        == 200
    )

    manual_streams = httpx.get(
        f"{mock_strava.api_base_url}/activities/9004/streams",
        headers=hdrs,
        params={"keys": "time,heartrate", "key_by_type": "true"},
    )
    assert manual_streams.status_code == 404

    mock_strava.set_token_state(expires_in=45)
    assert (
        mock_strava.get_active_tokens()["active_access_tokens"][DEFAULT_ACCESS_TOKEN]["expires_at"]
        <= int(time.time()) + 50
    )
    mock_strava.expire_access_token()
    assert httpx.get(f"{mock_strava.api_base_url}/athlete", headers=hdrs).status_code == 401

    assert len(mock_strava.requests) >= 7
    assert len(mock_strava.get_requests(path_prefix="/api/v3/activities/9004", method="GET")) == 1
    mock_strava.clear_requests()
    assert mock_strava.requests == []


def test_rich_dataset_endpoints(mock_strava: MockStravaServer) -> None:
    hdrs = {"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"}
    acts = httpx.get(f"{mock_strava.api_base_url}/athlete/activities", headers=hdrs).json()
    by_id = {a["id"]: a for a in acts}
    assert by_id[9001]["sport_type"] == "GravelRide"
    assert by_id[9001]["type"] == "Ride"
    assert by_id[9002]["sport_type"] == "TrailRun"
    assert by_id[9003]["sport_type"] == "MountainBikeRide"

    detail_default = httpx.get(f"{mock_strava.api_base_url}/activities/9001", headers=hdrs).json()
    detail_all = httpx.get(
        f"{mock_strava.api_base_url}/activities/9001",
        headers=hdrs,
        params={"include_all_efforts": "true"},
    ).json()
    assert len(detail_default["segment_efforts"]) == 1
    assert len(detail_all["segment_efforts"]) == 2

    streams = httpx.get(
        f"{mock_strava.api_base_url}/activities/9001/streams",
        headers=hdrs,
        params={
            "keys": (
                "time,distance,latlng,altitude,velocity_smooth,"
                "heartrate,cadence,watts,temp,moving,grade_smooth"
            ),
            "key_by_type": "true",
        },
    ).json()
    assert len(streams) == 11
    assert len(streams["time"]["data"]) == 1200

    assert (
        len(httpx.get(f"{mock_strava.api_base_url}/activities/9001/laps", headers=hdrs).json()) == 3
    )
    assert (
        len(httpx.get(f"{mock_strava.api_base_url}/activities/9001/zones", headers=hdrs).json())
        == 2
    )
    comments_p1 = httpx.get(
        f"{mock_strava.api_base_url}/activities/9001/comments",
        headers=hdrs,
        params={"page_size": 1},
    ).json()
    assert len(comments_p1) == 1
    comments_p2 = httpx.get(
        f"{mock_strava.api_base_url}/activities/9001/comments",
        headers=hdrs,
        params={"after_cursor": str(comments_p1[0]["id"])},
    ).json()
    assert len(comments_p2) == 1
    assert (
        len(httpx.get(f"{mock_strava.api_base_url}/activities/9001/kudos", headers=hdrs).json())
        == 2
    )

    gear = httpx.get(f"{mock_strava.api_base_url}/gear/b1001", headers=hdrs).json()
    assert gear["brand_name"] == "Canyon"

    gpx_resp = httpx.get(f"{mock_strava.api_base_url}/routes/7001/export_gpx", headers=hdrs)
    tcx_resp = httpx.get(f"{mock_strava.api_base_url}/routes/7001/export_tcx", headers=hdrs)
    assert gpx_resp.status_code == 200
    assert tcx_resp.status_code == 200
    assert ET.fromstring(gpx_resp.text).tag.endswith("gpx")
    assert ET.fromstring(tcx_resp.text).tag.endswith("TrainingCenterDatabase")
    route_streams = httpx.get(
        f"{mock_strava.api_base_url}/routes/7001/streams", headers=hdrs
    ).json()
    assert {"distance", "latlng", "altitude"} <= set(route_streams)

    assert len(httpx.get(f"{mock_strava.api_base_url}/segments/starred", headers=hdrs).json()) == 2
    assert httpx.get(f"{mock_strava.api_base_url}/segments/8001", headers=hdrs).json()["id"] == 8001
    assert len(httpx.get(f"{mock_strava.api_base_url}/athlete/clubs", headers=hdrs).json()) == 2
    assert (
        httpx.get(f"{mock_strava.api_base_url}/clubs/5001", headers=hdrs).json()["name"]
        == "Cascade Gravel Club"
    )


def test_conftest_mcp_imports_and_asyncio_run(mock_strava: MockStravaServer) -> None:
    assert callable(stdio_client)
    assert callable(streamable_http_client)
    assert callable(create_mcp_http_client)
    assert ClientSession is not None
    assert StdioServerParameters is not None

    async def _fetch() -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{mock_strava.api_base_url}/athlete",
                headers={"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"},
            )
            assert resp.status_code == 200
            return resp.json()

    data = asyncio.run(_fetch())
    assert data["id"] == 100001


def test_stdio_and_http_helpers_with_trivial_mock_server(
    tmp_path: Path,
    isolated_env: dict[str, str],
) -> None:
    server_script = tmp_path / "echo_mcp_server.py"
    server_script.write_text(
        textwrap.dedent(
            """\
            import sys
            from typing import Any
            import uvicorn
            from mcp.server.mcpserver import MCPServer
            from mcp.server.transport_security import TransportSecuritySettings

            mcp = MCPServer("echo-harness-test")

            @mcp.tool()
            def echo_structured(msg: str) -> dict[str, Any]:
                return {"echo": msg}

            @mcp.tool()
            def echo_text_only(msg: str) -> dict:
                return {"echo_json": msg}

            if sys.argv[1] == "stdio":
                print("echo-stdio-boot-log", file=sys.stderr, flush=True)
                mcp.run(transport="stdio")
            elif sys.argv[1] == "serve":
                host, port_str = sys.argv[3].split(":")
                app = mcp.streamable_http_app(
                    stateless_http=True,
                    json_response=True,
                    transport_security=TransportSecuritySettings(
                        enable_dns_rebinding_protection=True,
                        allowed_hosts=["127.0.0.1:*", "localhost:*"],
                    ),
                )
                expected_token = "harness-secret-token"

                async def auth_app(scope, receive, send):
                    if scope["type"] == "http":
                        hdrs = {k.lower(): v for k, v in scope.get("headers", [])}
                        if hdrs.get(b"authorization", b"").decode() != f"Bearer {expected_token}":
                            await send({
                                "type": "http.response.start",
                                "status": 401,
                                "headers": [(b"www-authenticate", b"Bearer")],
                            })
                            await send({"type": "http.response.body", "body": b"Unauthorized"})
                            return
                    await app(scope, receive, send)

                uvicorn.run(auth_app, host=host, port=int(port_str), log_level="warning")
            """
        ),
        encoding="utf-8",
    )

    cmd = [sys.executable, str(server_script)]
    with open_stdio_session([*cmd, "stdio"], isolated_env, tmp_path / "stdio.err") as sess:
        tools = [t.name for t in sess.list_tools().tools]
        assert "echo_structured" in tools
        assert "echo_text_only" in tools
        r1 = sess.call_tool("echo_structured", {"msg": "hello"})
        assert r1.structured_content == {"echo": "hello"}
        assert parse_tool_result(r1) == {"echo": "hello"}
        r2 = sess.call_tool("echo_text_only", {"msg": "world"})
        assert r2.structured_content is None
        assert parse_tool_result(r2) == {"echo_json": "world"}
    assert "echo-stdio-boot-log" in (tmp_path / "stdio.err").read_text(encoding="utf-8")

    with start_http_server(cmd, isolated_env, "harness-secret-token", tmp_path) as info:
        unauth = httpx.post(info.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert unauth.status_code == 401
        with open_mcp_http_session(info.mcp_url, info.token) as http_sess:
            res = http_sess.call_tool("echo_structured", {"msg": "streamable"})
            assert parse_tool_result(res) == {"echo": "streamable"}


def test_resolve_strava_mcp_cmd_fails_loudly_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(pytest.fail.Exception):
        resolve_strava_mcp_cmd(repo_root=tmp_path, path_env="")
    assert isinstance(subprocess.CompletedProcess, type)


def test_malformed_integer_query_params_return_400_and_record_requests(
    mock_strava: MockStravaServer,
) -> None:
    hdrs = {"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"}
    mock_strava.clear_requests()

    cases: list[tuple[str, dict[str, str], str]] = [
        (f"{mock_strava.api_base_url}/athlete/activities", {"before": "bad"}, "before"),
        (f"{mock_strava.api_base_url}/athlete/activities", {"after": "bad"}, "after"),
        (f"{mock_strava.api_base_url}/activities/9001/comments", {"page_size": "bad"}, "page_size"),
        (f"{mock_strava.api_base_url}/activities/9001/comments", {"page_size": ""}, "page_size"),
        (f"{mock_strava.api_base_url}/activities/9001/comments", {"page_size": "0"}, "page_size"),
        (f"{mock_strava.api_base_url}/activities/9001/comments", {"page_size": "-5"}, "page_size"),
        (
            f"{mock_strava.api_base_url}/activities/9001/comments",
            {"after_cursor": "bad"},
            "after_cursor",
        ),
        (f"{mock_strava.api_base_url}/segment_efforts", {"segment_id": "bad"}, "segment_id"),
        (f"{mock_strava.api_base_url}/segment_efforts", {"segment_id": "0"}, "segment_id"),
    ]
    for url, params, expected_field in cases:
        resp = httpx.get(url, headers=hdrs, params=params, timeout=3.0)
        assert resp.status_code == 400, (url, params, resp.status_code, resp.text)
        body = resp.json()
        assert isinstance(body["message"], str) and body["message"]
        assert body["errors"] and body["errors"][0]["field"] == expected_field

    recorded = mock_strava.requests
    assert len(recorded) == len(cases)
    assert all(req.status_code == 400 for req in recorded)


def test_inject_fault_times_non_positive_is_noop(mock_strava: MockStravaServer) -> None:
    hdrs = {"Authorization": f"Bearer {DEFAULT_ACCESS_TOKEN}"}
    mock_strava.inject_fault(fault_type="rate_limit_429", times=0)
    mock_strava.inject_fault(fault_type="cloudfront_403", times=-2)

    resp = httpx.get(f"{mock_strava.api_base_url}/athlete", headers=hdrs, timeout=3.0)
    assert resp.status_code == 200
