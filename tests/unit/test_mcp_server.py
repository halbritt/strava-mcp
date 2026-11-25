"""Unit and transport tests for strava_mcp.mcp_server and client/strava-mcp-headers."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import CallToolResult, TextContent

from strava_mcp.client import StravaClient
from strava_mcp.config import (
    ClientConfig,
    Config,
    ConfigError,
    TokenData,
    save_client,
    save_tokens,
    write_secret_text,
)
from strava_mcp.mcp_server import (
    build_http_app,
    create_mcp_server,
    resolve_allowed_hosts,
    validate_bind_host,
)
from strava_mcp.rate_limit import RateLimitTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADERS_SCRIPT = REPO_ROOT / "client" / "strava-mcp-headers"


def _make_config(tmp_path: Path, *, seed_secrets: bool = True) -> Config:
    cfg = Config(
        config_dir=tmp_path / "config",
        cache_dir=tmp_path / "cache",
        api_base_url="https://mock.strava.test/api/v3",
        oauth_base_url="https://mock.strava.test/oauth",
    )
    if seed_secrets:
        save_client(cfg, ClientConfig(client_id="123456", client_secret="unit_client_secret_val"))
        save_tokens(
            cfg,
            TokenData(
                access_token="unit_access_token_val",
                refresh_token="unit_refresh_token_val",
                expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
                scope="read,read_all,profile:read_all,activity:read_all",
                athlete_id=42,
            ),
        )
        write_secret_text(cfg.mcp_token_file, "unit_mcp_bearer_val\n")
    return cfg


@asynccontextmanager
async def _open_memory_session(cfg: Config, handler: Any):
    async with StravaClient(cfg, transport=httpx.MockTransport(handler)) as client:
        mcp, _state = create_mcp_server(cfg, client=client)
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            server_read, server_write = server_streams
            client_read, client_write = client_streams
            init_opts = mcp._lowlevel_server.create_initialization_options()
            server_task = asyncio.create_task(
                mcp._lowlevel_server.run(server_read, server_write, init_opts)
            )
            try:
                async with ClientSession(client_read, client_write) as session:
                    await session.initialize()
                    yield session
            finally:
                server_task.cancel()
                with suppress(asyncio.CancelledError):
                    await server_task


def _default_mock_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/v3/athlete":
        return httpx.Response(
            200,
            json={
                "id": 42,
                "username": "rider42",
                "firstname": "Ada",
                "lastname": "Lovelace",
                "weight": 62.0,
                "bikes": [{"id": "b12345", "name": "GravelRig", "distance": 2500000.0}],
                "shoes": [{"id": "g999", "name": "TrailShoe", "distance": 450000.0}],
            },
        )
    if path == "/api/v3/athletes/42/stats":
        return httpx.Response(
            200,
            json={
                "biggest_ride_distance": 120000.0,
                "recent_ride_totals": {"count": 3, "distance": 90000.0, "moving_time": 10800},
            },
        )
    if path == "/api/v3/athlete/zones":
        return httpx.Response(200, json={"heart_rate": {"custom_zones": False, "zones": []}})
    if path == "/api/v3/athlete/clubs":
        return httpx.Response(200, json=[{"id": 7, "name": "PNW Gravel"}])
    if path == "/api/v3/athlete/activities":
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 1001,
                        "name": "Gorge Gravel",
                        "sport_type": "GravelRide",
                        "type": "Ride",
                        "start_date": "2026-11-20T16:00:00Z",
                        "start_date_local": "2026-11-20T08:00:00Z",
                        "distance": 50000.0,
                        "moving_time": 7200,
                        "elapsed_time": 7500,
                        "total_elevation_gain": 800.0,
                        "average_heartrate": 152.0,
                        "private": True,
                        "visibility": "only_me",
                        "map": {"summary_polyline": "secretpoly"},
                    },
                    {
                        "id": 1002,
                        "name": "Forest Trail",
                        "sport_type": "TrailRun",
                        "type": "Run",
                        "start_date": "2026-11-21T16:00:00Z",
                        "start_date_local": "2026-11-21T08:00:00Z",
                        "distance": 12000.0,
                        "moving_time": 3600,
                        "elapsed_time": 3700,
                        "total_elevation_gain": 350.0,
                    },
                ],
            )
        return httpx.Response(200, json=[])
    if path == "/api/v3/activities/1001":
        return httpx.Response(
            200,
            json={
                "id": 1001,
                "name": "Gorge Gravel",
                "sport_type": "GravelRide",
                "type": "Ride",
                "start_date": "2026-11-20T16:00:00Z",
                "start_date_local": "2026-11-20T08:00:00Z",
                "distance": 50000.0,
                "moving_time": 7200,
                "elapsed_time": 7500,
                "total_elevation_gain": 800.0,
                "map": {"polyline": "fullpoly", "summary_polyline": "summarypoly"},
                "segment_efforts": [{"id": 55, "name": "Climb"}],
            },
        )
    if path == "/api/v3/activities/1001/laps":
        return httpx.Response(
            200,
            json=[{"id": 1, "lap_index": 1, "distance": 10000.0, "moving_time": 1440}],
        )
    if path == "/api/v3/activities/1001/zones":
        return httpx.Response(402, json={"message": "Payment Required"})
    if path == "/api/v3/activities/1001/streams":
        return httpx.Response(
            200,
            json={
                "time": {"data": list(range(500)), "original_size": 500, "resolution": "high"},
                "heartrate": {
                    "data": [140 + (i % 30) for i in range(500)],
                    "original_size": 500,
                    "resolution": "high",
                },
            },
        )
    if path == "/api/v3/activities/1002/streams":
        return httpx.Response(404, json={"message": "Record Not Found", "errors": []})
    if path == "/api/v3/gear/b12345":
        return httpx.Response(
            200, json={"id": "b12345", "name": "GravelRig", "distance": 2500000.0}
        )
    if path == "/api/v3/athletes/42/routes":
        return httpx.Response(200, json=[{"id": 88, "name": "Loop", "distance": 30000.0}])
    if path == "/api/v3/routes/88":
        return httpx.Response(200, json={"id": 88, "name": "Loop", "distance": 30000.0})
    if path == "/api/v3/routes/88/export_gpx":
        return httpx.Response(200, text="<gpx>route88</gpx>")
    if path == "/api/v3/routes/88/export_tcx":
        return httpx.Response(200, text="<TrainingCenterDatabase/>")
    if path == "/api/v3/segments/starred":
        return httpx.Response(200, json=[{"id": 300, "name": "Hill", "distance": 1500.0}])
    if path == "/api/v3/segments/300":
        return httpx.Response(200, json={"id": 300, "name": "Hill", "distance": 1500.0})
    if path == "/api/v3/segment_efforts":
        return httpx.Response(402, json={"message": "Subscription required"})
    return httpx.Response(404, json={"message": "Not Found"})


async def test_list_tools_annotations_and_output_schemas(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    async with _open_memory_session(cfg, _default_mock_handler) as session:
        res = await session.list_tools()
        assert len(res.tools) == 17
        for tool in res.tools:
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
            assert tool.annotations.idempotent_hint is True
            assert tool.annotations.open_world_hint is True
            assert tool.output_schema is not None
            assert tool.output_schema["type"] == "object"


async def test_structured_content_and_json_content_on_success(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    async with _open_memory_session(cfg, _default_mock_handler) as session:
        for tool_name in (
            "strava_get_athlete",
            "strava_get_athlete_stats",
            "strava_get_athlete_zones",
            "strava_list_clubs",
        ):
            call = await session.call_tool(tool_name, {})
            assert call.is_error is False
            assert call.structured_content is not None
            assert isinstance(call.content[0], TextContent)
            assert json.loads(call.content[0].text) == call.structured_content
            if tool_name == "strava_get_athlete_stats":
                assert "Everyone" in call.structured_content["visibility_caveat"]


async def test_auth_status_unconfigured_and_authenticated_no_secrets(tmp_path: Path) -> None:
    unconfigured = _make_config(tmp_path / "empty", seed_secrets=False)
    mcp_empty, state_empty = create_mcp_server(unconfigured)
    try:
        res_empty = await mcp_empty.call_tool("strava_auth_status", {"check_live": False})
        assert isinstance(res_empty, CallToolResult)
        payload_empty = res_empty.structured_content
        assert payload_empty is not None
        assert payload_empty["configured"] is False
        assert payload_empty["authenticated"] is False
        assert payload_empty["needs_reauth"] is True
        assert not unconfigured.cache_db_file.exists()
    finally:
        await state_empty.aclose()

    read_only = _make_config(tmp_path / "readonly", seed_secrets=True)
    save_tokens(
        read_only,
        TokenData(
            access_token="unit_access_token_val",
            refresh_token="unit_refresh_token_val",
            expires_at=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()),
            scope="read",
            athlete_id=42,
        ),
    )
    mcp_ro, state_ro = create_mcp_server(read_only)
    try:
        res_ro = await mcp_ro.call_tool("strava_auth_status", {"check_live": False})
        assert isinstance(res_ro, CallToolResult)
        payload_ro = res_ro.structured_content
        assert payload_ro is not None
        assert payload_ro["configured"] is True
        assert payload_ro["authenticated"] is True
        assert payload_ro["missing_recommended_scopes"] == ["activity:read_all", "profile:read_all"]
        assert payload_ro["needs_reauth"] is True
        dumped = json.dumps(payload_ro)
        for secret in ("unit_client_secret_val", "unit_access_token_val", "unit_refresh_token_val"):
            assert secret not in dumped
    finally:
        await state_ro.aclose()

    full_cfg = _make_config(tmp_path / "full", seed_secrets=True)
    async with _open_memory_session(full_cfg, _default_mock_handler) as session:
        res = await session.call_tool("strava_auth_status", {"check_live": True})
        assert res.is_error is False
        assert res.structured_content is not None
        assert res.structured_content["needs_reauth"] is False
        assert res.structured_content["live_check"]["ok"] is True
        dumped_full = json.dumps(res.structured_content)
        for secret in (
            "unit_client_secret_val",
            "unit_access_token_val",
            "unit_refresh_token_val",
            "unit_mcp_bearer_val",
        ):
            assert secret not in dumped_full


async def test_list_activities_filters_and_limit_autopagination(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)

    def _paged_handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(
                200,
                json=[
                    {
                        "id": i,
                        "name": f"Act {i}",
                        "sport_type": "GravelRide" if i % 2 == 0 else "TrailRun",
                        "type": "Ride" if i % 2 == 0 else "Run",
                        "start_date": "2026-11-20T12:00:00Z",
                        "start_date_local": "2026-11-20T04:00:00Z",
                        "distance": 10000.0,
                        "moving_time": 1800,
                        "map": {"summary_polyline": "poly"},
                    }
                    for i in range(1, 5)
                ],
            )
        return httpx.Response(200, json=[])

    async with _open_memory_session(cfg, _paged_handler) as session:
        res = await session.call_tool(
            "strava_list_activities",
            {"after": "2026-11-01", "before": "2026-11-25", "sport_type": "GravelRide", "limit": 1},
        )
        assert res.is_error is False
        assert res.structured_content is not None
        assert res.structured_content["count"] == 1
        assert res.structured_content["has_more"] is True
        assert res.structured_content["activities"][0]["sport_type"] == "GravelRide"
        assert "map" not in res.structured_content["activities"][0]

        bad = await session.call_tool(
            "strava_list_activities",
            {"after": "2026-11-25", "before": "2026-11-01"},
        )
        assert bad.is_error is True
        assert bad.structured_content is not None
        assert bad.structured_content["error"]["code"] == "invalid_argument"


async def test_activity_detail_laps_zones_and_streams_downsampling(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    async with _open_memory_session(cfg, _default_mock_handler) as session:
        detail = await session.call_tool("strava_get_activity", {"activity_id": 1001})
        assert detail.is_error is False
        assert detail.structured_content is not None
        assert "map" not in detail.structured_content
        assert "segment_efforts" not in detail.structured_content
        assert detail.structured_content["segment_efforts_count"] == 1

        detail_opt = await session.call_tool(
            "strava_get_activity",
            {"activity_id": 1001, "include_all_efforts": True, "include_polyline": True},
        )
        assert detail_opt.structured_content is not None
        assert detail_opt.structured_content["map"]["polyline"] == "fullpoly"
        assert len(detail_opt.structured_content["segment_efforts"]) == 1

        laps = await session.call_tool("strava_get_activity_laps", {"activity_id": 1001})
        assert laps.is_error is False
        assert laps.structured_content is not None
        assert laps.structured_content["count"] == 1

        zones = await session.call_tool("strava_get_activity_zones", {"activity_id": 1001})
        assert zones.is_error is True
        assert zones.structured_content is not None
        assert zones.structured_content["error"]["code"] == "subscription_required"

        streams = await session.call_tool(
            "strava_get_activity_streams",
            {"activity_id": 1001, "max_points": 50},
        )
        assert streams.is_error is False
        assert streams.structured_content is not None
        assert streams.structured_content["available"] is True
        assert streams.structured_content["streams"]["time"]["sampled_count"] == 50

        streams_sum = await session.call_tool(
            "strava_get_activity_streams",
            {"activity_id": 1001, "summary_only": True},
        )
        assert streams_sum.structured_content is not None
        assert "data" not in streams_sum.structured_content["streams"]["time"]

        missing_streams = await session.call_tool(
            "strava_get_activity_streams",
            {"activity_id": 1002},
        )
        assert missing_streams.is_error is False
        assert missing_streams.structured_content is not None
        assert missing_streams.structured_content["available"] is False


async def test_summarize_metrics_and_routes_and_segments(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    async with _open_memory_session(cfg, _default_mock_handler) as session:
        summary = await session.call_tool("strava_summarize_metrics", {"group_by": "sport"})
        assert summary.is_error is False
        assert summary.structured_content is not None
        assert summary.structured_content["activity_count"] == 2
        assert summary.structured_content["totals"]["private_count"] == 1

        routes = await session.call_tool("strava_list_routes", {})
        assert routes.is_error is False
        assert routes.structured_content is not None
        assert routes.structured_content["count"] == 1

        route_gpx = await session.call_tool(
            "strava_get_route", {"route_id": 88, "include_export": "gpx"}
        )
        assert route_gpx.is_error is False
        assert route_gpx.structured_content is not None
        assert route_gpx.structured_content["export_xml"] == "<gpx>route88</gpx>"

        starred = await session.call_tool("strava_list_starred_segments", {})
        assert starred.is_error is False
        assert starred.structured_content is not None
        assert starred.structured_content["count"] == 1

        seg = await session.call_tool("strava_get_segment", {"segment_id": 300})
        assert seg.is_error is False

        efforts = await session.call_tool("strava_list_segment_efforts", {"segment_id": 300})
        assert efforts.is_error is True
        assert efforts.structured_content is not None
        assert efforts.structured_content["error"]["code"] == "subscription_required"


async def test_gear_id_validation_rejects_traversal_and_special_chars(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    seen_urls: list[str] = []

    def _spy_handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return _default_mock_handler(request)

    async with _open_memory_session(cfg, _spy_handler) as session:
        for bad in (" ", "../athlete", "a/b", "b1?x=1", "b1#frag", "b1%2f", "b1\x00", "b 1"):
            res = await session.call_tool("strava_get_gear", {"gear_id": bad})
            assert res.is_error is True
            assert res.structured_content is not None
            assert res.structured_content["error"]["code"] == "invalid_argument"
        assert seen_urls == []

        ok = await session.call_tool("strava_get_gear", {"gear_id": "b12345"})
        assert ok.is_error is False
        assert ok.structured_content is not None
        assert ok.structured_content["distance_km"] == 2500.0
        assert len(seen_urls) == 1


async def test_error_mapping_and_secret_redaction(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    mode = {"case": "scope"}

    def _error_handler(request: httpx.Request) -> httpx.Response:
        if mode["case"] == "scope":
            return httpx.Response(
                403,
                json={
                    "message": "Authorization Error unit_access_token_val",
                    "errors": [{"field": "activity:read_all_permission"}],
                },
            )
        if mode["case"] == "rate_limit":
            return httpx.Response(
                429,
                headers={
                    "X-RateLimit-Limit": "400,4000",
                    "X-RateLimit-Usage": "200,500",
                    "X-ReadRateLimit-Limit": "200,2000",
                    "X-ReadRateLimit-Usage": "200,500",
                },
                json={"message": "Rate Limit Exceeded"},
            )
        if mode["case"] == "not_found":
            return httpx.Response(404, json={"message": "Not Found"})
        if mode["case"] == "auth_expired":
            return httpx.Response(401, json={"message": "Unauthorized"})
        return httpx.Response(500, json={"message": "Upstream Boom"})

    async with _open_memory_session(cfg, _error_handler) as session:
        for case, expected_code in (
            ("scope", "insufficient_scope"),
            ("not_found", "not_found"),
            ("upstream", "upstream_error"),
            ("rate_limit", "rate_limited"),
        ):
            mode["case"] = case
            res = await session.call_tool("strava_get_athlete", {"refresh": True})
            assert res.is_error is True
            assert res.structured_content is not None
            assert res.structured_content["error"]["code"] == expected_code
            assert json.loads(res.content[0].text) == res.structured_content  # type: ignore[union-attr]
            assert "unit_access_token_val" not in json.dumps(res.structured_content)


async def test_resources_and_resource_security(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    requests_made: list[str] = []

    def _spy_handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(request.url.path)
        return _default_mock_handler(request)

    async with _open_memory_session(cfg, _spy_handler) as session:
        resources = await session.list_resources()
        uris = {str(r.uri) for r in resources.resources}
        assert uris == {
            "strava://athlete/profile",
            "strava://athlete/stats",
            "strava://athlete/zones",
            "strava://status/rate-limits",
        }
        templates = await session.list_resource_templates()
        assert [t.uri_template for t in templates.resource_templates] == [
            "strava://activities/{activity_id}"
        ]

        rl_res = await session.read_resource("strava://status/rate-limits")  # type: ignore[arg-type]
        assert requests_made == []
        assert "rate_limits" in json.loads(rl_res.contents[0].text)  # type: ignore[union-attr]

        for uri in (
            "strava://athlete/profile",
            "strava://athlete/stats",
            "strava://athlete/zones",
            "strava://activities/1001",
        ):
            res = await session.read_resource(uri)  # type: ignore[arg-type]
            assert json.loads(res.contents[0].text)  # type: ignore[union-attr]

        for bad_uri in (
            "strava://activities/abc",
            "strava://activities/0",
            "strava://activities/../1",
        ):
            with pytest.raises(MCPError):
                await session.read_resource(bad_uri)  # type: ignore[arg-type]


async def test_prompts_list_and_render(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    async with _open_memory_session(cfg, _default_mock_handler) as session:
        prompts = await session.list_prompts()
        names = {p.name for p in prompts.prompts}
        assert names == {"weekly_training_review", "activity_deep_dive", "gear_wear_check"}

        review = await session.get_prompt("weekly_training_review", {"days": "14"})
        assert "14 days" in review.messages[0].content.text  # type: ignore[union-attr]
        deep = await session.get_prompt("activity_deep_dive", {"activity_id": "1001"})
        assert "1001" in deep.messages[0].content.text  # type: ignore[union-attr]
        gear = await session.get_prompt("gear_wear_check", {})
        assert "strava_get_gear" in gear.messages[0].content.text  # type: ignore[union-attr]


async def test_streamable_http_bearer_auth_and_dns_rebinding(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    async with StravaClient(cfg, transport=httpx.MockTransport(_default_mock_handler)) as client:
        app, mcp, _state = build_http_app(
            config=cfg,
            client=client,
            allowed_hosts=("mybox.ts.net",),
        )
        headers_base = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        rpc_list = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

        async with (
            mcp.session_manager.run(),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8772",
            ) as http,
        ):
            unauth = await http.post("/mcp", json=rpc_list, headers=headers_base)
            assert unauth.status_code == 401
            assert unauth.headers.get("www-authenticate") == "Bearer"

            wrong = await http.post(
                "/mcp",
                json=rpc_list,
                headers={**headers_base, "Authorization": "Bearer wrong-token"},
            )
            assert wrong.status_code == 401

            evil_host = await http.post(
                "/mcp",
                json=rpc_list,
                headers={
                    **headers_base,
                    "Authorization": "Bearer unit_mcp_bearer_val",
                    "Host": "evil.example",
                },
            )
            assert evil_host.status_code == 421

            evil_origin = await http.post(
                "/mcp",
                json=rpc_list,
                headers={
                    **headers_base,
                    "Authorization": "Bearer unit_mcp_bearer_val",
                    "Origin": "https://evil.example",
                },
            )
            assert evil_origin.status_code == 403

            ok_resp = await http.post(
                "/mcp",
                json=rpc_list,
                headers={
                    **headers_base,
                    "Authorization": "Bearer unit_mcp_bearer_val",
                },
            )
            assert ok_resp.status_code == 200
            assert len(ok_resp.json()["result"]["tools"]) == 17

            ts_resp = await http.post(
                "/mcp",
                json=rpc_list,
                headers={
                    **headers_base,
                    "Authorization": "Bearer unit_mcp_bearer_val",
                    "Host": "mybox.ts.net",
                },
            )
            assert ts_resp.status_code == 200


def test_loopback_bind_enforcement() -> None:
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
    assert validate_bind_host("localhost") == "localhost"
    assert validate_bind_host("[::1]") == "::1"
    with pytest.raises(ConfigError) as exc_info:
        validate_bind_host("0.0.0.0")
    assert exc_info.value.code == "invalid_config"
    assert validate_bind_host("0.0.0.0", allow_non_loopback=True) == "0.0.0.0"

    allowed = resolve_allowed_hosts("127.0.0.1", 8772, ("mybox.ts.net",))
    assert "127.0.0.1:8772" in allowed
    assert "mybox.ts.net" in allowed
    assert "mybox.ts.net:8772" in allowed


def test_stdio_subprocess_stdout_cleanliness(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    env = os.environ.copy()
    env["STRAVA_MCP_CONFIG_DIR"] = str(cfg.config_dir)
    env["STRAVA_MCP_CACHE_DIR"] = str(cfg.cache_dir)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    init_msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        }
    )
    initialized_notif = json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    )
    list_msg = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    proc = subprocess.Popen(
        [sys.executable, "-m", "strava_mcp.cli", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(init_msg + "\n")
        proc.stdin.flush()
        line1 = proc.stdout.readline()
        assert line1
        msg1 = json.loads(line1)
        assert msg1.get("id") == 1 and "result" in msg1

        proc.stdin.write(initialized_notif + "\n" + list_msg + "\n")
        proc.stdin.flush()
        line2 = proc.stdout.readline()
        assert line2
        msg2 = json.loads(line2)
        assert msg2.get("id") == 2 and len(msg2["result"]["tools"]) == 17

        proc.stdin.close()
        remaining_out = proc.stdout.read()
        for extra in remaining_out.splitlines():
            if extra.strip():
                json.loads(extra)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_headers_helper_script(tmp_path: Path) -> None:
    assert os.access(HEADERS_SCRIPT, os.X_OK)
    cfg = _make_config(tmp_path)
    env = os.environ.copy()
    env["STRAVA_MCP_CONFIG_DIR"] = str(cfg.config_dir)

    ok = subprocess.run(
        [str(HEADERS_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ok.returncode == 0
    assert json.loads(ok.stdout) == {"Authorization": "Bearer unit_mcp_bearer_val"}

    # Insecure file permissions (0644) must fail without leaking token
    os.chmod(cfg.mcp_token_file, 0o644)
    bad_file_perm = subprocess.run(
        [str(HEADERS_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad_file_perm.returncode == 1
    assert bad_file_perm.stdout == ""
    assert "insecure permissions (expected 0600)" in bad_file_perm.stderr
    assert "unit_mcp_bearer_val" not in bad_file_perm.stderr
    os.chmod(cfg.mcp_token_file, 0o600)

    # Insecure directory permissions (0755) must fail without leaking token
    os.chmod(cfg.config_dir, 0o755)
    bad_dir_perm = subprocess.run(
        [str(HEADERS_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad_dir_perm.returncode == 1
    assert bad_dir_perm.stdout == ""
    assert "insecure permissions" in bad_dir_perm.stderr
    assert "unit_mcp_bearer_val" not in bad_dir_perm.stderr
    os.chmod(cfg.config_dir, 0o700)

    # Symlink token file must fail
    sym_dir = tmp_path / "sym_cfg"
    sym_dir.mkdir(mode=0o700)
    (sym_dir / "mcp-token").symlink_to(cfg.mcp_token_file)
    env_sym = dict(env)
    env_sym["STRAVA_MCP_CONFIG_DIR"] = str(sym_dir)
    sym_res = subprocess.run(
        [str(HEADERS_SCRIPT)],
        env=env_sym,
        text=True,
        capture_output=True,
        check=False,
    )
    assert sym_res.returncode == 1
    assert sym_res.stdout == ""
    assert "unit_mcp_bearer_val" not in sym_res.stderr

    empty_dir = tmp_path / "missing_cfg"
    empty_dir.mkdir()
    env["STRAVA_MCP_CONFIG_DIR"] = str(empty_dir)
    missing = subprocess.run(
        [str(HEADERS_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0


def test_rate_limit_retry_after_overflow_and_malformed_date() -> None:
    now = datetime(2026, 11, 25, 4, 12, 0, tzinfo=UTC)
    tracker = RateLimitTracker(clock=lambda: now)

    huge_resp = httpx.Response(429, headers={"Retry-After": "999999999999999"})
    err_huge = tracker.error_from_429(huge_resp, is_read=True)
    assert err_huge.retry_after_seconds == 999999999999999
    assert err_huge.reset_at_utc == "9999-12-31T23:59:59Z"

    malformed_date_resp = httpx.Response(429, headers={"Retry-After": "Wed, 99 Xxx"})
    err_bad = tracker.error_from_429(malformed_date_resp, is_read=True)
    assert err_bad.reset_at_utc == "2026-11-25T04:15:00Z"


async def test_mcp_rejects_out_of_range_date_and_boolean_int_coercion(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    requests_made: list[str] = []

    def _spy_handler(request: httpx.Request) -> httpx.Response:
        requests_made.append(str(request.url))
        return _default_mock_handler(request)

    async with _open_memory_session(cfg, _spy_handler) as session:
        # Verify JSON schema constraints remain intact with BeforeValidator
        tools_res = await session.list_tools()
        tools_by_name = {t.name: t for t in tools_res.tools}
        act_props = tools_by_name["strava_get_activity"].input_schema["properties"]
        assert act_props["activity_id"]["minimum"] == 1
        assert act_props["activity_id"]["type"] == "integer"

        # Defect 2: out-of-range epoch string in strava_list_segment_efforts -> invalid_argument
        bad_epoch = await session.call_tool(
            "strava_list_segment_efforts",
            {"segment_id": 801, "start_date_local": "100000000000000000000"},
        )
        assert bad_epoch.is_error is True
        assert bad_epoch.structured_content is not None
        assert bad_epoch.structured_content["error"]["code"] == "invalid_argument"

        # Defect 4: bool -> int coercion rejected across all integer/date-filter tool params
        bool_cases: list[tuple[str, dict[str, Any]]] = [
            ("strava_get_athlete_stats", {"athlete_id": True}),
            ("strava_list_activities", {"after": True}),
            ("strava_list_activities", {"before": False}),
            ("strava_list_activities", {"page": True}),
            ("strava_list_activities", {"per_page": True}),
            ("strava_list_activities", {"limit": True}),
            ("strava_get_activity", {"activity_id": True}),
            ("strava_get_activity_laps", {"activity_id": False}),
            ("strava_get_activity_zones", {"activity_id": True}),
            ("strava_get_activity_streams", {"activity_id": True}),
            ("strava_get_activity_streams", {"activity_id": 1, "max_points": True}),
            ("strava_get_activity_streams", {"activity_id": 1, "max_points": False}),
            ("strava_summarize_metrics", {"after": True}),
            ("strava_summarize_metrics", {"before": False}),
            ("strava_list_routes", {"athlete_id": True}),
            ("strava_list_routes", {"page": True}),
            ("strava_list_routes", {"per_page": True}),
            ("strava_list_routes", {"limit": True}),
            ("strava_get_route", {"route_id": True}),
            ("strava_list_starred_segments", {"page": True}),
            ("strava_list_starred_segments", {"per_page": True}),
            ("strava_list_starred_segments", {"limit": True}),
            ("strava_get_segment", {"segment_id": True}),
            ("strava_list_segment_efforts", {"segment_id": True}),
            ("strava_list_segment_efforts", {"segment_id": 1, "start_date_local": True}),
            ("strava_list_segment_efforts", {"segment_id": 1, "end_date_local": False}),
            ("strava_list_segment_efforts", {"segment_id": 1, "per_page": True}),
            ("strava_list_segment_efforts", {"segment_id": 1, "limit": True}),
            ("strava_list_clubs", {"page": True}),
            ("strava_list_clubs", {"per_page": True}),
            ("strava_list_clubs", {"limit": True}),
        ]
        for tool_name, args in bool_cases:
            res = await session.call_tool(tool_name, args)
            assert res.is_error is True, f"{tool_name}({args}) unexpectedly succeeded"

        for bad_date_args in (
            {"after": "9999-12-31T23:59:59-12:00"},
            {"before": "0001-01-01T00:00:00+12:00"},
        ):
            res_dt = await session.call_tool("strava_list_activities", bad_date_args)
            assert res_dt.is_error is True
            assert res_dt.structured_content is not None
            assert res_dt.structured_content["error"]["code"] == "invalid_argument"

        assert requests_made == []


async def test_mcp_auth_status_handles_out_of_range_expires_at(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    save_tokens(
        cfg,
        TokenData(
            access_token="unit_extreme_access_val",
            refresh_token="unit_extreme_refresh_val",
            expires_at=99999999999999999999,
            scope="read,activity:read_all,profile:read_all",
            athlete_id=42,
        ),
    )
    async with _open_memory_session(cfg, _default_mock_handler) as session:
        res = await session.call_tool("strava_auth_status", {"check_live": False})
        assert res.is_error is False
        assert res.structured_content is not None
        assert res.structured_content["authenticated"] is True
        assert res.structured_content["expires_at"] == 99999999999999999999
        assert res.structured_content["expires_at_utc"] is None
        assert res.structured_content["is_expired"] is False


async def test_mcp_resource_and_tool_surface_sqlite_error_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(tmp_path)
    cfg.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    cfg.cache_db_file.write_bytes(b"")
    os.chmod(cfg.cache_db_file, 0o600)

    def _raise_sqlite(*_args: Any, **_kwargs: Any) -> Any:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("strava_mcp.mcp_server.ResponseCache", _raise_sqlite)
    mcp, _state = create_mcp_server(cfg)

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        server_read, server_write = server_streams
        client_read, client_write = client_streams
        init_opts = mcp._lowlevel_server.create_initialization_options()
        server_task = asyncio.create_task(
            mcp._lowlevel_server.run(server_read, server_write, init_opts)
        )
        try:
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                with pytest.raises(MCPError) as exc_info:
                    await session.read_resource("strava://status/rate-limits")
                err_msg = str(exc_info.value)
                assert "cache_error" in err_msg
                assert "Traceback" not in err_msg
        finally:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task

    async with _open_memory_session(cfg, _default_mock_handler) as session:
        async def _broken_get_athlete(*_args: Any, **_kwargs: Any) -> Any:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(StravaClient, "get_athlete", _broken_get_athlete)
        tool_res = await session.call_tool("strava_get_athlete", {})
        assert tool_res.is_error is True
        assert tool_res.structured_content is not None
        assert tool_res.structured_content["error"]["code"] == "cache_error"
        assert "database is locked" in tool_res.structured_content["error"]["message"]


async def test_segment_efforts_limit_200_has_more_and_resource_sanitization_and_dangling_cache(
    tmp_path: Path,
) -> None:
    cfg = _make_config(tmp_path)

    def _efforts_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/segment_efforts"):
            return httpx.Response(
                200,
                json=[{"id": i, "elapsed_time": 100 + i} for i in range(1, 201)],
            )
        if request.url.path.endswith("/athlete"):
            raise RuntimeError("boom unit_client_secret_val")
        return _default_mock_handler(request)

    async with _open_memory_session(cfg, _efforts_handler) as session:
        res = await session.call_tool(
            "strava_list_segment_efforts",
            {"segment_id": 8001, "limit": 200},
        )
        assert res.is_error is False
        assert res.structured_content is not None
        assert res.structured_content["count"] == 200
        assert res.structured_content["has_more"] is True

        with pytest.raises(MCPError) as exc_res:
            await session.read_resource("strava://athlete/profile")
        assert "Unexpected internal error." in str(exc_res.value)
        assert "unit_client_secret_val" not in str(exc_res.value)

    # Dangling symlink at cache.sqlite must fail in auth_status and rate_limit_status_resource
    cfg_sym = _make_config(tmp_path / "dangling_cache")
    cfg_sym.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    (cfg_sym.cache_db_file).symlink_to(tmp_path / "does_not_exist.sqlite")
    mcp_sym, _state = create_mcp_server(cfg_sym)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        server_read, server_write = server_streams
        client_read, client_write = client_streams
        init_opts = mcp_sym._lowlevel_server.create_initialization_options()
        server_task = asyncio.create_task(
            mcp_sym._lowlevel_server.run(server_read, server_write, init_opts)
        )
        try:
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                auth_res = await session.call_tool("strava_auth_status", {})
                assert auth_res.is_error is True
                assert auth_res.structured_content is not None
                assert auth_res.structured_content["error"]["code"] == "insecure_permissions"

                with pytest.raises(MCPError) as exc_rl:
                    await session.read_resource("strava://status/rate-limits")
                assert "insecure_permissions" in str(exc_rl.value)
        finally:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task


def test_headers_helper_trims_whitespace_and_rejects_bad_tokens(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, seed_secrets=False)
    cfg.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRAVA_")}
    env["STRAVA_MCP_CONFIG_DIR"] = str(cfg.config_dir)

    # Valid token with surrounding ASCII whitespace trims cleanly
    cfg.mcp_token_file.write_text("  valid_bearer-tok_123 \t\r\n", encoding="utf-8")
    os.chmod(cfg.mcp_token_file, 0o600)
    ok_res = subprocess.run(
        [str(HEADERS_SCRIPT)], env=env, text=True, capture_output=True, check=False
    )
    assert ok_res.returncode == 0
    assert json.loads(ok_res.stdout) == {"Authorization": "Bearer valid_bearer-tok_123"}

    bad_tokens: list[bytes] = [
        b"   \n",
        b"\t\r\n",
        b"tok1\ntok2\n",
        b"tok with space\n",
        b'tok"with\\chars\n',
        b"tok\x00nul\n",
    ]
    for raw_bytes in bad_tokens:
        cfg.mcp_token_file.write_bytes(raw_bytes)
        os.chmod(cfg.mcp_token_file, 0o600)
        bad_res = subprocess.run(
            [str(HEADERS_SCRIPT)], env=env, capture_output=True, check=False
        )
        assert bad_res.returncode != 0, (
            f"Expected failure for {raw_bytes!r}, got {bad_res.stdout!r}"
        )
        assert bad_res.stdout == b""
