"""Unit tests for strava_mcp.client."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from strava_mcp import __version__
from strava_mcp.auth import TokenManager
from strava_mcp.cache import ResponseCache, make_cache_key
from strava_mcp.client import StravaClient
from strava_mcp.config import (
    AuthError,
    ClientConfig,
    Config,
    InsufficientScopeError,
    NotFoundError,
    RateLimitError,
    SubscriptionRequiredError,
    TierRestrictedError,
    TokenData,
    UpstreamAPIError,
    save_client,
    save_tokens,
)
from strava_mcp.rate_limit import RateLimitTracker


def _make_config(
    tmp_path: Path,
    *,
    scope: str = "read,read_all,profile:read_all,activity:read_all",
    athlete_id: int | None = 4242,
    access_token: str = "test-access-token",
    expires_at: int = 9_999_999_999,
) -> Config:
    cfg = Config(
        config_dir=tmp_path / "cfg",
        cache_dir=tmp_path / "cache",
        api_base_url="https://mock.strava.test/api/v3",
        oauth_base_url="https://mock.strava.test/oauth",
    )
    save_client(cfg, ClientConfig(client_id="293001", client_secret="mock-client-secret"))
    save_tokens(
        cfg,
        TokenData(
            access_token=access_token,
            refresh_token="mock-refresh-token",
            expires_at=expires_at,
            expires_in=21600,
            scope=scope,
            athlete_id=athlete_id,
        ),
    )
    return cfg


def _rate_headers(u15: int = 1, ud: int = 10) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": "400,4000",
        "X-RateLimit-Usage": f"{u15},{ud}",
        "X-ReadRateLimit-Limit": "200,2000",
        "X-ReadRateLimit-Usage": f"{u15},{ud}",
    }


async def test_client_lifecycle_headers_and_owned_vs_injected_resources(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    seen_headers: list[dict[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append({k.lower(): v for k, v in request.headers.items()})
        return httpx.Response(200, headers=_rate_headers(), json={"id": 4242, "firstname": "Ada"})

    # Owned http_client + owned cache
    async with StravaClient(cfg, transport=httpx.MockTransport(handler)) as client:
        athlete = await client.get_athlete()
        assert athlete["id"] == 4242
        assert seen_headers[0]["authorization"] == "Bearer test-access-token"
        assert f"strava-mcp/{__version__}" in seen_headers[0]["user-agent"]

    assert client._http_client is None
    assert client.cache._closed is True

    # Caller-injected http_client and cache must remain open after client.aclose()
    injected_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    injected_cache = ResponseCache(cfg)
    try:
        async with StravaClient(
            cfg,
            http_client=injected_http,
            cache=injected_cache,
        ) as client2:
            res = await client2.get_athlete()
            assert res["id"] == 4242
        assert not injected_http.is_closed
        assert injected_cache._closed is False
    finally:
        await injected_http.aclose()
        injected_cache.close()


async def test_all_22_endpoints_contracts_and_athlete_id_resolution(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, athlete_id=None)
    recorded: list[tuple[str, dict[str, str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v3")
        params = dict(request.url.params)
        recorded.append((path, params))
        headers = _rate_headers(len(recorded), len(recorded))

        if path == "/routes/55/export_gpx":
            return httpx.Response(200, headers=headers, text="<gpx>route55</gpx>")
        if path == "/routes/55/export_tcx":
            return httpx.Response(200, headers=headers, text="<tcx>route55</tcx>")
        if path == "/athlete":
            return httpx.Response(200, headers=headers, json={"id": 7788, "username": "runner"})
        if path in {
            "/athlete/activities",
            "/activities/101/laps",
            "/activities/101/zones",
            "/activities/101/comments",
            "/activities/101/kudos",
            "/athletes/7788/routes",
            "/segments/starred",
            "/segment_efforts",
            "/athlete/clubs",
        }:
            return httpx.Response(200, headers=headers, json=[{"path": path}])
        return httpx.Response(200, headers=headers, json={"path": path, "params": params})

    async with StravaClient(cfg, transport=httpx.MockTransport(handler)) as client:
        assert (await client.get_athlete())["id"] == 7788
        assert (await client.get_athlete_stats())["path"] == "/athletes/7788/stats"
        assert (await client.get_athlete_zones())["path"] == "/athlete/zones"
        assert await client.list_activities(before=200, after=100, page=2, per_page=50) == [
            {"path": "/athlete/activities"}
        ]
        assert (await client.get_activity(101, include_all_efforts=True))["path"] == (
            "/activities/101"
        )
        assert await client.get_activity_laps(101) == [{"path": "/activities/101/laps"}]
        assert await client.get_activity_zones(101) == [{"path": "/activities/101/zones"}]
        assert await client.get_activity_comments(101, page_size=15, after_cursor="cur-9") == [
            {"path": "/activities/101/comments"}
        ]
        assert await client.get_activity_kudos(101, page=2, per_page=10) == [
            {"path": "/activities/101/kudos"}
        ]
        assert (
            await client.get_activity_streams(101, keys=["watts", "time", "watts"])
        )["path"] == "/activities/101/streams"
        assert (await client.get_gear("b12345"))["path"] == "/gear/b12345"
        assert await client.list_routes(page=2, per_page=20) == [{"path": "/athletes/7788/routes"}]
        assert (await client.get_route(55))["path"] == "/routes/55"
        assert await client.export_route_gpx(55) == "<gpx>route55</gpx>"
        assert await client.export_route_tcx(55) == "<tcx>route55</tcx>"
        route_streams = await client.get_route_streams(55)
        assert isinstance(route_streams, dict)
        assert route_streams["path"] == "/routes/55/streams"
        assert await client.list_starred_segments(page=1, per_page=25) == [
            {"path": "/segments/starred"}
        ]
        assert (await client.get_segment(909))["path"] == "/segments/909"
        assert (
            await client.get_segment_streams(909, keys=["altitude", "distance"])
        )["path"] == "/segments/909/streams"
        assert await client.list_segment_efforts(
            909,
            start_date_local="2026-01-01T00:00:00Z",
            end_date_local="2026-02-01T00:00:00Z",
            per_page=12,
        ) == [{"path": "/segment_efforts"}]
        assert await client.list_clubs(page=1, per_page=15) == [{"path": "/athlete/clubs"}]
        assert (await client.get_club(333))["path"] == "/clubs/333"

    by_path = dict(recorded)
    assert by_path["/athlete/activities"] == {
        "before": "200",
        "after": "100",
        "page": "2",
        "per_page": "50",
    }
    assert by_path["/activities/101"] == {"include_all_efforts": "true"}
    assert by_path["/activities/101/comments"] == {"page_size": "15", "after_cursor": "cur-9"}
    assert "page" not in by_path["/activities/101/comments"]
    assert by_path["/activities/101/kudos"] == {"page": "2", "per_page": "10"}
    assert by_path["/activities/101/streams"] == {"keys": "time,watts", "key_by_type": "true"}
    assert by_path["/athletes/7788/routes"] == {"page": "2", "per_page": "20"}
    assert by_path["/routes/55/streams"] == {}
    assert by_path["/segments/909/streams"] == {"keys": "altitude,distance", "key_by_type": "true"}
    assert by_path["/segment_efforts"] == {
        "segment_id": "909",
        "start_date_local": "2026-01-01T00:00:00Z",
        "end_date_local": "2026-02-01T00:00:00Z",
        "per_page": "12",
    }
    assert "page" not in by_path["/segment_efforts"]

    # Input validation checks
    async with StravaClient(cfg, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await client.get_activity_streams(101, keys="invalid_key")
        with pytest.raises(ValueError):
            await client.get_activity_streams(101, keys=[])
        with pytest.raises(ValueError):
            await client.get_segment_streams(909, keys="heartrate")
        with pytest.raises(ValueError):
            await client.get_gear("   ")


async def test_401_refresh_and_retry_once_and_concurrent_dedup(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, access_token="token-old")
    refresh_calls = 0
    api_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_calls, api_calls
        if request.url.path.endswith("/oauth/token"):
            refresh_calls += 1
            form = parse_qs(request.content.decode("utf-8"))
            assert form["grant_type"] == ["refresh_token"]
            await asyncio.sleep(0.03)
            return httpx.Response(
                200,
                json={
                    "access_token": "token-new",
                    "refresh_token": "refresh-new",
                    "expires_at": 9_999_999_999,
                    "expires_in": 21600,
                },
            )

        api_calls += 1
        auth = request.headers.get("authorization")
        if auth == "Bearer token-old":
            await asyncio.sleep(0.02)
            return httpx.Response(
                401,
                headers=_rate_headers(),
                json={"message": "Authorization Error", "errors": []},
            )
        if auth == "Bearer token-new":
            return httpx.Response(
                200,
                headers=_rate_headers(),
                json={"path": request.url.path},
            )
        return httpx.Response(401, json={"message": "Unexpected token"})

    transport = httpx.MockTransport(handler)
    tm = TokenManager(cfg, transport=transport)
    async with StravaClient(cfg, token_manager=tm, transport=transport) as client:
        r1, r2, r3 = await asyncio.gather(
            client.get_athlete(),
            client.get_athlete_zones(),
            client.get_club(77),
        )
        assert r1["path"].endswith("/athlete")
        assert r2["path"].endswith("/athlete/zones")
        assert r3["path"].endswith("/clubs/77")
        assert refresh_calls == 1
        assert api_calls == 6  # 3 initial 401s + 3 retried 200s

    # Persistent 401 after retry raises AuthError without infinite loop
    cfg_fail = _make_config(tmp_path / "fail", access_token="bad-token")
    persistent_api_calls = 0

    async def persistent_401(request: httpx.Request) -> httpx.Response:
        nonlocal persistent_api_calls
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(
                200,
                json={
                    "access_token": "still-bad",
                    "refresh_token": "r2",
                    "expires_at": 9_999_999_999,
                },
            )
        persistent_api_calls += 1
        return httpx.Response(401, json={"message": "Revoked", "errors": []})

    async with StravaClient(cfg_fail, transport=httpx.MockTransport(persistent_401)) as client:
        with pytest.raises(AuthError) as exc_auth:
            await client.get_athlete()
        assert exc_auth.value.code == "auth_expired"
        assert persistent_api_calls == 2


async def test_rate_limit_preemptive_429_and_cache_refresh(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    now = datetime(2026, 11, 25, 4, 10, 0, tzinfo=UTC)
    tracker = RateLimitTracker(clock=lambda: now)
    http_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        if request.url.path.endswith("/routes/12/export_gpx"):
            return httpx.Response(200, headers=_rate_headers(10, 100), text="<gpx>v1</gpx>")
        if request.url.path.endswith("/activities/88/streams"):
            return httpx.Response(
                200,
                headers=_rate_headers(11, 101),
                json={"time": {"data": [0, 1, 2]}},
            )
        return httpx.Response(
            200,
            headers=_rate_headers(12, 102),
            json=[{"id": http_calls, "name": f"Run {http_calls}"}],
        )

    async with StravaClient(
        cfg,
        rate_limiter=tracker,
        transport=httpx.MockTransport(handler),
    ) as client:
        acts1 = await client.list_activities()
        acts2 = await client.list_activities()
        assert acts1 == acts2
        assert http_calls == 1

        # refresh=True bypasses cache and updates stored value
        acts3 = await client.list_activities(refresh=True)
        assert acts3[0]["id"] == 2
        assert http_calls == 2
        assert await client.list_activities() == acts3
        assert http_calls == 2

        # Immutable GPX export and activity streams use ttl_seconds=None
        assert await client.export_route_gpx(12) == "<gpx>v1</gpx>"
        assert await client.export_route_gpx(12) == "<gpx>v1</gpx>"
        assert await client.get_activity_streams(88, keys="time") == {"time": {"data": [0, 1, 2]}}
        assert http_calls == 4

        gpx_key = make_cache_key("/routes/12/export_gpx")
        gpx_entry = client.cache.get_entry(gpx_key)
        assert gpx_entry is not None
        assert gpx_entry.expires_at is None

        list_key = make_cache_key("/athlete/activities", {"page": 1, "per_page": 30})
        list_entry = client.cache.get_entry(list_key)
        assert list_entry is not None
        assert list_entry.expires_at is not None

        # Preemptive block when tracker is exhausted makes 0 HTTP calls
        tracker.update_from_headers(_rate_headers(200, 500), now=now)
        with pytest.raises(RateLimitError) as exc_pre:
            await client.get_athlete(refresh=True)
        assert exc_pre.value.limit_type == "read_15m"
        assert http_calls == 4

    # Upstream 429 raises RateLimitError immediately without retry loop
    cfg_429 = _make_config(tmp_path / "r429")
    calls_429 = 0

    async def handler_429(_request: httpx.Request) -> httpx.Response:
        nonlocal calls_429
        calls_429 += 1
        return httpx.Response(
            429,
            headers=_rate_headers(201, 500),
            json={"message": "Rate Limit Exceeded", "errors": []},
        )

    async with StravaClient(
        cfg_429,
        rate_limiter=RateLimitTracker(clock=lambda: now),
        transport=httpx.MockTransport(handler_429),
    ) as client_429:
        with pytest.raises(RateLimitError) as exc_429:
            await client_429.get_athlete()
        assert exc_429.value.limit_type == "read_15m"
        assert exc_429.value.reset_at_utc == "2026-11-25T04:15:00Z"
        assert calls_429 == 1


async def test_proactive_scope_and_upstream_error_mapping(tmp_path: Path) -> None:
    # 1. Known read-only token blocks locally before HTTP
    cfg_read = _make_config(tmp_path / "read_only", scope="read")
    http_calls = 0

    async def fail_if_called(_request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(200, json={})

    async with StravaClient(cfg_read, transport=httpx.MockTransport(fail_if_called)) as client:
        with pytest.raises(InsufficientScopeError) as exc_list:
            await client.list_activities()
        assert exc_list.value.code == "insufficient_scope"
        assert exc_list.value.required_scopes == ["activity:read"]
        assert exc_list.value.granted_scopes == ["read"]
        assert exc_list.value.remediation == "strava-mcp auth"

        with pytest.raises(InsufficientScopeError) as exc_zones:
            await client.get_athlete_zones()
        assert exc_zones.value.required_scopes == ["profile:read_all"]
        assert http_calls == 0

    # 2. Empty scope skips local check and maps 401 permission fault directly (no refresh)
    cfg_empty_scope = _make_config(tmp_path / "empty_scope", scope="")
    refresh_attempts = 0

    async def scope_401_handler(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_attempts
        if request.url.path.endswith("/oauth/token"):
            refresh_attempts += 1
            return httpx.Response(200, json={})
        return httpx.Response(
            401,
            headers=_rate_headers(),
            json={
                "message": "Authorization Error",
                "errors": [
                    {
                        "resource": "Athlete",
                        "field": "activity:read_permission",
                        "code": "invalid",
                    }
                ],
            },
        )

    async with StravaClient(
        cfg_empty_scope,
        transport=httpx.MockTransport(scope_401_handler),
    ) as client_empty:
        with pytest.raises(InsufficientScopeError) as exc_fault_scope:
            await client_empty.list_activities()
        assert exc_fault_scope.value.required_scopes == ["activity:read"]
        assert exc_fault_scope.value.remediation == "strava-mcp auth"
        assert refresh_attempts == 0

    # 2b. 401 permission faults encoding scope in err["code"] ("missing:<scope>")
    for missing_code, expected_scope in (
        ("missing:activity:read_all", "activity:read_all"),
        ("missing:read_all", "read_all"),
    ):
        cfg_code_scope = _make_config(tmp_path / f"code_{expected_scope}", scope="")

        async def _code_401_handler(
            _request: httpx.Request,
            code_val: str = missing_code,
        ) -> httpx.Response:
            return httpx.Response(
                401,
                headers=_rate_headers(),
                json={
                    "message": "Authorization Error",
                    "errors": [
                        {
                            "resource": "Athlete",
                            "field": "permission",
                            "code": code_val,
                        }
                    ],
                },
            )

        async with StravaClient(
            cfg_code_scope,
            transport=httpx.MockTransport(_code_401_handler),
        ) as client_code:
            with pytest.raises(InsufficientScopeError) as exc_code_scope:
                await client_code.get_route(7002)
            assert exc_code_scope.value.required_scopes == [expected_scope]
            assert exc_code_scope.value.remediation == "strava-mcp auth"

    # 3. 402, 403 (CloudFront HTML / subscription / tier),
    # 404 (standard vs manual streams), 5xx, ConnectError
    cfg_err = _make_config(tmp_path / "errors", access_token="SUPER_SECRET_TOKEN_XYZ")

    async def error_router(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v3")
        if path == "/activities/1/zones":
            return httpx.Response(402, json={"message": "Payment Required", "errors": []})
        if path == "/segment_efforts":
            return httpx.Response(403, json={"message": "Forbidden", "errors": []})
        if path == "/clubs/1":
            return httpx.Response(
                403,
                headers={"content-type": "text/html"},
                text="<html><title>403 Forbidden</title><body>CloudFront</body></html>",
            )
        if path == "/segments/1":
            return httpx.Response(
                403,
                json={"message": "Restricted by Extended Access Tier", "errors": []},
            )
        if path == "/activities/2":
            return httpx.Response(
                404,
                json={
                    "message": "Record Not Found",
                    "errors": [{"resource": "Activity", "field": "id", "code": "invalid"}],
                },
            )
        if path == "/activities/300/streams":
            return httpx.Response(
                404,
                json={
                    "message": "Record Not Found",
                    "errors": [{"resource": "StreamSet", "field": "", "code": "not found"}],
                },
            )
        if path == "/activities/404/streams":
            return httpx.Response(
                404,
                json={
                    "message": "Record Not Found",
                    "errors": [{"resource": "Activity", "field": "id", "code": "invalid"}],
                },
            )
        if path == "/gear/b999":
            return httpx.Response(502, text="Bad Gateway")
        raise httpx.ConnectError("network down", request=request)

    async with StravaClient(cfg_err, transport=httpx.MockTransport(error_router)) as client_err:
        with pytest.raises(SubscriptionRequiredError):
            await client_err.get_activity_zones(1)
        with pytest.raises(SubscriptionRequiredError):
            await client_err.list_segment_efforts(10)
        with pytest.raises(UpstreamAPIError) as exc_cf:
            await client_err.get_club(1)
        assert exc_cf.value.status_code == 403
        assert "CloudFront" in exc_cf.value.details["body_snippet"]
        with pytest.raises(TierRestrictedError):
            await client_err.get_segment(1)
        with pytest.raises(NotFoundError):
            await client_err.get_activity(2)

        # Manual activity streams 404 returns {} when missing_ok=True
        assert await client_err.get_activity_streams(300, keys="time") == {}
        with pytest.raises(NotFoundError):
            await client_err.get_activity_streams(300, keys="time", missing_ok=False)
        # Missing Activity 404 still raises NotFoundError even with missing_ok=True
        with pytest.raises(NotFoundError):
            await client_err.get_activity_streams(404, keys="time")

        with pytest.raises(UpstreamAPIError) as exc_502:
            await client_err.get_gear("b999")
        assert exc_502.value.status_code == 502

        with pytest.raises(UpstreamAPIError) as exc_conn:
            await client_err.get_route(999)
        assert exc_conn.value.status_code == 502
        assert "SUPER_SECRET_TOKEN_XYZ" not in json.dumps(exc_conn.value.to_dict())


async def test_client_scope_case_normalization_and_get_gear_traversal_guard(
    tmp_path: Path,
) -> None:
    cfg = _make_config(tmp_path, scope="")
    seen_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            401,
            headers=_rate_headers(),
            json={
                "message": "Authorization Error",
                "errors": [
                    {
                        "resource": "Athlete",
                        "field": "Activity:Read_All_Permission",
                        "code": "missing:activity:read_all",
                    }
                ],
            },
        )

    async with StravaClient(cfg, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(InsufficientScopeError) as exc_scope:
            await client.list_activities()
        assert exc_scope.value.required_scopes == ["activity:read_all"]

        for bad_gear in ("../athlete", "b1/extra", "b1?x=1", "b1#frag", "b 1", ""):
            with pytest.raises(ValueError):
                await client.get_gear(bad_gear)

    assert len(seen_urls) == 1
    assert "/athlete/activities" in seen_urls[0]
