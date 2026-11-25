"""Async Strava API v3 client with caching, rate-limit tracking, and 401 refresh retry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import httpx

from strava_mcp.auth import (
    DEFAULT_USER_AGENT,
    TokenManager,
    parse_scopes,
    require_scope,
)
from strava_mcp.cache import ResponseCache, make_cache_key
from strava_mcp.config import (
    AuthError,
    Config,
    InsufficientScopeError,
    NotFoundError,
    SubscriptionRequiredError,
    TierRestrictedError,
    UpstreamAPIError,
    load_config,
)
from strava_mcp.rate_limit import RateLimitTracker

ACTIVITY_STREAM_KEYS: tuple[str, ...] = (
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
)
SEGMENT_STREAM_KEYS: tuple[str, ...] = ("distance", "latlng", "altitude")

TTL_SHORT_SECONDS = 300
TTL_PROFILE_SECONDS = 900
TTL_MEDIUM_SECONDS = 3600
TTL_LONG_SECONDS = 86400
TTL_IMMUTABLE_SECONDS: float | None = None

_KNOWN_SCOPES = frozenset(
    {
        "read",
        "read_all",
        "profile:read_all",
        "profile:write",
        "activity:read",
        "activity:read_all",
        "activity:write",
    }
)


def _normalize_stream_keys(
    keys: str | Iterable[str],
    *,
    allowed: tuple[str, ...],
    stream_kind: str,
) -> str:
    raw_items: list[str] = []
    if isinstance(keys, str):
        raw_items.extend(keys.replace(",", " ").split())
    else:
        for item in keys:
            raw_items.extend(str(item).replace(",", " ").split())

    cleaned = sorted({part.strip() for part in raw_items if part.strip()})
    if not cleaned:
        raise ValueError(f"{stream_kind} stream keys must not be empty.")
    allowed_set = set(allowed)
    invalid = [key for key in cleaned if key not in allowed_set]
    if invalid:
        raise ValueError(
            f"Unsupported {stream_kind} stream key(s): {', '.join(invalid)}. "
            f"Allowed: {', '.join(allowed)}."
        )
    return ",".join(cleaned)


def _try_parse_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _extract_fault_fields(
    body: Any,
) -> tuple[str | None, list[dict[str, Any]]]:
    if not isinstance(body, dict):
        return (None, [])
    raw_msg = body.get("message")
    message = raw_msg.strip()[:300] if isinstance(raw_msg, str) and raw_msg.strip() else None
    raw_errors = body.get("errors")
    errors: list[dict[str, Any]] = []
    if isinstance(raw_errors, list):
        for item in raw_errors:
            if isinstance(item, dict):
                errors.append(item)
    return (message, errors)


def _extract_scope_error_info(
    message: str | None,
    errors: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    inferred: list[str] = []
    is_scope = False

    for err in errors:
        field_raw = err.get("field")
        if not isinstance(field_raw, str):
            continue
        field = field_raw.strip()
        lowered = field.lower()
        if lowered.endswith("_permission"):
            is_scope = True
            candidate = field[: -len("_permission")].strip()
            if candidate and candidate not in inferred:
                inferred.append(candidate)
        elif lowered in _KNOWN_SCOPES:
            is_scope = True
            if lowered not in inferred:
                inferred.append(lowered)
        elif "permission" in lowered or "scope" in lowered:
            is_scope = True

    if not is_scope and message is not None:
        lower_msg = message.lower()
        if (
            "missing" in lower_msg and ("permission" in lower_msg or "scope" in lower_msg)
        ) or "insufficient scope" in lower_msg:
            is_scope = True

    return (is_scope, inferred)


def _is_subscription_fault(message: str | None, errors: list[dict[str, Any]]) -> bool:
    haystack_parts: list[str] = []
    if message:
        haystack_parts.append(message.lower())
    for err in errors:
        for key in ("resource", "field", "code"):
            val = err.get(key)
            if isinstance(val, str):
                haystack_parts.append(val.lower())
    haystack = " ".join(haystack_parts)
    return "subscription" in haystack or "summit" in haystack


def _is_tier_fault(message: str | None, errors: list[dict[str, Any]]) -> bool:
    haystack_parts: list[str] = []
    if message:
        haystack_parts.append(message.lower())
    for err in errors:
        for key in ("resource", "field", "code"):
            val = err.get(key)
            if isinstance(val, str):
                haystack_parts.append(val.lower())
    haystack = " ".join(haystack_parts)
    return "tier" in haystack or "extended access" in haystack


def _fault_names_activity_resource(errors: list[dict[str, Any]]) -> bool:
    for err in errors:
        resource = err.get("resource")
        if isinstance(resource, str) and resource.strip().lower() == "activity":
            return True
    return False


class StravaClient:
    """Async Strava API v3 read client with caching, rate limiting, and error mapping."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        token_manager: TokenManager | None = None,
        rate_limiter: RateLimitTracker | None = None,
        cache: ResponseCache | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.config = config if config is not None else load_config()
        self._transport = transport
        self._timeout = timeout
        self._user_agent = user_agent
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._token_manager = token_manager or TokenManager(
            self.config,
            http_client=http_client,
            transport=transport,
        )
        self._rate_limiter = rate_limiter or RateLimitTracker()
        self._owns_cache = cache is None
        self._cache = cache if cache is not None else ResponseCache(self.config)

    @property
    def token_manager(self) -> TokenManager:
        return self._token_manager

    @property
    def rate_limiter(self) -> RateLimitTracker:
        return self._rate_limiter

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent},
            )
            self._owns_http_client = True
        return self._http_client

    async def __aenter__(self) -> StravaClient:
        self._get_http_client()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close internally-owned HTTP client and cache resources."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._owns_cache:
            self._cache.close()

    def _raise_for_error_response(
        self,
        resp: httpx.Response,
        *,
        path: str,
        required_scopes: tuple[str, ...],
        granted_scopes: list[str],
        subscription_gated: bool,
    ) -> None:
        status = resp.status_code
        body = _try_parse_json(resp)
        fault_msg, errors = _extract_fault_fields(body)
        base_details: dict[str, Any] = {"status_code": status, "path": path}
        if fault_msg:
            base_details["strava_message"] = fault_msg

        if status == 402:
            raise SubscriptionRequiredError(
                fault_msg or "This Strava endpoint requires a Strava subscription.",
                details=base_details,
            )

        if status == 403:
            if body is None:
                raise UpstreamAPIError(
                    "Strava API returned non-JSON HTTP 403.",
                    status_code=403,
                    details={
                        **base_details,
                        "content_type": resp.headers.get("content-type", ""),
                        "body_snippet": resp.text[:200],
                    },
                )
            is_scope, inferred = _extract_scope_error_info(fault_msg, errors)
            if is_scope:
                raise InsufficientScopeError(
                    fault_msg,
                    required_scopes=inferred or required_scopes,
                    granted_scopes=granted_scopes,
                    remediation="strava-mcp auth",
                    details=base_details,
                )
            if _is_tier_fault(fault_msg, errors) or path.startswith("/segments/explore"):
                raise TierRestrictedError(
                    fault_msg or "This Strava resource is restricted by API application tier.",
                    details=base_details,
                )
            if _is_subscription_fault(fault_msg, errors) or subscription_gated:
                raise SubscriptionRequiredError(
                    fault_msg or "This Strava endpoint requires a Strava subscription.",
                    details=base_details,
                )
            raise UpstreamAPIError(
                fault_msg or "Strava API returned HTTP 403.",
                status_code=403,
                details=base_details,
            )

        if status == 404:
            raise NotFoundError(
                fault_msg or f"Requested Strava resource not found: {path}",
                details=base_details,
            )

        raise UpstreamAPIError(
            fault_msg or f"Strava API request failed with HTTP {status}.",
            status_code=status,
            details=base_details,
        )

    async def _request(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        required_scopes: tuple[str, ...] = ("read",),
        ttl_seconds: float | int | None = TTL_SHORT_SECONDS,
        refresh: bool = False,
        response_type: str = "json",
        subscription_gated: bool = False,
        allow_missing_streams_404: bool = False,
        context: str | None = None,
    ) -> Any:
        clean_params = (
            {k: v for k, v in params.items() if v is not None} if params is not None else {}
        )
        cache_key = make_cache_key(path, clean_params)
        if not refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        tokens = await self._token_manager.get_valid_tokens()
        granted_scopes = parse_scopes(tokens.scope)
        if required_scopes and granted_scopes:
            require_scope(granted_scopes, required_scopes, context=context)

        client = self._get_http_client()
        url = f"{self.config.api_base_url}{path}"
        resp: httpx.Response | None = None

        for attempt in range(2):
            self._rate_limiter.check_available(is_read=True)
            headers = {
                "Authorization": f"Bearer {tokens.access_token}",
                "User-Agent": self._user_agent,
            }
            try:
                resp = await client.get(
                    url,
                    params=clean_params or None,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise UpstreamAPIError(
                    f"Strava API request failed ({type(exc).__name__}).",
                    status_code=502,
                    details={"error_type": type(exc).__name__, "path": path},
                ) from None

            self._rate_limiter.update_from_headers(resp.headers)
            if resp.status_code == 429:
                raise self._rate_limiter.error_from_429(resp, is_read=True)

            if resp.status_code == 401:
                body = _try_parse_json(resp)
                fault_msg, errors = _extract_fault_fields(body)
                is_scope, inferred = _extract_scope_error_info(fault_msg, errors)
                if is_scope:
                    raise InsufficientScopeError(
                        fault_msg,
                        required_scopes=inferred or required_scopes,
                        granted_scopes=parse_scopes(tokens.scope),
                        remediation="strava-mcp auth",
                        details={"status_code": 401, "path": path},
                    )
                if attempt == 0:
                    tokens = await self._token_manager.get_valid_tokens(
                        if_access_token_matches=tokens.access_token
                    )
                    granted_scopes = parse_scopes(tokens.scope)
                    continue
                raise AuthError(
                    "auth_expired",
                    fault_msg or "Strava access token was rejected with HTTP 401 after refresh.",
                    details={"status_code": 401, "path": path},
                )

            break

        assert resp is not None
        if resp.status_code == 404 and allow_missing_streams_404:
            body = _try_parse_json(resp)
            _msg, errors = _extract_fault_fields(body)
            if not _fault_names_activity_resource(errors):
                return {}

        if resp.status_code >= 400:
            self._raise_for_error_response(
                resp,
                path=path,
                required_scopes=required_scopes,
                granted_scopes=granted_scopes,
                subscription_gated=subscription_gated,
            )

        if response_type == "text":
            result: Any = resp.text
        else:
            try:
                result = resp.json()
            except ValueError:
                raise UpstreamAPIError(
                    "Strava API returned invalid JSON response.",
                    status_code=resp.status_code,
                    details={"status_code": resp.status_code, "path": path},
                ) from None

        self._cache.set(
            cache_key,
            result,
            ttl_seconds=ttl_seconds,
            etag=resp.headers.get("etag"),
            metadata={"endpoint": path},
        )
        return result

    async def _resolve_athlete_id(self, athlete_id: int | None) -> int:
        if athlete_id is not None:
            return int(athlete_id)
        tokens = await self._token_manager.get_valid_tokens()
        if tokens.athlete_id is not None:
            return int(tokens.athlete_id)
        athlete = await self.get_athlete(refresh=False)
        raw_id = athlete.get("id")
        if raw_id is None or isinstance(raw_id, bool):
            raise UpstreamAPIError(
                "Strava athlete profile response is missing an integer 'id'.",
                status_code=502,
                details={"path": "/athlete"},
            )
        return int(raw_id)

    async def get_athlete(self, *, refresh: bool = False) -> dict[str, Any]:
        """Get the authenticated athlete profile (GET /athlete)."""
        return await self._request(
            "/athlete",
            required_scopes=("read",),
            ttl_seconds=TTL_PROFILE_SECONDS,
            refresh=refresh,
            context="get_athlete",
        )

    async def get_athlete_stats(
        self,
        athlete_id: int | None = None,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get recent, YTD, and all-time stats for the athlete (GET /athletes/{id}/stats)."""
        resolved_id = await self._resolve_athlete_id(athlete_id)
        return await self._request(
            f"/athletes/{resolved_id}/stats",
            required_scopes=("read",),
            ttl_seconds=TTL_PROFILE_SECONDS,
            refresh=refresh,
            context="get_athlete_stats",
        )

    async def get_athlete_zones(self, *, refresh: bool = False) -> dict[str, Any]:
        """Get the authenticated athlete's heart-rate and power zones (GET /athlete/zones)."""
        return await self._request(
            "/athlete/zones",
            required_scopes=("profile:read_all",),
            ttl_seconds=TTL_PROFILE_SECONDS,
            refresh=refresh,
            context="get_athlete_zones",
        )

    async def list_activities(
        self,
        before: int | None = None,
        after: int | None = None,
        page: int = 1,
        per_page: int = 30,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """List authenticated athlete activities (GET /athlete/activities)."""
        return await self._request(
            "/athlete/activities",
            params={
                "before": before,
                "after": after,
                "page": page,
                "per_page": per_page,
            },
            required_scopes=("activity:read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            context="list_activities",
        )

    async def get_activity(
        self,
        activity_id: int,
        include_all_efforts: bool = False,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get detailed activity representation (GET /activities/{id})."""
        return await self._request(
            f"/activities/{activity_id}",
            params={"include_all_efforts": "true" if include_all_efforts else "false"},
            required_scopes=("activity:read",),
            ttl_seconds=TTL_LONG_SECONDS,
            refresh=refresh,
            context="get_activity",
        )

    async def get_activity_laps(
        self,
        activity_id: int,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Get laps for an activity (GET /activities/{id}/laps)."""
        return await self._request(
            f"/activities/{activity_id}/laps",
            required_scopes=("activity:read",),
            ttl_seconds=TTL_LONG_SECONDS,
            refresh=refresh,
            context="get_activity_laps",
        )

    async def get_activity_zones(
        self,
        activity_id: int,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Get heart-rate and power zone distributions for an activity (subscription required)."""
        return await self._request(
            f"/activities/{activity_id}/zones",
            required_scopes=("activity:read",),
            ttl_seconds=TTL_LONG_SECONDS,
            refresh=refresh,
            subscription_gated=True,
            context="get_activity_zones",
        )

    async def get_activity_comments(
        self,
        activity_id: int,
        page_size: int = 30,
        after_cursor: str | None = None,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Get comments on an activity using cursor pagination (GET /activities/{id}/comments)."""
        return await self._request(
            f"/activities/{activity_id}/comments",
            params={"page_size": page_size, "after_cursor": after_cursor},
            required_scopes=("activity:read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            context="get_activity_comments",
        )

    async def get_activity_kudos(
        self,
        activity_id: int,
        page: int = 1,
        per_page: int = 30,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Get kudoers for an activity (GET /activities/{id}/kudos)."""
        return await self._request(
            f"/activities/{activity_id}/kudos",
            params={"page": page, "per_page": per_page},
            required_scopes=("activity:read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            context="get_activity_kudos",
        )

    async def get_activity_streams(
        self,
        activity_id: int,
        keys: str | Iterable[str] = ACTIVITY_STREAM_KEYS,
        key_by_type: bool = True,
        *,
        missing_ok: bool = True,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get time-series streams for an activity (GET /activities/{id}/streams)."""
        normalized_keys = _normalize_stream_keys(
            keys,
            allowed=ACTIVITY_STREAM_KEYS,
            stream_kind="activity",
        )
        return await self._request(
            f"/activities/{activity_id}/streams",
            params={
                "keys": normalized_keys,
                "key_by_type": "true" if key_by_type else "false",
            },
            required_scopes=("activity:read",),
            ttl_seconds=TTL_IMMUTABLE_SECONDS,
            refresh=refresh,
            allow_missing_streams_404=missing_ok,
            context="get_activity_streams",
        )

    async def get_gear(
        self,
        gear_id: str,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get gear details for a bike or shoe (GET /gear/{id})."""
        cleaned_id = gear_id.strip()
        if not cleaned_id:
            raise ValueError("gear_id must not be empty.")
        return await self._request(
            f"/gear/{cleaned_id}",
            required_scopes=("read",),
            ttl_seconds=TTL_MEDIUM_SECONDS,
            refresh=refresh,
            context="get_gear",
        )

    async def list_routes(
        self,
        athlete_id: int | None = None,
        page: int = 1,
        per_page: int = 30,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """List routes created by the athlete (GET /athletes/{id}/routes)."""
        resolved_id = await self._resolve_athlete_id(athlete_id)
        return await self._request(
            f"/athletes/{resolved_id}/routes",
            params={"page": page, "per_page": per_page},
            required_scopes=("read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            context="list_routes",
        )

    async def get_route(
        self,
        route_id: int,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get route details (GET /routes/{id})."""
        return await self._request(
            f"/routes/{route_id}",
            required_scopes=("read",),
            ttl_seconds=TTL_LONG_SECONDS,
            refresh=refresh,
            context="get_route",
        )

    async def export_route_gpx(
        self,
        route_id: int,
        *,
        refresh: bool = False,
    ) -> str:
        """Export a route as GPX XML string (GET /routes/{id}/export_gpx)."""
        return await self._request(
            f"/routes/{route_id}/export_gpx",
            required_scopes=("read",),
            ttl_seconds=TTL_IMMUTABLE_SECONDS,
            refresh=refresh,
            response_type="text",
            context="export_route_gpx",
        )

    async def export_route_tcx(
        self,
        route_id: int,
        *,
        refresh: bool = False,
    ) -> str:
        """Export a route as TCX XML string (GET /routes/{id}/export_tcx)."""
        return await self._request(
            f"/routes/{route_id}/export_tcx",
            required_scopes=("read",),
            ttl_seconds=TTL_IMMUTABLE_SECONDS,
            refresh=refresh,
            response_type="text",
            context="export_route_tcx",
        )

    async def get_route_streams(
        self,
        route_id: int,
        *,
        refresh: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Get distance/latlng/altitude streams for a route (GET /routes/{id}/streams)."""
        return await self._request(
            f"/routes/{route_id}/streams",
            required_scopes=("read",),
            ttl_seconds=TTL_IMMUTABLE_SECONDS,
            refresh=refresh,
            context="get_route_streams",
        )

    async def list_starred_segments(
        self,
        page: int = 1,
        per_page: int = 30,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """List starred segments for the authenticated athlete (GET /segments/starred)."""
        return await self._request(
            "/segments/starred",
            params={"page": page, "per_page": per_page},
            required_scopes=("read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            context="list_starred_segments",
        )

    async def get_segment(
        self,
        segment_id: int,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get segment details (GET /segments/{id})."""
        return await self._request(
            f"/segments/{segment_id}",
            required_scopes=("read",),
            ttl_seconds=TTL_MEDIUM_SECONDS,
            refresh=refresh,
            context="get_segment",
        )

    async def get_segment_streams(
        self,
        segment_id: int,
        keys: str | Iterable[str] = SEGMENT_STREAM_KEYS,
        key_by_type: bool = True,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get streams for a segment (GET /segments/{id}/streams)."""
        normalized_keys = _normalize_stream_keys(
            keys,
            allowed=SEGMENT_STREAM_KEYS,
            stream_kind="segment",
        )
        return await self._request(
            f"/segments/{segment_id}/streams",
            params={
                "keys": normalized_keys,
                "key_by_type": "true" if key_by_type else "false",
            },
            required_scopes=("read",),
            ttl_seconds=TTL_MEDIUM_SECONDS,
            refresh=refresh,
            context="get_segment_streams",
        )

    async def list_segment_efforts(
        self,
        segment_id: int,
        start_date_local: str | None = None,
        end_date_local: str | None = None,
        per_page: int = 30,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """List segment efforts for a segment (GET /segment_efforts, subscription required)."""
        return await self._request(
            "/segment_efforts",
            params={
                "segment_id": segment_id,
                "start_date_local": start_date_local,
                "end_date_local": end_date_local,
                "per_page": per_page,
            },
            required_scopes=("read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            subscription_gated=True,
            context="list_segment_efforts",
        )

    async def list_clubs(
        self,
        page: int = 1,
        per_page: int = 30,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """List clubs the authenticated athlete belongs to (GET /athlete/clubs)."""
        return await self._request(
            "/athlete/clubs",
            params={"page": page, "per_page": per_page},
            required_scopes=("read",),
            ttl_seconds=TTL_SHORT_SECONDS,
            refresh=refresh,
            context="list_clubs",
        )

    async def get_club(
        self,
        club_id: int,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Get club details (GET /clubs/{id})."""
        return await self._request(
            f"/clubs/{club_id}",
            required_scopes=("read",),
            ttl_seconds=TTL_MEDIUM_SECONDS,
            refresh=refresh,
            context="get_club",
        )
