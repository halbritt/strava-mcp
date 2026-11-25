"""CLI entrypoint for strava-mcp."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import time
import webbrowser
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from strava_mcp import __version__, mcp_server, metrics
from strava_mcp.auth import (
    CORE_RECOMMENDED_SCOPES,
    DEFAULT_CALLBACK_PORT,
    DEFAULT_SCOPE_STRING,
    TokenManager,
    missing_scopes,
    parse_scopes,
    revoke_token,
    run_loopback_auth,
)
from strava_mcp.cache import ResponseCache
from strava_mcp.client import StravaClient
from strava_mcp.config import (
    ClientConfig,
    ConfigError,
    StravaMCPError,
    TokenData,
    ensure_mcp_token,
    ensure_secure_dir,
    load_client,
    load_config,
    load_mcp_token,
    load_tokens,
    save_client,
    save_tokens,
)
from strava_mcp.rate_limit import RateLimitTracker


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _format_utc(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def _parse_listen_address(raw: str) -> tuple[str, int]:
    text = raw.strip()
    if not text:
        raise ConfigError("invalid_config", "--listen address must not be empty.")
    if text.startswith("["):
        end_bracket = text.find("]")
        if end_bracket == -1:
            raise ConfigError("invalid_config", f"Invalid IPv6 --listen address: {raw!r}")
        host = text[1:end_bracket].strip()
        rest = text[end_bracket + 1 :].strip()
        if not rest.startswith(":"):
            raise ConfigError("invalid_config", f"Missing :port in --listen address: {raw!r}")
        port_str = rest[1:].strip()
    else:
        if ":" not in text:
            raise ConfigError("invalid_config", f"Expected host:port in --listen: {raw!r}")
        host, port_str = text.rsplit(":", 1)
        host = host.strip()
        port_str = port_str.strip()

    if not host:
        raise ConfigError("invalid_config", f"Missing host in --listen address: {raw!r}")
    if not port_str.isascii() or not port_str.isdigit():
        raise ConfigError("invalid_config", f"Invalid port in --listen address: {raw!r}")
    port = int(port_str)
    if port < 1 or port > 65535:
        raise ConfigError("invalid_config", f"Port out of range (1-65535): {port}")
    return (host, port)


def _handle_init(args: argparse.Namespace) -> int:
    cfg = load_config()
    ensure_secure_dir(cfg.config_dir)

    client_cfg: ClientConfig
    if args.client_id is not None:
        if args.client_secret_stdin:
            secret = sys.stdin.read().strip()
        elif sys.stdin.isatty():
            secret = getpass.getpass("Strava Client Secret: ").strip()
        else:
            raise ConfigError(
                "invalid_config",
                "Use --client-secret-stdin when supplying --client-id non-interactively.",
            )
        if not secret:
            raise ConfigError("invalid_config", "client_secret must not be empty.")
        try:
            client_cfg = ClientConfig(client_id=args.client_id, client_secret=secret)
        except ValidationError:
            raise ConfigError("invalid_config", "Invalid client_id or client_secret.") from None
        save_client(cfg, client_cfg)
    elif cfg.client_file.exists() or cfg.client_file.is_symlink():
        client_cfg = load_client(cfg)
    elif sys.stdin.isatty():
        entered_id = input("Strava Client ID: ").strip()
        entered_secret = getpass.getpass("Strava Client Secret: ").strip()
        try:
            client_cfg = ClientConfig(client_id=entered_id, client_secret=entered_secret)
        except ValidationError:
            raise ConfigError("invalid_config", "Invalid client_id or client_secret.") from None
        save_client(cfg, client_cfg)
    else:
        raise ConfigError("config_missing", "Missing --client-id for non-interactive init.")

    tokens_imported = False
    if args.import_tokens_file is not None:
        src = Path(args.import_tokens_file)
        if src.is_symlink() or not src.is_file():
            raise ConfigError(
                "invalid_config",
                f"Token import path must be a regular file: {src}",
                details={"path": str(src)},
            )
        try:
            raw_payload = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(
                "invalid_config",
                f"Failed to parse token import file {src}",
                details={"path": str(src), "error_type": type(exc).__name__},
            ) from None
        if not isinstance(raw_payload, dict):
            raise ConfigError(
                "invalid_config",
                "Token import file must contain a JSON object.",
                details={"path": str(src)},
            )
        try:
            normalized = dict(raw_payload)
            for num_field in ("athlete_id", "expires_in", "expires_at", "updated_at"):
                if isinstance(normalized.get(num_field), bool):
                    raise ValueError(f"{num_field} must be an integer, not a boolean.")
            if normalized.get("athlete_id") is None and isinstance(normalized.get("athlete"), dict):
                ath_id = normalized["athlete"].get("id")
                if ath_id is not None:
                    if isinstance(ath_id, bool):
                        raise ValueError("athlete.id must be an integer, not a boolean.")
                    normalized["athlete_id"] = int(ath_id)
            if normalized.get("expires_at") is None and normalized.get("expires_in") is not None:
                raw_exp_in = normalized["expires_in"]
                if isinstance(raw_exp_in, bool):
                    raise ValueError("expires_in must be an integer, not a boolean.")
                normalized["expires_at"] = int(time.time()) + int(raw_exp_in)
            tokens = TokenData.model_validate(normalized)
            datetime.fromtimestamp(tokens.expires_at, tz=UTC)
            if tokens.updated_at is not None:
                datetime.fromtimestamp(tokens.updated_at, tz=UTC)
        except (ValidationError, ValueError, TypeError, OverflowError, OSError) as exc:
            raise ConfigError(
                "invalid_config",
                "Invalid token import payload.",
                details={"path": str(src), "error_type": type(exc).__name__},
            ) from None
        save_tokens(cfg, tokens)
        tokens_imported = True

    ensure_mcp_token(cfg)

    result = {
        "initialized": True,
        "config_dir": str(cfg.config_dir),
        "client_id": client_cfg.client_id,
        "client_file": str(cfg.client_file),
        "tokens_imported": tokens_imported,
        "tokens_file": str(cfg.tokens_file),
        "mcp_token_configured": True,
        "mcp_token_file": str(cfg.mcp_token_file),
    }
    if args.json:
        _emit_json(result)
    else:
        print(f"Initialized strava-mcp in {cfg.config_dir} (client_id={client_cfg.client_id}).")
        if tokens_imported:
            print(f"Imported tokens to {cfg.tokens_file} (mode 0600).")
        print(f"MCP bearer token ready at {cfg.mcp_token_file} (mode 0600).")
    return 0


async def _handle_auth(args: argparse.Namespace) -> int:
    cfg = load_config()
    manager = TokenManager(cfg)

    def _on_url(url: str) -> None:
        print(f"Open this URL in your browser to authorize Strava:\n{url}", file=sys.stderr)
        if not args.no_browser:
            with suppress(Exception):
                webbrowser.open(url)

    auth_result = await run_loopback_auth(
        manager,
        scope=args.scopes,
        port=args.port,
        redirect_uri=args.redirect_uri,
        approval_prompt=args.approval_prompt,
        timeout_seconds=args.timeout,
        on_authorize_url=_on_url,
    )
    payload = {
        "authenticated": True,
        "athlete_id": auth_result.tokens.athlete_id,
        "expires_at": auth_result.tokens.expires_at,
        "expires_at_utc": _format_utc(auth_result.tokens.expires_at),
        "granted_scopes": auth_result.granted_scopes,
        "missing_recommended_scopes": auth_result.missing_recommended_scopes,
        "warnings": auth_result.warnings,
    }
    if args.json:
        _emit_json(payload)
    else:
        for warning in auth_result.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        print(
            f"Authenticated athlete_id={auth_result.tokens.athlete_id} "
            f"scopes={','.join(auth_result.granted_scopes) or 'none'} "
            f"expires={payload['expires_at_utc'] or '(out-of-range)'}"
        )
    return 0


async def _handle_status(args: argparse.Namespace) -> int:
    cfg = load_config()

    client_configured = False
    client_id: str | None = None
    if cfg.client_file.exists() or cfg.client_file.is_symlink():
        client_cfg = load_client(cfg)
        client_configured = True
        client_id = client_cfg.client_id

    mcp_token_configured = False
    if cfg.mcp_token_file.exists() or cfg.mcp_token_file.is_symlink():
        load_mcp_token(cfg)
        mcp_token_configured = True

    authenticated = False
    athlete_id: int | None = None
    expires_at: int | None = None
    expires_at_utc: str | None = None
    is_expired: bool | None = None
    granted_scopes: list[str] = []
    missing_rec: list[str] = list(CORE_RECOMMENDED_SCOPES)

    if cfg.tokens_file.exists() or cfg.tokens_file.is_symlink():
        tokens = load_tokens(cfg)
        authenticated = True
        athlete_id = tokens.athlete_id
        expires_at = tokens.expires_at
        expires_at_utc = _format_utc(tokens.expires_at)
        is_expired = tokens.expires_at <= int(time.time())
        granted_scopes = parse_scopes(tokens.scope)
        missing_rec = missing_scopes(tokens.scope, CORE_RECOMMENDED_SCOPES)

    live_check: dict[str, Any] | None = None
    if args.check_live:
        async with StravaClient(cfg) as client:
            athlete = await client.get_athlete(refresh=True)
            live_check = {
                "ok": True,
                "athlete_id": athlete.get("id"),
                "username": athlete.get("username"),
            }
            rate_limits = client.rate_limiter.snapshot()
            cache_stats = client.cache.stats()
    else:
        rate_limits = RateLimitTracker().snapshot()
        if cfg.cache_db_file.exists():
            with ResponseCache(cfg) as cache:
                cache_stats = cache.stats()
        else:
            cache_stats = {
                "db_path": str(cfg.cache_db_file),
                "exists": False,
                "entry_count": 0,
            }

    payload: dict[str, Any] = {
        "config_dir": str(cfg.config_dir),
        "client_configured": client_configured,
        "client_id": client_id,
        "mcp_token_configured": mcp_token_configured,
        "authenticated": authenticated,
        "athlete_id": athlete_id,
        "expires_at": expires_at,
        "expires_at_utc": expires_at_utc,
        "is_expired": is_expired,
        "granted_scopes": granted_scopes,
        "missing_recommended_scopes": missing_rec,
        "rate_limits": rate_limits,
        "cache": cache_stats,
    }
    if live_check is not None:
        payload["live_check"] = live_check

    if args.json:
        _emit_json(payload)
    else:
        print(f"Config dir:    {cfg.config_dir}")
        print(f"Client ID:     {client_id or 'not configured'}")
        print(f"MCP token:     {'configured' if mcp_token_configured else 'missing'}")
        print(f"Authenticated: {authenticated}")
        if authenticated:
            print(f"Athlete ID:    {athlete_id}")
            expires_display = expires_at_utc or "(out-of-range)"
            print(f"Expires at:    {expires_display} (expired={is_expired})")
            print(f"Scopes:        {','.join(granted_scopes) or 'none'}")
            if missing_rec:
                print(f"Missing scope: {','.join(missing_rec)} (run 'strava-mcp auth')")
        print(f"Cache entries: {cache_stats.get('entry_count', 0)}")
        if live_check is not None:
            print(f"Live check:    OK (athlete_id={live_check['athlete_id']})")
    return 0


async def _handle_refresh(args: argparse.Namespace) -> int:
    cfg = load_config()
    tokens = await TokenManager(cfg).get_valid_tokens(force=True)
    payload = {
        "refreshed": True,
        "athlete_id": tokens.athlete_id,
        "expires_at": tokens.expires_at,
        "expires_at_utc": _format_utc(tokens.expires_at),
        "granted_scopes": parse_scopes(tokens.scope),
    }
    if args.json:
        _emit_json(payload)
    else:
        expires_display = payload["expires_at_utc"] or "(out-of-range)"
        print(
            f"Refreshed tokens for athlete_id={tokens.athlete_id} "
            f"(expires={expires_display})."
        )
    return 0


async def _handle_revoke(args: argparse.Namespace) -> int:
    cfg = load_config()
    res = await revoke_token(
        cfg,
        token_type_hint=args.token_type,
        clear_local=not args.keep_local,
        oauth_base_url=cfg.oauth_base_url,
    )
    if args.json:
        _emit_json(res)
    else:
        print(
            f"Revoked Strava {res['token_type_hint']} "
            f"(local_cleared={res['local_cleared']})."
        )
    return 0


def _handle_stdio(_args: argparse.Namespace) -> int:
    cfg = load_config()
    mcp_server.run_stdio(config=cfg)
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    host, port = _parse_listen_address(args.listen)
    mcp_server.validate_bind_host(host, allow_non_loopback=args.allow_non_loopback)
    cfg = load_config()
    mcp_server.serve_http(
        host=host,
        port=port,
        allowed_hosts=args.allowed_host or (),
        allow_non_loopback=args.allow_non_loopback,
        config=cfg,
    )
    return 0


def _handle_cache(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.cache_command == "stats":
        with ResponseCache(cfg) as cache:
            stats = cache.stats()
        if args.json:
            _emit_json(stats)
        else:
            print(
                f"Cache DB: {stats['db_path']} "
                f"(entries={stats['entry_count']}, active={stats['active_count']}, "
                f"expired={stats['expired_count']}, size={stats['db_size_bytes']}B)"
            )
        return 0

    if args.cache_command == "clear":
        with ResponseCache(cfg) as cache:
            removed = cache.clear()
        payload = {"cleared": True, "removed_entries": removed, "db_path": str(cfg.cache_db_file)}
        if args.json:
            _emit_json(payload)
        else:
            print(f"Cleared {removed} cached entries from {cfg.cache_db_file}.")
        return 0

    raise ValueError(f"Unknown cache subcommand: {args.cache_command}")


async def _handle_activities(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be >= 1.")
    after_ts = metrics.parse_date_filter(args.after, is_end=False)
    before_ts = metrics.parse_date_filter(args.before, is_end=True)
    if after_ts is not None and before_ts is not None and after_ts > before_ts:
        raise ValueError("--after must be less than or equal to --before.")

    cfg = load_config()
    async with StravaClient(cfg) as client:
        items, pages_fetched, has_more = await metrics.fetch_activities_paginated(
            client,
            after=after_ts,
            before=before_ts,
            sport_type=args.sport_type,
            page=1,
            per_page=min(200, args.limit),
            limit=args.limit,
            max_pages=10,
            refresh=args.refresh,
        )
    formatted = [
        metrics.format_activity_summary(act, include_polyline=args.include_polyline)
        for act in items
    ]
    payload = {
        "count": len(formatted),
        "activities": formatted,
        "pages_fetched": pages_fetched,
        "has_more": has_more,
    }
    if args.json:
        _emit_json(payload)
    else:
        for act in formatted:
            dist = f"{act['distance_km']} km" if act["distance_km"] is not None else "-"
            dur = act["moving_time_formatted"] or "-"
            print(
                f"{act['id']}  {act['start_date_local']}  {act['sport_type']:<14}  "
                f"{dist:>10}  {dur:>8}  {act['name']}"
            )
    return 0


async def _handle_activity(args: argparse.Namespace) -> int:
    if args.activity_id < 1:
        raise ValueError("activity_id must be >= 1.")
    cfg = load_config()
    async with StravaClient(cfg) as client:
        raw = await client.get_activity(
            args.activity_id,
            include_all_efforts=args.include_all_efforts,
            refresh=args.refresh,
        )
    detail = metrics.format_activity_detail(
        raw,
        include_all_efforts=args.include_all_efforts,
        include_polyline=args.include_polyline,
    )
    if args.json:
        _emit_json(detail)
    else:
        print(f"{detail['id']}: {detail['name']} ({detail['sport_type']})")
        print(f"Start:     {detail['start_date_local']}")
        print(f"Distance:  {detail['distance_km']} km ({detail['distance_mi']} mi)")
        print(f"Moving:    {detail['moving_time_formatted']}")
        print(f"Elevation: {detail['elevation_gain_m']} m ({detail['elevation_gain_ft']} ft)")
        print(f"Efforts:   {detail['segment_efforts_count']}")
    return 0


async def _handle_stats(args: argparse.Namespace) -> int:
    cfg = load_config()
    async with StravaClient(cfg) as client:
        raw = await client.get_athlete_stats(athlete_id=args.athlete_id, refresh=args.refresh)
    payload = metrics.normalize_athlete_stats(raw)
    payload["visibility_caveat"] = metrics.STATS_VISIBILITY_CAVEAT
    if args.json:
        _emit_json(payload)
    else:
        for key in ("recent_run_totals", "recent_ride_totals", "ytd_run_totals", "ytd_ride_totals"):
            bucket = payload.get(key)
            if isinstance(bucket, dict):
                print(
                    f"{key:<20} count={bucket.get('count')} "
                    f"dist={bucket.get('distance_km')} km ({bucket.get('distance_mi')} mi)"
                )
        print(f"Note: {payload['visibility_caveat']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level strava-mcp argument parser with all 11 subcommands."""
    parser = argparse.ArgumentParser(
        prog="strava-mcp",
        description="Read-only Strava MCP connector",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize config directory, client, and MCP token")
    p_init.add_argument("--client-id", help="Strava API Client ID")
    p_init.add_argument(
        "--client-secret-stdin",
        action="store_true",
        help="Read Strava Client Secret from stdin",
    )
    p_init.add_argument("--import-tokens-file", help="Import initial tokens from JSON file")
    p_init.add_argument("--json", action="store_true", help="Emit JSON output")

    p_auth = sub.add_parser("auth", help="Run loopback OAuth2 authorization to upgrade scopes")
    p_auth.add_argument("--port", type=int, default=DEFAULT_CALLBACK_PORT, help="Callback port")
    p_auth.add_argument("--redirect-uri", default=None, help="Override callback redirect URI")
    p_auth.add_argument("--scopes", default=DEFAULT_SCOPE_STRING, help="Requested OAuth scopes")
    p_auth.add_argument(
        "--approval-prompt",
        choices=["force", "auto"],
        default="force",
        help="Strava OAuth approval prompt mode",
    )
    p_auth.add_argument("--timeout", type=float, default=300.0, help="Callback timeout seconds")
    p_auth.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    p_auth.add_argument("--json", action="store_true", help="Emit JSON output")

    p_status = sub.add_parser("status", help="Show auth, scope, rate-limit, and cache status")
    p_status.add_argument("--check-live", action="store_true", help="Verify live GET /athlete")
    p_status.add_argument("--json", action="store_true", help="Emit JSON output")

    p_refresh = sub.add_parser("refresh", help="Force OAuth2 token refresh")
    p_refresh.add_argument("--json", action="store_true", help="Emit JSON output")

    p_revoke = sub.add_parser("revoke", help="Revoke Strava OAuth token")
    p_revoke.add_argument(
        "--token-type",
        choices=["access_token", "refresh_token"],
        default="refresh_token",
        help="Token hint sent to POST /oauth/revoke",
    )
    p_revoke.add_argument(
        "--keep-local",
        action="store_true",
        help="Do not delete local tokens.json after remote revocation",
    )
    p_revoke.add_argument("--json", action="store_true", help="Emit JSON output")

    sub.add_parser("stdio", help="Run MCP server over stdio")

    p_serve = sub.add_parser("serve", help="Run authenticated Streamable HTTP MCP server")
    p_serve.add_argument("--listen", default="127.0.0.1:8772", help="Listen host:port")
    p_serve.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Additional allowed HTTP Host header (e.g. Tailscale Serve hostname)",
    )
    p_serve.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow binding to a non-loopback address",
    )

    p_cache = sub.add_parser("cache", help="Inspect or clear the local SQLite cache")
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True)
    p_cache_stats = cache_sub.add_parser("stats", help="Show SQLite cache statistics")
    p_cache_stats.add_argument("--json", action="store_true", help="Emit JSON output")
    p_cache_clear = cache_sub.add_parser("clear", help="Clear all SQLite cache entries")
    p_cache_clear.add_argument("--json", action="store_true", help="Emit JSON output")

    p_acts = sub.add_parser("activities", help="List recent Strava activities")
    p_acts.add_argument("--after", default=None, help="Start date filter (ISO, 7d/30d, epoch)")
    p_acts.add_argument("--before", default=None, help="End date filter (ISO, 7d/30d, epoch)")
    p_acts.add_argument("--sport-type", default=None, help="Filter by Strava sport_type")
    p_acts.add_argument("--limit", type=int, default=30, help="Max activities to return")
    p_acts.add_argument("--include-polyline", action="store_true", help="Include map polyline")
    p_acts.add_argument("--refresh", action="store_true", help="Bypass local cache")
    p_acts.add_argument("--json", action="store_true", help="Emit JSON output")

    p_act = sub.add_parser("activity", help="Show detailed Strava activity by ID")
    p_act.add_argument("activity_id", type=int, help="Strava activity ID")
    p_act.add_argument("--include-all-efforts", action="store_true", help="Include segment_efforts")
    p_act.add_argument("--include-polyline", action="store_true", help="Include map polyline")
    p_act.add_argument("--refresh", action="store_true", help="Bypass local cache")
    p_act.add_argument("--json", action="store_true", help="Emit JSON output")

    p_stats = sub.add_parser("stats", help="Show athlete stats (Everyone-visibility caveat)")
    p_stats.add_argument("--athlete-id", type=int, default=None, help="Strava athlete ID")
    p_stats.add_argument("--refresh", action="store_true", help="Bypass local cache")
    p_stats.add_argument("--json", action="store_true", help="Emit JSON output")

    return parser


def _emit_cli_error(
    code: str,
    message: str,
    details: dict[str, Any],
    *,
    as_json: bool,
) -> int:
    secrets_list: list[str] = []
    with suppress(Exception):
        secrets_list = mcp_server._collect_secrets(load_config())
    safe_msg = mcp_server._redact_text(message, secrets_list)
    safe_details = mcp_server._redact_value(details, secrets_list)
    if as_json:
        print(
            json.dumps(
                {"error": {"code": code, "message": safe_msg, "details": safe_details}},
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    else:
        print(f"Error [{code}]: {safe_msg}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    mcp_server.configure_stderr_logging(verbose=bool(args.verbose))
    as_json = bool(getattr(args, "json", False))

    try:
        cmd = args.command
        if cmd == "init":
            return _handle_init(args)
        if cmd == "auth":
            return asyncio.run(_handle_auth(args))
        if cmd == "status":
            return asyncio.run(_handle_status(args))
        if cmd == "refresh":
            return asyncio.run(_handle_refresh(args))
        if cmd == "revoke":
            return asyncio.run(_handle_revoke(args))
        if cmd == "stdio":
            return _handle_stdio(args)
        if cmd == "serve":
            return _handle_serve(args)
        if cmd == "cache":
            return _handle_cache(args)
        if cmd == "activities":
            return asyncio.run(_handle_activities(args))
        if cmd == "activity":
            return asyncio.run(_handle_activity(args))
        if cmd == "stats":
            return asyncio.run(_handle_stats(args))
        raise ValueError(f"Unknown command: {cmd}")
    except StravaMCPError as exc:
        return _emit_cli_error(exc.code, exc.message, exc.details, as_json=as_json)
    except (ValueError, OverflowError) as exc:
        return _emit_cli_error("invalid_argument", str(exc), {}, as_json=as_json)


if __name__ == "__main__":
    raise SystemExit(main())
