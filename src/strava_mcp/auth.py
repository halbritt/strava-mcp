"""OAuth2 token management, locking, loopback authorization, and scope utilities."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import inspect
import os
import re
import secrets
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from strava_mcp import __version__
from strava_mcp.config import (
    AuthError,
    Config,
    ConfigError,
    InsufficientScopeError,
    TokenData,
    delete_tokens,
    ensure_secure_dir,
    load_client,
    load_tokens,
    resolve_config_dir,
    resolve_oauth_base_url,
    save_tokens,
    verify_secure_permissions,
)

DEFAULT_SCOPE_STRING = "read,read_all,profile:read_all,activity:read_all"
CORE_RECOMMENDED_SCOPES: tuple[str, ...] = ("activity:read_all", "profile:read_all")
ALL_RECOMMENDED_SCOPES: tuple[str, ...] = (
    "read",
    "read_all",
    "profile:read_all",
    "activity:read_all",
)

DEFAULT_CALLBACK_HOST = "127.0.0.1"
DEFAULT_CALLBACK_PORT = 38762
DEFAULT_CALLBACK_PATH = "/callback"

REFRESH_BUFFER_SECONDS = 300
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_LOCK_TIMEOUT = 30.0
DEFAULT_USER_AGENT = f"strava-mcp/{__version__} (https://github.com/halbritt/strava-mcp)"

SCOPE_IMPLICATIONS: dict[str, set[str]] = {
    "activity:read_all": {"activity:read"},
}

_SCOPE_SPLIT_RE = re.compile(r"[,\s]+")


def parse_scopes(raw: str | Iterable[str] | None) -> list[str]:
    """Parse comma- or whitespace-delimited OAuth scopes preserving order and uniqueness."""
    if raw is None:
        return []
    tokens: list[str] = []
    if isinstance(raw, str):
        parts = _SCOPE_SPLIT_RE.split(raw.strip())
    else:
        parts = []
        for item in raw:
            parts.extend(_SCOPE_SPLIT_RE.split(str(item).strip()))
    for part in parts:
        cleaned = part.strip()
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def format_scopes(scopes: str | Iterable[str] | None) -> str:
    """Format scopes into Strava's comma-delimited scope string."""
    return ",".join(parse_scopes(scopes))


def _expand_scopes(granted: Iterable[str]) -> set[str]:
    expanded: set[str] = set(granted)
    for scope in list(expanded):
        expanded.update(SCOPE_IMPLICATIONS.get(scope, ()))
    return expanded


def has_scope(
    granted: str | Iterable[str] | None,
    required: str | Iterable[str],
) -> bool:
    """Return True if all required scopes are satisfied by granted scopes."""
    return len(missing_scopes(granted, required)) == 0


def missing_scopes(
    granted: str | Iterable[str] | None,
    required: str | Iterable[str],
) -> list[str]:
    """Return required scopes not satisfied by granted scopes, preserving order."""
    granted_expanded = _expand_scopes(parse_scopes(granted))
    return [scope for scope in parse_scopes(required) if scope not in granted_expanded]


def require_scope(
    granted: str | Iterable[str] | None,
    required: str | Iterable[str],
    *,
    context: str | None = None,
) -> None:
    """Raise InsufficientScopeError if any required scope is missing."""
    required_list = parse_scopes(required)
    granted_list = parse_scopes(granted)
    missing = missing_scopes(granted_list, required_list)
    if not missing:
        return
    prefix = f"{context}: " if context else ""
    message = (
        f"{prefix}Missing required Strava scope(s): {', '.join(missing)}. "
        "Run 'strava-mcp auth' to grant missing scopes."
    )
    raise InsufficientScopeError(
        message,
        required_scopes=required_list,
        granted_scopes=granted_list,
    )


@asynccontextmanager
async def _acquire_file_lock(
    lock_path: Path | str,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    poll_interval: float = 0.02,
):
    """Acquire an exclusive non-blocking fcntl.flock without ever unlinking the lock file."""
    lock_file = Path(lock_path)
    parent = lock_file.parent
    if parent.is_symlink() or parent.exists():
        verify_secure_permissions(parent)
        if not parent.is_dir():
            raise ConfigError(
                "insecure_permissions",
                f"Expected regular directory at {parent}",
                details={"path": str(parent)},
            )
    else:
        ensure_secure_dir(parent)
    if lock_file.is_symlink():
        raise ConfigError(
            "insecure_permissions",
            f"Refusing symlinked lock file: {lock_file}",
            details={"path": str(lock_file)},
        )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_file, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ConfigError(
                "insecure_permissions",
                f"Refusing symlinked lock file: {lock_file}",
                details={"path": str(lock_file)},
            ) from None
        raise
    acquired = False
    try:
        os.fchmod(fd, 0o600)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AuthError(
                        "refresh_failed",
                        "Timed out waiting for token lock.",
                        details={"lock_path": str(lock_file)},
                    ) from None
                await asyncio.sleep(poll_interval)
        yield
    finally:
        if acquired:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def build_authorize_url(
    client_id: str | int,
    *,
    redirect_uri: str,
    state: str,
    scope: str | Iterable[str] = DEFAULT_SCOPE_STRING,
    approval_prompt: str = "force",
    oauth_base_url: str | None = None,
) -> str:
    """Build a Strava OAuth2 authorization URL."""
    client_id_str = str(client_id).strip()
    if not client_id_str:
        raise ValueError("client_id must not be empty")
    if not redirect_uri.strip():
        raise ValueError("redirect_uri must not be empty")
    if not state.strip():
        raise ValueError("state must not be empty")
    if approval_prompt not in {"force", "auto"}:
        raise ValueError("approval_prompt must be 'force' or 'auto'")

    base = (oauth_base_url or resolve_oauth_base_url()).rstrip("/")
    params = {
        "client_id": client_id_str,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": approval_prompt,
        "scope": format_scopes(scope),
        "state": state,
    }
    return f"{base}/authorize?{urlencode(params)}"


def _extract_athlete_id(payload: dict[str, Any]) -> int | None:
    athlete = payload.get("athlete")
    if isinstance(athlete, dict) and athlete.get("id") is not None:
        raw_id = athlete["id"]
        if not isinstance(raw_id, bool):
            return int(raw_id)
    return None


def _extract_fault_summary(resp: httpx.Response, *, secrets_to_redact: Iterable[str]) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    cleaned = message.strip()
    for secret in secrets_to_redact:
        if secret and secret in cleaned:
            cleaned = cleaned.replace(secret, "[REDACTED]")
    return cleaned


class TokenManager:
    """Coroutine- and process-safe Strava OAuth2 token lifecycle manager."""

    def __init__(
        self,
        config_dir: Config | Path | str | None = None,
        *,
        oauth_base_url: str | None = None,
        refresh_margin_seconds: int = REFRESH_BUFFER_SECONDS,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if isinstance(config_dir, Config):
            self._config_dir: Path | None = config_dir.config_dir
            self._oauth_base_url = oauth_base_url or config_dir.oauth_base_url
        elif config_dir is None:
            self._config_dir = None
            self._oauth_base_url = oauth_base_url
        else:
            self._config_dir = Path(config_dir)
            self._oauth_base_url = oauth_base_url
        self.refresh_margin_seconds = refresh_margin_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self._http_client = http_client
        self._transport = transport
        self._clock = clock
        self._lock = asyncio.Lock()

    @property
    def config_dir(self) -> Path:
        return self._config_dir if self._config_dir is not None else resolve_config_dir()

    @property
    def lock_path(self) -> Path:
        return self.config_dir / "tokens.lock"

    @property
    def oauth_base_url(self) -> str:
        if self._oauth_base_url:
            return self._oauth_base_url.rstrip("/")
        return resolve_oauth_base_url()

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.time()

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        if self._http_client is not None:
            return await self._http_client.request(method, url, **kwargs)
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=DEFAULT_HTTP_TIMEOUT,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            return await client.request(method, url, **kwargs)

    async def _post_token(
        self,
        form_data: dict[str, str],
        *,
        secrets_to_redact: Iterable[str],
    ) -> dict[str, Any]:
        url = f"{self.oauth_base_url}/token"
        try:
            resp = await self._request("POST", url, data=form_data)
        except httpx.HTTPError as exc:
            raise AuthError(
                "refresh_failed",
                "Failed to contact Strava OAuth token endpoint.",
                details={"error_type": type(exc).__name__},
            ) from None

        if resp.status_code >= 400:
            details: dict[str, Any] = {"status_code": resp.status_code}
            fault = _extract_fault_summary(resp, secrets_to_redact=secrets_to_redact)
            if fault:
                details["strava_message"] = fault
            raise AuthError(
                "refresh_failed",
                f"Strava OAuth token request failed with HTTP {resp.status_code}.",
                details=details,
            )

        try:
            payload = resp.json()
        except ValueError:
            raise AuthError(
                "refresh_failed",
                "Invalid JSON response from Strava OAuth token endpoint.",
                details={"status_code": resp.status_code},
            ) from None

        if not isinstance(payload, dict):
            raise AuthError(
                "refresh_failed",
                "Unexpected response format from Strava OAuth token endpoint.",
                details={"status_code": resp.status_code},
            )
        return payload

    def _build_token_data(
        self,
        payload: dict[str, Any],
        *,
        fallback_scope: str = "",
        fallback_athlete_id: int | None = None,
        fallback_token_type: str = "Bearer",
    ) -> TokenData:
        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token")
        raw_expires_at = payload.get("expires_at")
        raw_expires_in = payload.get("expires_in")
        if not isinstance(access_token, str) or not access_token.strip():
            raise AuthError("refresh_failed", "OAuth response missing access_token.")
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            raise AuthError("refresh_failed", "OAuth response missing refresh_token.")

        now = int(self._now())
        expires_in: int | None = None
        if raw_expires_in is not None and not isinstance(raw_expires_in, bool):
            try:
                expires_in = int(raw_expires_in)
            except (TypeError, ValueError) as exc:
                raise AuthError(
                    "refresh_failed",
                    "OAuth response contained invalid expires_in.",
                    details={"error_type": type(exc).__name__},
                ) from None

        if raw_expires_at is not None and not isinstance(raw_expires_at, bool):
            try:
                expires_at = int(raw_expires_at)
            except (TypeError, ValueError) as exc:
                raise AuthError(
                    "refresh_failed",
                    "OAuth response contained invalid expires_at.",
                    details={"error_type": type(exc).__name__},
                ) from None
        elif expires_in is not None:
            expires_at = now + expires_in
        else:
            raise AuthError("refresh_failed", "OAuth response missing expires_at/expires_in.")

        scope = format_scopes(payload.get("scope")) or fallback_scope
        athlete_id = _extract_athlete_id(payload)
        if athlete_id is None:
            athlete_id = fallback_athlete_id

        token_type_raw = payload.get("token_type")
        token_type = (
            token_type_raw.strip()
            if isinstance(token_type_raw, str) and token_type_raw.strip()
            else fallback_token_type
        )

        return TokenData(
            access_token=access_token.strip(),
            refresh_token=refresh_token.strip(),
            expires_at=expires_at,
            expires_in=expires_in,
            token_type=token_type,
            scope=scope,
            athlete_id=athlete_id,
            updated_at=now,
        )

    def _validate_existing_config_dir(self, *, require_exists: bool) -> None:
        config_dir = self.config_dir
        if require_exists:
            verify_secure_permissions(
                config_dir,
                missing_error=ConfigError,
                missing_code="config_missing",
                missing_message=f"Configuration directory does not exist: {config_dir}",
            )
        elif config_dir.is_symlink() or config_dir.exists():
            verify_secure_permissions(config_dir)
        else:
            return
        if not config_dir.is_dir():
            raise ConfigError(
                "insecure_permissions",
                f"Expected regular directory at {config_dir}",
                details={"path": str(config_dir)},
            )

    async def get_valid_tokens(
        self,
        *,
        force: bool = False,
        if_access_token_matches: str | None = None,
    ) -> TokenData:
        """Return valid TokenData, refreshing under lock when expired, near expiry, or forced."""
        self._validate_existing_config_dir(require_exists=True)
        async with (
            self._lock,
            _acquire_file_lock(self.lock_path, timeout=self.lock_timeout_seconds),
        ):
            client = load_client(self.config_dir)
            tokens = load_tokens(self.config_dir)
            now = int(self._now())

            if (
                if_access_token_matches is not None
                and tokens.access_token != if_access_token_matches
                and (tokens.expires_at - now) > self.refresh_margin_seconds
            ):
                return tokens

            needs_refresh = (
                force
                or (
                    if_access_token_matches is not None
                    and tokens.access_token == if_access_token_matches
                )
                or (tokens.expires_at - now <= self.refresh_margin_seconds)
            )
            if not needs_refresh:
                return tokens

            payload = await self._post_token(
                {
                    "client_id": client.client_id,
                    "client_secret": client.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                },
                secrets_to_redact=(
                    client.client_secret,
                    tokens.refresh_token,
                    tokens.access_token,
                ),
            )
            new_tokens = self._build_token_data(
                payload,
                fallback_scope=tokens.scope,
                fallback_athlete_id=tokens.athlete_id,
                fallback_token_type=tokens.token_type,
            )
            save_tokens(self.config_dir, new_tokens)
            return new_tokens

    async def get_valid_token(
        self,
        *,
        force: bool = False,
        if_access_token_matches: str | None = None,
    ) -> str:
        """Return a valid Strava access token string."""
        tokens = await self.get_valid_tokens(
            force=force,
            if_access_token_matches=if_access_token_matches,
        )
        return tokens.access_token

    async def exchange_code(
        self,
        code: str,
        *,
        callback_scope: str | None = None,
    ) -> TokenData:
        """Exchange an OAuth2 authorization code for tokens and persist them securely."""
        cleaned_code = code.strip()
        if not cleaned_code:
            raise AuthError("auth_missing", "Authorization code must not be empty.")

        self._validate_existing_config_dir(require_exists=False)
        async with (
            self._lock,
            _acquire_file_lock(self.lock_path, timeout=self.lock_timeout_seconds),
        ):
            client = load_client(self.config_dir)
            payload = await self._post_token(
                {
                    "client_id": client.client_id,
                    "client_secret": client.client_secret,
                    "code": cleaned_code,
                    "grant_type": "authorization_code",
                },
                secrets_to_redact=(client.client_secret, cleaned_code),
            )
            new_tokens = self._build_token_data(
                payload,
                fallback_scope=format_scopes(callback_scope),
            )
            save_tokens(self.config_dir, new_tokens)
            return new_tokens

    async def revoke(
        self,
        *,
        token_type_hint: Literal["access_token", "refresh_token"] = "refresh_token",
        clear_local: bool = True,
    ) -> dict[str, Any]:
        """Revoke the stored Strava token via POST /oauth/revoke and delete local tokens on 200."""
        self._validate_existing_config_dir(require_exists=True)
        async with (
            self._lock,
            _acquire_file_lock(self.lock_path, timeout=self.lock_timeout_seconds),
        ):
            client = load_client(self.config_dir)
            tokens = load_tokens(self.config_dir)
            token_value = (
                tokens.refresh_token
                if token_type_hint == "refresh_token"
                else tokens.access_token
            )
            url = f"{self.oauth_base_url}/revoke"
            try:
                resp = await self._request(
                    "POST",
                    url,
                    auth=(client.client_id, client.client_secret),
                    data={
                        "token": token_value,
                        "token_type_hint": token_type_hint,
                    },
                )
            except httpx.HTTPError as exc:
                raise AuthError(
                    "revoke_failed",
                    "Failed to contact Strava OAuth revoke endpoint.",
                    details={"error_type": type(exc).__name__},
                ) from None

            if resp.status_code != 200:
                details: dict[str, Any] = {"status_code": resp.status_code}
                fault = _extract_fault_summary(
                    resp,
                    secrets_to_redact=(
                        client.client_secret,
                        tokens.access_token,
                        tokens.refresh_token,
                    ),
                )
                if fault:
                    details["strava_message"] = fault
                raise AuthError(
                    "revoke_failed",
                    f"Strava token revocation failed with HTTP {resp.status_code}.",
                    details=details,
                )

            local_cleared = delete_tokens(self.config_dir) if clear_local else False
            return {
                "revoked": True,
                "token_type_hint": token_type_hint,
                "local_cleared": local_cleared,
            }


async def revoke_token(
    config_dir: Config | Path | str | None = None,
    *,
    token_type_hint: Literal["access_token", "refresh_token"] = "refresh_token",
    clear_local: bool = True,
    oauth_base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Convenience helper to revoke the stored Strava token."""
    manager = TokenManager(
        config_dir,
        oauth_base_url=oauth_base_url,
        http_client=http_client,
        transport=transport,
    )
    return await manager.revoke(token_type_hint=token_type_hint, clear_local=clear_local)


class OAuthCallbackState:
    """Single-use, constant-time CSRF-checked OAuth2 callback state holder."""

    def __init__(
        self,
        expected_state: str,
        *,
        callback_path: str = DEFAULT_CALLBACK_PATH,
        ttl_seconds: float = 600.0,
        loop: asyncio.AbstractEventLoop | None = None,
        async_event: asyncio.Event | None = None,
    ) -> None:
        self.expected_state: str | None = expected_state
        self.callback_path = callback_path
        self.ttl_seconds = ttl_seconds
        self.started_monotonic = time.monotonic()
        self._lock = threading.Lock()
        self.code: str | None = None
        self.callback_scope: str = ""
        self.error: AuthError | None = None
        self.done = threading.Event()
        self._loop = loop
        self._async_event = async_event

    def _signal_done(self) -> None:
        self.done.set()
        if self._loop is not None and self._async_event is not None:
            with suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._async_event.set)

    def handle_callback_query(self, query: dict[str, list[str]]) -> tuple[int, str]:
        """Validate and consume single-use OAuth callback query parameters."""
        with self._lock:
            states = query.get("state", [])
            active_state = self.expected_state
            within_ttl = (time.monotonic() - self.started_monotonic) <= self.ttl_seconds
            state_valid = (
                active_state is not None
                and within_ttl
                and len(states) == 1
                and secrets.compare_digest(states[0], active_state)
            )
            # Single-use clearance: any callback-path attempt consumes expected_state.
            self.expected_state = None

            if not state_valid:
                if self.code is None and self.error is None:
                    self.error = AuthError(
                        "oauth_state_mismatch",
                        "Invalid, expired, or already-used OAuth state.",
                    )
                self._signal_done()
                return (400, "Authorization failed: invalid or expired OAuth state.")

            errors = query.get("error", [])
            if errors:
                oauth_error = errors[0]
                if self.code is None and self.error is None:
                    self.error = AuthError(
                        "access_denied",
                        f"Strava authorization was denied ({oauth_error}).",
                        details={"oauth_error": oauth_error},
                    )
                self._signal_done()
                return (400, "Strava authorization was denied. You can close this window.")

            codes = query.get("code", [])
            if len(codes) != 1 or not codes[0].strip():
                if self.code is None and self.error is None:
                    self.error = AuthError(
                        "auth_missing",
                        "OAuth callback did not include an authorization code.",
                    )
                self._signal_done()
                return (400, "Authorization failed: missing authorization code.")

            self.code = codes[0].strip()
            self.callback_scope = format_scopes(query.get("scope", [""])[0])
            self._signal_done()
            return (200, "Strava authorization complete. You can close this window.")


class _ReuseHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request: object, client_address: object) -> None:
        # Never dump raw request errors or tracebacks to stderr on client timeout/disconnect.
        pass


def _make_callback_handler(
    state_box: OAuthCallbackState,
    *,
    request_timeout: float = 2.0,
) -> type[BaseHTTPRequestHandler]:
    class _CallbackHandler(BaseHTTPRequestHandler):
        timeout = request_timeout

        def log_message(self, format: str, *args: object) -> None:
            # Silence HTTP access logs so query strings containing code/state are never logged.
            pass

        def handle(self) -> None:
            with suppress(TimeoutError, ConnectionResetError, BrokenPipeError, OSError):
                super().handle()

        def _respond(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path != state_box.callback_path:
                self._respond(404, "Not found.")
                return
            query = parse_qs(parsed.query, keep_blank_values=True)
            status, message = state_box.handle_callback_query(query)
            self._respond(status, message)

    return _CallbackHandler


class LoopbackAuthServer:
    """Ephemeral 127.0.0.1 HTTP server for OAuth2 authorization code callbacks."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_CALLBACK_HOST,
        port: int = DEFAULT_CALLBACK_PORT,
        callback_path: str = DEFAULT_CALLBACK_PATH,
        expected_state: str | None = None,
        ttl_seconds: float = 600.0,
        request_timeout: float = 2.0,
        loop: asyncio.AbstractEventLoop | None = None,
        async_event: asyncio.Event | None = None,
    ) -> None:
        normalized_path = callback_path or "/"
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
        self.host = host
        self.callback_path = normalized_path
        resolved_state = expected_state or secrets.token_urlsafe(32)
        self.state = resolved_state
        self.state_box = OAuthCallbackState(
            resolved_state,
            callback_path=normalized_path,
            ttl_seconds=ttl_seconds,
            loop=loop,
            async_event=async_event,
        )
        handler_cls = _make_callback_handler(
            self.state_box,
            request_timeout=request_timeout,
        )
        self._server = _ReuseHTTPServer((host, port), handler_cls)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    @property
    def redirect_uri(self) -> str:
        netloc_host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{netloc_host}:{self.port}{self.callback_path}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> LoopbackAuthServer:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True)
class AuthResult:
    """Result returned by run_loopback_auth."""

    tokens: TokenData
    authorize_url: str
    granted_scopes: list[str]
    missing_recommended_scopes: list[str]
    warnings: list[str]


async def run_loopback_auth(
    token_manager: TokenManager,
    *,
    scope: str | Iterable[str] = DEFAULT_SCOPE_STRING,
    port: int = DEFAULT_CALLBACK_PORT,
    redirect_uri: str | None = None,
    approval_prompt: str = "force",
    timeout_seconds: float = 300.0,
    on_authorize_url: Callable[[str], Awaitable[None] | None] | None = None,
) -> AuthResult:
    """Run the loopback OAuth2 authorization code flow and exchange for persisted tokens."""
    client = load_client(token_manager.config_dir)

    if redirect_uri is not None:
        parsed_redirect = urlsplit(redirect_uri)
        scheme = parsed_redirect.scheme or "http"
        bind_host = parsed_redirect.hostname or DEFAULT_CALLBACK_HOST
        callback_path = parsed_redirect.path or "/"
        if not callback_path.startswith("/"):
            callback_path = f"/{callback_path}"
        bind_port = (
            0
            if port == 0
            else (parsed_redirect.port if parsed_redirect.port is not None else port)
        )
    else:
        scheme = "http"
        bind_host = DEFAULT_CALLBACK_HOST
        callback_path = DEFAULT_CALLBACK_PATH
        bind_port = port

    loop = asyncio.get_running_loop()
    async_event = asyncio.Event()
    state = secrets.token_urlsafe(32)

    server = LoopbackAuthServer(
        host=bind_host,
        port=bind_port,
        callback_path=callback_path,
        expected_state=state,
        loop=loop,
        async_event=async_event,
    )
    try:
        netloc_host = f"[{bind_host}]" if ":" in bind_host else bind_host
        effective_redirect_uri = f"{scheme}://{netloc_host}:{server.port}{callback_path}"
        authorize_url = build_authorize_url(
            client.client_id,
            redirect_uri=effective_redirect_uri,
            state=state,
            scope=scope,
            approval_prompt=approval_prompt,
            oauth_base_url=token_manager.oauth_base_url,
        )
        if on_authorize_url is not None:
            callback_result = on_authorize_url(authorize_url)
            if inspect.isawaitable(callback_result):
                await callback_result

        try:
            await asyncio.wait_for(async_event.wait(), timeout=timeout_seconds)
        except TimeoutError:
            raise AuthError(
                "auth_expired",
                "Timed out waiting for OAuth authorization callback.",
                details={"timeout_seconds": timeout_seconds},
            ) from None

        if server.state_box.error is not None:
            raise server.state_box.error
        if not server.state_box.code:
            raise AuthError(
                "auth_missing",
                "OAuth callback did not yield an authorization code.",
            )

        tokens = await token_manager.exchange_code(
            server.state_box.code,
            callback_scope=server.state_box.callback_scope,
        )
        missing = missing_scopes(tokens.scope, CORE_RECOMMENDED_SCOPES)
        warnings = [
            (
                f"Recommended Strava scope '{missing_scope}' was not granted; "
                "related athlete or activity data will be unavailable. "
                "Re-run 'strava-mcp auth' and approve it to enable full access."
            )
            for missing_scope in missing
        ]
        return AuthResult(
            tokens=tokens,
            authorize_url=authorize_url,
            granted_scopes=parse_scopes(tokens.scope),
            missing_recommended_scopes=missing,
            warnings=warnings,
        )
    finally:
        server.close()
