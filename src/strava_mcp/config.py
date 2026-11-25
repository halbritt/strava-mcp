"""Configuration, secure secret storage, and typed error models for strava-mcp."""

from __future__ import annotations

import json
import os
import secrets
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_API_BASE_URL = "https://www.strava.com/api/v3"
DEFAULT_OAUTH_BASE_URL = "https://www.strava.com/oauth"

ENV_CONFIG_DIR = "STRAVA_MCP_CONFIG_DIR"
ENV_CACHE_DIR = "STRAVA_MCP_CACHE_DIR"
ENV_API_BASE_URL = "STRAVA_API_BASE_URL"
ENV_OAUTH_BASE_URL = "STRAVA_OAUTH_BASE_URL"


class StravaMCPError(Exception):
    """Base structured error for all strava-mcp operations."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details) if details is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize error into a structured JSON-safe payload."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": dict(self.details),
            }
        }


class ConfigError(StravaMCPError):
    """Configuration or secret store permission error."""


class AuthError(StravaMCPError):
    """OAuth2 authentication, token refresh, or revocation error."""


class InsufficientScopeError(StravaMCPError):
    """Token lacks required Strava OAuth scope(s)."""

    def __init__(
        self,
        message: str | None = None,
        *,
        required_scopes: Iterable[str] = (),
        granted_scopes: Iterable[str] = (),
        remediation: str = "strava-mcp auth",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.required_scopes = list(required_scopes)
        self.granted_scopes = list(granted_scopes)
        self.remediation = remediation
        merged_details: dict[str, Any] = {
            "required_scopes": self.required_scopes,
            "granted_scopes": self.granted_scopes,
            "remediation": self.remediation,
        }
        if details:
            merged_details.update(details)
        resolved_message = message or (
            f"Missing required Strava scope(s): {', '.join(self.required_scopes) or 'unknown'}. "
            f"Run '{self.remediation}' to authorize."
        )
        super().__init__("insufficient_scope", resolved_message, details=merged_details)


class RateLimitError(StravaMCPError):
    """Strava API rate limit exceeded."""

    def __init__(
        self,
        message: str = "Strava API rate limit exceeded.",
        *,
        limit_type: str | None = None,
        usage: int | None = None,
        limit: int | None = None,
        reset_at_utc: str | None = None,
        retry_after_seconds: int | float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.limit_type = limit_type
        self.usage = usage
        self.limit = limit
        self.reset_at_utc = reset_at_utc
        self.retry_after_seconds = retry_after_seconds
        merged_details: dict[str, Any] = {
            "limit_type": limit_type,
            "usage": usage,
            "limit": limit,
            "reset_at_utc": reset_at_utc,
            "retry_after_seconds": retry_after_seconds,
        }
        if details:
            merged_details.update(details)
        super().__init__("rate_limited", message, details=merged_details)


class SubscriptionRequiredError(StravaMCPError):
    """Strava feature requires an active Strava subscription."""

    def __init__(
        self,
        message: str = "This Strava endpoint requires a Strava subscription.",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("subscription_required", message, details=details)


class TierRestrictedError(StravaMCPError):
    """Endpoint or athlete access restricted by Strava API application tier."""

    def __init__(
        self,
        message: str = "This Strava resource is restricted by API application tier.",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("tier_restricted", message, details=details)


class NotFoundError(StravaMCPError):
    """Requested Strava resource was not found."""

    def __init__(
        self,
        message: str = "Requested Strava resource was not found.",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("not_found", message, details=details)


class UpstreamAPIError(StravaMCPError):
    """Unexpected upstream Strava HTTP error."""

    def __init__(
        self,
        message: str = "Strava API request failed.",
        *,
        status_code: int = 500,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        merged_details: dict[str, Any] = {"status_code": status_code}
        if details:
            merged_details.update(details)
        super().__init__("upstream_error", message, details=merged_details)


def resolve_config_dir() -> Path:
    """Resolve the configuration directory dynamically from environment or XDG default."""
    raw = os.environ.get(ENV_CONFIG_DIR, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".config" / "strava-mcp"


def resolve_cache_dir() -> Path:
    """Resolve the cache directory dynamically from environment or XDG default."""
    raw = os.environ.get(ENV_CACHE_DIR, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share" / "strava-mcp"


def resolve_api_base_url() -> str:
    """Resolve the Strava REST API v3 base URL."""
    raw = os.environ.get(ENV_API_BASE_URL, "").strip()
    return (raw or DEFAULT_API_BASE_URL).rstrip("/")


def resolve_oauth_base_url() -> str:
    """Resolve the Strava OAuth base URL."""
    raw = os.environ.get(ENV_OAUTH_BASE_URL, "").strip()
    return (raw or DEFAULT_OAUTH_BASE_URL).rstrip("/")


@dataclass(frozen=True)
class Config:
    """Runtime path and endpoint configuration (contains zero secret values)."""

    config_dir: Path
    cache_dir: Path
    api_base_url: str
    oauth_base_url: str

    @property
    def client_file(self) -> Path:
        return self.config_dir / "client.json"

    @property
    def tokens_file(self) -> Path:
        return self.config_dir / "tokens.json"

    @property
    def mcp_token_file(self) -> Path:
        return self.config_dir / "mcp-token"

    @property
    def lock_file(self) -> Path:
        return self.config_dir / "tokens.lock"

    @property
    def cache_db_file(self) -> Path:
        return self.cache_dir / "cache.sqlite"


def load_config() -> Config:
    """Build runtime Config dynamically from current environment variables."""
    return Config(
        config_dir=resolve_config_dir(),
        cache_dir=resolve_cache_dir(),
        api_base_url=resolve_api_base_url(),
        oauth_base_url=resolve_oauth_base_url(),
    )


def ensure_secure_dir(path: Path | str) -> Path:
    """Create directory if needed and enforce 0700 permissions without following symlinks."""
    target = Path(path)
    if target.is_symlink():
        raise ConfigError(
            "insecure_permissions",
            f"Refusing symlinked directory: {target}",
            details={"path": str(target)},
        )
    if target.exists() and not target.is_dir():
        raise ConfigError(
            "insecure_permissions",
            f"Expected regular directory at {target}",
            details={"path": str(target)},
        )
    try:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
    except (FileExistsError, NotADirectoryError):
        raise ConfigError(
            "insecure_permissions",
            f"Expected regular directory at {target}",
            details={"path": str(target)},
        ) from None
    if target.is_symlink() or not target.is_dir():
        raise ConfigError(
            "insecure_permissions",
            f"Expected regular directory at {target}",
            details={"path": str(target)},
        )
    os.chmod(target, 0o700)
    return target


def verify_secure_permissions(
    path: Path | str,
    *,
    missing_error: type[StravaMCPError] = ConfigError,
    missing_code: str = "config_missing",
    missing_message: str | None = None,
) -> None:
    """Verify path exists, is not a symlink, and has no group/other permissions."""
    target = Path(path)
    if target.is_symlink():
        raise ConfigError(
            "insecure_permissions",
            f"Refusing symlink at secure path: {target}",
            details={"path": str(target)},
        )
    if not target.exists():
        raise missing_error(
            missing_code,
            missing_message or f"Required path does not exist: {target}",
            details={"path": str(target)},
        )
    mode = stat.S_IMODE(target.lstat().st_mode)
    if mode & 0o077 != 0:
        raise ConfigError(
            "insecure_permissions",
            f"Insecure permissions ({oct(mode)}) on {target}; group/other access must be 0",
            details={"path": str(target), "mode": oct(mode)},
        )


def write_secret_text(path: Path | str, content: str) -> None:
    """Atomically write secret text to a 0600 file inside a 0700 directory."""
    target = Path(path)
    ensure_secure_dir(target.parent)
    if target.is_symlink():
        raise ConfigError(
            "insecure_permissions",
            f"Refusing to overwrite symlink at secret path: {target}",
            details={"path": str(target)},
        )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    tmp_path = Path(tmp_name)
    closed = False
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            closed = True
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if not closed:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise


def write_secret_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically serialize JSON payload to a 0600 secret file."""
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    write_secret_text(path, serialized)


class ClientConfig(BaseModel):
    """Strava OAuth2 API application credentials."""

    model_config = ConfigDict(extra="ignore")

    client_id: str
    client_secret: str = Field(repr=False)

    @field_validator("client_id", mode="before")
    @classmethod
    def _validate_client_id(cls, value: Any) -> str:
        if isinstance(value, bool) or value is None:
            raise ValueError("client_id must be a non-empty string or integer")
        if isinstance(value, int):
            text = str(value)
        elif isinstance(value, str):
            text = value.strip()
        else:
            raise ValueError("client_id must be a string or integer")
        if not text:
            raise ValueError("client_id must not be empty")
        return text

    @field_validator("client_secret")
    @classmethod
    def _validate_client_secret(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("client_secret must not be empty")
        return text


class TokenData(BaseModel):
    """Strava OAuth2 access/refresh token state."""

    model_config = ConfigDict(extra="ignore")

    access_token: str = Field(repr=False)
    refresh_token: str = Field(repr=False)
    expires_at: int
    expires_in: int | None = None
    token_type: str = "Bearer"
    scope: str = ""
    athlete_id: int | None = None
    updated_at: int | None = None

    @field_validator("access_token", "refresh_token")
    @classmethod
    def _validate_non_empty_secret(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("token field must not be empty")
        return text

    @field_validator("expires_at", mode="before")
    @classmethod
    def _validate_expires_at(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("expires_at must be an integer epoch timestamp")
        return int(value)

    @property
    def scopes(self) -> tuple[str, ...]:
        """Return deduplicated scopes parsed from comma/whitespace-delimited scope string."""
        seen: list[str] = []
        for token in self.scope.replace(",", " ").split():
            cleaned = token.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return tuple(seen)


def _resolve_config_dir_arg(config_or_dir: Config | Path | str | None) -> Path:
    if config_or_dir is None:
        return resolve_config_dir()
    if isinstance(config_or_dir, Config):
        return config_or_dir.config_dir
    return Path(config_or_dir)


def load_client(config_or_dir: Config | Path | str) -> ClientConfig:
    """Load and validate client.json after checking 0700/0600 permissions."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    verify_secure_permissions(
        config_dir,
        missing_error=ConfigError,
        missing_code="config_missing",
        missing_message=f"Configuration directory does not exist: {config_dir}",
    )
    path = config_dir / "client.json"
    verify_secure_permissions(
        path,
        missing_error=ConfigError,
        missing_code="config_missing",
        missing_message=f"Client configuration file not found: {path}. Run 'strava-mcp init'.",
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ClientConfig.model_validate(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(
            "invalid_config",
            f"Invalid client configuration in {path}",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from None


load_client_config = load_client


def save_client(config_or_dir: Config | Path | str, client: ClientConfig) -> Path:
    """Atomically persist ClientConfig to client.json with 0600 permissions."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    path = config_dir / "client.json"
    write_secret_json(path, client.model_dump(mode="json"))
    return path


def load_tokens(config_or_dir: Config | Path | str) -> TokenData:
    """Load and validate tokens.json after checking 0700/0600 permissions."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    verify_secure_permissions(
        config_dir,
        missing_error=AuthError,
        missing_code="auth_missing",
        missing_message=f"Authentication directory does not exist: {config_dir}",
    )
    path = config_dir / "tokens.json"
    verify_secure_permissions(
        path,
        missing_error=AuthError,
        missing_code="auth_missing",
        missing_message=(
            f"Tokens file not found: {path}. Run 'strava-mcp auth' or 'strava-mcp init'."
        ),
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TokenData.model_validate(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(
            "invalid_config",
            f"Invalid tokens configuration in {path}",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from None


def save_tokens(config_or_dir: Config | Path | str, tokens: TokenData) -> Path:
    """Atomically persist TokenData to tokens.json with 0600 permissions."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    path = config_dir / "tokens.json"
    write_secret_json(path, tokens.model_dump(mode="json"))
    return path


def delete_tokens(config_or_dir: Config | Path | str) -> bool:
    """Delete tokens.json if present without following symlinks."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    path = config_dir / "tokens.json"
    if path.is_symlink():
        raise ConfigError(
            "insecure_permissions",
            f"Refusing to delete symlinked tokens file: {path}",
            details={"path": str(path)},
        )
    if path.exists():
        path.unlink()
        return True
    return False


def load_mcp_token(config_or_dir: Config | Path | str) -> str:
    """Load the local Streamable HTTP bearer token after checking 0700/0600 permissions."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    verify_secure_permissions(
        config_dir,
        missing_error=ConfigError,
        missing_code="config_missing",
        missing_message=f"Configuration directory does not exist: {config_dir}",
    )
    path = config_dir / "mcp-token"
    verify_secure_permissions(
        path,
        missing_error=ConfigError,
        missing_code="config_missing",
        missing_message=f"MCP bearer token file not found: {path}. Run 'strava-mcp init'.",
    )
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(
            "invalid_config",
            f"Failed to read MCP bearer token from {path}",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from None
    if not token:
        raise ConfigError(
            "invalid_config",
            f"MCP bearer token file is empty: {path}",
            details={"path": str(path)},
        )
    return token


def ensure_mcp_token(config_or_dir: Config | Path | str) -> str:
    """Return existing valid mcp-token or atomically generate a new 0600 bearer token."""
    config_dir = _resolve_config_dir_arg(config_or_dir)
    path = config_dir / "mcp-token"
    if path.is_symlink() or path.exists():
        return load_mcp_token(config_dir)
    if config_dir.exists() or config_dir.is_symlink():
        verify_secure_permissions(config_dir)
    token = secrets.token_urlsafe(32)
    write_secret_text(path, token + "\n")
    return token
