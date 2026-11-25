"""Unit tests for strava_mcp.config."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from strava_mcp.config import (
    AuthError,
    ClientConfig,
    Config,
    ConfigError,
    InsufficientScopeError,
    NotFoundError,
    RateLimitError,
    StravaMCPError,
    SubscriptionRequiredError,
    TierRestrictedError,
    TokenData,
    UpstreamAPIError,
    delete_tokens,
    ensure_mcp_token,
    ensure_secure_dir,
    load_client,
    load_client_config,
    load_config,
    load_mcp_token,
    load_tokens,
    save_client,
    save_tokens,
)


def _clear_strava_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "STRAVA_MCP_CONFIG_DIR",
        "STRAVA_MCP_CACHE_DIR",
        "STRAVA_API_BASE_URL",
        "STRAVA_OAUTH_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_config_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_strava_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = load_config()
    assert cfg.config_dir == tmp_path / ".config" / "strava-mcp"
    assert cfg.cache_dir == tmp_path / ".local" / "share" / "strava-mcp"
    assert cfg.api_base_url == "https://www.strava.com/api/v3"
    assert cfg.oauth_base_url == "https://www.strava.com/oauth"
    assert cfg.client_file == cfg.config_dir / "client.json"
    assert cfg.tokens_file == cfg.config_dir / "tokens.json"
    assert cfg.mcp_token_file == cfg.config_dir / "mcp-token"
    assert cfg.lock_file == cfg.config_dir / "tokens.lock"
    assert cfg.cache_db_file == cfg.cache_dir / "cache.sqlite"


def test_load_config_env_overrides_are_dynamic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_strava_env(monkeypatch)
    cfg1 = tmp_path / "cfg1"
    cache1 = tmp_path / "cache1"
    monkeypatch.setenv("STRAVA_MCP_CONFIG_DIR", str(cfg1))
    monkeypatch.setenv("STRAVA_MCP_CACHE_DIR", str(cache1))
    monkeypatch.setenv("STRAVA_API_BASE_URL", "http://127.0.0.1:9001/api/v3/")
    monkeypatch.setenv("STRAVA_OAUTH_BASE_URL", "http://127.0.0.1:9001/oauth/")

    loaded1 = load_config()
    assert loaded1.config_dir == cfg1
    assert loaded1.cache_dir == cache1
    assert loaded1.api_base_url == "http://127.0.0.1:9001/api/v3"
    assert loaded1.oauth_base_url == "http://127.0.0.1:9001/oauth"

    cfg2 = tmp_path / "cfg2"
    monkeypatch.setenv("STRAVA_MCP_CONFIG_DIR", str(cfg2))
    loaded2 = load_config()
    assert loaded2.config_dir == cfg2


def test_secret_env_vars_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_strava_env(monkeypatch)
    config_dir = tmp_path / "isolated_cfg"
    monkeypatch.setenv("STRAVA_MCP_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("STRAVA_CLIENT_SECRET", "env_secret_should_be_ignored")
    monkeypatch.setenv("STRAVA_ACCESS_TOKEN", "env_access_should_be_ignored")
    monkeypatch.setenv("STRAVA_REFRESH_TOKEN", "env_refresh_should_be_ignored")
    monkeypatch.setenv("STRAVA_MCP_TOKEN", "env_mcp_should_be_ignored")

    cfg = load_config()
    assert cfg.config_dir == config_dir

    with pytest.raises(ConfigError) as exc_client:
        load_client(cfg)
    assert exc_client.value.code == "config_missing"

    with pytest.raises(AuthError) as exc_tokens:
        load_tokens(cfg)
    assert exc_tokens.value.code == "auth_missing"

    with pytest.raises(ConfigError) as exc_mcp:
        load_mcp_token(cfg)
    assert exc_mcp.value.code == "config_missing"


def test_secure_dir_and_atomic_secret_writes(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    cfg = Config(
        config_dir=config_dir,
        cache_dir=tmp_path / "cache",
        api_base_url="https://www.strava.com/api/v3",
        oauth_base_url="https://www.strava.com/oauth",
    )
    client = ClientConfig(client_id="12345", client_secret="fake-client-secret-value")
    tokens = TokenData(
        access_token="fake-access-token",
        refresh_token="fake-refresh-token",
        expires_at=1_900_000_000,
        expires_in=21600,
        scope="read,activity:read_all",
        athlete_id=777,
        updated_at=1_899_990_000,
    )

    client_path = save_client(cfg, client)
    tokens_path = save_tokens(config_dir, tokens)

    assert stat.S_IMODE(config_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(client_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tokens_path.stat().st_mode) == 0o600
    assert [p.name for p in config_dir.iterdir() if ".tmp" in p.name] == []

    loaded_client = load_client_config(config_dir)
    loaded_tokens = load_tokens(cfg)
    assert loaded_client == client
    assert loaded_tokens == tokens
    assert loaded_tokens.scopes == ("read", "activity:read_all")


def test_insecure_permissions_and_symlinks_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    ensure_secure_dir(config_dir)
    client = ClientConfig(client_id="101", client_secret="secret-xyz")
    client_path = save_client(config_dir, client)
    tokens_path = save_tokens(
        config_dir,
        TokenData(access_token="acc-1", refresh_token="ref-1", expires_at=1_900_000_000),
    )

    os.chmod(client_path, 0o644)
    with pytest.raises(ConfigError) as exc_info:
        load_client(config_dir)
    assert exc_info.value.code == "insecure_permissions"

    os.chmod(client_path, 0o600)
    os.chmod(tokens_path, 0o640)
    with pytest.raises(ConfigError) as exc_tok:
        load_tokens(config_dir)
    assert exc_tok.value.code == "insecure_permissions"

    os.chmod(tokens_path, 0o600)
    os.chmod(config_dir, 0o755)
    with pytest.raises(ConfigError) as exc_dir:
        load_client(config_dir)
    assert exc_dir.value.code == "insecure_permissions"

    os.chmod(config_dir, 0o700)
    client_path.unlink()
    target = tmp_path / "real-client.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    client_path.symlink_to(target)
    with pytest.raises(ConfigError) as exc_sym:
        load_client(config_dir)
    assert exc_sym.value.code == "insecure_permissions"

    sym_dir = tmp_path / "sym-dir"
    sym_dir.symlink_to(config_dir)
    with pytest.raises(ConfigError) as exc_sym_dir:
        ensure_secure_dir(sym_dir)
    assert exc_sym_dir.value.code == "insecure_permissions"


def test_missing_and_invalid_configs_do_not_leak_secrets(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    ensure_secure_dir(config_dir)

    with pytest.raises(ConfigError) as exc_missing:
        load_client(config_dir)
    assert exc_missing.value.code == "config_missing"

    with pytest.raises(AuthError) as exc_auth_missing:
        load_tokens(config_dir)
    assert exc_auth_missing.value.code == "auth_missing"

    secret_marker = "DO_NOT_LEAK_CLIENT_SECRET_12345"
    bad_client = config_dir / "client.json"
    bad_client.write_text(
        json.dumps({"client_id": "", "client_secret": secret_marker}),
        encoding="utf-8",
    )
    os.chmod(bad_client, 0o600)
    with pytest.raises(ConfigError) as exc_invalid_client:
        load_client(config_dir)
    assert exc_invalid_client.value.code == "invalid_config"
    assert secret_marker not in str(exc_invalid_client.value)
    assert secret_marker not in json.dumps(exc_invalid_client.value.to_dict())

    token_marker = "DO_NOT_LEAK_TOKEN_99999"
    bad_tokens = config_dir / "tokens.json"
    bad_tokens.write_text(f"{{not-valid-json:{token_marker}", encoding="utf-8")
    os.chmod(bad_tokens, 0o600)
    with pytest.raises(ConfigError) as exc_invalid_tokens:
        load_tokens(config_dir)
    assert exc_invalid_tokens.value.code == "invalid_config"
    assert token_marker not in str(exc_invalid_tokens.value)
    assert token_marker not in json.dumps(exc_invalid_tokens.value.to_dict())


def test_models_coerce_and_redact_repr() -> None:
    client = ClientConfig.model_validate({"client_id": 293000, "client_secret": "super-secret-val"})
    assert client.client_id == "293000"
    assert "super-secret-val" not in repr(client)
    assert "super-secret-val" not in str(client)

    tokens = TokenData(
        access_token="secret-access-abc",
        refresh_token="secret-refresh-def",
        expires_at=1_900_000_000,
        scope="read, read_all , activity:read_all, read",
    )
    assert tokens.scopes == ("read", "read_all", "activity:read_all")
    assert "secret-access-abc" not in repr(tokens)
    assert "secret-refresh-def" not in repr(tokens)
    assert "secret-access-abc" not in str(tokens)
    assert "secret-refresh-def" not in str(tokens)


def test_delete_tokens_and_mcp_token_lifecycle(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    assert delete_tokens(config_dir) is False

    token1 = ensure_mcp_token(config_dir)
    assert len(token1) >= 32
    mcp_path = config_dir / "mcp-token"
    assert stat.S_IMODE(config_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(mcp_path.stat().st_mode) == 0o600

    token2 = ensure_mcp_token(config_dir)
    assert token2 == token1
    assert load_mcp_token(config_dir) == token1

    mcp_path.write_text("   \n", encoding="utf-8")
    os.chmod(mcp_path, 0o600)
    with pytest.raises(ConfigError) as exc_empty:
        load_mcp_token(config_dir)
    assert exc_empty.value.code == "invalid_config"

    mcp_path.write_text(token1 + "\n", encoding="utf-8")
    os.chmod(mcp_path, 0o644)
    with pytest.raises(ConfigError) as exc_mode:
        load_mcp_token(config_dir)
    assert exc_mode.value.code == "insecure_permissions"

    save_tokens(
        config_dir,
        TokenData(access_token="acc", refresh_token="ref", expires_at=1_900_000_000),
    )
    assert delete_tokens(config_dir) is True
    assert delete_tokens(config_dir) is False


def test_error_hierarchy_and_to_dict() -> None:
    base = StravaMCPError("custom", "bad thing", details={"x": 1})
    assert base.to_dict() == {
        "error": {"code": "custom", "message": "bad thing", "details": {"x": 1}}
    }

    scope_err = InsufficientScopeError(
        required_scopes=["activity:read_all"],
        granted_scopes=["read"],
    )
    assert scope_err.code == "insufficient_scope"
    assert scope_err.details["remediation"] == "strava-mcp auth"
    assert "strava-mcp auth" in scope_err.message

    rate_err = RateLimitError(
        limit_type="read_15m",
        usage=200,
        limit=200,
        reset_at_utc="2026-11-25T04:15:00Z",
        retry_after_seconds=120,
    )
    assert rate_err.code == "rate_limited"
    assert rate_err.limit_type == "read_15m"
    assert rate_err.to_dict()["error"]["details"]["retry_after_seconds"] == 120

    assert SubscriptionRequiredError().code == "subscription_required"
    assert TierRestrictedError().code == "tier_restricted"
    assert NotFoundError().code == "not_found"
    upstream = UpstreamAPIError("bad gateway", status_code=502)
    assert upstream.code == "upstream_error"
    assert upstream.status_code == 502
    assert upstream.details["status_code"] == 502


def test_ensure_secure_dir_rejects_regular_file_and_coerces_str(tmp_path: Path) -> None:
    regular_file = tmp_path / "not_a_dir"
    regular_file.write_text("file-content", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_file:
        ensure_secure_dir(regular_file)
    assert exc_file.value.code == "insecure_permissions"

    with pytest.raises(ConfigError) as exc_nested:
        ensure_secure_dir(regular_file / "nested")
    assert exc_nested.value.code == "insecure_permissions"

    str_dir = str(tmp_path / "str_cfg")
    created = ensure_secure_dir(str_dir)
    assert created == Path(str_dir)
    assert stat.S_IMODE(created.stat().st_mode) == 0o700

    save_client(str_dir, ClientConfig(client_id="42", client_secret="sec-42"))
    assert load_client(str_dir).client_id == "42"
    save_tokens(
        str_dir,
        TokenData(access_token="acc-str", refresh_token="ref-str", expires_at=1_900_000_000),
    )
    assert load_tokens(str_dir).access_token == "acc-str"
    mcp_tok = ensure_mcp_token(str_dir)
    assert load_mcp_token(str_dir) == mcp_tok
    assert delete_tokens(str_dir) is True


def test_ensure_secure_dir_and_verify_permissions_handle_oserror_and_fifo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ro_parent = tmp_path / "ro_parent"
    ro_parent.mkdir(mode=0o700)
    os.chmod(ro_parent, 0o500)
    try:
        with pytest.raises(ConfigError) as exc_parent:
            ensure_secure_dir(ro_parent / "child")
        assert exc_parent.value.code == "insecure_permissions"
    finally:
        os.chmod(ro_parent, 0o700)

    unchmoddable = tmp_path / "unchmoddable"
    unchmoddable.mkdir(mode=0o500)
    orig_chmod = os.chmod

    def _fail_chmod(target: Any, mode: int, *args: Any, **kwargs: Any) -> None:
        if Path(target) == unchmoddable:
            raise PermissionError(1, "Operation not permitted")
        orig_chmod(target, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", _fail_chmod)
    try:
        with pytest.raises(ConfigError) as exc_chmod:
            ensure_secure_dir(unchmoddable)
        assert exc_chmod.value.code == "insecure_permissions"
    finally:
        monkeypatch.setattr(os, "chmod", orig_chmod)
        os.chmod(unchmoddable, 0o700)

    cfg_dir = tmp_path / "fifo_cfg"
    ensure_secure_dir(cfg_dir)
    for fname in ("client.json", "tokens.json", "mcp-token"):
        fifo_path = cfg_dir / fname
        os.mkfifo(fifo_path, 0o600)
        os.chmod(fifo_path, 0o600)

    with pytest.raises(ConfigError) as exc_client_fifo:
        load_client(cfg_dir)
    assert exc_client_fifo.value.code == "insecure_permissions"

    with pytest.raises(ConfigError) as exc_tok_fifo:
        load_tokens(cfg_dir)
    assert exc_tok_fifo.value.code == "insecure_permissions"

    with pytest.raises(ConfigError) as exc_mcp_fifo:
        load_mcp_token(cfg_dir)
    assert exc_mcp_fifo.value.code == "insecure_permissions"
