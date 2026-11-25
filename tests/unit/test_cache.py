"""Unit tests for strava_mcp.cache."""

from __future__ import annotations

import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from strava_mcp.cache import ResponseCache, make_cache_key
from strava_mcp.config import Config, ConfigError


def test_cache_path_resolution_permissions_and_wal_pragmas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(
        config_dir=tmp_path / "cfg",
        cache_dir=tmp_path / "cache_from_cfg",
        api_base_url="https://www.strava.com/api/v3",
        oauth_base_url="https://www.strava.com/oauth",
    )
    with ResponseCache(cfg) as cache_cfg:
        assert cache_cfg.db_path == cfg.cache_db_file
        assert stat.S_IMODE(cfg.cache_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(cache_cfg.db_path.stat().st_mode) == 0o600
        conn = sqlite3.connect(cache_cfg.db_path)
        try:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert str(journal_mode).lower() == "wal"
        finally:
            conn.close()
        assert cache_cfg._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    # Explicit .sqlite file path, str dir path, and STRAVA_MCP_CACHE_DIR env var
    explicit_db = tmp_path / "custom" / "my_cache.sqlite"
    with ResponseCache(explicit_db) as cache_file:
        assert cache_file.db_path == explicit_db
        assert stat.S_IMODE(explicit_db.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(explicit_db.stat().st_mode) == 0o600

    str_dir = str(tmp_path / "str_cache")
    with ResponseCache(str_dir) as cache_str:
        assert cache_str.db_path == Path(str_dir) / "cache.sqlite"

    env_dir = tmp_path / "env_cache"
    monkeypatch.setenv("STRAVA_MCP_CACHE_DIR", str(env_dir))
    with ResponseCache() as cache_env:
        assert cache_env.db_path == env_dir / "cache.sqlite"


def test_cache_rejects_symlinks_and_non_regular_paths(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir(mode=0o700)
    sym_dir = tmp_path / "sym_dir"
    sym_dir.symlink_to(real_dir)
    with pytest.raises(ConfigError) as exc_dir:
        ResponseCache(sym_dir)
    assert exc_dir.value.code == "insecure_permissions"

    regular_file = tmp_path / "not_a_dir"
    regular_file.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_file:
        ResponseCache(regular_file)
    assert exc_file.value.code == "insecure_permissions"

    db_dir = tmp_path / "db_dir"
    db_dir.mkdir(mode=0o700)
    target = tmp_path / "target.sqlite"
    target.write_text("", encoding="utf-8")
    (db_dir / "cache.sqlite").symlink_to(target)
    with pytest.raises(ConfigError) as exc_db_sym:
        ResponseCache(db_dir)
    assert exc_db_sym.value.code == "insecure_permissions"

    # Hardens permissive existing db file back to 0600
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir(mode=0o755)
    existing_db = existing_dir / "cache.sqlite"
    existing_db.write_bytes(b"")
    os.chmod(existing_db, 0o644)
    with ResponseCache(existing_dir) as cache:
        assert stat.S_IMODE(existing_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(cache.db_path.stat().st_mode) == 0o600


def test_make_cache_key_determinism_and_collisions() -> None:
    k1 = make_cache_key("/athlete/activities", {"page": 1, "per_page": 30, "after": None})
    k2 = make_cache_key("athlete/activities", {"per_page": 30, "page": 1})
    assert k1 == k2
    assert len(k1) == 64

    assert make_cache_key("/athlete", None) == make_cache_key("/athlete", {})
    assert make_cache_key("/x", {"items": (1, 2), "nested": {"b": 2, "a": 1, "skip": None}}) == (
        make_cache_key("/x", {"nested": {"a": 1, "b": 2}, "items": [1, 2]})
    )
    assert make_cache_key("/athlete", {"page": 1}) != make_cache_key("/athlete", {"page": 2})
    assert make_cache_key("/activities/1") != make_cache_key("/activities/2")


def test_cache_round_trip_ttl_stale_and_metadata(tmp_path: Path) -> None:
    now = 1_700_000_000.0
    with ResponseCache(tmp_path / "cache", clock=lambda: now) as cache:
        k_json = cache.make_key("/activities/123", {"include_all_efforts": False})
        cache.set(
            k_json,
            {"id": 123, "name": "Morning Ride"},
            ttl_seconds=60.0,
            etag='"etag-1"',
            metadata={"endpoint": "/activities/123"},
        )
        entry = cache.get_entry(k_json)
        assert entry is not None
        assert entry.value == {"id": 123, "name": "Morning Ride"}
        assert entry.etag == '"etag-1"'
        assert entry.metadata == {"endpoint": "/activities/123"}
        assert entry.is_stale is False
        assert entry.expires_at == 1_700_000_060.0

        # Raw XML string round-trip and immutable ttl_seconds=None
        k_gpx = cache.make_key("/routes/99/export_gpx")
        gpx_xml = '<?xml version="1.0"?><gpx><trk><name>Loop</name></trk></gpx>'
        cache.set(k_gpx, gpx_xml, ttl_seconds=None)

        # Advance past TTL -> expired unless allow_stale=True
        now = 1_700_000_060.0
        assert cache.get(k_json) is None
        assert cache.get(k_json, allow_stale=False) is None
        stale_entry = cache.get_entry(k_json, allow_stale=True)
        assert stale_entry is not None
        assert stale_entry.is_stale is True
        assert stale_entry.value == {"id": 123, "name": "Morning Ride"}
        assert cache.get(k_gpx) == gpx_xml

        # Overwrite upsert refreshes value and TTL
        cache.set(k_json, {"id": 123, "name": "Updated Ride"}, ttl_seconds=120.0)
        assert cache.get(k_json) == {"id": 123, "name": "Updated Ride"}


def test_cache_delete_clear_purge_expired_and_stats(tmp_path: Path) -> None:
    now = 1_700_000_000.0
    with ResponseCache(tmp_path / "cache", clock=lambda: now) as cache:
        empty_stats = cache.stats()
        assert empty_stats["entry_count"] == 0
        assert empty_stats["active_count"] == 0
        assert empty_stats["expired_count"] == 0

        cache.set("k1", {"v": 1}, ttl_seconds=10.0)
        now = 1_700_000_005.0
        cache.set("k2", {"v": 2}, ttl_seconds=100.0)
        cache.set("k3", {"v": 3}, ttl_seconds=None)

        assert cache.get("k2") == {"v": 2}
        assert cache.get("missing") is None

        now = 1_700_000_020.0
        assert cache.get("k1") is None
        assert cache.get("k1", allow_stale=True) == {"v": 1}

        stats = cache.stats()
        assert stats["entry_count"] == 3
        assert stats["active_count"] == 2
        assert stats["expired_count"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["stale_hits"] == 1
        assert stats["db_size_bytes"] > 0
        assert stats["oldest_created_at"] == 1_700_000_000.0
        assert stats["newest_created_at"] == 1_700_000_005.0
        assert stats["oldest_created_at_utc"] is not None
        assert stats["newest_created_at_utc"] is not None

        assert cache.purge_expired() == 1
        assert cache.stats()["entry_count"] == 2
        assert cache.delete("k2") is True
        assert cache.delete("k2") is False
        assert cache.clear() == 1
        assert cache.stats()["entry_count"] == 0


def test_cache_concurrent_multi_instance_and_close_semantics(tmp_path: Path) -> None:
    cache_dir = tmp_path / "shared_cache"
    c1 = ResponseCache(cache_dir)
    c2 = ResponseCache(cache_dir)

    try:
        def worker(idx: int) -> None:
            target = c1 if idx % 2 == 0 else c2
            key = make_cache_key("/activities", {"idx": idx})
            target.set(key, {"idx": idx}, ttl_seconds=300.0)
            assert target.get(key) == {"idx": idx}

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(40)))

        assert c1.stats()["entry_count"] == 40
        assert c2.stats()["entry_count"] == 40
    finally:
        c1.close()
        c2.close()
        c2.close()  # idempotent

    with pytest.raises(RuntimeError):
        c1.get("any-key")
    with pytest.raises(ValueError):
        c2 = ResponseCache(cache_dir)
        try:
            c2.get("   ")
        finally:
            c2.close()
