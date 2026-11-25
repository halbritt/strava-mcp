"""Unit tests for strava_mcp.cache."""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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


def test_cache_recovers_cleanly_from_corrupt_sqlite_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache_dir = tmp_path / "corrupt_cache"
    cache_dir.mkdir(mode=0o700)
    db_file = cache_dir / "cache.sqlite"
    db_file.write_bytes(b"not-a-valid-sqlite-db")
    os.chmod(db_file, 0o600)
    wal_file = Path(f"{db_file}-wal")
    wal_file.write_bytes(b"corrupt-wal")
    os.chmod(wal_file, 0o600)

    with caplog.at_level("WARNING", logger="strava_mcp.cache"):
        with ResponseCache(cache_dir) as cache:
            assert stat.S_IMODE(cache.db_path.stat().st_mode) == 0o600
            assert cache.stats()["entry_count"] == 0
            cache.set("k_recovered", {"ok": True}, ttl_seconds=60.0)
            assert cache.get("k_recovered") == {"ok": True}
            assert cache.clear() == 1

    assert any("Corrupt SQLite cache" in rec.message for rec in caplog.records)
    assert all(rec.exc_info is None for rec in caplog.records)


def _mp_cold_start_worker(
    cache_dir_str: str,
    idx: int,
    barrier: Any,
    queue: Any,
) -> None:
    try:
        barrier.wait(timeout=15.0)
        with ResponseCache(Path(cache_dir_str)) as cache:
            key = f"cold-{idx}"
            cache.set(key, {"idx": idx}, ttl_seconds=60.0)
            val = cache.get(key)
            if val != {"idx": idx}:
                queue.put(("err", idx, "AssertionError", f"unexpected value {val!r}"))
                return
        queue.put(("ok", idx))
    except Exception as exc:
        queue.put(("err", idx, type(exc).__name__, str(exc)))


def _mp_corrupt_recovery_worker(
    cache_dir_str: str,
    idx: int,
    barrier: Any,
    queue: Any,
) -> None:
    try:
        barrier.wait(timeout=15.0)
        with ResponseCache(Path(cache_dir_str)) as cache:
            key = f"k-{idx}"
            cache.set(key, {"idx": idx}, ttl_seconds=60.0)
            val = cache.get(key)
            if val != {"idx": idx}:
                queue.put(("err", idx, "AssertionError", f"unexpected value {val!r}"))
                return
        queue.put(("ok", idx))
    except Exception as exc:
        queue.put(("err", idx, type(exc).__name__, str(exc)))


def test_cache_concurrent_cold_start_multiprocess(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    trials = 10
    n_procs = 6
    for trial in range(trials):
        cache_dir = tmp_path / f"cold_trial_{trial}"
        barrier = ctx.Barrier(n_procs)
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_mp_cold_start_worker,
                args=(str(cache_dir), idx, barrier, queue),
            )
            for idx in range(n_procs)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=20.0)
            assert proc.exitcode == 0, f"trial={trial} worker exited with {proc.exitcode}"

        results = [queue.get(timeout=5.0) for _ in range(n_procs)]
        errors = [res for res in results if res[0] != "ok"]
        assert errors == [], f"trial={trial} cold-start errors: {errors}"

        lock_file = cache_dir / "cache.lock"
        db_file = cache_dir / "cache.sqlite"
        assert lock_file.exists()
        assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600
        assert db_file.exists()
        assert stat.S_IMODE(db_file.stat().st_mode) == 0o600
        with ResponseCache(cache_dir) as check_cache:
            assert check_cache.stats()["entry_count"] == n_procs


def test_cache_concurrent_corrupt_recovery_multiprocess(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    n_procs = 4
    for trial in range(3):
        cache_dir = tmp_path / f"corrupt_trial_{trial}"
        cache_dir.mkdir(mode=0o700)
        db_file = cache_dir / "cache.sqlite"
        db_file.write_bytes(b"not-a-valid-sqlite-db")
        os.chmod(db_file, 0o600)
        wal_file = Path(f"{db_file}-wal")
        wal_file.write_bytes(b"corrupt-wal")
        os.chmod(wal_file, 0o600)

        barrier = ctx.Barrier(n_procs)
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_mp_corrupt_recovery_worker,
                args=(str(cache_dir), idx, barrier, queue),
            )
            for idx in range(n_procs)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=20.0)
            assert proc.exitcode == 0, f"trial={trial} worker exited with {proc.exitcode}"

        results = [queue.get(timeout=5.0) for _ in range(n_procs)]
        errors = [res for res in results if res[0] != "ok"]
        assert errors == [], f"trial={trial} corrupt-recovery errors: {errors}"
        assert stat.S_IMODE(db_file.stat().st_mode) == 0o600
        with ResponseCache(cache_dir) as check_cache:
            assert check_cache.stats()["entry_count"] == n_procs


def test_cache_clear_does_not_starve_writers_with_active_reader(tmp_path: Path) -> None:
    cache_dir = tmp_path / "clear_starve"
    cache1 = ResponseCache(cache_dir, busy_timeout_seconds=5.0)
    cache2 = ResponseCache(cache_dir, busy_timeout_seconds=0.3)
    reader = sqlite3.connect(cache1.db_path, isolation_level=None)
    try:
        cache1.set("seed", {"v": 1}, ttl_seconds=60.0)
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] == 1

        errors: list[BaseException] = []
        ready = threading.Barrier(2)

        def _clear_worker() -> None:
            try:
                ready.wait(timeout=5.0)
                cache1.clear()
            except BaseException as exc:
                errors.append(exc)

        def _write_worker() -> None:
            try:
                ready.wait(timeout=5.0)
                time.sleep(0.02)
                cache2.set("after_clear", {"v": 2}, ttl_seconds=60.0)
            except BaseException as exc:
                errors.append(exc)

        t0 = time.monotonic()
        t_clear = threading.Thread(target=_clear_worker)
        t_write = threading.Thread(target=_write_worker)
        t_clear.start()
        t_write.start()
        t_clear.join(timeout=5.0)
        t_write.join(timeout=5.0)
        elapsed = time.monotonic() - t0

        assert errors == []
        assert elapsed < 1.5, f"clear() stalled concurrent writer for {elapsed:.2f}s"
        assert cache1._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        reader.execute("ROLLBACK")
        reader.close()
        cache2.close()
        cache1.close()


def test_cache_rejects_symlinked_lock_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "sym_lock_cache"
    cache_dir.mkdir(mode=0o700)
    target = tmp_path / "outside.lock"
    target.write_text("", encoding="utf-8")
    (cache_dir / "cache.lock").symlink_to(target)

    with pytest.raises(ConfigError) as exc_info:
        ResponseCache(cache_dir)
    assert exc_info.value.code == "insecure_permissions"


def test_cache_stats_handles_vanishing_wal_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "vanish_wal"
    with ResponseCache(cache_dir) as cache:
        cache.set("k1", {"v": 1}, ttl_seconds=60.0)
        wal_path = Path(f"{cache.db_path}-wal")
        orig_stat = Path.stat

        def _vanishing_wal_stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
            if self == wal_path:
                raise FileNotFoundError(str(self))
            return orig_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", _vanishing_wal_stat)
        st = cache.stats()
        assert st["wal_size_bytes"] == 0
        assert st["db_size_bytes"] > 0


def test_cache_rejects_symlinked_journal_and_non_writable_parent(tmp_path: Path) -> None:
    cache_dir = tmp_path / "sym_journal_cache"
    cache_dir.mkdir(mode=0o700)
    outside = tmp_path / "outside.journal"
    outside.write_text("keep", encoding="utf-8")
    (cache_dir / "cache.sqlite-journal").symlink_to(outside)

    with pytest.raises(ConfigError) as exc_sym:
        ResponseCache(cache_dir)
    assert exc_sym.value.code == "insecure_permissions"
    assert outside.read_text(encoding="utf-8") == "keep"

    ro_parent = tmp_path / "ro_parent"
    ro_parent.mkdir(mode=0o700)
    os.chmod(ro_parent, 0o500)
    try:
        with pytest.raises(ConfigError) as exc_ro:
            ResponseCache(ro_parent / "sub")
        assert exc_ro.value.code == "insecure_permissions"
    finally:
        os.chmod(ro_parent, 0o700)


def _seed_multipage_db(cache_dir: Path) -> Path:
    with ResponseCache(cache_dir) as cache:
        for i in range(40):
            cache.set(f"key-{i}", {"idx": i, "blob": "x" * 512}, ttl_seconds=600.0)
        cache._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return cache.db_path


def test_cache_recovers_page2_init_post_init_corruption_and_corrupt_row_json(
    tmp_path: Path,
) -> None:
    # 1. Page-2+ corruption with intact page-1 sqlite_master caught and reset at __init__
    init_dir = tmp_path / "page2_init"
    db_path = _seed_multipage_db(init_dir)
    raw = bytearray(db_path.read_bytes())
    assert len(raw) >= 8192
    raw[4096:8192] = b"\xff" * 4096
    db_path.write_bytes(bytes(raw))
    os.chmod(db_path, 0o600)

    with ResponseCache(init_dir) as recovered:
        assert recovered.get("key-0") is None
        assert recovered.stats()["entry_count"] == 0
        recovered.set("fresh", {"ok": True}, ttl_seconds=60.0)
        assert recovered.get("fresh") == {"ok": True}

    # 2. Post-init corruption on an already-open ResponseCache auto-recovers on get/set/stats
    live_dir = tmp_path / "page2_live"
    with ResponseCache(live_dir) as live:
        with ResponseCache(live_dir) as peer:
            for i in range(40):
                peer.set(f"key-{i}", {"idx": i, "blob": "x" * 512}, ttl_seconds=600.0)
            peer._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        live_raw = bytearray(live.db_path.read_bytes())
        assert len(live_raw) >= 8192
        live_raw[4096:] = b"\x00" * (len(live_raw) - 4096)
        live.db_path.write_bytes(bytes(live_raw))
        assert live.get("key-0") is None
        assert live.stats()["entry_count"] == 0
        live.set("after_live_corrupt", {"ok": 2}, ttl_seconds=60.0)
        assert live.get("after_live_corrupt") == {"ok": 2}

    # 3. Corrupt JSON inside row payload / metadata returns None and evicts corrupt row
    row_dir = tmp_path / "corrupt_json_row"
    with ResponseCache(row_dir) as cache:
        cache.set("bad-payload", {"ok": 1}, ttl_seconds=60.0)
        cache.set("bad-meta", {"ok": 2}, ttl_seconds=60.0, metadata={"a": 1})
        cache._conn.execute(
            "UPDATE cache_entries SET payload = '{not-json' WHERE cache_key = 'bad-payload'"
        )
        cache._conn.execute(
            "UPDATE cache_entries SET metadata_json = '{bad-meta' WHERE cache_key = 'bad-meta'"
        )
        assert cache.get("bad-payload") is None
        assert cache.get("bad-meta") is None
        assert cache.stats()["entry_count"] == 0
