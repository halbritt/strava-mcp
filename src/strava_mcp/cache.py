"""Local SQLite response cache with 0700/0600 permissions and WAL concurrency."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strava_mcp.config import Config, ConfigError, ensure_secure_dir, resolve_cache_dir

_SQLITE_FILE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    payload_type TEXT NOT NULL DEFAULT 'json',
    created_at REAL NOT NULL,
    expires_at REAL,
    etag TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at
    ON cache_entries(expires_at);
"""


def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return {str(k): _normalize_param_value(v) for k, v in items if v is not None}
    if isinstance(value, (list, tuple)):
        return [_normalize_param_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_param_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def make_cache_key(endpoint: str, params: Mapping[str, Any] | None = None) -> str:
    """Derive a deterministic SHA-256 cache key from normalized endpoint and params."""
    cleaned = endpoint.strip()
    norm_ep = "/" + cleaned.lstrip("/") if cleaned else "/"
    norm_params = _normalize_param_value(params) if params is not None else {}
    canonical = json.dumps(
        {"endpoint": norm_ep, "params": norm_params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _format_epoch_utc(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CacheEntry:
    """Cached response entry including freshness, ETag, and metadata."""

    key: str
    value: Any
    created_at: float
    expires_at: float | None
    is_stale: bool
    etag: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ResponseCache:
    """SQLite-backed response cache with strict local permissions and WAL concurrency."""

    make_key = staticmethod(make_cache_key)

    def __init__(
        self,
        cache_dir: Config | Path | str | None = None,
        *,
        db_path: Path | str | None = None,
        busy_timeout_seconds: float = 5.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._db_path = self._resolve_db_path(cache_dir, db_path)
        self._busy_timeout_seconds = max(0.0, float(busy_timeout_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False
        self._hits = 0
        self._misses = 0
        self._stale_hits = 0

        self._prepare_secure_files()
        self._conn = sqlite3.connect(
            self._db_path,
            timeout=self._busy_timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._configure_pragmas()
            self._init_schema()
            self._harden_sidecar_permissions()
        except Exception:
            self._conn.close()
            raise

    @staticmethod
    def _resolve_db_path(
        cache_dir: Config | Path | str | None,
        db_path: Path | str | None,
    ) -> Path:
        if db_path is not None:
            return Path(db_path)
        if cache_dir is None:
            return resolve_cache_dir() / "cache.sqlite"
        if isinstance(cache_dir, Config):
            return cache_dir.cache_db_file
        candidate = Path(cache_dir)
        if candidate.suffix in _SQLITE_FILE_SUFFIXES and not candidate.is_dir():
            return candidate
        return candidate / "cache.sqlite"

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def cache_dir(self) -> Path:
        return self._db_path.parent

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.time()

    def _sidecar_paths(self) -> tuple[Path, Path]:
        return (Path(f"{self._db_path}-wal"), Path(f"{self._db_path}-shm"))

    def _reject_symlink(self, path: Path) -> None:
        if path.is_symlink():
            raise ConfigError(
                "insecure_permissions",
                f"Refusing symlink at cache path: {path}",
                details={"path": str(path)},
            )

    def _prepare_secure_files(self) -> None:
        ensure_secure_dir(self._db_path.parent)
        self._reject_symlink(self._db_path)
        for sidecar in self._sidecar_paths():
            self._reject_symlink(sidecar)

        if self._db_path.exists():
            if not self._db_path.is_file():
                raise ConfigError(
                    "insecure_permissions",
                    f"Expected regular file at cache DB path: {self._db_path}",
                    details={"path": str(self._db_path)},
                )
            os.chmod(self._db_path, 0o600)
        else:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self._db_path, flags, 0o600)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ConfigError(
                        "insecure_permissions",
                        f"Refusing symlink at cache path: {self._db_path}",
                        details={"path": str(self._db_path)},
                    ) from None
                raise
            try:
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)

    def _harden_sidecar_permissions(self) -> None:
        for sidecar in self._sidecar_paths():
            if sidecar.is_symlink():
                raise ConfigError(
                    "insecure_permissions",
                    f"Refusing symlink at cache sidecar path: {sidecar}",
                    details={"path": str(sidecar)},
                )
            if sidecar.exists() and sidecar.is_file():
                with suppress(OSError):
                    os.chmod(sidecar, 0o600)

    def _configure_pragmas(self) -> None:
        busy_ms = int(self._busy_timeout_seconds * 1000)
        self._conn.execute(f"PRAGMA busy_timeout = {busy_ms}")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA temp_store = MEMORY")

    def _init_schema(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for stmt in _SCHEMA_SQL.strip().split(";"):
                sql = stmt.strip()
                if sql:
                    self._conn.execute(sql)
            self._conn.execute("COMMIT")
        except Exception:
            with suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ResponseCache is closed")

    @staticmethod
    def _validate_key(key: str) -> str:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Cache key must be a non-empty string.")
        return key

    def get_entry(self, key: str, allow_stale: bool = False) -> CacheEntry | None:
        """Fetch a CacheEntry by key, respecting TTL unless allow_stale=True."""
        validated_key = self._validate_key(key)
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                """
                SELECT payload, payload_type, created_at, expires_at, etag, metadata_json
                FROM cache_entries
                WHERE cache_key = ?
                """,
                (validated_key,),
            ).fetchone()
            if row is None:
                self._misses += 1
                return None

            now = self._now()
            raw_expires = row["expires_at"]
            expires_at = float(raw_expires) if raw_expires is not None else None
            is_stale = expires_at is not None and now >= expires_at
            if is_stale and not allow_stale:
                self._misses += 1
                return None

            payload_type = row["payload_type"]
            raw_payload = row["payload"]
            value: Any = raw_payload if payload_type == "text" else json.loads(raw_payload)
            raw_meta = row["metadata_json"]
            metadata: dict[str, Any] = json.loads(raw_meta) if raw_meta else {}

            if is_stale:
                self._stale_hits += 1
            else:
                self._hits += 1

            return CacheEntry(
                key=validated_key,
                value=value,
                created_at=float(row["created_at"]),
                expires_at=expires_at,
                is_stale=is_stale,
                etag=row["etag"],
                metadata=metadata,
            )

    def get(self, key: str, allow_stale: bool = False) -> Any | None:
        """Fetch a cached value by key, or None if missing or expired."""
        entry = self.get_entry(key, allow_stale=allow_stale)
        return entry.value if entry is not None else None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | int | None = None,
        *,
        etag: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Store or overwrite a cached value with optional TTL (None = immutable)."""
        validated_key = self._validate_key(key)
        if isinstance(value, str):
            payload = value
            payload_type = "text"
        else:
            payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            payload_type = "json"

        metadata_json = (
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if metadata is not None
            else None
        )

        with self._lock:
            self._ensure_open()
            now = self._now()
            expires_at = None if ttl_seconds is None else now + float(ttl_seconds)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO cache_entries (
                        cache_key, payload, payload_type, created_at,
                        expires_at, etag, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload = excluded.payload,
                        payload_type = excluded.payload_type,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        etag = excluded.etag,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        validated_key,
                        payload,
                        payload_type,
                        now,
                        expires_at,
                        etag,
                        metadata_json,
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise
            self._harden_sidecar_permissions()

    def delete(self, key: str) -> bool:
        """Delete a single cache entry; returns True if an entry was removed."""
        validated_key = self._validate_key(key)
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?",
                    (validated_key,),
                )
                deleted = cur.rowcount > 0
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise
            self._harden_sidecar_permissions()
            return deleted

    def clear(self) -> int:
        """Delete all cached entries and truncate the WAL; returns removed row count."""
        with self._lock:
            self._ensure_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute("DELETE FROM cache_entries")
                removed = max(0, cur.rowcount)
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise
            with suppress(sqlite3.Error):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._harden_sidecar_permissions()
            return removed

    def purge_expired(self) -> int:
        """Delete all expired entries; returns removed row count."""
        with self._lock:
            self._ensure_open()
            now = self._now()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
                removed = max(0, cur.rowcount)
                self._conn.execute("COMMIT")
            except Exception:
                with suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise
            self._harden_sidecar_permissions()
            return removed

    def stats(self) -> dict[str, Any]:
        """Return JSON-safe cache statistics and database file sizes."""
        with self._lock:
            self._ensure_open()
            now = self._now()
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS entry_count,
                    SUM(CASE WHEN expires_at IS NULL OR expires_at > ? THEN 1 ELSE 0 END)
                        AS active_count,
                    SUM(CASE WHEN expires_at IS NOT NULL AND expires_at <= ? THEN 1 ELSE 0 END)
                        AS expired_count,
                    MIN(created_at) AS oldest_created_at,
                    MAX(created_at) AS newest_created_at
                FROM cache_entries
                """,
                (now, now),
            ).fetchone()
            db_size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0
            wal_path = Path(f"{self._db_path}-wal")
            wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0
            raw_oldest = row["oldest_created_at"]
            raw_newest = row["newest_created_at"]
            oldest = float(raw_oldest) if raw_oldest is not None else None
            newest = float(raw_newest) if raw_newest is not None else None
            return {
                "db_path": str(self._db_path),
                "entry_count": int(row["entry_count"]),
                "active_count": int(row["active_count"] or 0),
                "expired_count": int(row["expired_count"] or 0),
                "db_size_bytes": db_size_bytes,
                "wal_size_bytes": wal_size_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "stale_hits": self._stale_hits,
                "oldest_created_at": oldest,
                "newest_created_at": newest,
                "oldest_created_at_utc": _format_epoch_utc(oldest),
                "newest_created_at_utc": _format_epoch_utc(newest),
            }

    def close(self) -> None:
        """Close the underlying SQLite connection idempotently."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
