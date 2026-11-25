"""Local SQLite response cache with 0700/0600 permissions and WAL concurrency."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from strava_mcp.config import Config, ConfigError, ensure_secure_dir, resolve_cache_dir

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
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


def _is_sqlite_locked_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _is_corrupt_db_error(exc: sqlite3.DatabaseError) -> bool:
    msg = str(exc).lower()
    if any(tok in msg for tok in ("malformed", "not a database", "corrupt")):
        return True
    return not isinstance(
        exc,
        (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.ProgrammingError),
    )


@contextmanager
def _acquire_cache_lock(
    lock_path: Path,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.01,
) -> Iterator[None]:
    """Acquire an exclusive non-blocking fcntl.flock without ever unlinking the lock file."""
    ensure_secure_dir(lock_path.parent)
    if lock_path.is_symlink():
        raise ConfigError(
            "insecure_permissions",
            f"Refusing symlinked lock file: {lock_path}",
            details={"path": str(lock_path)},
        )
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ConfigError(
                "insecure_permissions",
                f"Refusing symlinked lock file: {lock_path}",
                details={"path": str(lock_path)},
            ) from None
        raise ConfigError(
            "insecure_permissions",
            f"Cannot open cache lock file {lock_path}: {exc.strerror or exc}",
            details={"path": str(lock_path), "error_type": type(exc).__name__},
        ) from None
    acquired = False
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError as exc:
            raise ConfigError(
                "insecure_permissions",
                f"Cannot secure cache lock file {lock_path}: {exc.strerror or exc}",
                details={"path": str(lock_path), "error_type": type(exc).__name__},
            ) from None
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise sqlite3.OperationalError("database is locked") from None
                time.sleep(poll_interval)
        yield
    finally:
        if acquired:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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

        with _acquire_cache_lock(self._lock_path, timeout=self._busy_timeout_seconds):
            self._prepare_secure_files()
            try:
                self._retry_on_busy(self._open_and_init)
            except sqlite3.DatabaseError as exc:
                if not _is_corrupt_db_error(exc):
                    raise
                self._reset_corrupt_db()
                self._prepare_secure_files()
                self._retry_on_busy(self._open_and_init)

    def _open_and_init(self) -> None:
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

    def _reset_corrupt_db(self) -> None:
        self._reject_symlink(self._db_path)
        for sidecar in self._sidecar_paths():
            self._reject_symlink(sidecar)
        self._db_path.unlink(missing_ok=True)
        for sidecar in self._sidecar_paths():
            sidecar.unlink(missing_ok=True)
        logger.warning("Corrupt SQLite cache at %s; recreating.", self._db_path)

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

    @property
    def _lock_path(self) -> Path:
        if self._db_path.suffix == ".lock":
            return Path(f"{self._db_path}.flock")
        if self._db_path.suffix:
            return self._db_path.with_suffix(".lock")
        return Path(f"{self._db_path}.lock")

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.time()

    def _sidecar_paths(self) -> tuple[Path, Path, Path]:
        return (
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
            Path(f"{self._db_path}-journal"),
        )

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
            try:
                os.chmod(self._db_path, 0o600)
            except OSError as exc:
                raise ConfigError(
                    "insecure_permissions",
                    f"Cannot secure cache DB file {self._db_path}: {exc.strerror or exc}",
                    details={"path": str(self._db_path), "error_type": type(exc).__name__},
                ) from None
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
                raise ConfigError(
                    "insecure_permissions",
                    f"Cannot open cache DB file {self._db_path}: {exc.strerror or exc}",
                    details={"path": str(self._db_path), "error_type": type(exc).__name__},
                ) from None
            try:
                os.fchmod(fd, 0o600)
            except OSError as exc:
                raise ConfigError(
                    "insecure_permissions",
                    f"Cannot secure cache DB file {self._db_path}: {exc.strerror or exc}",
                    details={"path": str(self._db_path), "error_type": type(exc).__name__},
                ) from None
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
        row = self._conn.execute("PRAGMA journal_mode").fetchone()
        if row is None or str(row[0]).lower() != "wal":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA temp_store = MEMORY")

    def _init_schema(self) -> None:
        check_row = self._conn.execute("PRAGMA quick_check(1)").fetchone()
        if check_row is None or str(check_row[0]).lower() != "ok":
            raise sqlite3.DatabaseError(
                f"SQLite quick_check failed: {check_row[0] if check_row else 'empty'}"
            )
        existing = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'index') "
                "AND name IN ('cache_entries', 'idx_cache_entries_expires_at')"
            ).fetchall()
        }
        if existing == {"cache_entries", "idx_cache_entries_expires_at"}:
            self._conn.execute("SELECT 1 FROM cache_entries LIMIT 1").fetchone()
            return
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

    def _recover_corrupt_db_locked(self) -> None:
        with suppress(Exception):
            self._conn.close()
        with _acquire_cache_lock(self._lock_path, timeout=self._busy_timeout_seconds):
            self._prepare_secure_files()
            try:
                self._retry_on_busy(self._open_and_init)
            except sqlite3.DatabaseError as exc:
                if not _is_corrupt_db_error(exc):
                    raise
                self._reset_corrupt_db()
                self._prepare_secure_files()
                self._retry_on_busy(self._open_and_init)

    def _run_locked_with_recovery(self, op: Callable[[], _T]) -> _T:
        with self._lock:
            self._ensure_open()
            try:
                return op()
            except sqlite3.DatabaseError as exc:
                if not _is_corrupt_db_error(exc):
                    raise
                self._recover_corrupt_db_locked()
                return op()

    def _retry_on_busy(self, fn: Callable[[], _T]) -> _T:
        deadline = time.monotonic() + self._busy_timeout_seconds
        attempt = 0
        while True:
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_locked_error(exc):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise
                delay = min(0.05, 0.005 * (2**attempt))
                time.sleep(min(delay, remaining))
                attempt += 1

    def _run_write_tx(self, fn: Callable[[], _T]) -> _T:
        def _attempt() -> _T:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn()
                self._conn.execute("COMMIT")
                return result
            except Exception:
                with suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

        return self._retry_on_busy(_attempt)

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

        def _op() -> CacheEntry | None:
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
            raw_meta = row["metadata_json"]
            try:
                value: Any = raw_payload if payload_type == "text" else json.loads(raw_payload)
                metadata: dict[str, Any] = json.loads(raw_meta) if raw_meta else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupt JSON in cache entry %s; evicting row.", validated_key)
                with suppress(sqlite3.Error):
                    self._run_write_tx(
                        lambda: self._conn.execute(
                            "DELETE FROM cache_entries WHERE cache_key = ?",
                            (validated_key,),
                        )
                    )
                self._misses += 1
                return None

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

        return self._run_locked_with_recovery(_op)

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

        def _op() -> None:
            now = self._now()
            expires_at = None if ttl_seconds is None else now + float(ttl_seconds)

            def _write() -> None:
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

            self._run_write_tx(_write)
            self._harden_sidecar_permissions()

        self._run_locked_with_recovery(_op)

    def delete(self, key: str) -> bool:
        """Delete a single cache entry; returns True if an entry was removed."""
        validated_key = self._validate_key(key)

        def _op() -> bool:
            def _delete_tx() -> bool:
                cur = self._conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?",
                    (validated_key,),
                )
                return cur.rowcount > 0

            deleted = self._run_write_tx(_delete_tx)
            self._harden_sidecar_permissions()
            return deleted

        return self._run_locked_with_recovery(_op)

    def clear(self) -> int:
        """Delete all cached entries and truncate the WAL; returns removed row count."""

        def _op() -> int:
            def _clear_tx() -> int:
                cur = self._conn.execute("DELETE FROM cache_entries")
                return max(0, cur.rowcount)

            removed = self._run_write_tx(_clear_tx)
            busy_ms = int(self._busy_timeout_seconds * 1000)
            try:
                with suppress(sqlite3.Error):
                    self._conn.execute("PRAGMA busy_timeout = 0")
                    row = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if row is not None and row[0]:
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            finally:
                with suppress(sqlite3.Error):
                    self._conn.execute(f"PRAGMA busy_timeout = {busy_ms}")
            self._harden_sidecar_permissions()
            return removed

        return self._run_locked_with_recovery(_op)

    def purge_expired(self) -> int:
        """Delete all expired entries; returns removed row count."""

        def _op() -> int:
            now = self._now()

            def _purge_tx() -> int:
                cur = self._conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
                return max(0, cur.rowcount)

            removed = self._run_write_tx(_purge_tx)
            self._harden_sidecar_permissions()
            return removed

        return self._run_locked_with_recovery(_op)

    def stats(self) -> dict[str, Any]:
        """Return JSON-safe cache statistics and database file sizes."""

        def _op() -> dict[str, Any]:
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
            try:
                db_size_bytes = self._db_path.stat().st_size
            except FileNotFoundError:
                db_size_bytes = 0
            wal_path = Path(f"{self._db_path}-wal")
            try:
                wal_size_bytes = wal_path.stat().st_size
            except FileNotFoundError:
                wal_size_bytes = 0
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

        return self._run_locked_with_recovery(_op)

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
