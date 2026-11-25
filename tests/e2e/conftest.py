"""Opaque-box E2E pytest fixtures, isolated env, and sync MCP client helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

_E2E_DIR = Path(__file__).resolve().parent
if str(_E2E_DIR) not in sys.path:
    sys.path.insert(0, str(_E2E_DIR))

REPO_ROOT = _E2E_DIR.parent.parent

if TYPE_CHECKING:
    from mock_strava import MockStravaServer

DEFAULT_TEST_CLIENT_ID = "999001"
DEFAULT_TEST_CLIENT_SECRET = "mock-client-secret-999001"
DEFAULT_TEST_ACCESS_TOKEN = "synthetic-access-token-initial"
DEFAULT_TEST_REFRESH_TOKEN = "synthetic-refresh-token-initial"
DEFAULT_TEST_SCOPE = "read,read_all,profile:read_all,activity:read_all"
DEFAULT_TEST_ATHLETE_ID = 100001
DEFAULT_TEST_MCP_TOKEN = "synthetic-mcp-bearer-token-0001"


def resolve_strava_mcp_cmd(
    repo_root: Path = REPO_ROOT,
    path_env: str | None = None,
) -> list[str]:
    """Locate the strava-mcp executable or fail loudly (never skip)."""
    venv_bin = repo_root / ".venv" / "bin" / "strava-mcp"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return [str(venv_bin)]

    if repo_root == REPO_ROOT:
        py_bin = Path(sys.executable).parent / "strava-mcp"
        if py_bin.is_file() and os.access(py_bin, os.X_OK):
            return [str(py_bin)]

    which_path = os.environ.get("PATH") if path_env is None else path_env
    found = shutil.which("strava-mcp", path=which_path)
    if found:
        return [found]

    if repo_root == REPO_ROOT and importlib.util.find_spec("strava_mcp") is not None:
        return [sys.executable, "-m", "strava_mcp"]

    venv_py = repo_root / ".venv" / "bin" / "python"
    if repo_root != REPO_ROOT and venv_py.is_file() and os.access(venv_py, os.X_OK):
        return [str(venv_py), "-m", "strava_mcp"]

    pytest.fail(
        "strava-mcp binary/package not found; E2E tests require an installed strava-mcp",
        pytrace=False,
    )


@dataclass(frozen=True)
class SeededConfig:
    """Paths and synthetic credential values seeded for a single test."""

    config_dir: Path
    cache_dir: Path
    client_file: Path
    tokens_file: Path
    mcp_token_file: Path
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str
    expires_at: int
    scope: str
    athlete_id: int
    mcp_token: str


@dataclass
class HttpServerInfo:
    """Running strava-mcp Streamable HTTP server handle."""

    host: str
    port: int
    base_url: str
    mcp_url: str
    token: str
    process: subprocess.Popen[str]
    stderr_path: Path
    stdout_path: Path

    @property
    def stderr_text(self) -> str:
        if not self.stderr_path.exists():
            return ""
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")

    @property
    def stdout_text(self) -> str:
        if not self.stdout_path.exists():
            return ""
        return self.stdout_path.read_text(encoding="utf-8", errors="replace")


class SyncMCPSession:
    """Synchronous wrapper around an AnyIO single-task MCP ClientSession actor."""

    def __init__(
        self,
        transport_cm_factory: Callable[[], AbstractAsyncContextManager[Any]],
        *,
        timeout: float = 15.0,
        stderr_path: Path | None = None,
    ) -> None:
        self._transport_cm_factory = transport_cm_factory
        self._timeout = timeout
        self._stderr_path = stderr_path
        self._portal_cm: Any = None
        self._portal: BlockingPortal | None = None
        self._send_stream: Any = None
        self._actor_future: Future[None] | None = None
        self._init_result: Any = None

    @property
    def init_result(self) -> Any:
        return self._init_result

    @property
    def stderr_text(self) -> str:
        if self._stderr_path is None or not self._stderr_path.exists():
            return ""
        return self._stderr_path.read_text(encoding="utf-8", errors="replace")

    def __enter__(self) -> SyncMCPSession:
        portal_cm = start_blocking_portal()
        portal = portal_cm.__enter__()
        self._portal_cm = portal_cm
        self._portal = portal
        ready: Future[tuple[Any, Any]] = Future()

        async def _actor() -> None:
            send_stream, receive_stream = anyio.create_memory_object_stream[
                tuple[str, tuple[Any, ...], dict[str, Any], Future[Any]]
            ](32)
            try:
                async with self._transport_cm_factory() as streams:
                    read_stream, write_stream = streams[0], streams[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        init_res = await session.initialize()
                        ready.set_result((init_res, send_stream))
                        async with receive_stream:
                            async for method, args, kwargs, fut in receive_stream:
                                if fut.cancelled():
                                    continue
                                try:
                                    res = await getattr(session, method)(*args, **kwargs)
                                    fut.set_result(res)
                                except BaseException as exc:
                                    fut.set_exception(exc)
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                else:
                    raise

        self._actor_future = portal.start_task_soon(_actor)
        try:
            init_res, send_stream = ready.result(timeout=self._timeout)
        except BaseException as exc:
            try:
                portal_cm.__exit__(type(exc), exc, exc.__traceback__)
            finally:
                self._portal_cm = None
                self._portal = None
            stderr = self.stderr_text
            if stderr:
                raise RuntimeError(
                    f"MCP session failed to initialize: {exc}\nstderr:\n{stderr}"
                ) from exc
            raise
        self._init_result = init_res
        self._send_stream = send_stream
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            if self._portal is not None and self._send_stream is not None:
                try:
                    self._portal.call(self._send_stream.aclose)
                except Exception:
                    pass
            if self._actor_future is not None:
                try:
                    self._actor_future.result(timeout=5.0)
                except Exception:
                    pass
        finally:
            if self._portal_cm is not None:
                self._portal_cm.__exit__(*exc_info)
            self._portal_cm = None
            self._portal = None
            self._send_stream = None
            self._actor_future = None

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._portal is None or self._send_stream is None:
            raise RuntimeError("SyncMCPSession is not active")
        fut: Future[Any] = Future()
        self._portal.call(self._send_stream.send, (method, args, kwargs, fut))
        return fut.result(timeout=self._timeout)

    def list_tools(self) -> Any:
        return self._call("list_tools")

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        return self._call("call_tool", name, arguments=arguments)

    def list_resources(self) -> Any:
        return self._call("list_resources")

    def read_resource(self, uri: Any) -> Any:
        return self._call("read_resource", uri)

    def list_prompts(self) -> Any:
        return self._call("list_prompts")

    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> Any:
        return self._call("get_prompt", name, arguments=arguments)


@contextmanager
def open_stdio_session(
    command: list[str],
    env: dict[str, str],
    stderr_path: Path,
    timeout: float = 15.0,
) -> Iterator[SyncMCPSession]:
    """Launch an MCP stdio server subprocess and yield a synchronous session."""
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w+", encoding="utf-8") as err_fh:
        params = StdioServerParameters(
            command=command[0],
            args=command[1:],
            env=env,
        )
        with SyncMCPSession(
            lambda: stdio_client(params, errlog=err_fh),
            timeout=timeout,
            stderr_path=stderr_path,
        ) as session:
            yield session


def _allocate_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def start_http_server(
    command: list[str],
    env: dict[str, str],
    token: str,
    tmp_dir: Path,
    timeout: float = 10.0,
) -> Iterator[HttpServerInfo]:
    """Spawn `strava-mcp serve` on a free loopback port and wait until ready."""
    port = _allocate_free_port()
    stdout_path = tmp_dir / "http_server_stdout.log"
    stderr_path = tmp_dir / "http_server_stderr.log"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w+", encoding="utf-8") as out_fh,
        stderr_path.open("w+", encoding="utf-8") as err_fh,
    ):
        proc = subprocess.Popen(
            [*command, "serve", "--listen", f"127.0.0.1:{port}"],
            env=env,
            stdout=out_fh,
            stderr=err_fh,
            text=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"HTTP server exited early with code {proc.returncode}.\n"
                        f"stdout:\n{stdout_path.read_text(encoding='utf-8', errors='replace')}\n"
                        f"stderr:\n{stderr_path.read_text(encoding='utf-8', errors='replace')}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise TimeoutError(
                    f"HTTP server failed to bind 127.0.0.1:{port} within {timeout}s.\n"
                    f"stderr:\n{stderr_path.read_text(encoding='utf-8', errors='replace')}"
                )
            yield HttpServerInfo(
                host="127.0.0.1",
                port=port,
                base_url=f"http://127.0.0.1:{port}",
                mcp_url=f"http://127.0.0.1:{port}/mcp",
                token=token,
                process=proc,
                stderr_path=stderr_path,
                stdout_path=stdout_path,
            )
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)


@contextmanager
def open_mcp_http_session(
    mcp_url: str,
    token: str | None = None,
    timeout: float = 15.0,
) -> Iterator[SyncMCPSession]:
    """Open a synchronous Streamable HTTP MCP ClientSession using mcp's httpx2 client."""

    @asynccontextmanager
    async def _transport() -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else None
        async with create_mcp_http_client(headers=headers) as http_client:
            async with streamable_http_client(mcp_url, http_client=http_client) as streams:
                yield streams

    with SyncMCPSession(_transport, timeout=timeout) as session:
        yield session


def parse_tool_result(result: Any) -> dict[str, Any]:
    """Extract structured dict from CallToolResult (structuredContent or JSON content)."""
    if getattr(result, "structured_content", None) is not None:
        sc = result.structured_content
        if isinstance(sc, dict):
            return sc
    content = getattr(result, "content", None) or []
    assert content, f"Tool result had neither structured_content nor content: {result!r}"
    first = content[0]
    text = getattr(first, "text", None)
    assert isinstance(text, str) and text.strip(), f"Empty/non-text tool content: {first!r}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool content[0].text is not valid JSON: {text!r}") from exc
    assert isinstance(parsed, dict), (
        f"Expected JSON object in tool output, got {type(parsed).__name__}"
    )
    return parsed


def _atomic_write_text(path: Path, text: str, file_mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    closed = False
    try:
        os.fchmod(fd, file_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            closed = True
            handle.write(text)
        os.replace(tmp_path, path)
        os.chmod(path, file_mode)
    except BaseException:
        if not closed:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise


@pytest.fixture
def mock_strava() -> Iterator[MockStravaServer]:
    """Start an isolated local MockStravaServer on 127.0.0.1:0."""
    from mock_strava import MockStravaServer

    server = MockStravaServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def isolated_env(tmp_path: Path, mock_strava: MockStravaServer) -> dict[str, str]:
    """Create isolated 0700 config/cache/HOME/XDG dirs and strip host STRAVA_* vars."""
    home_dir = tmp_path / "home"
    config_dir = tmp_path / "config"
    cache_dir = tmp_path / "cache"
    xdg_config = tmp_path / "xdg_config"
    xdg_data = tmp_path / "xdg_data"
    for d in (home_dir, config_dir, cache_dir, xdg_config, xdg_data):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)

    env = {k: v for k, v in os.environ.items() if not k.startswith("STRAVA_")}
    env.update(
        {
            "STRAVA_MCP_CONFIG_DIR": str(config_dir),
            "STRAVA_MCP_CACHE_DIR": str(cache_dir),
            "STRAVA_API_BASE_URL": mock_strava.api_base_url,
            "STRAVA_OAUTH_BASE_URL": mock_strava.oauth_base_url,
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(xdg_config),
            "XDG_DATA_HOME": str(xdg_data),
            "NO_COLOR": "1",
        }
    )
    return env


@pytest.fixture
def seed_config(
    isolated_env: dict[str, str],
    mock_strava: MockStravaServer,
) -> Callable[..., SeededConfig]:
    """Seed client.json, tokens.json, and mcp-token into isolated_env."""

    def _seed(
        *,
        client_id: str = DEFAULT_TEST_CLIENT_ID,
        client_secret: str = DEFAULT_TEST_CLIENT_SECRET,
        access_token: str = DEFAULT_TEST_ACCESS_TOKEN,
        refresh_token: str = DEFAULT_TEST_REFRESH_TOKEN,
        expires_at: int | None = None,
        scope: str = DEFAULT_TEST_SCOPE,
        athlete_id: int = DEFAULT_TEST_ATHLETE_ID,
        mcp_token: str = DEFAULT_TEST_MCP_TOKEN,
        include_tokens: bool = True,
        include_mcp_token: bool = True,
        dir_mode: int = 0o700,
        file_mode: int = 0o600,
        sync_mock: bool = True,
    ) -> SeededConfig:
        config_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
        cache_dir = Path(isolated_env["STRAVA_MCP_CACHE_DIR"])
        config_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(config_dir, 0o700)
        os.chmod(cache_dir, 0o700)

        resolved_exp = int(time.time()) + 21600 if expires_at is None else int(expires_at)
        client_file = config_dir / "client.json"
        tokens_file = config_dir / "tokens.json"
        mcp_token_file = config_dir / "mcp-token"

        _atomic_write_text(
            client_file,
            json.dumps({"client_id": client_id, "client_secret": client_secret}, indent=2) + "\n",
            file_mode,
        )
        if include_tokens:
            _atomic_write_text(
                tokens_file,
                json.dumps(
                    {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "expires_at": resolved_exp,
                        "expires_in": max(0, resolved_exp - int(time.time())),
                        "token_type": "Bearer",
                        "scope": scope,
                        "athlete_id": athlete_id,
                    },
                    indent=2,
                )
                + "\n",
                file_mode,
            )
        elif tokens_file.exists():
            tokens_file.unlink()

        if include_mcp_token:
            _atomic_write_text(mcp_token_file, f"{mcp_token}\n", file_mode)
        elif mcp_token_file.exists():
            mcp_token_file.unlink()

        os.chmod(config_dir, dir_mode)
        os.chmod(cache_dir, dir_mode)

        if sync_mock:
            mock_strava.set_client_credentials(client_id, client_secret)
            if include_tokens:
                mock_strava.set_token_state(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    scopes=scope,
                    expires_at=resolved_exp,
                    athlete_id=athlete_id,
                )

        return SeededConfig(
            config_dir=config_dir,
            cache_dir=cache_dir,
            client_file=client_file,
            tokens_file=tokens_file,
            mcp_token_file=mcp_token_file,
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=resolved_exp,
            scope=scope,
            athlete_id=athlete_id,
            mcp_token=mcp_token,
        )

    return _seed


@pytest.fixture
def run_cli(isolated_env: dict[str, str]) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Invoke `strava-mcp` CLI subprocess inside isolated_env."""

    def _run(
        *args: str,
        input_text: str | None = None,
        env_overrides: dict[str, str] | None = None,
        timeout: float = 30.0,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [*resolve_strava_mcp_cmd(), *args]
        merged_env = dict(isolated_env)
        if env_overrides:
            merged_env.update(env_overrides)
        proc = subprocess.run(
            cmd,
            input=input_text,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and proc.returncode != 0:
            raise AssertionError(
                f"Command {cmd!r} exited {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc

    return _run


@pytest.fixture
def stdio_session(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    tmp_path: Path,
) -> Iterator[SyncMCPSession]:
    """Yield an initialized synchronous MCP stdio session against `strava-mcp stdio`."""
    config_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    if not (config_dir / "client.json").exists():
        seed_config()
    cmd = [*resolve_strava_mcp_cmd(), "stdio"]
    with open_stdio_session(cmd, isolated_env, tmp_path / "stdio_stderr.log") as session:
        yield session


@pytest.fixture
def http_server(
    isolated_env: dict[str, str],
    seed_config: Callable[..., SeededConfig],
    tmp_path: Path,
) -> Iterator[HttpServerInfo]:
    """Start `strava-mcp serve` on a free loopback port and yield HttpServerInfo."""
    config_dir = Path(isolated_env["STRAVA_MCP_CONFIG_DIR"])
    mcp_token_path = config_dir / "mcp-token"
    if not mcp_token_path.exists():
        seed_config()
    token = mcp_token_path.read_text(encoding="utf-8").strip()
    cmd = resolve_strava_mcp_cmd()
    with start_http_server(cmd, isolated_env, token, tmp_path) as info:
        yield info


@pytest.fixture
def mcp_http_client(http_server: HttpServerInfo) -> Iterator[SyncMCPSession]:
    """Yield an initialized synchronous Streamable HTTP MCP ClientSession."""
    with open_mcp_http_session(http_server.mcp_url, http_server.token) as session:
        yield session
