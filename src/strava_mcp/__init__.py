"""Read-only Strava MCP connector."""

__version__ = "0.1.0"

from strava_mcp.cache import CacheEntry, ResponseCache, make_cache_key
from strava_mcp.client import StravaClient
from strava_mcp.config import StravaMCPError
from strava_mcp.mcp_server import build_http_app, create_mcp_server
from strava_mcp.rate_limit import BucketState, RateLimitTracker

__all__ = [
    "BucketState",
    "CacheEntry",
    "RateLimitTracker",
    "ResponseCache",
    "StravaClient",
    "StravaMCPError",
    "__version__",
    "build_http_app",
    "create_mcp_server",
    "make_cache_key",
]
