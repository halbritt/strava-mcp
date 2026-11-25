# strava-mcp

A standalone, read-only Python MCP connector exposing Strava activities, athlete profile/zones, gear, routes, segments, clubs, and rolling training metrics to arbitrary MCP agents via `stdio` and authenticated Streamable HTTP (`/mcp`).

## Overview & Architecture

```
MCP Client (stdio / Streamable HTTP + BearerAuth)
  -> mcp_server.py (MCPServer, 17 read-only tools, 5 resources, 3 prompts)
  -> metrics.py (date parsing, dual metric/imperial units, stream downsampling, rollups)
  -> client.py (async httpx Strava API v3, 401 refresh-retry, error mapping)
     <-> cache.py (SQLite WAL cache, 0600, TTLs + immutable historical data)
     <-> rate_limit.py (15m/daily X-RateLimit-* & X-ReadRateLimit-* UTC tracking)
     <-> auth.py (locked OAuth2 token refresh/rotation + loopback consent server)
     <-> config.py (0700/0600 XDG secret store)
```

Key design choices:
- **Strictly read-only**: Every MCP tool carries `readOnlyHint=true`, `destructiveHint=false`, `idempotentHint=true`.
- **Context-safe responses**: Activity lists omit polylines by default, activity details omit `segment_efforts` unless requested, and time-series streams support `summary_only=True` and uniform downsampling (`max_points=200`).
- **Dual units & `sport_type` preference**: Every distance, elevation, speed, and pace output provides metric and imperial values and prefers `sport_type` (e.g., `GravelRide`, `TrailRun`) over legacy `type`.
- **Private-activity blind spot solved**: Strava's `/athletes/{id}/stats` only aggregates `Everyone`-visibility activities. `strava_summarize_metrics` aggregates directly from `/athlete/activities` across `Everyone`, `Followers`, and `Only Me`.
- **Zero secrets in environment or client configs**: Secrets (`client_secret`, `access_token`, `refresh_token`, `mcp-token`) live strictly in `0600` files inside `~/.config/strava-mcp/` (`0700`).

## Installation & Setup

```bash
cd /home/halbritt/git/strava-mcp
uv sync
mkdir -p ~/.local/bin
ln -sf "$(pwd)/.venv/bin/strava-mcp" ~/.local/bin/strava-mcp
install -m 0755 client/strava-mcp-headers ~/.local/bin/strava-mcp-headers
```

Initialize your local secret store (never pass secrets as CLI flags or env vars):

```bash
# Interactive setup (prompts securely for Client ID and Client Secret):
strava-mcp init

# Or non-interactive client init + token import:
printf '%s\n' '<YOUR_CLIENT_SECRET>' | strava-mcp init --client-id <YOUR_CLIENT_ID> --client-secret-stdin --import-tokens-file /path/to/tokens.json
```

This creates:
- `~/.config/strava-mcp/` (`0700`)
- `~/.config/strava-mcp/client.json` (`0600`)
- `~/.config/strava-mcp/tokens.json` (`0600`, if imported or after `auth`)
- `~/.config/strava-mcp/mcp-token` (`0600`, auto-generated bearer token for HTTP transport)

## OAuth Scope Upgrade Guide

Strava's static API Settings page issues a `read`-only token, which cannot read `/athlete/activities` (`activity:read` / `activity:read_all`) or `/athlete/zones` (`profile:read_all`).

Upgrade scopes with the built-in loopback consent server:

```bash
strava-mcp auth
```

What this does:
1. Starts an ephemeral single-use HTTP callback server on `http://127.0.0.1:38762/callback` with constant-time CSRF `state` validation and `Cache-Control: no-store`.
2. Requests `read,read_all,profile:read_all,activity:read_all` with `approval_prompt=force`.
3. Exchanges the authorization code, verifies granted scopes, warns if any recommended scope was unchecked, and persists rotated tokens atomically (`0600`).

Verify live connectivity and scopes:

```bash
strava-mcp status --check-live
```

To revoke tokens remotely and clear local `tokens.json`:

```bash
strava-mcp revoke
```

## CLI Quick Reference

| Command | Purpose |
|---|---|
| `strava-mcp init` | Initialize `0700`/`0600` config store, client creds, optional token import, and `mcp-token` |
| `strava-mcp auth` | Run loopback OAuth2 flow (`127.0.0.1:38762/callback`) to grant full read scopes |
| `strava-mcp status [--check-live] [--json]` | Inspect config, scopes, token expiry, rate limits, and cache stats (zero secrets) |
| `strava-mcp refresh [--json]` | Force locked OAuth2 token refresh |
| `strava-mcp revoke [--token-type ...] [--keep-local]` | Revoke Strava token (`POST /oauth/revoke`) and clear `tokens.json` |
| `strava-mcp stdio` | Run MCP server over stdin/stdout (logs strictly on stderr) |
| `strava-mcp serve [--listen 127.0.0.1:8772] [--allowed-host HOST]` | Run Bearer-authenticated Streamable HTTP `/mcp` server |
| `strava-mcp cache {stats\|clear} [--json]` | Inspect or clear local SQLite response cache |
| `strava-mcp activities [--after ...] [--before ...] [--sport-type ...] [--limit N]` | Query normalized activities |
| `strava-mcp activity <id> [--include-all-efforts] [--include-polyline]` | Inspect single activity details |
| `strava-mcp stats [--athlete-id ID]` | Inspect athlete stats (includes `Everyone`-only caveat) |

Non-secret environment overrides (for isolated testing or custom endpoints):
- `STRAVA_MCP_CONFIG_DIR` (default `~/.config/strava-mcp`)
- `STRAVA_MCP_CACHE_DIR` (default `~/.local/share/strava-mcp`)
- `STRAVA_API_BASE_URL` (default `https://www.strava.com/api/v3`)
- `STRAVA_OAUTH_BASE_URL` (default `https://www.strava.com/oauth`)

## MCP Client Registration (Zero Secrets in Configs)

### Antigravity (Agy) — `~/.gemini/config/mcp_config.json`

```json
{
  "mcpServers": {
    "strava": {
      "command": "/home/<user>/.local/bin/strava-mcp",
      "args": ["stdio"]
    }
  }
}
```

### Claude Code — `~/.claude.json`

Ensure `CLAUDE_CONFIG_DIR` is unset when registering globally.

- **stdio**:
  ```bash
  claude mcp add-json --scope user strava \
    '{"type":"stdio","command":"'"$HOME"'/.local/bin/strava-mcp","args":["stdio"]}'
  ```
- **Streamable HTTP** (uses `strava-mcp-headers` so the bearer token stays out of `~/.claude.json`):
  ```bash
  claude mcp add-json --scope user strava \
    '{"type":"http","url":"http://127.0.0.1:8772/mcp","headersHelper":"'"$HOME"'/.local/bin/strava-mcp-headers"}'
  ```

### Codex — `~/.codex/config.toml`

```toml
[mcp_servers.strava]
command = "/home/<user>/.local/bin/strava-mcp"
args = ["stdio"]
```

### Hermes — `~/.hermes/config.yaml`

```yaml
mcp_servers:
  strava:
    command: /home/<user>/.local/bin/strava-mcp
    args:
      - stdio
```

### OpenCode — `~/.config/opencode/opencode.json`

```json
{
  "mcp": {
    "strava": {
      "type": "local",
      "command": ["/home/<user>/.local/bin/strava-mcp", "stdio"],
      "enabled": true
    }
  }
}
```

### Systemd User Service, Tailscale Serve & ChatGPT ("Frank") Tunnel

Install the hardened systemd user service (`deploy/systemd/strava-mcp.service`):

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/strava-mcp.service ~/.config/systemd/user/strava-mcp.service
systemctl --user daemon-reload
systemctl --user enable --now strava-mcp.service
```

To expose over a private tailnet with Tailscale Serve (keeping the server bound to `127.0.0.1:8772`):
1. Append `--allowed-host <machine>.<tailnet>.ts.net` to `ExecStart` in `~/.config/systemd/user/strava-mcp.service` (DNS-rebinding protection remains active).
2. Publish via Tailscale Serve:
   ```bash
   tailscale serve --bg --https=8772 http://127.0.0.1:8772
   ```
3. For ChatGPT ("Frank") private MCP tunnel (`tunnel-client`), configure a local profile targeting `http://127.0.0.1:8772/mcp` with its `Authorization: Bearer <token>` header sourced from a `0600` credentials file matching `~/.config/strava-mcp/mcp-token`.

## Tools, Resources & Prompts

### Tools (17, All Read-Only)

- `strava_auth_status`: OAuth config, scopes, expiry, rate limits, cache stats, optional live check.
- `strava_get_athlete` / `strava_get_athlete_stats` / `strava_get_athlete_zones`: Athlete profile, stats (`Everyone`-only caveat), and HR/power zones.
- `strava_list_activities` / `strava_get_activity` / `strava_get_activity_laps` / `strava_get_activity_zones` / `strava_get_activity_streams`: Compact activity queries, telemetry, laps, subscription-gated zones, and summarized/downsampled streams.
- `strava_summarize_metrics`: Rolling weekly/monthly/sport/total training rollups across all activity visibilities.
- `strava_get_gear`: Bike and shoe mileage (strictly validated `gear_id`).
- `strava_list_routes` / `strava_get_route`: Athlete routes with optional GPX/TCX XML export.
- `strava_list_starred_segments` / `strava_get_segment` / `strava_list_segment_efforts`: Starred segments, details/streams, and subscription-gated efforts.
- `strava_list_clubs`: Athlete clubs.

### Resources (5) & Prompts (3)

- Resources: `strava://athlete/profile`, `strava://athlete/stats`, `strava://athlete/zones`, `strava://status/rate-limits`, `strava://activities/{activity_id}`.
- Prompts: `weekly_training_review`, `activity_deep_dive`, `gear_wear_check`.
