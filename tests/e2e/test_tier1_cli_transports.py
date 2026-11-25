"""Tier 1 opaque-box E2E tests for F11: CLI query subcommands and MCP transports."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from typing import Any

import pytest
from conftest import (
    DEFAULT_TEST_ATHLETE_ID,
    REPO_ROOT,
    SeededConfig,
    open_mcp_http_session,
    parse_tool_result,
)
from mock_strava import MockStravaServer

_CONTRACT_TOOLS: frozenset[str] = frozenset(
    {
        "strava_auth_status",
        "strava_get_athlete",
        "strava_get_athlete_stats",
        "strava_get_athlete_zones",
        "strava_list_activities",
        "strava_get_activity",
        "strava_get_activity_laps",
        "strava_get_activity_zones",
        "strava_get_activity_streams",
        "strava_summarize_metrics",
        "strava_get_gear",
        "strava_list_routes",
        "strava_get_route",
        "strava_list_starred_segments",
        "strava_get_segment",
        "strava_list_segment_efforts",
        "strava_list_clubs",
    }
)


def _assert_no_secrets_leaked(
    text: str,
    mock_strava: MockStravaServer,
    *extra_secrets: str,
) -> None:
    candidates = [s for s in extra_secrets if s]
    active = mock_strava.get_active_tokens()
    candidates.extend(active.get("active_access_tokens", {}).keys())
    candidates.extend(active.get("active_refresh_tokens", {}).keys())
    candidates.extend(active.get("invalidated_refresh_tokens", []))
    for secret in dict.fromkeys(candidates):
        assert secret not in text, "Sensitive credential value leaked in output"


# =============================================================================
# F11: CLI & MCP Transports (7 Tier 1 tests)
# =============================================================================


def test_f11_tier1_stdio_initialize_and_list_all_read_only_tools(
    stdio_session: Any,
) -> None:
    init_res = stdio_session.init_result
    assert init_res.protocol_version
    assert init_res.server_info and init_res.server_info.name
    assert init_res.capabilities and init_res.capabilities.tools is not None

    tools = stdio_session.list_tools().tools
    assert len(tools) >= 15
    tool_names = {t.name for t in tools}
    assert _CONTRACT_TOOLS <= tool_names, f"Missing tools: {_CONTRACT_TOOLS - tool_names}"


def test_f11_tier1_all_tools_have_read_only_annotations(
    stdio_session: Any,
) -> None:
    tools = stdio_session.list_tools().tools
    assert tools
    for tool in tools:
        ann = tool.annotations
        assert ann is not None, f"Tool {tool.name} missing annotations"
        assert ann.read_only_hint is True, f"Tool {tool.name} not read_only_hint=True"
        assert ann.destructive_hint is False, f"Tool {tool.name} not destructive_hint=False"
        assert ann.idempotent_hint is True, f"Tool {tool.name} not idempotent_hint=True"


def test_f11_tier1_dual_structured_content_and_json_text_output(
    stdio_session: Any,
) -> None:
    tools_by_name = {t.name: t for t in stdio_session.list_tools().tools}
    for name in ("strava_auth_status", "strava_get_athlete"):
        tool = tools_by_name[name]
        assert tool.output_schema is not None
        assert tool.output_schema.get("type") == "object"

    for name in ("strava_auth_status", "strava_get_athlete"):
        res = stdio_session.call_tool(name, {})
        assert res.is_error is not True
        assert isinstance(res.structured_content, dict)
        assert res.content and isinstance(res.content[0].text, str)
        assert json.loads(res.content[0].text) == res.structured_content
        assert parse_tool_result(res) == res.structured_content


def test_f11_tier1_list_and_read_strava_resources(
    stdio_session: Any,
) -> None:
    listed = stdio_session.list_resources()
    uris = {str(r.uri) for r in listed.resources}
    expected_uris = {
        "strava://athlete/profile",
        "strava://athlete/stats",
        "strava://athlete/zones",
        "strava://status/rate-limits",
    }
    assert expected_uris <= uris, f"Missing resources: {expected_uris - uris}"

    for uri in ("strava://athlete/profile", "strava://status/rate-limits", "strava://activities/9001"):
        res = stdio_session.read_resource(uri)
        assert res.contents, f"Empty resource contents for {uri}"
        text = getattr(res.contents[0], "text", "")
        assert isinstance(text, str) and text.strip()
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


def test_f11_tier1_list_and_get_prompts(
    stdio_session: Any,
) -> None:
    prompts = stdio_session.list_prompts().prompts
    names = {p.name for p in prompts}
    expected = {"weekly_training_review", "activity_deep_dive", "gear_wear_check"}
    assert expected <= names, f"Missing prompts: {expected - names}"

    p_review = stdio_session.get_prompt("weekly_training_review", {})
    assert p_review.messages

    p_deep = stdio_session.get_prompt("activity_deep_dive", {"activity_id": "9001"})
    assert p_deep.messages

    p_gear = stdio_session.get_prompt("gear_wear_check", {})
    assert p_gear.messages


def test_f11_tier1_streamable_http_bearer_auth_and_headers_helper(
    http_server: Any,
    isolated_env: dict[str, str],
) -> None:
    headers_script = REPO_ROOT / "client" / "strava-mcp-headers"
    if not headers_script.is_file() or not os.access(headers_script, os.X_OK):
        pytest.fail(
            f"client/strava-mcp-headers missing or not executable: {headers_script}",
            pytrace=False,
        )

    proc = subprocess.run(
        [str(headers_script)],
        env=isolated_env,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    assert proc.returncode == 0, f"strava-mcp-headers failed: {proc.stdout}\n{proc.stderr}"
    hdr_payload = json.loads(proc.stdout)
    expected_auth = f"Bearer {http_server.token}"
    assert hdr_payload.get("Authorization") == expected_auth
    token = hdr_payload["Authorization"].removeprefix("Bearer ").strip()

    with open_mcp_http_session(http_server.mcp_url, token) as session:
        tools = session.list_tools().tools
        assert len(tools) >= 15
        res = session.call_tool("strava_get_athlete", {})
        assert res.is_error is not True
        athlete = parse_tool_result(res)
        assert athlete["id"] == DEFAULT_TEST_ATHLETE_ID


def test_f11_tier1_cli_query_and_help_subcommands(
    seed_config: Callable[..., SeededConfig],
    mock_strava: MockStravaServer,
    run_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    seeded = seed_config()

    help_proc = run_cli("--help", check=True)
    assert "strava-mcp" in help_proc.stdout.lower()

    status_proc = run_cli("status", check=True)
    assert status_proc.stdout.strip()

    acts_proc = run_cli("activities", "--limit", "2", check=True)
    assert "Morning Gravel Grind" in acts_proc.stdout or "9001" in acts_proc.stdout

    act_proc = run_cli("activity", "9001", check=True)
    assert "Morning Gravel Grind" in act_proc.stdout
    assert "9001" in act_proc.stdout

    stats_proc = run_cli("stats", check=True)
    assert stats_proc.stdout.strip()
    assert any(k in stats_proc.stdout.lower() for k in ("ride", "distance", "count", "totals"))

    combined = "\n".join(
        p.stdout + p.stderr
        for p in (help_proc, status_proc, acts_proc, act_proc, stats_proc)
    )
    _assert_no_secrets_leaked(
        combined,
        mock_strava,
        seeded.client_secret,
        seeded.access_token,
        seeded.refresh_token,
        seeded.mcp_token,
    )
