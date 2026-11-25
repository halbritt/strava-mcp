"""Date parsing, unit normalization, stream downsampling, and metric rollups for strava-mcp."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

if TYPE_CHECKING:
    from strava_mcp.client import StravaClient

METERS_PER_KM = 1000.0
METERS_PER_MILE = 1609.344
FEET_PER_METER = 1.0 / 0.3048
MPS_TO_KPH = 3.6
MPS_TO_MPH = 3600.0 / 1609.344
KG_TO_LBS = 2.20462262185

VALID_GROUP_BY: tuple[str, ...] = ("week", "month", "sport", "total")

STATS_VISIBILITY_CAVEAT = (
    "Strava /athletes/{id}/stats only aggregates 'Everyone'-visibility activities. "
    "Use strava_summarize_metrics for accurate totals including 'Followers' and 'Only Me' "
    "activities."
)

_RELATIVE_DURATION_RE = re.compile(r"^(\d+)\s*([hdwmyHDWMY])$")
_NUMERIC_EPOCH_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SPORT_FOLD_RE = re.compile(r"[\s_-]+")
_GEAR_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_real_number(val: Any) -> TypeGuard[int | float]:
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(float(val))
    )


def _normalize_utc(
    dt: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> datetime:
    resolved = dt if dt is not None else (clock() if clock is not None else datetime.now(UTC))
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=UTC)
    try:
        return resolved.astimezone(UTC)
    except OverflowError as exc:
        raise ValueError(f"Datetime value out of UTC range: {resolved.isoformat()!r}") from exc


def parse_date_filter(
    value: int | float | str | datetime | date | None,
    *,
    is_end: bool = False,
    now: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> int | None:
    """Parse epoch, YYYY-MM-DD, ISO-8601 datetime, or relative duration to UTC epoch seconds."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Date filter must not be a boolean.")
    if isinstance(value, (int, float)):
        try:
            num = float(value)
        except OverflowError as exc:
            raise ValueError(f"Invalid numeric epoch date filter: {value!r}") from exc
        if not math.isfinite(num) or num < 0:
            raise ValueError(f"Invalid numeric epoch date filter: {value!r}")
        return int(num)
    if isinstance(value, datetime):
        return int(_normalize_utc(value).timestamp())
    if isinstance(value, date):
        t = time(23, 59, 59) if is_end else time(0, 0, 0)
        return int(datetime.combine(value, t, tzinfo=UTC).timestamp())
    if not isinstance(value, str):
        raise ValueError(f"Unsupported date filter type: {type(value).__name__}")

    text = value.strip()
    if not text:
        raise ValueError("Date filter string must not be empty.")

    if _NUMERIC_EPOCH_RE.fullmatch(text):
        num = float(text)
        if not math.isfinite(num) or num < 0:
            raise ValueError(f"Invalid numeric epoch date filter: {value!r}")
        return int(num)

    rel_match = _RELATIVE_DURATION_RE.fullmatch(text)
    if rel_match is not None:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2).lower()
        ref_now = _normalize_utc(now, clock)
        try:
            if unit == "h":
                delta = timedelta(hours=amount)
            elif unit == "d":
                delta = timedelta(days=amount)
            elif unit == "w":
                delta = timedelta(weeks=amount)
            elif unit == "m":
                delta = timedelta(days=30 * amount)
            else:
                delta = timedelta(days=365 * amount)
            return int((ref_now - delta).timestamp())
        except OverflowError:
            raise ValueError(f"Relative duration out of range: {value!r}") from None

    if _DATE_ONLY_RE.fullmatch(text):
        parsed_date = date.fromisoformat(text)
        t = time(23, 59, 59) if is_end else time(0, 0, 0)
        return int(datetime.combine(parsed_date, t, tzinfo=UTC).timestamp())

    try:
        parsed_dt = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"Invalid date filter {value!r}; expected epoch, YYYY-MM-DD, ISO-8601 datetime, "
            "or relative duration (e.g. 7d, 30d, 3m, 1y)."
        ) from None
    return int(_normalize_utc(parsed_dt).timestamp())


def parse_date_iso(
    value: int | float | str | datetime | date | None,
    *,
    is_end: bool = False,
    now: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> str | None:
    """Parse any supported date filter into a UTC ISO-8601 timestamp string."""
    epoch = parse_date_filter(value, is_end=is_end, now=now, clock=clock)
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"Date filter timestamp out of range: {value!r}") from exc


def format_duration(seconds: int | float | None) -> str | None:
    """Format duration in seconds as H:MM:SS or M:SS."""
    if not _is_real_number(seconds) or float(seconds) < 0:
        return None
    total = round(float(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_pace(
    speed_mps: int | float | None,
    *,
    meters_per_unit: float = METERS_PER_KM,
) -> str | None:
    """Format speed in m/s as pace string (M:SS per km or mi), or None if non-positive."""
    if not _is_real_number(speed_mps) or float(speed_mps) <= 0:
        return None
    total_seconds = round(meters_per_unit / float(speed_mps))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}:{secs:02d}"


def normalize_activity_units(activity: Mapping[str, Any]) -> dict[str, Any]:
    """Compute dual metric/imperial distance, elevation, speed, duration, and pace fields."""
    raw_dist = activity.get("distance")
    raw_mt = activity.get("moving_time")
    raw_et = activity.get("elapsed_time")
    raw_elev = activity.get("total_elevation_gain")
    raw_avg_speed = activity.get("average_speed")

    distance_m = round(float(raw_dist), 2) if _is_real_number(raw_dist) else None
    distance_km = round(distance_m / METERS_PER_KM, 2) if distance_m is not None else None
    distance_mi = round(distance_m / METERS_PER_MILE, 2) if distance_m is not None else None

    moving_time_s = round(float(raw_mt)) if _is_real_number(raw_mt) else None
    elapsed_time_s = round(float(raw_et)) if _is_real_number(raw_et) else None

    elevation_gain_m = round(float(raw_elev), 1) if _is_real_number(raw_elev) else None
    elevation_gain_ft = (
        round(float(raw_elev) * FEET_PER_METER, 1) if _is_real_number(raw_elev) else None
    )

    if _is_real_number(raw_avg_speed):
        avg_speed_mps: float | None = float(raw_avg_speed)
    elif (
        distance_m is not None
        and distance_m > 0
        and moving_time_s is not None
        and moving_time_s > 0
    ):
        avg_speed_mps = distance_m / moving_time_s
    else:
        avg_speed_mps = None

    avg_speed_kph = round(avg_speed_mps * MPS_TO_KPH, 2) if avg_speed_mps is not None else None
    avg_speed_mph = round(avg_speed_mps * MPS_TO_MPH, 2) if avg_speed_mps is not None else None

    if (
        avg_speed_mps is None
        or avg_speed_mps <= 0
        or (distance_m is not None and distance_m <= 0)
    ):
        pace_min_per_km = None
        pace_min_per_mi = None
    else:
        pace_min_per_km = format_pace(avg_speed_mps, meters_per_unit=METERS_PER_KM)
        pace_min_per_mi = format_pace(avg_speed_mps, meters_per_unit=METERS_PER_MILE)

    return {
        "distance_m": distance_m,
        "distance_km": distance_km,
        "distance_mi": distance_mi,
        "moving_time_seconds": moving_time_s,
        "moving_time_formatted": format_duration(moving_time_s),
        "elapsed_time_seconds": elapsed_time_s,
        "elapsed_time_formatted": format_duration(elapsed_time_s),
        "elevation_gain_m": elevation_gain_m,
        "elevation_gain_ft": elevation_gain_ft,
        "avg_speed_kph": avg_speed_kph,
        "avg_speed_mph": avg_speed_mph,
        "pace_min_per_km": pace_min_per_km,
        "pace_min_per_mi": pace_min_per_mi,
    }


def normalize_sport_type(activity: Mapping[str, Any]) -> str:
    """Return Strava sport_type if present, falling back to legacy type or 'Unknown'."""
    sport_type = activity.get("sport_type")
    if isinstance(sport_type, str) and sport_type.strip():
        return sport_type.strip()
    legacy_type = activity.get("type")
    if isinstance(legacy_type, str) and legacy_type.strip():
        return legacy_type.strip()
    return "Unknown"


def _fold_sport(value: str) -> str:
    return _SPORT_FOLD_RE.sub("", value.strip()).casefold()


def matches_sport_type(activity: Mapping[str, Any], sport_filter: str | None) -> bool:
    """Return True if activity matches sport_filter via sport_type or legacy type."""
    if sport_filter is None or not sport_filter.strip():
        return True
    target = _fold_sport(sport_filter)
    for field_name in ("sport_type", "type"):
        raw = activity.get(field_name)
        if isinstance(raw, str) and raw.strip() and _fold_sport(raw) == target:
            return True
    return False


def validate_gear_id(gear_id: str) -> str:
    """Validate Strava gear_id against path traversal, query injection, and whitespace."""
    if not isinstance(gear_id, str):
        raise ValueError("gear_id must be a string.")
    cleaned = gear_id.strip()
    if not cleaned:
        raise ValueError("gear_id must not be empty.")
    if (
        cleaned in {".", ".."}
        or ".." in cleaned
        or any(ch in cleaned for ch in ("/", "\\", "?", "#", "%", "\x00"))
        or any(ch.isspace() for ch in cleaned)
        or _GEAR_ID_RE.fullmatch(cleaned) is None
    ):
        raise ValueError(
            "Invalid gear_id: must be a single identifier without '/', '?', '#', '..', "
            "or whitespace."
        )
    return cleaned


def normalize_gear(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a Strava gear (bike/shoe) object with metric and imperial distances."""
    raw_dist = raw.get("distance")
    distance_m = round(float(raw_dist), 1) if _is_real_number(raw_dist) else None
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "primary": raw.get("primary"),
        "retired": raw.get("retired"),
        "brand_name": raw.get("brand_name"),
        "model_name": raw.get("model_name"),
        "description": raw.get("description"),
        "frame_type": raw.get("frame_type"),
        "weight": raw.get("weight"),
        "distance_m": distance_m,
        "distance_km": round(distance_m / METERS_PER_KM, 1) if distance_m is not None else None,
        "distance_mi": round(distance_m / METERS_PER_MILE, 1) if distance_m is not None else None,
    }


def normalize_athlete(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Strava athlete profile with metric/imperial weight and gear summaries."""
    raw_weight = raw.get("weight")
    weight_kg = round(float(raw_weight), 1) if _is_real_number(raw_weight) else None
    weight_lbs = round(float(raw_weight) * KG_TO_LBS, 1) if _is_real_number(raw_weight) else None

    raw_bikes = raw.get("bikes")
    bikes = (
        [normalize_gear(b) for b in raw_bikes if isinstance(b, Mapping)]
        if isinstance(raw_bikes, list)
        else []
    )
    raw_shoes = raw.get("shoes")
    shoes = (
        [normalize_gear(s) for s in raw_shoes if isinstance(s, Mapping)]
        if isinstance(raw_shoes, list)
        else []
    )
    raw_clubs = raw.get("clubs")
    clubs = (
        [
            {"id": c.get("id"), "name": c.get("name"), "sport_type": c.get("sport_type")}
            for c in raw_clubs
            if isinstance(c, Mapping)
        ]
        if isinstance(raw_clubs, list)
        else []
    )

    return {
        "id": raw.get("id"),
        "username": raw.get("username"),
        "firstname": raw.get("firstname"),
        "lastname": raw.get("lastname"),
        "city": raw.get("city"),
        "state": raw.get("state"),
        "country": raw.get("country"),
        "sex": raw.get("sex"),
        "premium": raw.get("premium"),
        "summit": raw.get("summit"),
        "measurement_preference": raw.get("measurement_preference"),
        "ftp": raw.get("ftp"),
        "weight_kg": weight_kg,
        "weight_lbs": weight_lbs,
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "bikes": bikes,
        "shoes": shoes,
        "clubs": clubs,
    }


def _normalize_stat_totals(bucket: Any) -> dict[str, Any] | None:
    if not isinstance(bucket, Mapping):
        return None
    raw_dist = bucket.get("distance")
    raw_mt = bucket.get("moving_time")
    raw_et = bucket.get("elapsed_time")
    raw_elev = bucket.get("elevation_gain")

    distance_m = round(float(raw_dist), 2) if _is_real_number(raw_dist) else None
    moving_s = round(float(raw_mt)) if _is_real_number(raw_mt) else None
    elapsed_s = round(float(raw_et)) if _is_real_number(raw_et) else None
    elev_m = round(float(raw_elev), 1) if _is_real_number(raw_elev) else None

    return {
        "count": bucket.get("count"),
        "distance_m": distance_m,
        "distance_km": round(distance_m / METERS_PER_KM, 2) if distance_m is not None else None,
        "distance_mi": round(distance_m / METERS_PER_MILE, 2) if distance_m is not None else None,
        "moving_time_seconds": moving_s,
        "moving_time_formatted": format_duration(moving_s),
        "elapsed_time_seconds": elapsed_s,
        "elevation_gain_m": elev_m,
        "elevation_gain_ft": round(elev_m * FEET_PER_METER, 1) if elev_m is not None else None,
        "achievement_count": bucket.get("achievement_count"),
    }


def normalize_athlete_stats(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Strava athlete stats buckets with metric and imperial units."""
    raw_big_ride = raw.get("biggest_ride_distance")
    big_ride_m = round(float(raw_big_ride), 2) if _is_real_number(raw_big_ride) else None
    raw_big_climb = raw.get("biggest_climb_elevation_gain")
    big_climb_m = round(float(raw_big_climb), 1) if _is_real_number(raw_big_climb) else None

    result: dict[str, Any] = {
        "biggest_ride_distance_m": big_ride_m,
        "biggest_ride_distance_km": (
            round(big_ride_m / METERS_PER_KM, 2) if big_ride_m is not None else None
        ),
        "biggest_ride_distance_mi": (
            round(big_ride_m / METERS_PER_MILE, 2) if big_ride_m is not None else None
        ),
        "biggest_climb_elevation_gain_m": big_climb_m,
        "biggest_climb_elevation_gain_ft": (
            round(big_climb_m * FEET_PER_METER, 1) if big_climb_m is not None else None
        ),
    }
    for key in (
        "recent_run_totals",
        "recent_ride_totals",
        "recent_swim_totals",
        "ytd_run_totals",
        "ytd_ride_totals",
        "ytd_swim_totals",
        "all_run_totals",
        "all_ride_totals",
        "all_swim_totals",
    ):
        result[key] = _normalize_stat_totals(raw.get(key))
    return result


def normalize_lap(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an activity lap with dual units and telemetry fields."""
    units = normalize_activity_units(raw)
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "lap_index": raw.get("lap_index"),
        "split": raw.get("split"),
        "start_date": raw.get("start_date"),
        "start_date_local": raw.get("start_date_local"),
        "start_index": raw.get("start_index"),
        "end_index": raw.get("end_index"),
        **units,
        "average_heartrate": raw.get("average_heartrate"),
        "max_heartrate": raw.get("max_heartrate"),
        "average_watts": raw.get("average_watts"),
        "average_cadence": raw.get("average_cadence"),
        "pace_zone": raw.get("pace_zone"),
    }


def normalize_route(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a Strava route without raw polyline bloat."""
    raw_dist = raw.get("distance")
    distance_m = round(float(raw_dist), 2) if _is_real_number(raw_dist) else None
    raw_elev = raw.get("elevation_gain")
    elev_m = round(float(raw_elev), 1) if _is_real_number(raw_elev) else None
    raw_est = raw.get("estimated_moving_time")
    est_s = round(float(raw_est)) if _is_real_number(raw_est) else None
    raw_segments = raw.get("segments")
    segments_count = len(raw_segments) if isinstance(raw_segments, list) else None

    return {
        "id": raw.get("id"),
        "id_str": raw.get("id_str"),
        "name": raw.get("name"),
        "description": raw.get("description"),
        "type": raw.get("type"),
        "sub_type": raw.get("sub_type"),
        "distance_m": distance_m,
        "distance_km": round(distance_m / METERS_PER_KM, 2) if distance_m is not None else None,
        "distance_mi": round(distance_m / METERS_PER_MILE, 2) if distance_m is not None else None,
        "elevation_gain_m": elev_m,
        "elevation_gain_ft": round(elev_m * FEET_PER_METER, 1) if elev_m is not None else None,
        "estimated_moving_time_seconds": est_s,
        "estimated_moving_time_formatted": format_duration(est_s),
        "private": raw.get("private"),
        "starred": raw.get("starred"),
        "timestamp": raw.get("timestamp"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "segments_count": segments_count,
    }


def normalize_segment(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a Strava segment with metric and imperial distances/elevation."""
    raw_dist = raw.get("distance")
    distance_m = round(float(raw_dist), 2) if _is_real_number(raw_dist) else None
    raw_elev = raw.get("total_elevation_gain")
    elev_m = round(float(raw_elev), 1) if _is_real_number(raw_elev) else None
    raw_high = raw.get("elevation_high")
    raw_low = raw.get("elevation_low")

    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "activity_type": raw.get("activity_type"),
        "distance_m": distance_m,
        "distance_km": round(distance_m / METERS_PER_KM, 2) if distance_m is not None else None,
        "distance_mi": round(distance_m / METERS_PER_MILE, 2) if distance_m is not None else None,
        "average_grade": raw.get("average_grade"),
        "maximum_grade": raw.get("maximum_grade"),
        "elevation_high_m": round(float(raw_high), 1) if _is_real_number(raw_high) else None,
        "elevation_low_m": round(float(raw_low), 1) if _is_real_number(raw_low) else None,
        "total_elevation_gain_m": elev_m,
        "total_elevation_gain_ft": (
            round(elev_m * FEET_PER_METER, 1) if elev_m is not None else None
        ),
        "climb_category": raw.get("climb_category"),
        "city": raw.get("city"),
        "state": raw.get("state"),
        "country": raw.get("country"),
        "private": raw.get("private"),
        "starred": raw.get("starred"),
        "hazardous": raw.get("hazardous"),
        "effort_count": raw.get("effort_count"),
        "athlete_count": raw.get("athlete_count"),
        "star_count": raw.get("star_count"),
        "start_latlng": raw.get("start_latlng"),
        "end_latlng": raw.get("end_latlng"),
        "athlete_pr_effort": raw.get("athlete_pr_effort"),
    }


def format_activity_summary(
    activity: Mapping[str, Any],
    *,
    include_polyline: bool = False,
) -> dict[str, Any]:
    """Format a compact, unit-normalized activity summary (polylines omitted by default)."""
    units = normalize_activity_units(activity)
    result: dict[str, Any] = {
        "id": activity.get("id"),
        "name": activity.get("name"),
        "sport_type": normalize_sport_type(activity),
        "type": activity.get("type"),
        "start_date": activity.get("start_date"),
        "start_date_local": activity.get("start_date_local"),
        "timezone": activity.get("timezone"),
        **units,
        "average_heartrate": activity.get("average_heartrate"),
        "max_heartrate": activity.get("max_heartrate"),
        "average_watts": activity.get("average_watts"),
        "weighted_average_watts": activity.get("weighted_average_watts"),
        "max_watts": activity.get("max_watts"),
        "kilojoules": activity.get("kilojoules"),
        "average_cadence": activity.get("average_cadence"),
        "suffer_score": activity.get("suffer_score"),
        "calories": activity.get("calories"),
        "gear_id": activity.get("gear_id"),
        "trainer": activity.get("trainer"),
        "commute": activity.get("commute"),
        "manual": activity.get("manual"),
        "private": activity.get("private"),
        "visibility": activity.get("visibility"),
        "kudos_count": activity.get("kudos_count"),
        "comment_count": activity.get("comment_count"),
        "achievement_count": activity.get("achievement_count"),
        "pr_count": activity.get("pr_count"),
        "start_latlng": activity.get("start_latlng"),
        "end_latlng": activity.get("end_latlng"),
    }
    if include_polyline and isinstance(activity.get("map"), Mapping):
        result["map"] = dict(activity["map"])
    return result


def format_activity_detail(
    activity: Mapping[str, Any],
    *,
    include_all_efforts: bool = False,
    include_polyline: bool = False,
) -> dict[str, Any]:
    """Format detailed activity payload with opt-in segment_efforts and polylines."""
    result = format_activity_summary(activity, include_polyline=include_polyline)
    raw_gear = activity.get("gear")
    raw_laps = activity.get("laps")
    raw_efforts = activity.get("segment_efforts")
    efforts = (
        [dict(item) for item in raw_efforts if isinstance(item, Mapping)]
        if isinstance(raw_efforts, list)
        else []
    )

    result.update(
        {
            "description": activity.get("description"),
            "device_name": activity.get("device_name"),
            "workout_type": activity.get("workout_type"),
            "perceived_exertion": activity.get("perceived_exertion"),
            "gear": normalize_gear(raw_gear) if isinstance(raw_gear, Mapping) else None,
            "splits_metric": activity.get("splits_metric"),
            "splits_standard": activity.get("splits_standard"),
            "laps": (
                [normalize_lap(lap) for lap in raw_laps if isinstance(lap, Mapping)]
                if isinstance(raw_laps, list)
                else None
            ),
            "best_efforts": activity.get("best_efforts"),
            "segment_efforts_count": len(efforts),
        }
    )
    if include_all_efforts:
        result["segment_efforts"] = efforts
    return result


def _coerce_streams_mapping(
    streams: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(streams, Mapping):
        return {str(k): v for k, v in streams.items()}
    mapped: dict[str, Any] = {}
    for item in streams:
        if isinstance(item, Mapping):
            key = item.get("type")
            if isinstance(key, str) and key.strip():
                mapped[key.strip()] = item
    return mapped


def _extract_stream_parts(stream_obj: Any) -> tuple[list[Any], int, Any, Any]:
    if isinstance(stream_obj, Mapping):
        raw_data = stream_obj.get("data")
        data = list(raw_data) if isinstance(raw_data, list) else []
        raw_orig = stream_obj.get("original_size")
        orig_size = (
            int(raw_orig)
            if isinstance(raw_orig, int) and not isinstance(raw_orig, bool)
            else len(data)
        )
        return (data, orig_size, stream_obj.get("resolution"), stream_obj.get("series_type"))
    if isinstance(stream_obj, list):
        return (list(stream_obj), len(stream_obj), None, None)
    return ([], 0, None, None)


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _is_latlng_pair(val: Any) -> bool:
    return (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and _is_real_number(val[0])
        and _is_real_number(val[1])
    )


def summarize_streams(
    streams: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize numeric, boolean, and latlng activity/segment streams."""
    mapped = _coerce_streams_mapping(streams)
    summaries: dict[str, dict[str, Any]] = {}

    for key, stream_obj in mapped.items():
        data, original_size, resolution, series_type = _extract_stream_parts(stream_obj)
        non_none = [v for v in data if v is not None]
        base: dict[str, Any] = {
            "count": len(data),
            "original_size": original_size,
            "resolution": resolution,
            "series_type": series_type,
            "min": None,
            "max": None,
            "avg": None,
            "median": None,
            "p95": None,
        }

        if key == "moving" or (non_none and all(isinstance(v, bool) for v in non_none)):
            true_count = sum(1 for v in data if v is True)
            false_count = sum(1 for v in data if v is False)
            total_bool = true_count + false_count
            base["stream_type"] = "boolean"
            base["true_count"] = true_count
            base["false_count"] = false_count
            base["true_fraction"] = (
                round(true_count / total_bool, 4) if total_bool > 0 else None
            )
        elif key == "latlng" or (non_none and all(_is_latlng_pair(v) for v in non_none)):
            pairs = [[float(v[0]), float(v[1])] for v in non_none if _is_latlng_pair(v)]
            base["stream_type"] = "latlng"
            if pairs:
                lats = [p[0] for p in pairs]
                lngs = [p[1] for p in pairs]
                base["start_latlng"] = pairs[0]
                base["end_latlng"] = pairs[-1]
                base["bounds"] = {
                    "min_lat": min(lats),
                    "max_lat": max(lats),
                    "min_lng": min(lngs),
                    "max_lng": max(lngs),
                }
            else:
                base["start_latlng"] = None
                base["end_latlng"] = None
                base["bounds"] = None
        else:
            nums = [float(v) for v in non_none if _is_real_number(v)]
            base["stream_type"] = "numeric"
            if nums:
                ordered = sorted(nums)
                base["min"] = round(ordered[0], 4)
                base["max"] = round(ordered[-1], 4)
                base["avg"] = round(sum(ordered) / len(ordered), 4)
                base["median"] = round(float(statistics.median(ordered)), 4)
                base["p95"] = round(_percentile(ordered, 0.95), 4)

        summaries[key] = base
    return summaries


def downsample_streams(
    streams: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    max_points: int = 200,
    summary_only: bool = False,
) -> dict[str, Any]:
    """Summarize streams and uniformly downsample point arrays preserving endpoints."""
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")

    mapped = _coerce_streams_mapping(streams)
    summaries = summarize_streams(mapped)
    result_streams: dict[str, dict[str, Any]] = {}

    for key, summary in summaries.items():
        data, _orig, _res, _st = _extract_stream_parts(mapped[key])
        if summary_only:
            result_streams[key] = {**summary, "sampled_count": 0}
            continue

        n = len(data)
        if n <= max_points:
            sampled = list(data)
        elif max_points == 1:
            sampled = [data[0]]
        else:
            indices = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
            sampled = [data[idx] for idx in indices]

        result_streams[key] = {
            **summary,
            "sampled_count": len(sampled),
            "data": sampled,
        }

    return {
        "max_points": max_points,
        "summary_only": summary_only,
        "stream_keys": list(summaries.keys()),
        "streams": result_streams,
    }


def _parse_activity_datetime(raw: Any, *, as_utc: bool) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC) if as_utc else dt


def _activity_epoch_utc(activity: Mapping[str, Any]) -> int | None:
    dt = _parse_activity_datetime(activity.get("start_date"), as_utc=True)
    if dt is None:
        dt = _parse_activity_datetime(activity.get("start_date_local"), as_utc=True)
    return int(dt.timestamp()) if dt is not None else None


def _activity_local_dt(activity: Mapping[str, Any]) -> datetime | None:
    dt = _parse_activity_datetime(activity.get("start_date_local"), as_utc=False)
    if dt is None:
        dt = _parse_activity_datetime(activity.get("start_date"), as_utc=True)
    return dt


def _weighted_average(pairs: list[tuple[float, float]]) -> float | None:
    if not pairs:
        return None
    total_weight = sum(w for _, w in pairs if w > 0)
    if total_weight > 0:
        return round(sum(val * w for val, w in pairs if w > 0) / total_weight, 1)
    return round(sum(val for val, _ in pairs) / len(pairs), 1)


def _aggregate_bucket(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    total_dist_m = 0.0
    total_elev_m = 0.0
    total_moving_s = 0
    total_elapsed_s = 0
    total_kj = 0.0
    total_suffer = 0.0
    hr_pairs: list[tuple[float, float]] = []
    watts_pairs: list[tuple[float, float]] = []
    weighted_watts_pairs: list[tuple[float, float]] = []
    max_hrs: list[float] = []
    private_count = 0

    for act in items:
        raw_dist = act.get("distance")
        if _is_real_number(raw_dist):
            total_dist_m += float(raw_dist)
        raw_elev = act.get("total_elevation_gain")
        if _is_real_number(raw_elev):
            total_elev_m += float(raw_elev)
        raw_mt = act.get("moving_time")
        mt = round(float(raw_mt)) if _is_real_number(raw_mt) else 0
        if mt > 0:
            total_moving_s += mt
        raw_et = act.get("elapsed_time")
        if _is_real_number(raw_et):
            total_elapsed_s += round(float(raw_et))

        weight = float(mt) if mt > 0 else 0.0
        raw_hr = act.get("average_heartrate")
        if _is_real_number(raw_hr):
            hr_pairs.append((float(raw_hr), weight))
        raw_max_hr = act.get("max_heartrate")
        if _is_real_number(raw_max_hr):
            max_hrs.append(float(raw_max_hr))
        raw_w = act.get("average_watts")
        if _is_real_number(raw_w):
            watts_pairs.append((float(raw_w), weight))
        raw_ww = act.get("weighted_average_watts")
        if _is_real_number(raw_ww):
            weighted_watts_pairs.append((float(raw_ww), weight))
        raw_kj = act.get("kilojoules")
        if _is_real_number(raw_kj):
            total_kj += float(raw_kj)
        raw_suffer = act.get("suffer_score")
        if _is_real_number(raw_suffer):
            total_suffer += float(raw_suffer)

        if act.get("private") is True or act.get("visibility") in {
            "only_me",
            "followers_only",
        }:
            private_count += 1

    dist_m = round(total_dist_m, 2)
    elev_m = round(total_elev_m, 1)
    return {
        "activity_count": len(items),
        "distance_m": dist_m,
        "distance_km": round(dist_m / METERS_PER_KM, 2),
        "distance_mi": round(dist_m / METERS_PER_MILE, 2),
        "elevation_gain_m": elev_m,
        "elevation_gain_ft": round(elev_m * FEET_PER_METER, 1),
        "moving_time_seconds": total_moving_s,
        "moving_time_formatted": format_duration(total_moving_s),
        "elapsed_time_seconds": total_elapsed_s,
        "avg_heartrate": _weighted_average(hr_pairs),
        "max_heartrate": round(max(max_hrs), 1) if max_hrs else None,
        "avg_watts": _weighted_average(watts_pairs),
        "weighted_avg_watts": _weighted_average(weighted_watts_pairs),
        "kilojoules": round(total_kj, 1),
        "suffer_score": round(total_suffer, 1),
        "private_count": private_count,
    }


def summarize_metrics(
    activities: Iterable[Mapping[str, Any]],
    *,
    after: int | float | str | datetime | date | None = None,
    before: int | float | str | datetime | date | None = None,
    sport_type: str | None = None,
    group_by: Literal["week", "month", "sport", "total"] = "week",
    now: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Aggregate activity volume, intensity, and sport breakdowns across all visibilities."""
    if group_by not in VALID_GROUP_BY:
        raise ValueError(
            f"Invalid group_by {group_by!r}; expected one of {', '.join(VALID_GROUP_BY)}."
        )

    after_ts = parse_date_filter(after, is_end=False, now=now, clock=clock)
    before_ts = parse_date_filter(before, is_end=True, now=now, clock=clock)
    if after_ts is not None and before_ts is not None and after_ts > before_ts:
        raise ValueError("'after' filter must be less than or equal to 'before' filter.")

    filtered: list[Mapping[str, Any]] = []
    for act in activities:
        if not matches_sport_type(act, sport_type):
            continue
        act_epoch = _activity_epoch_utc(act)
        if after_ts is not None and (act_epoch is None or act_epoch < after_ts):
            continue
        if before_ts is not None and (act_epoch is None or act_epoch > before_ts):
            continue
        filtered.append(act)

    grouped_items: dict[str, list[Mapping[str, Any]]] = {}
    sport_items: dict[str, list[Mapping[str, Any]]] = {}
    for act in filtered:
        sport = normalize_sport_type(act)
        sport_items.setdefault(sport, []).append(act)

        if group_by == "total":
            key = "total"
        elif group_by == "sport":
            key = sport
        else:
            local_dt = _activity_local_dt(act)
            if local_dt is None:
                key = "unknown"
            elif group_by == "week":
                iso_year, iso_week, _ = local_dt.isocalendar()
                key = f"{iso_year}-W{iso_week:02d}"
            else:
                key = local_dt.strftime("%Y-%m")
        grouped_items.setdefault(key, []).append(act)

    buckets = {k: _aggregate_bucket(v) for k, v in grouped_items.items()}
    if group_by == "sport":
        ordered_keys = sorted(
            buckets.keys(),
            key=lambda k: (-buckets[k]["activity_count"], -buckets[k]["distance_m"], k),
        )
    else:
        ordered_keys = sorted(buckets.keys())
    ordered_groups = {k: buckets[k] for k in ordered_keys}

    sport_buckets = {k: _aggregate_bucket(v) for k, v in sport_items.items()}
    ordered_sport_keys = sorted(
        sport_buckets.keys(),
        key=lambda k: (-sport_buckets[k]["activity_count"], -sport_buckets[k]["distance_m"], k),
    )
    sport_breakdown = {k: sport_buckets[k] for k in ordered_sport_keys}

    return {
        "group_by": group_by,
        "activity_count": len(filtered),
        "filters": {
            "after": after,
            "before": before,
            "after_epoch": after_ts,
            "before_epoch": before_ts,
            "sport_type": sport_type,
        },
        "totals": _aggregate_bucket(filtered),
        "groups": ordered_groups,
        "sport_breakdown": sport_breakdown,
        "visibility_note": STATS_VISIBILITY_CAVEAT,
    }


async def fetch_activities_paginated(
    client: StravaClient,
    *,
    after: int | None = None,
    before: int | None = None,
    sport_type: str | None = None,
    page: int = 1,
    per_page: int = 30,
    limit: int | None = None,
    max_pages: int = 10,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Fetch Strava activities with optional sport_type filtering and rate-limit-capped paging."""
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise ValueError("page must be >= 1.")
    if (
        not isinstance(per_page, int)
        or isinstance(per_page, bool)
        or per_page < 1
        or per_page > 200
    ):
        raise ValueError("per_page must be between 1 and 200.")
    if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
        raise ValueError("max_pages must be >= 1.")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be >= 1.")

    if limit is None:
        batch = await client.list_activities(
            before=before,
            after=after,
            page=page,
            per_page=per_page,
            refresh=refresh,
        )
        matched = [act for act in batch if matches_sport_type(act, sport_type)]
        has_more = len(batch) >= per_page
        return (matched, 1, has_more)

    fetch_per_page = 200 if (sport_type is not None and sport_type.strip()) else min(
        200, max(per_page, limit)
    )
    collected: list[dict[str, Any]] = []
    pages_fetched = 0
    exhausted_upstream = False
    current_page = page

    while pages_fetched < max_pages and len(collected) <= limit:
        batch = await client.list_activities(
            before=before,
            after=after,
            page=current_page,
            per_page=fetch_per_page,
            refresh=refresh,
        )
        pages_fetched += 1
        for act in batch:
            if matches_sport_type(act, sport_type):
                collected.append(act)
        if len(batch) < fetch_per_page:
            exhausted_upstream = True
            break
        if len(collected) >= limit:
            break
        current_page += 1

    if len(collected) > limit:
        return (collected[:limit], pages_fetched, True)
    return (collected, pages_fetched, not exhausted_upstream)
