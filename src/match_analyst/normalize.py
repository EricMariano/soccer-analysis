"""Validate StatsBomb shot origins without changing their coordinate frame.

StatsBomb uses a 120 by 80 pitch, with y increasing from the top touchline.
Shot coordinates already put the attacking goal at x=120 for either team.
Neither the away team nor second-half events need to be rotated here.
"""

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

import pandas as pd


SHOT_EVENT_ID = 16
PENALTY_SHOT_ID = 88
GOAL_OUTCOME_ID = 97
SHOT_OUTCOME_IDS = frozenset({96, 97, 98, 99, 100, 101, 115, 116})

SHOT_COLUMNS = (
    "event_id", "match_id", "competition_id", "season_id", "team_id",
    "team_name", "player_id", "player_name", "period", "index", "minute",
    "second", "x", "y", "outcome", "is_goal", "is_penalty", "event_type",
    "xg_statsbomb", "body_part", "shot_type",
)

EVENT_COLUMNS = SHOT_COLUMNS + (
    "end_x", "end_y", "is_complete", "duel_type", "defensive_action",
    "is_own_goal", "related_event_ids", "timestamp",
)


@dataclass(frozen=True)
class ValidationReport:
    """Mutually exclusive decisions account for every input record.

    ``shot_events`` counts input records identifiable as shots, including any
    later rejected. ``excluded_by_reason`` includes malformed records and
    duplicate IDs across *all* event types; valid non-shots are just ignored.
    The first occurrence of each ID wins, even if it subsequently fails checks.
    """

    total_events: int
    shot_events: int
    included_shots: int
    ignored_non_shots: int
    excluded_by_reason: dict[str, int]
    warnings_by_reason: dict[str, int] = field(default_factory=dict)
    included_events: int | None = None

    @property
    def excluded_count(self) -> int:
        return sum(self.excluded_by_reason.values())

    @property
    def duplicate_count(self) -> int:
        return self.excluded_by_reason.get("duplicate_event_id", 0)

    @property
    def shootout_count(self) -> int:
        return self.excluded_by_reason.get("penalty_shootout", 0)

    @property
    def invalid_count(self) -> int:
        return self.excluded_count - self.duplicate_count - self.shootout_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "shot_events": self.shot_events,
            "included_shots": self.included_shots,
            "ignored_non_shots": self.ignored_non_shots,
            "excluded_count": self.excluded_count,
            "duplicate_count": self.duplicate_count,
            "shootout_count": self.shootout_count,
            "invalid_count": self.invalid_count,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "warnings_by_reason": dict(self.warnings_by_reason),
            "included_events": self.included_events if self.included_events is not None else self.included_shots,
        }


@dataclass(frozen=True)
class ShotNormalization:
    shots: pd.DataFrame
    report: ValidationReport


@dataclass(frozen=True)
class EventNormalization:
    events: pd.DataFrame
    report: ValidationReport


def _integer(value: Any, minimum: int = 0, maximum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _named_id(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and _integer(value.get("id"), minimum=1)
        and _text(value.get("name"))
    )


def _location(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, Real) and not isinstance(v, bool) for v in value)
        and 0 <= value[0] <= 120
        and 0 <= value[1] <= 80
        and all(isfinite(v) for v in value)
    )


def _shot_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    for column in (
        "match_id", "competition_id", "season_id", "team_id", "player_id",
        "period", "index", "minute", "second",
    ):
        frame[column] = frame[column].astype("Int64")
    for column in ("event_id", "team_name", "player_name", "outcome", "event_type", "body_part", "shot_type"):
        frame[column] = frame[column].astype("string")
    for column in ("x", "y", "xg_statsbomb"):
        frame[column] = frame[column].astype("float64")
    for column in ("is_goal", "is_penalty"):
        frame[column] = frame[column].astype("bool")
    return frame.sort_values(["period", "index"], kind="stable").reset_index(drop=True)


def _player_valid(player: Any) -> bool:
    return player is None or (
        isinstance(player, dict)
        and (player.get("id") is None or _integer(player["id"], minimum=1))
        and (player.get("name") is None or _text(player["name"]))
    )


def _shot_problem(event: dict[str, Any]) -> str | None:
    if not _location(event.get("location")):
        return "invalid_coordinates"
    shot = event.get("shot")
    outcome = shot.get("outcome") if isinstance(shot, dict) else None
    if (
        not _named_id(outcome)
        or outcome["id"] not in SHOT_OUTCOME_IDS
        or ((outcome["id"] == GOAL_OUTCOME_ID) != (outcome["name"] == "Goal"))
    ):
        return "invalid_shot_outcome"
    shot_type = shot.get("type")
    if not _named_id(shot_type) or (
        (shot_type["id"] == PENALTY_SHOT_ID) != (shot_type["name"] == "Penalty")
    ):
        return "invalid_shot_type"
    if not _player_valid(event.get("player")):
        return "invalid_player"
    return None


def _shot_fields(event: dict[str, Any], warnings: Counter[str]) -> dict[str, Any]:
    shot = event["shot"]
    xg = shot.get("statsbomb_xg")
    if xg is None:
        warnings["missing_xg_statsbomb"] += 1
    elif not isinstance(xg, Real) or isinstance(xg, bool) or not 0 <= xg <= 1:
        warnings["invalid_xg_statsbomb"] += 1
        xg = None
    body_part = shot.get("body_part")
    if body_part is not None and not _named_id(body_part):
        warnings["invalid_body_part"] += 1
        body_part = None
    return {
        "outcome": shot["outcome"]["name"],
        "is_goal": shot["outcome"]["id"] == GOAL_OUTCOME_ID,
        "is_penalty": shot["type"]["id"] == PENALTY_SHOT_ID,
        "xg_statsbomb": xg,
        "body_part": body_part["name"] if body_part else None,
        "shot_type": shot["type"]["name"],
    }


def normalize_shots(
    events: list[dict[str, Any]],
    match_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> ShotNormalization:
    """Return eligible shot origins plus a complete exclusion audit.

    Periods 1–4 include normal time and extra time; period 5 is a shootout.
    In-game penalties remain eligible. The source's period and event index,
    rather than clock minutes, define order because clocks overlap at the end
    of added time. Missing player data is represented by nullable values.

    Own-goal events are not shots and never generate an invented shot here.
    The match's final score must come from match metadata, not these rows.
    The optional source xG is retained separately from any future model output.
    An invalid or absent source probability remains missing, never zero.
    """
    if not isinstance(events, list):
        raise TypeError("events must be a list of StatsBomb event objects")
    for name, value in (
        ("match_id", match_id), ("competition_id", competition_id),
        ("season_id", season_id),
    ):
        if (name == "match_id" or value is not None) and not _integer(value, minimum=1):
            raise ValueError(f"{name} must be a positive integer")

    seen_ids: set[str] = set()
    excluded: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    shot_events = 0
    ignored_non_shots = 0

    for event in events:
        if not isinstance(event, dict):
            excluded["invalid_event"] += 1
            continue
        event_type = event.get("type")
        if isinstance(event_type, dict) and (
            event_type.get("id") == SHOT_EVENT_ID or event_type.get("name") == "Shot"
        ):
            shot_events += 1
        event_id = event.get("id")
        if not _text(event_id):
            excluded["missing_event_id"] += 1
            continue
        if event_id in seen_ids:
            excluded["duplicate_event_id"] += 1
            continue
        seen_ids.add(event_id)
        if not _named_id(event_type) or (
            (event_type["id"] == SHOT_EVENT_ID) != (event_type["name"] == "Shot")
        ):
            excluded["invalid_event_type"] += 1
            continue
        if event_type["id"] != SHOT_EVENT_ID:
            ignored_non_shots += 1
            continue

        period = event.get("period")
        if not _integer(period, minimum=1, maximum=5):
            excluded["invalid_period"] += 1
            continue
        if period == 5:
            excluded["penalty_shootout"] += 1
            continue
        if not _integer(event.get("index"), minimum=1):
            excluded["invalid_event_index"] += 1
            continue
        if not _integer(event.get("minute")) or not _integer(event.get("second"), maximum=59):
            excluded["invalid_clock"] += 1
            continue
        if not _named_id(event.get("team")):
            excluded["invalid_team"] += 1
            continue
        problem = _shot_problem(event)
        if problem:
            excluded[problem] += 1
            continue
        player = event.get("player") or {}
        rows.append({
            "event_id": event_id,
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "team_id": event["team"]["id"],
            "team_name": event["team"]["name"],
            "player_id": player.get("id"),
            "player_name": player.get("name"),
            "period": period,
            "index": event["index"],
            "minute": event["minute"],
            "second": event["second"],
            "x": event["location"][0],
            "y": event["location"][1],
            **_shot_fields(event, warnings),
            "event_type": "Shot",
        })

    return ShotNormalization(
        shots=_shot_frame(rows),
        report=ValidationReport(
            total_events=len(events),
            shot_events=shot_events,
            included_shots=len(rows),
            ignored_non_shots=ignored_non_shots,
            excluded_by_reason=dict(sorted(excluded.items())),
            warnings_by_reason=dict(sorted(warnings.items())),
        ),
    )


def normalize_events(
    events: list[dict[str, Any]],
    match_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> EventNormalization:
    """Normalize valid events, retaining passes with unavailable coordinates.

    Pass completion follows the source convention: an *absent* ``outcome``
    key means complete; a present outcome never means complete. Missing or
    invalid optional coordinates are audited and replaced by nulls. Such
    passes remain in attempt/completion denominators but cannot contribute to
    spatial entry counts or maps. Shot origins remain mandatory.

    Pressure and defensive-event coordinates retain the acting team's source
    frame. No team/half-based rotation is applied. Period 5 is excluded for
    every event type. Related IDs are preserved for own-goal reconciliation.
    """
    if not isinstance(events, list):
        raise TypeError("events must be a list of StatsBomb event objects")
    for name, value in (("match_id", match_id), ("competition_id", competition_id), ("season_id", season_id)):
        if (name == "match_id" or value is not None) and not _integer(value, minimum=1):
            raise ValueError(f"{name} must be a positive integer")

    excluded: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    shot_events = 0
    for event in events:
        if not isinstance(event, dict):
            excluded["invalid_event"] += 1
            continue
        kind = event.get("type")
        if isinstance(kind, dict) and (kind.get("id") == SHOT_EVENT_ID or kind.get("name") == "Shot"):
            shot_events += 1
        event_id = event.get("id")
        if not _text(event_id):
            excluded["missing_event_id"] += 1
            continue
        if event_id in seen:
            excluded["duplicate_event_id"] += 1
            continue
        seen.add(event_id)
        if not _named_id(kind) or ((kind["id"] == SHOT_EVENT_ID) != (kind["name"] == "Shot")):
            excluded["invalid_event_type"] += 1
            continue
        if not _integer(event.get("period"), minimum=1, maximum=5):
            excluded["invalid_period"] += 1
            continue
        if event["period"] == 5:
            excluded["penalty_shootout"] += 1
            continue
        if not _integer(event.get("index"), minimum=1):
            excluded["invalid_event_index"] += 1
            continue
        if not _integer(event.get("minute")) or not _integer(event.get("second"), maximum=59):
            excluded["invalid_clock"] += 1
            continue
        if not _named_id(event.get("team")):
            excluded["invalid_team"] += 1
            continue
        if kind["name"] == "Shot":
            problem = _shot_problem(event)
            if problem:
                excluded[problem] += 1
                continue
        elif not _player_valid(event.get("player")):
            excluded["invalid_player"] += 1
            continue
        player = event.get("player") or {}
        location = event.get("location")
        if not _location(location):
            if kind["name"] in {"Pass", "Pressure", "Interception", "Duel", "Clearance"}:
                warnings["missing_origin_coordinates" if location is None else "invalid_origin_coordinates"] += 1
            location = [None, None]
        row = {
            "event_id": event_id, "match_id": match_id,
            "competition_id": competition_id, "season_id": season_id,
            "team_id": event["team"]["id"], "team_name": event["team"]["name"],
            "player_id": player.get("id"), "player_name": player.get("name"),
            "period": event["period"], "index": event["index"],
            "minute": event["minute"], "second": event["second"],
            "x": location[0], "y": location[1],
            "event_type": kind["name"], "outcome": None, "is_goal": False,
            "is_penalty": False, "is_complete": None, "xg_statsbomb": None,
            "body_part": None, "shot_type": None,
            "end_x": None, "end_y": None, "duel_type": None,
            "defensive_action": None,
            "is_own_goal": kind["name"] in {"Own Goal For", "Own Goal Against"},
            "related_event_ids": [value for value in event.get("related_events", []) if _text(value)]
            if isinstance(event.get("related_events", []), list) else [],
            "timestamp": event.get("timestamp") if _text(event.get("timestamp")) else None,
        }
        if kind["name"] == "Shot":
            row.update(_shot_fields(event, warnings))
        elif kind["name"] == "Pass":
            details = event.get("pass")
            if not isinstance(details, dict):
                excluded["invalid_pass"] += 1
                continue
            outcome = details.get("outcome")
            row["is_complete"] = "outcome" not in details
            if "outcome" in details:
                row["outcome"] = outcome["name"] if _named_id(outcome) else "Unknown outcome"
                if not _named_id(outcome):
                    warnings["invalid_pass_outcome"] += 1
            else:
                row["outcome"] = "Complete"
            end = details.get("end_location")
            if _location(end):
                row["end_x"], row["end_y"] = end
            else:
                warnings["missing_pass_end_coordinates" if end is None else "invalid_pass_end_coordinates"] += 1
        if kind["name"] in {"Pressure", "Interception", "Clearance"}:
            row["defensive_action"] = kind["name"]
            nested = event.get(kind["name"].lower(), {})
            if isinstance(nested, dict) and _named_id(nested.get("outcome")):
                row["outcome"] = nested["outcome"]["name"]
        elif kind["name"] == "Duel":
            duel = event.get("duel", {})
            duel_type = duel.get("type") if isinstance(duel, dict) else None
            if _named_id(duel_type):
                row["duel_type"] = duel_type["name"]
                if duel_type["id"] == 11 and duel_type["name"] == "Tackle":
                    row["defensive_action"] = "Tackle"
            elif duel_type is not None:
                warnings["invalid_duel_type"] += 1
        rows.append(row)

    base = _shot_frame(rows)
    # _shot_frame supplies the shared, typed schema; extras preserve nullable
    # booleans and lists rather than relying on pandas' inferred object types.
    raw = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    raw = raw.sort_values(["period", "index"], kind="stable").reset_index(drop=True)
    for column in SHOT_COLUMNS:
        raw[column] = base[column]
    for column in ("end_x", "end_y"):
        raw[column] = raw[column].astype("float64")
    for column in ("duel_type", "defensive_action", "timestamp"):
        raw[column] = raw[column].astype("string")
    raw["is_complete"] = raw["is_complete"].astype("boolean")
    raw["is_own_goal"] = raw["is_own_goal"].astype("bool")
    return EventNormalization(
        events=raw,
        report=ValidationReport(
            total_events=len(events), shot_events=shot_events,
            included_shots=int(raw["event_type"].eq("Shot").sum()),
            ignored_non_shots=0, included_events=len(raw),
            excluded_by_reason=dict(sorted(excluded.items())),
            warnings_by_reason=dict(sorted(warnings.items())),
        ),
    )
