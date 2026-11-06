"""Small, offline fixtures exercise the football and coordinate rules."""

from copy import deepcopy

import pandas as pd
import pytest

from match_analyst.normalize import SHOT_COLUMNS, normalize_events, normalize_shots


def shot(event_id="shot-1", **changes):
    event = {
        "id": event_id,
        "type": {"id": 16, "name": "Shot"},
        "team": {"id": 781, "name": "Brazil"},
        "player": {"id": 123, "name": "Player"},
        "period": 1,
        "index": 10,
        "minute": 1,
        "second": 30,
        "location": [108.0, 40.0],
        "shot": {
            "type": {"id": 87, "name": "Open Play"},
            "outcome": {"id": 100, "name": "Saved"},
        },
    }
    event.update(changes)
    return event


def test_normalizes_ids_outcomes_and_nullable_player():
    event = shot(player=None)
    event["shot"]["outcome"] = {"id": 97, "name": "Goal"}
    result = normalize_shots([event], 12345, 43, 106)
    row = result.shots.iloc[0]
    assert row["match_id"] == 12345
    assert row["competition_id"] == 43
    assert row["season_id"] == 106
    assert row["team_id"] == 781
    assert pd.isna(row["player_id"])
    assert pd.isna(row["player_name"])
    assert row["is_goal"]
    assert not row["is_penalty"]
    assert row["outcome"] == "Goal"
    assert result.report.excluded_count == 0


@pytest.mark.parametrize("location", [[0, 0], [120, 80], [120, 0], [0, 80], [108.2, 40.7]])
def test_coordinate_boundaries_preserved_without_rotation(location):
    events = [
        shot("home", location=location),
        shot("away", location=location, period=2, team={"id": 782, "name": "Serbia"}),
    ]
    result = normalize_shots(events, 1)
    assert result.shots["x"].tolist() == [location[0], location[0]]
    assert result.shots["y"].tolist() == [location[1], location[1]]


@pytest.mark.parametrize(
    "location",
    [None, [], [10], [1, 2, 3], [-0.1, 40], [120.1, 40], [50, -0.1], [50, 80.1],
     [float("nan"), 40], [float("inf"), 40], [50, -float("inf")], ["10", 40], [True, 40]],
)
def test_rejects_invalid_coordinates_with_reason(location):
    result = normalize_shots([shot(location=location)], 1)
    assert result.shots.empty
    assert result.report.excluded_by_reason == {"invalid_coordinates": 1}
    assert result.report.invalid_count == 1


def test_excludes_shootouts_but_keeps_in_game_penalties_and_extra_time():
    penalty = {"type": {"id": 88, "name": "Penalty"}, "outcome": {"id": 97, "name": "Goal"}}
    events = [
        shot("regular-penalty", shot=deepcopy(penalty)),
        shot("extra-penalty", period=4, minute=118, shot=deepcopy(penalty)),
        shot("shootout", period=5, minute=120, shot=deepcopy(penalty)),
    ]
    result = normalize_shots(events, 1)
    assert result.shots["event_id"].tolist() == ["regular-penalty", "extra-penalty"]
    assert result.shots["is_penalty"].all()
    assert result.shots["is_goal"].all()
    assert result.report.shootout_count == 1
    assert result.report.invalid_count == 0


def test_order_uses_period_then_index_and_preserves_added_time():
    events = [
        shot("extra2", period=4, index=9, minute=120),
        shot("second", period=2, index=1, minute=45),
        shot("first-later", period=1, index=8, minute=49),
        shot("extra1", period=3, index=1, minute=90),
        shot("first-earlier", period=1, index=7, minute=48),
    ]
    result = normalize_shots(events, 1)
    assert result.shots["event_id"].tolist() == [
        "first-earlier", "first-later", "second", "extra1", "extra2",
    ]
    assert result.shots["minute"].tolist() == [48, 49, 45, 90, 120]


def test_duplicate_ids_removed_even_across_non_shot_events():
    first = shot()
    duplicate = deepcopy(first)
    duplicate["location"] = [100, 20]
    own_goal = {"id": "own-goal", "type": {"id": 20, "name": "Own Goal Against"}}
    result = normalize_shots([first, duplicate, own_goal, deepcopy(own_goal)], 1)
    assert len(result.shots) == 1
    assert result.shots.iloc[0]["x"] == 108
    assert result.report.duplicate_count == 2
    assert result.report.ignored_non_shots == 1
    assert result.report.shot_events == 2
    assert result.report.excluded_count == 2


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"id": None}, "missing_event_id"),
        ({"type": {"id": 16, "name": "Pass"}}, "invalid_event_type"),
        ({"period": 6}, "invalid_period"),
        ({"period": True}, "invalid_period"),
        ({"index": 0}, "invalid_event_index"),
        ({"minute": -1}, "invalid_clock"),
        ({"second": 60}, "invalid_clock"),
        ({"team": {"id": True, "name": "Team"}}, "invalid_team"),
        ({"shot": {}}, "invalid_shot_outcome"),
        ({"shot": {"outcome": {"id": 97, "name": "Saved"}}}, "invalid_shot_outcome"),
        ({"shot": {"outcome": {"id": 999, "name": "Unknown"}}}, "invalid_shot_outcome"),
        ({"shot": {"outcome": {"id": 97, "name": "Goal"}}}, "invalid_shot_type"),
        ({"player": {"id": "123", "name": "Player"}}, "invalid_player"),
    ],
)
def test_invalid_essential_fields_are_audited(changes, reason):
    result = normalize_shots([shot(**changes)], 1)
    assert result.shots.empty
    assert result.report.excluded_by_reason == {reason: 1}


def test_empty_result_has_typed_schema_and_auditable_counts():
    result = normalize_shots([], 1)
    assert result.shots.empty
    assert list(result.shots.columns) == list(SHOT_COLUMNS)
    assert str(result.shots["x"].dtype) == "float64"
    assert str(result.shots["player_id"].dtype) == "Int64"
    assert result.report.as_dict()["excluded_by_reason"] == {}


def test_all_input_records_are_accounted_for_and_input_is_not_mutated():
    events = [shot("a"), shot("a"), None, shot("b", location=None)]
    snapshot = deepcopy(events)
    result = normalize_shots(events, 1)
    report = result.report
    assert report.total_events == report.included_shots + report.ignored_non_shots + report.excluded_count
    assert events == snapshot


@pytest.mark.parametrize("match_id", [0, -1, True, "1", None])
def test_requires_valid_match_identity(match_id):
    with pytest.raises(ValueError, match="match_id"):
        normalize_shots([], match_id)


@pytest.mark.parametrize("value", [None, -0.1, 1.1, float("nan"), float("inf"), "0.5", True])
def test_optional_xg_missing_or_invalid_is_not_zero_and_preserves_shot(value):
    event = shot()
    event["shot"]["statsbomb_xg"] = value
    event["shot"]["body_part"] = {"id": 40, "name": "Right Foot"}
    result = normalize_shots([event], 1)
    assert len(result.shots) == 1
    assert pd.isna(result.shots.iloc[0]["xg_statsbomb"])
    assert result.shots.iloc[0]["body_part"] == "Right Foot"
    assert result.shots.iloc[0]["shot_type"] == "Open Play"
    key = "missing_xg_statsbomb" if value is None else "invalid_xg_statsbomb"
    assert result.report.warnings_by_reason[key] == 1


def test_source_xg_endpoints_are_valid():
    events = [shot("zero"), shot("one")]
    for event, value in zip(events, [0, 1]):
        event["shot"]["statsbomb_xg"] = value
    assert normalize_shots(events, 1).shots["xg_statsbomb"].tolist() == [0, 1]


def test_pass_missing_locations_kept_for_non_spatial_denominators():
    event = shot()
    event["type"] = {"id": 30, "name": "Pass"}
    event.pop("shot")
    event["location"] = None
    event["pass"] = {}
    result = normalize_events([event], 1)
    row = result.events.iloc[0]
    assert row["is_complete"]
    assert pd.isna(row["x"]) and pd.isna(row["end_x"])
    assert result.report.included_events == 1
    assert result.report.excluded_count == 0
    assert result.report.warnings_by_reason == {"missing_origin_coordinates": 1, "missing_pass_end_coordinates": 1}


def test_pass_outcome_key_semantics_and_defensive_action_types():
    events = []
    for index, outcome in enumerate(["absent", None, {"id": 9, "name": "Incomplete"}]):
        event = shot(f"pass-{index}")
        event["type"] = {"id": 30, "name": "Pass"}
        event["pass"] = {"end_location": [90, 40]}
        if outcome != "absent":
            event["pass"]["outcome"] = outcome
        events.append(event)
    for index, (kind, identifier, subtype) in enumerate([
        ("Pressure", 17, None), ("Interception", 10, None), ("Clearance", 9, None),
        ("Duel", 4, {"id": 11, "name": "Tackle"}),
        ("Duel", 4, {"id": 10, "name": "Aerial Lost"}),
    ]):
        event = shot(f"defence-{index}")
        event["type"] = {"id": identifier, "name": kind}
        if subtype:
            event["duel"] = {"type": subtype}
        events.append(event)
    result = normalize_events(events, 1)
    assert result.events.iloc[:3]["is_complete"].tolist() == [True, False, False]
    assert result.events["defensive_action"].dropna().tolist() == ["Pressure", "Interception", "Clearance", "Tackle"]
    assert result.report.warnings_by_reason["invalid_pass_outcome"] == 1


def test_normalize_events_excludes_all_shootout_types_and_keeps_added_time():
    events = [shot("normal", minute=49), shot("shootout", period=5)]
    nonshot = shot("shootout-pass", period=5)
    nonshot["type"] = {"id": 30, "name": "Pass"}
    nonshot["pass"] = {}
    events.append(nonshot)
    result = normalize_events(events, 1)
    assert result.events["event_id"].tolist() == ["normal"]
    assert result.events.iloc[0]["minute"] == 49
    assert result.report.shootout_count == 2
    assert result.report.total_events == result.report.included_events + result.report.excluded_count


def test_event_normalization_empty_schema_and_huge_coordinates_do_not_crash():
    assert normalize_events([], 1).events.empty
    result = normalize_shots([shot(location=[10**1000, 40])], 1)
    assert result.report.excluded_by_reason == {"invalid_coordinates": 1}
