"""Football semantics, missingness and consistent filtered facts."""

import json

import pandas as pd
import pytest

from match_analyst.metrics import build_facts, filter_events, team_statistics
from match_analyst.normalize import normalize_events


MATCH = {
    "match_id": 99, "match_date": "2022-11-24", "home_score": 2, "away_score": 0,
    "home_team": {"home_team_id": 1, "home_team_name": "Home"},
    "away_team": {"away_team_id": 2, "away_team_name": "Away"},
}


def event(event_id, kind="Shot", team=1, **kwargs):
    ids = {"Shot": 16, "Pass": 30, "Own Goal For": 25, "Own Goal Against": 20,
           "Pressure": 17, "Interception": 10, "Clearance": 9, "Duel": 4}
    value = {
        "id": event_id, "index": 1, "type": {"id": ids[kind], "name": kind},
        "period": 1, "minute": 10, "second": 0, "location": [108, 40],
        "team": {"id": team, "name": "Home" if team == 1 else "Away"},
        "player": {"id": team * 10, "name": f"Player {team}"},
    }
    if kind == "Shot":
        value["shot"] = {
            "type": {"id": 87, "name": "Open Play"},
            "outcome": {"id": 100, "name": "Saved"},
            "statsbomb_xg": 0.1,
        }
    elif kind == "Pass":
        value["pass"] = {"end_location": [90, 40]}
    value.update(kwargs)
    return value


def frame(raw):
    return normalize_events(raw, MATCH["match_id"]).events


def home_stats(raw, source="xg_statsbomb"):
    return team_statistics(frame(raw), MATCH, source).iloc[0]


def test_shot_area_boundaries_and_same_xg_source_totals():
    first = event("inside", location=[102, 18])
    second = event("outside", location=[101.99, 40])
    second["shot"]["statsbomb_xg"] = 0.2
    penalty = event("penalty", location=[108, 40])
    penalty["shot"]["type"] = {"id": 88, "name": "Penalty"}
    penalty["shot"]["statsbomb_xg"] = 0.8
    values = home_stats([first, second, penalty])
    assert values["shots"] == 3
    assert values["shots_inside_box"] == 2
    assert values["shots_outside_box"] == 1
    assert values["xg"] == pytest.approx(1.1)
    assert values["npxg"] == pytest.approx(0.3)
    assert values["xg_per_shot"] == pytest.approx(1.1 / 3)


def test_pass_semantics_final_third_and_corridor_boundaries():
    passes = []
    for index, y in enumerate([0, 80 / 3, 160 / 3, 80]):
        passes.append(event(f"entry-{index}", "Pass", location=[79.9, 40], **{"pass": {"end_location": [80, y]}}))
    passes += [
        event("already-there", "Pass", location=[80, 40]),
        event("incomplete", "Pass", location=[30, 40], **{"pass": {"end_location": [90, 40], "outcome": {"id": 9, "name": "Incomplete"}}}),
        event("null-outcome", "Pass", location=[30, 40], **{"pass": {"end_location": [90, 40], "outcome": None}}),
        event("missing-xy", "Pass", location=None, **{"pass": {}}),
    ]
    values = home_stats(passes)
    assert values["passes_attempted"] == 8
    assert values["passes_completed"] == 6
    assert values["pass_accuracy"] == 0.75
    assert values["final_third_entries"] == 4
    assert values["entries_left"] == 1
    assert values["entries_center"] == 1
    assert values["entries_right"] == 2
    assert values["passes_with_coordinates"] == 7


def test_defensive_events_are_separate_and_aerial_duel_is_not_tackle():
    raw = [event("pressure", "Pressure"), event("interception", "Interception"), event("clearance", "Clearance")]
    raw += [event("tackle", "Duel", duel={"type": {"id": 11, "name": "Tackle"}}),
            event("aerial", "Duel", duel={"type": {"id": 10, "name": "Aerial Lost"}})]
    values = home_stats(raw)
    assert [values[key] for key in ("pressures", "interceptions", "tackles", "clearances")] == [1, 1, 1, 1]


def test_missing_xg_never_becomes_false_zero_or_partial_total():
    missing = event("missing")
    missing["shot"].pop("statsbomb_xg")
    values = home_stats([event("known"), missing])
    assert pd.isna(values["xg"])
    assert pd.isna(values["xg_per_shot"])
    assert values["xg_observed_shots"] == 1
    assert values["xg_missing_shots"] == 1
    facts = build_facts(frame([event("known"), missing]), MATCH)
    assert facts["teams"][0]["xg"] is None
    assert any("indisponível" in limitation for limitation in facts["limitations"])
    json.dumps(facts, allow_nan=False)


def test_own_model_is_separate_and_unavailable_unless_column_present():
    events = frame([event("a")])
    facts = build_facts(events, MATCH, xg_source="xg_model", xg_version="test-v1")
    assert facts["teams"][0]["xg"] is None
    assert not facts["xg"]["available"]
    events["xg_model"] = 0.7
    facts = build_facts(events, MATCH, xg_source="xg_model", xg_version="test-v1")
    assert facts["teams"][0]["xg"] == 0.7
    assert facts["xg"]["source"] == "xg_model"
    assert facts["xg"]["version"] == "test-v1"


@pytest.mark.parametrize("use_links", [True, False])
def test_own_goal_mirror_reconciled_once_and_credited_before_team_filter(use_links):
    against = event("against", "Own Goal Against", team=2)
    own_for = event("for", "Own Goal For", team=1)
    if use_links:
        against["related_events"] = ["for"]
        own_for["related_events"] = ["against"]
        own_for["second"] = 1  # linked IDs still pair despite a clock discrepancy
    shot_goal = event("shot-goal", minute=20)
    shot_goal["shot"]["outcome"] = {"id": 97, "name": "Goal"}
    events = frame([against, own_for, shot_goal])
    facts = build_facts(events, MATCH, filters={"team_id": 1})
    assert facts["teams"][0]["goals"] == 2
    assert facts["teams"][0]["shots"] == 1
    away = build_facts(events, MATCH, filters={"team_id": 2})
    assert away["teams"][0]["goals"] == 0
    assert facts["match"]["final_score"] == {"home": 2, "away": 0}
    assert facts["score_reconciliation"]["home"]["matches_catalogue"]
    assert facts["score_reconciliation"]["away"]["matches_catalogue"]


def test_own_goal_against_without_mirror_and_period_filter():
    events = frame([event("against", "Own Goal Against", team=2, period=2), event("first", period=1)])
    assert build_facts(events, MATCH, filters={"period": 2})["teams"][0]["goals"] == 1
    assert build_facts(events, MATCH, filters={"period": 1})["teams"][0]["goals"] == 0


def test_filters_empty_denominators_and_shootout_exclusion():
    events = frame([event("first"), event("second", period=2, player={"id": 12, "name": "Other"})])
    assert filter_events(events, period=2, player_id=12)["event_id"].tolist() == ["second"]
    facts = build_facts(events, MATCH, filters={"player_id": 999})
    assert facts["teams"][0]["shots"] == 0
    assert facts["teams"][0]["xg"] == 0
    assert facts["teams"][0]["xg_per_shot"] is None
    assert facts["teams"][0]["pass_accuracy"] is None
    assert facts["teams"][0]["denominators"]["pass_accuracy"] == 0
    # Even a caller-supplied period-5 row cannot enter normal metrics.
    events.loc[events.index[0], "period"] = 5
    assert int(team_statistics(events, MATCH).iloc[0]["shots"]) == 1


def test_facts_include_evidence_coverage_and_same_slice_comparisons():
    result = normalize_events([event("a"), event("b", team=2)], 99)
    facts = build_facts(result.events, MATCH, validation=result.report)
    assert facts["comparisons"]["shot_difference"] == 0
    assert facts["evidence"]["teams.1.shots"]["value"] == 1
    assert facts["teams"][0]["denominators"]["xg_coverage"] == 1
    assert facts["validation"]["included_events"] == 2
    assert facts["metrics_version"]
    json.dumps(facts, allow_nan=False)


def test_unknown_source_and_filters_rejected():
    events = frame([])
    with pytest.raises(ValueError, match="source"):
        build_facts(events, MATCH, xg_source="mixed")
    with pytest.raises(ValueError, match="filters"):
        build_facts(events, MATCH, filters={"nonsense": 1})
