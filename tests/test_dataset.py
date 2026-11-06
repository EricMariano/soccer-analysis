"""Reproducibility and split-safety tests using a tiny offline tournament source."""

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from match_analyst.data import RAW_BASE_URL, StatsBombClient
from match_analyst.dataset import (
    DatasetError,
    build_catalog_manifest,
    load_dataset_config,
    prepare_dataset,
    validate_match_splits,
)

REVISION = "a" * 40


def config():
    return {
        "schema_version": 1,
        "name": "small-adult-male-international-test-cohort",
        "source_revision": REVISION,
        "scope": {
            "competition_gender": "male",
            "competition_youth": False,
            "competition_international": True,
        },
        "demonstration_match_ids": [300],
        "splits": {
            "train": {"start_date": "2015-01-01", "end_date": "2020-12-31", "competitions": [{"competition_id": 10, "season_id": 1, "expected_matches": 1}]},
            "validation": {"start_date": "2021-01-01", "end_date": "2022-11-19", "competitions": [{"competition_id": 20, "season_id": 2, "expected_matches": 1}]},
            "test": {"start_date": "2022-11-20", "end_date": "2022-12-31", "competitions": [{"competition_id": 30, "season_id": 3, "expected_matches": 1}]},
        },
        "coverage_notes": ["Synthetic tournaments; no claim of wider coverage."],
    }


def shot(mid, index, *, penalty=False, period=1, goal=False):
    event = {
        "id": f"{mid}-event-{index}",
        "type": {"id": 16, "name": "Shot"},
        "team": {"id": 1, "name": "Home"},
        "player": {"id": 5, "name": "Player"},
        "period": period,
        "index": index,
        "minute": index,
        "second": 1,
        "location": [110.0, 40.0],
        "shot": {
            "type": {"id": 88 if penalty else 87, "name": "Penalty" if penalty else "Open Play"},
            "outcome": {"id": 97 if goal else 98, "name": "Goal" if goal else "Off T"},
            "statsbomb_xg": 0.78 if penalty else 0.2,
        },
    }
    if penalty:
        event["shot"]["body_part"] = {"id": 40, "name": "Right Foot"}
    return event


def source_files():
    files = {}
    root = f"{RAW_BASE_URL}/{REVISION}/data/"
    competitions = []
    dates = ("2018-06-14", "2021-06-11", "2022-11-24")
    for number, day in enumerate(dates, start=1):
        cid, sid, mid = number * 10, number, number * 100
        competitions.append({
            "competition_id": cid, "season_id": sid,
            "competition_name": f"Tournament {number}", "season_name": str(number),
            "competition_gender": "male", "competition_youth": False,
            "competition_international": True,
        })
        matches = [{
            "match_id": mid, "match_date": day, "match_status": "available",
            "home_team": {"home_team_name": "Home", "home_team_id": 1},
            "away_team": {"away_team_name": "Away", "away_team_id": 2},
            "home_score": 1, "away_score": 0,
        }]
        events = [shot(mid, 1, goal=True), shot(mid, 2, penalty=True), shot(mid, 3, penalty=True, period=5)]
        files[root + f"matches/{cid}/{sid}.json"] = json.dumps(matches).encode()
        files[root + f"events/{mid}.json"] = json.dumps(events).encode()
    files[root + "competitions.json"] = json.dumps(competitions).encode()
    return files


@pytest.fixture
def source(monkeypatch):
    files = source_files()
    calls = []

    def fetch(self, url):
        if self.offline:
            pytest.fail("Offline preparation attempted a network request")
        calls.append(url)
        return files[url]

    monkeypatch.setattr(StatsBombClient, "_download_bytes", fetch)
    return files, calls


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    return path


def test_catalogue_inspection_confirms_dates_and_never_downloads_events(tmp_path, source):
    _, calls = source
    manifest = build_catalog_manifest(config(), StatsBombClient(tmp_path, revision=REVISION))
    assert manifest["split_match_ids"] == {"train": [100], "validation": [200], "test": [300]}
    assert manifest["coverage"][1]["first_match_date"] == "2021-06-11"
    assert len(manifest["catalog_sources"]) == 4
    assert all(meta["sha256"] for meta in manifest["catalog_sources"])
    assert len(calls) == 4
    assert not any("/events/" in url for url in calls)


def test_repeated_offline_preparation_has_identical_identity_rows_and_no_match_leak(tmp_path, source, config_path):
    _, calls = source
    cache = tmp_path / "raw"
    first = prepare_dataset(config_path, cache, tmp_path / "first", workers=2)
    first_csv = (tmp_path / "first" / "shots.csv").read_bytes()
    initial_calls = len(calls)
    second = prepare_dataset(config_path, cache, tmp_path / "second", offline=True, workers=1)
    assert first["dataset_id"] == second["dataset_id"]
    assert first_csv == (tmp_path / "second" / "shots.csv").read_bytes()
    assert len(calls) == initial_calls == 7
    manifest = json.loads((tmp_path / "second" / "manifest.json").read_text())
    frame = pd.read_csv(tmp_path / "second" / "shots.csv")
    assert frame.groupby("match_id")["split"].nunique().eq(1).all()
    assert frame["period"].lt(5).all()
    assert frame["is_penalty"].sum() == 3
    assert {"distance_statsbomb", "angle_radians", "body_part", "shot_type", "xg_statsbomb"}.issubset(frame.columns)
    assert manifest["shot_table"]["sha256"] == hashlib.sha256(first_csv).hexdigest()
    assert manifest["split_counts"]["train"] == {
        "matches": 1, "shots": 2, "goals": 1, "non_penalty_shots": 1,
        "non_penalty_goals": 1, "penalties": 1, "penalty_goals": 0,
    }
    diagnostics = json.loads((tmp_path / "second" / "diagnostics.json").read_text())
    assert diagnostics["shootout_shots_excluded"] == 3
    assert diagnostics["missing_values"]["body_part"] == 3


def test_incremental_preparation_downloads_only_uncached_matches(tmp_path, source, config_path):
    _, calls = source
    cache = tmp_path / "raw"
    StatsBombClient(cache, revision=REVISION).events(100)
    prepare_dataset(config_path, cache, tmp_path / "processed", workers=2)
    event_calls = [url for url in calls if "/events/" in url]
    assert len(event_calls) == 3
    assert len(set(event_calls)) == 3
    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert all(row["events_sha256"] for row in manifest["matches"])


def test_expected_match_count_mismatch_stops_before_event_download(tmp_path, source):
    _, calls = source
    settings = config()
    settings["splits"]["train"]["competitions"][0]["expected_matches"] = 64
    with pytest.raises(DatasetError, match="Expected 64"):
        build_catalog_manifest(settings, StatsBombClient(tmp_path, revision=REVISION))
    assert not any("/events/" in url for url in calls)


def test_scope_mismatch_is_not_silently_included(tmp_path, source):
    _, calls = source
    settings = config()
    settings["scope"]["competition_youth"] = True
    with pytest.raises(DatasetError, match="violates the configured scope"):
        build_catalog_manifest(settings, StatsBombClient(tmp_path, revision=REVISION))
    assert not any("/events/" in url for url in calls)


def test_demonstration_match_is_never_allowed_into_training_or_validation(tmp_path, source):
    _, calls = source
    settings = config()
    settings["demonstration_match_ids"] = [100]
    with pytest.raises(DatasetError, match="cannot enter train"):
        build_catalog_manifest(settings, StatsBombClient(tmp_path, revision=REVISION))
    assert not any("/events/" in url for url in calls)


@pytest.mark.parametrize("split_ids", [
    {"train": [1], "validation": [1], "test": [3]},
    {"train": [1, 1], "validation": [2], "test": [3]},
])
def test_split_manifest_rejects_match_overlap_or_repetition(split_ids):
    with pytest.raises(DatasetError, match="more than once"):
        validate_match_splits(split_ids)


def test_overlapping_date_windows_are_rejected_before_data_access(tmp_path):
    path = tmp_path / "bad.json"
    settings = config()
    settings["splits"]["validation"]["start_date"] = "2020-12-31"
    path.write_text(json.dumps(settings))
    with pytest.raises(DatasetError, match="must not overlap"):
        load_dataset_config(path)


def test_repeated_competition_selection_cannot_duplicate_training_matches(tmp_path, source):
    settings = config()
    settings["splits"]["train"]["competitions"] *= 2
    with pytest.raises(DatasetError, match="more than once"):
        build_catalog_manifest(settings, StatsBombClient(tmp_path, revision=REVISION))


def test_duplicate_event_ids_across_matches_prevent_a_completed_dataset(tmp_path, source, config_path):
    files, _ = source
    url = f"{RAW_BASE_URL}/{REVISION}/data/events/200.json"
    events = json.loads(files[url])
    events[0]["id"] = "100-event-1"
    files[url] = json.dumps(events).encode()
    with pytest.raises(DatasetError, match="more than one selected match"):
        prepare_dataset(config_path, tmp_path / "raw", tmp_path / "output", workers=2)
    assert not (tmp_path / "output" / "manifest.json").exists()


def test_dataset_identity_changes_when_source_event_content_changes(tmp_path, source, config_path):
    files, _ = source
    first = prepare_dataset(config_path, tmp_path / "cache1", tmp_path / "output1", workers=1)
    url = f"{RAW_BASE_URL}/{REVISION}/data/events/100.json"
    events = json.loads(files[url])
    events[0]["location"][0] = 100.0
    files[url] = json.dumps(events).encode()
    second = prepare_dataset(config_path, tmp_path / "cache2", tmp_path / "output2", workers=1)
    assert first["dataset_id"] != second["dataset_id"]
