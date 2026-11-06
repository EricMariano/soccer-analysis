"""The command-line workflow exports the same cached facts without Streamlit."""

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from match_analyst.data import RAW_BASE_URL, DataError, StatsBombClient
from scripts.analyze_match import analyze_match

REVISION = "a" * 40


@pytest.fixture
def cached_match(tmp_path, monkeypatch):
    cache = tmp_path / "raw"
    competition = {
        "competition_id": 43, "season_id": 106,
        "competition_name": "FIFA World Cup", "season_name": "2022",
        "competition_gender": "male",
    }
    match = {
        "match_id": 123, "match_date": "2022-11-24",
        "home_team": {"home_team_id": 1, "home_team_name": "Brazil"},
        "away_team": {"away_team_id": 2, "away_team_name": "Serbia"},
        "home_score": 1, "away_score": 0,
    }
    events = [
        {
            "id": "shot-1", "index": 1, "period": 1, "minute": 10, "second": 0,
            "type": {"id": 16, "name": "Shot"}, "team": {"id": 1, "name": "Brazil"},
            "player": {"id": 5, "name": "Scorer"}, "location": [110, 40],
            "shot": {"type": {"id": 87, "name": "Open Play"},
                     "outcome": {"id": 97, "name": "Goal"}, "statsbomb_xg": 0.2},
        },
        {
            "id": "pass-1", "index": 2, "period": 2, "minute": 46, "second": 0,
            "type": {"id": 30, "name": "Pass"}, "team": {"id": 1, "name": "Brazil"},
            "player": {"id": 6, "name": "Passer"}, "location": [70, 20],
            "pass": {"end_location": [90, 20]},
        },
        {
            "id": "shootout", "index": 3, "period": 5, "minute": 120, "second": 0,
            "type": {"id": 16, "name": "Shot"}, "team": {"id": 1, "name": "Brazil"},
            "player": {"id": 5, "name": "Scorer"}, "location": [108, 40],
            "shot": {"type": {"id": 88, "name": "Penalty"},
                     "outcome": {"id": 97, "name": "Goal"}, "statsbomb_xg": 0.78},
        },
    ]
    files = {"competitions.json": [competition], "matches/43/106.json": [match], "events/123.json": events}
    prefix = f"{RAW_BASE_URL}/{REVISION}/data/"

    def seed(self, url):
        return json.dumps(files[url.removeprefix(prefix)]).encode()

    monkeypatch.setattr(StatsBombClient, "_download_bytes", seed)
    client = StatsBombClient(cache, revision=REVISION)
    client.load_match(client.find_demo_match())
    (cache / "source.json").write_text(json.dumps({"source_revision": REVISION}))

    def forbidden(*args, **kwargs):
        pytest.fail("Offline analysis attempted a download")

    monkeypatch.setattr(StatsBombClient, "_download_bytes", forbidden)
    return cache


def test_offline_export_contains_reviewable_facts_shots_and_pngs_and_preserves_reviews(tmp_path, cached_match):
    options = {"offline": True, "cache_dir": cached_match, "output_dir": tmp_path / "output", "model_dir": tmp_path / "no-model"}
    result = analyze_match(**options)
    destination = Path(result["output_dir"])
    expected = {"facts.json", "statistics.csv", "shots.csv", "report.json", "report.md", "human_review.json"}
    with ZipFile(result["archive_path"]) as bundle:
        assert expected.issubset(bundle.namelist())
        pngs = [filename for filename in bundle.namelist() if filename.endswith(".png")]
        assert len(pngs) == 5
        assert all(bundle.read(filename).startswith(b"\x89PNG") for filename in pngs)
    facts = json.loads((destination / "facts.json").read_text())
    assert facts["xg"]["version"] == REVISION
    assert facts["model"] is None
    assert facts["teams"][0]["shots"] == 1
    assert facts["teams"][0]["xg"] == pytest.approx(0.2)
    assert facts["provenance"]["sha256"]
    shots = pd.read_csv(destination / "shots.csv")
    assert shots["event_id"].tolist() == ["shot-1"]
    assert result["validation"]["shootout_count"] == 1
    review_path = destination / "human_review.json"
    reviewed = {"approved_correction": "Human-reviewed version"}
    review_path.write_text(json.dumps(reviewed))
    second = analyze_match(**options)
    assert second["output_dir"] != result["output_dir"]
    assert json.loads(review_path.read_text()) == reviewed


def test_filtered_pass_only_scope_exports_zero_shots_and_no_shootout(tmp_path, cached_match):
    result = analyze_match(
        offline=True, match_id=123, team_id=1, period=2, player_id=6,
        cache_dir=cached_match, output_dir=tmp_path / "filtered", model_dir=tmp_path / "no-model",
    )
    destination = Path(result["output_dir"])
    facts = json.loads((destination / "facts.json").read_text())
    assert facts["filters"] == {"team_id": 1, "period": 2, "player_id": 6}
    assert facts["teams"][0]["shots"] == 0
    assert facts["teams"][0]["xg_per_shot"] is None
    assert facts["teams"][0]["passes_attempted"] == 1
    assert pd.read_csv(destination / "shots.csv").empty


def test_explicit_own_source_requires_an_existing_artifact(tmp_path, cached_match):
    with pytest.raises(DataError, match="No saved own model"):
        analyze_match(
            offline=True, xg_source="xg_model", cache_dir=cached_match,
            output_dir=tmp_path / "output", model_dir=tmp_path / "no-model",
        )
    assert not (tmp_path / "output").exists()


def test_unknown_match_is_rejected_in_the_selected_catalogue(tmp_path, cached_match):
    with pytest.raises(DataError, match="not in competition"):
        analyze_match(offline=True, match_id=999, cache_dir=cached_match, output_dir=tmp_path / "output")
