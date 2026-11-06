"""Exercise the complete cached-data path through Streamlit, with no network."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from streamlit.testing.v1 import AppTest

from match_analyst.data import RAW_BASE_URL, StatsBombClient


APP = Path(__file__).resolve().parents[1] / "app.py"
REVISION = "a" * 40


def test_cached_match_opens_offline_and_renders_shots(tmp_path, monkeypatch):
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
    event = {
        "id": "shot-1", "type": {"id": 16, "name": "Shot"},
        "period": 1, "index": 1, "minute": 10, "second": 0,
        "team": {"id": 1, "name": "Brazil"}, "location": [110, 40],
        "shot": {
            "type": {"id": 87, "name": "Open Play"},
            "outcome": {"id": 97, "name": "Goal"},
        },
    }
    files = {
        "competitions.json": [competition],
        "matches/43/106.json": [match],
        "events/123.json": [event],
    }
    root_url = f"{RAW_BASE_URL}/{REVISION}/data/"

    def download(self, url):
        return json.dumps(files[url.removeprefix(root_url)]).encode()

    with monkeypatch.context() as seed_patch:
        seed_patch.setattr(StatsBombClient, "_download_bytes", download)
        client = StatsBombClient(tmp_path, revision=REVISION)
        client.load_match(client.find_demo_match())
    # Explicit revision clients do not replace the default revision pointer.
    (tmp_path / "source.json").write_text(json.dumps({"source_revision": REVISION}))

    def forbidden(*args, **kwargs):
        raise AssertionError("The cached offline app attempted a download")

    monkeypatch.setattr(StatsBombClient, "_download_bytes", forbidden)
    monkeypatch.setenv("MATCH_ANALYST_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MATCH_ANALYST_OFFLINE", "true")
    monkeypatch.setenv("MATCH_ANALYST_MODEL_DIR", str(tmp_path / "no-model"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert app.selectbox(key="match").value == 123
    assert app.checkbox[0].value is True
    assert any("Final score" in item.value and "123" in item.value for item in app.caption)
    assert any("Brazil" in item.value and "Serbia" in item.value for item in app.markdown)
    assert any("included_shots" in item.value and '1' in item.value for item in app.json)
    assert len(app.get("download_button")) == 5
    app.selectbox(key="team").select(2).run()
    assert not app.exception
    assert any("No eligible shots" in item.value for item in app.info)
    assert any("Serbia não registrou finalizações" in item.value for item in app.markdown)

    # An artifact can load successfully but fail inference; the source remains usable.
    import match_analyst.xg as xg
    from types import SimpleNamespace

    broken = tmp_path / "broken-model"
    broken.mkdir()
    (broken / "metadata.json").write_text("{}")
    monkeypatch.setenv("MATCH_ANALYST_MODEL_DIR", str(broken))
    monkeypatch.setattr(xg, "load_model", lambda path: SimpleNamespace(metadata={}))

    def bad_prediction(*args):
        raise ValueError("Incompatible saved model")

    monkeypatch.setattr(xg, "predict_xg", bad_prediction)
    fallback = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not fallback.exception
    assert not fallback.error
    assert len(fallback.warning) == 1
    assert fallback.selectbox(key="xg_source").options == ["xG StatsBomb"]


def test_empty_offline_cache_displays_recovery_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_ANALYST_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MATCH_ANALYST_OFFLINE", "true")
    monkeypatch.setenv("MATCH_ANALYST_MODEL_DIR", str(tmp_path / "no-model"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert len(app.error) == 1
    assert "No cached source revision" in app.error[0].value
    assert "Turn off Offline mode" in app.info[0].value


def test_empty_catalogue_is_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(StatsBombClient, "competitions", lambda self: [])
    monkeypatch.setenv("MATCH_ANALYST_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MATCH_ANALYST_OFFLINE", "true")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "catalogue is empty" in app.error[0].value
