"""Offline cache and provenance tests with tiny synthetic source files."""

import hashlib
import json
from pathlib import Path

import pytest

from match_analyst.data import (
    CacheError,
    DataError,
    DownloadError,
    MatchNotFoundError,
    RAW_BASE_URL,
    REVISION_URL,
    StatsBombClient,
)

REVISION = "a" * 40
COMPETITION = {
    "competition_id": 43,
    "season_id": 106,
    "competition_name": "FIFA World Cup",
    "season_name": "2022",
    "competition_gender": "male",
}
MATCH = {
    "match_id": 123,
    "home_team": {"home_team_id": 1, "home_team_name": "Brazil"},
    "away_team": {"away_team_id": 2, "away_team_name": "Serbia"},
    "home_score": 2,
    "away_score": 0,
}
EVENTS = [{"id": "shot-1", "type": {"id": 16, "name": "Shot"}}]


def source_files():
    root = f"{RAW_BASE_URL}/{REVISION}/data/"
    return {
        REVISION_URL: json.dumps({"sha": REVISION}).encode(),
        root + "competitions.json": json.dumps([COMPETITION], indent=1).encode(),
        root + "matches/43/106.json": json.dumps([MATCH], indent=1).encode(),
        root + "events/123.json": json.dumps(EVENTS, indent=1).encode(),
    }


def fake_download(monkeypatch, files=None):
    files = source_files() if files is None else files
    calls = []

    def download(self, url):
        if self.offline:
            pytest.fail("offline client attempted a download")
        calls.append(url)
        return files[url]

    monkeypatch.setattr(StatsBombClient, "_download_bytes", download)
    return calls


def test_discovers_demo_then_reuses_unchanged_raw_files_without_network(tmp_path, monkeypatch):
    calls = fake_download(monkeypatch)
    client = StatsBombClient(tmp_path)
    match = client.find_demo_match()
    assert match["match_id"] == 123
    assert match["competition_id"] == 43
    assert match["season_id"] == 106
    loaded = client.load_match(match)
    assert loaded.events == EVENTS
    assert loaded.event_path.read_bytes() == source_files()[
        f"{RAW_BASE_URL}/{REVISION}/data/events/123.json"
    ]
    assert loaded.provenance["sha256"] == hashlib.sha256(loaded.event_path.read_bytes()).hexdigest()
    assert loaded.provenance["source_revision"] == REVISION
    assert loaded.provenance["competition_id"] == 43
    assert loaded.provenance["season_id"] == 106
    assert loaded.provenance["match_id"] == 123
    assert loaded.provenance["downloaded_at"].endswith("+00:00")
    assert len(calls) == 4

    offline = StatsBombClient(tmp_path, offline=True)
    assert offline.load_match(offline.find_demo_match()).events == EVENTS
    assert len(calls) == 4
    assert offline.revision == REVISION


def test_online_client_also_prefers_cache(tmp_path, monkeypatch):
    calls = fake_download(monkeypatch)
    initial = StatsBombClient(tmp_path)
    initial.load_match(initial.find_demo_match())
    second = StatsBombClient(tmp_path)
    second.load_match(second.find_demo_match())
    assert len(calls) == 4


def test_empty_offline_cache_has_actionable_error(tmp_path, monkeypatch):
    fake_download(monkeypatch)
    with pytest.raises(CacheError, match="Run once online"):
        StatsBombClient(tmp_path, offline=True).competitions()


def test_offline_missing_match_does_not_fetch(tmp_path, monkeypatch):
    fake_download(monkeypatch)
    online = StatsBombClient(tmp_path)
    online.find_demo_match()
    with pytest.raises(CacheError, match="not cached"):
        StatsBombClient(tmp_path, offline=True).events(123)


@pytest.mark.parametrize("corruption", ["contents", "checksum", "revision", "url", "sidecar"])
def test_cache_tampering_and_incomplete_writes_are_rejected(tmp_path, monkeypatch, corruption):
    fake_download(monkeypatch)
    client = StatsBombClient(tmp_path)
    loaded = client.load_match(client.find_demo_match())
    sidecar = loaded.event_path.with_suffix(".meta.json")
    if corruption == "contents":
        loaded.event_path.write_text("[]")
    elif corruption == "sidecar":
        sidecar.unlink()
    else:
        meta = json.loads(sidecar.read_text())
        key = {"checksum": "sha256", "revision": "source_revision", "url": "source_url"}[corruption]
        meta[key] = "invalid"
        sidecar.write_text(json.dumps(meta))
    with pytest.raises(CacheError, match="cache|Cache"):
        StatsBombClient(tmp_path, offline=True).events(123)


def test_invalid_remote_json_never_becomes_cached(tmp_path, monkeypatch):
    files = source_files()
    files[f"{RAW_BASE_URL}/{REVISION}/data/competitions.json"] = b"not json"
    fake_download(monkeypatch, files)
    with pytest.raises(DataError, match="Invalid JSON"):
        StatsBombClient(tmp_path).competitions()
    assert not (tmp_path / REVISION / "competitions.json").exists()


def test_explicit_revisions_do_not_share_files(tmp_path, monkeypatch):
    fake_download(monkeypatch)
    StatsBombClient(tmp_path, revision=REVISION).competitions()
    with pytest.raises(CacheError, match="not cached"):
        StatsBombClient(tmp_path, revision="b" * 40, offline=True).competitions()


def test_absent_demo_is_reported_without_guessing_an_id(tmp_path, monkeypatch):
    files = source_files()
    files[f"{RAW_BASE_URL}/{REVISION}/data/matches/43/106.json"] = b"[]"
    fake_download(monkeypatch, files)
    with pytest.raises(MatchNotFoundError, match="not found"):
        StatsBombClient(tmp_path).find_demo_match()


def test_events_loaded_by_id_gain_competition_and_season_provenance(tmp_path, monkeypatch):
    fake_download(monkeypatch)
    client = StatsBombClient(tmp_path)
    client.events(123)
    data = client.load_match(client.find_demo_match())
    assert data.provenance["competition_id"] == 43
    assert data.provenance["season_id"] == 106


def test_download_failure_is_actionable(tmp_path, monkeypatch):
    import requests

    def fail(*args, **kwargs):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(DownloadError, match="Check the connection"):
        StatsBombClient(tmp_path).competitions()


def test_malformed_revision_record_is_rejected(tmp_path):
    (tmp_path / "source.json").write_text('{"source_revision": "master"}')
    with pytest.raises(CacheError, match="Invalid revision record"):
        StatsBombClient(tmp_path, offline=True).competitions()


def test_manifest_provenance_verifies_raw_file_before_returning_metadata(tmp_path, monkeypatch):
    fake_download(monkeypatch)
    client = StatsBombClient(tmp_path)
    client.competitions()
    provenance = client.file_provenance("competitions.json")
    assert provenance["source_revision"] == REVISION
    (tmp_path / REVISION / "competitions.json").write_text("[]")
    with pytest.raises(CacheError, match="integrity check failed"):
        client.file_provenance("competitions.json")
