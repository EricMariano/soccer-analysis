"""Evidence-bound Portuguese text and exact, separately reviewable exports."""

from io import BytesIO
import json
from zipfile import ZipFile

from matplotlib.figure import Figure
import pandas as pd
import pytest

from match_analyst.exports import bundle_export
from match_analyst.metrics import build_facts
from match_analyst.narrative import generate_report
from match_analyst.normalize import normalize_events


MATCH = {
    "match_id": 99, "home_score": 0, "away_score": 0,
    "home_team": {"home_team_id": 1, "home_team_name": "Casa"},
    "away_team": {"away_team_id": 2, "away_team_name": "Fora"},
}


def facts_with_entries(left, center, right):
    raw = []
    for y, count in [(10, left), (40, center), (70, right)]:
        for _ in range(count):
            raw.append({
                "id": str(len(raw)), "index": len(raw) + 1,
                "type": {"id": 30, "name": "Pass"}, "period": 1,
                "minute": 20, "second": 10,
                "team": {"id": 1, "name": "Casa"}, "location": [70, 40],
                "pass": {"end_location": [90, y]},
            })
    result = normalize_events(raw, 99)
    return build_facts(result.events, MATCH)


def test_every_generated_sentence_has_resolvable_evidence_and_matches_facts():
    facts = facts_with_entries(2, 2, 6)
    report = generate_report(facts)
    assert "Casa registrou 10 entradas" in report["text"]
    assert "6 pela direita" in report["text"]
    assert "concentrou as entradas pelo corredor da direita" in report["text"]
    assert "não registrou finalizações" in report["text"]
    for sentence in report["sentences"]:
        assert sentence["evidence_keys"]
        assert all(key in facts["evidence"] for key in sentence["evidence_keys"])
    assert "dominou" not in report["text"]
    assert report["rules"] == {"min_entries": 10, "corridor_margin": 0.10}


@pytest.mark.parametrize("counts", [(1, 1, 7), (3, 3, 4), (5, 5, 0)])
def test_concentration_requires_minimum_sample_and_margin(counts):
    # (3,3,4) qualifies exactly at a 10-point gap, so raise the configured
    # margin here to prove the rule actually uses its configuration.
    report = generate_report(facts_with_entries(*counts), corridor_margin=0.11)
    assert "concentrou" not in report["text"]


def test_exact_threshold_qualifies_and_report_fails_for_missing_evidence():
    facts = facts_with_entries(3, 3, 4)
    assert "concentrou" in generate_report(facts)["text"]
    del facts["evidence"]["teams.1.shots"]
    with pytest.raises(ValueError, match="evidence"):
        generate_report(facts)


def test_export_preserves_facts_stats_report_and_unapproved_human_slot():
    facts = facts_with_entries(2, 2, 6)
    report = generate_report(facts)
    figure = Figure(figsize=(2, 2))
    figure.subplots().plot([0, 1], [0, 1])
    payload = bundle_export(facts, report, {"../../shot map": figure})
    with ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "facts.json", "report.json", "statistics.csv", "report.md", "human_review.json", "figures/shot_map.png",
        }
        assert json.loads(archive.read("facts.json")) == facts
        assert json.loads(archive.read("report.json")) == report
        assert report["text"] in archive.read("report.md").decode()
        review = json.loads(archive.read("human_review.json"))
        assert review["approved_correction"] is None
        assert review["approved_by"] is None
        table = pd.read_csv(BytesIO(archive.read("statistics.csv")))
        assert table.iloc[0]["final_third_entries"] == 10
        assert archive.read("figures/shot_map.png").startswith(b"\x89PNG\r\n\x1a\n")


def test_report_missing_model_cannot_claim_source_zero():
    facts = facts_with_entries(0, 0, 0)
    events = normalize_events([], 99).events
    facts = build_facts(events, MATCH, xg_source="xg_model")
    text = generate_report(facts)["text"]
    assert "não está disponível" in text
    assert "somou 0,00" not in text


def test_export_includes_optional_per_shot_probabilities_without_mixing_sources():
    facts = facts_with_entries(0, 0, 0)
    report = generate_report(facts)
    shots = pd.DataFrame({
        "event_id": ["shot-a", "shot-b"],
        "xg_statsbomb": [0.12, 0.34],
        "xg_model": [0.23, float("nan")],
    })
    snapshot = shots.copy(deep=True)
    with ZipFile(BytesIO(bundle_export(facts, report, {}, shots=shots))) as archive:
        exported = pd.read_csv(BytesIO(archive.read("shots.csv")))
    pd.testing.assert_frame_equal(exported, shots)
    pd.testing.assert_frame_equal(shots, snapshot)
