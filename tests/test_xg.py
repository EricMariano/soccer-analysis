"""Guard geometry, match separation, train-only fitting and saved probabilities."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from match_analyst import xg


def test_geometry_extremes_and_units():
    shots = pd.DataFrame({"x": [108, 120, 120, 120, 120, 0, np.nan, 121],
                          "y": [40, 40, 36, 44, 0, 40, 40, 40]}, index=list("abcdefgh"))
    geometry = xg.geometry_features(shots)
    assert geometry.index.equals(shots.index)
    assert geometry.loc["a", "distance_statsbomb"] == 12
    assert geometry.loc["a", "angle_radians"] == pytest.approx(2 * np.arctan2(4, 12))
    assert geometry.loc["b", "distance_statsbomb"] == 0
    assert geometry.loc["b", "angle_radians"] == pytest.approx(np.pi)
    assert geometry.loc[["c", "d", "e"], "angle_radians"].tolist() == [0, 0, 0]
    assert geometry.loc["f", "distance_statsbomb"] == 120
    assert np.isfinite(geometry.iloc[:6].to_numpy()).all()
    assert geometry.iloc[6:].isna().all().all()


def test_features_are_a_strict_pre_shot_whitelist():
    shots = pd.DataFrame({
        "x": [100, 110], "y": [40, 35], "body_part": ["Right Foot", pd.NA],
        "shot_type": ["Open Play", "Open Play"], "outcome": ["Goal", "Saved"],
        "is_goal": [True, False], "end_x": [120, 123], "end_location": [[120, 40], [123, 30]],
        "xg_statsbomb": [0.95, 0.01], "distance_statsbomb": [999, 999],
    })
    features = xg.prepare_features(shots)
    assert list(features) == list(xg.FEATURES)
    assert features.loc[0, "distance_statsbomb"] == 20
    assert pd.isna(features.loc[1, "body_part"])
    changed = shots.assign(outcome="Goal", is_goal=True, xg_statsbomb=1.0, end_x=0)
    pd.testing.assert_frame_equal(features, xg.prepare_features(changed))


def test_pipeline_fits_preprocessing_only_on_supplied_training_rows():
    train = pd.DataFrame({"x": [90, 100, 90, 100], "y": [40, 40, 40, 40],
                          "body_part": ["Head", "Foot", None, "Foot"], "shot_type": ["Open Play"] * 4})
    pipeline = xg.build_pipeline(seed=7)
    pipeline.fit(train, [0, 1, 0, 1])
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 25
    validation = pd.DataFrame({"x": [119, np.nan], "y": [40, np.nan],
                               "body_part": ["Previously unseen", None], "shot_type": [None, None]})
    before = imputer.statistics_.copy()
    probabilities = pipeline.predict_proba(validation)[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(imputer.statistics_, before)


def test_penalty_component_excludes_shootout_and_rejects_other_splits():
    shots = pd.DataFrame({"period": [1, 2, 2, 5, 1], "is_penalty": [True, True, True, True, False],
                          "is_goal": [True, True, False, True, True], "split": ["train"] * 5})
    component = xg.estimate_penalties(shots)
    assert component["n"] == 3
    assert component["goals"] == 2
    assert component["conversion_rate"] == pytest.approx(3 / 5)
    assert component["available"]
    with pytest.raises(ValueError, match="training split"):
        xg.estimate_penalties(shots.assign(split="test"))
    assert not xg.estimate_penalties(shots.loc[shots["is_penalty"].eq(False)])["available"]


def test_split_validation_rejects_manifest_and_row_match_leakage():
    manifest = {"split_match_ids": {"train": [1], "validation": [2], "test": [3]}}
    shots = pd.DataFrame({"match_id": [1, 2, 3], "split": ["train", "validation", "test"],
                          "event_id": ["a", "b", "c"]})
    assert xg.validate_splits(shots, manifest) == manifest["split_match_ids"]
    overlapping = {"split_match_ids": {"train": [1], "validation": [2], "test": [1, 3]}}
    with pytest.raises(ValueError, match="both"):
        xg.validate_splits(shots, overlapping)
    with pytest.raises(ValueError, match="different splits"):
        xg.validate_splits(shots.assign(match_id=[1, 1, 3]), manifest)
    with pytest.raises(ValueError, match="undeclared"):
        xg.validate_splits(shots.assign(match_id=[1, 2, 4]), manifest)


def _write_dataset(path, *, invert_test=False):
    path.mkdir()
    records = []
    for match_id, split in enumerate(xg.SPLITS, start=1):
        for i in range(24):
            penalty = i >= 20
            goal = (i % 5 == 0) if not penalty else (i % 3 != 0)
            if invert_test and split == "test":
                goal = not goal
            records.append({
                "event_id": f"{split}-{i}", "match_id": match_id, "split": split,
                "period": 1 + i % 2, "x": 110 if goal and not (invert_test and split == "test") else 80 + i,
                "y": 32 + i % 16, "body_part": "Head" if i % 3 == 0 else "Right Foot",
                "shot_type": "Penalty" if penalty else "Open Play", "is_goal": goal,
                "is_penalty": penalty, "event_type": "Shot", "competition_name": f"Competition {match_id}",
                "match_date": f"{2018 + match_id}-06-01", "xg_statsbomb": 0.1 + i / 100,
            })
    frame = pd.DataFrame(records)
    # Identical test features for the label-independence assertion.
    if invert_test:
        original = _synthetic_test_features()
        frame.loc[frame["split"].eq("test"), "x"] = original
    frame.to_csv(path / "shots.csv", index=False)
    manifest = {
        "dataset_id": "synthetic-dataset", "source_revision": "a" * 40,
        "split_match_ids": {"train": [1], "validation": [2], "test": [3]},
        "coverage": {"limitations": ["Synthetic fixture; no football performance claim."]},
        "shot_table": {"filename": "shots.csv", "sha256": hashlib.sha256((path / "shots.csv").read_bytes()).hexdigest()},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return frame


def _synthetic_test_features():
    return [110 if ((i % 5 == 0) if i < 20 else (i % 3 != 0)) else 80 + i for i in range(24)]


def test_training_freezes_selection_before_test_and_persists(tmp_path, monkeypatch):
    frame = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "model"
    original_evaluate = xg._evaluate_split
    visited = []

    def checked_evaluate(group, artifact, base_rate):
        split = group["split"].iloc[0]
        visited.append(split)
        if split == "test":
            selection = json.loads((output / "selection.json").read_text())
            assert selection["test_evaluated_at_selection"] is False
            assert selection["selected_C"] == artifact.metadata["config"]["C"]
        return original_evaluate(group, artifact, base_rate)

    monkeypatch.setattr(xg, "_evaluate_split", checked_evaluate)
    trained = xg.train_model(tmp_path / "dataset", output, candidates=[0.1, 1], seed=17)
    assert visited == ["train", "validation", "test"]
    loaded = xg.load_model(output)
    first, second = xg.predict_xg(frame, trained), xg.predict_xg(frame, loaded)
    np.testing.assert_array_equal(first["xg_model"], second["xg_model"])
    assert first["xg_model"].between(0, 1).all()
    assert first.loc[frame["is_penalty"], "npxg_model"].isna().all()
    assert loaded.metadata["penalty_component"]["n"] == 4
    assert loaded.metadata["features"] == list(xg.FEATURES)
    assert (output / "calibration.png").stat().st_size > 0
    assert xg.model_membership(1, loaded.metadata) == "train"
    assert xg.model_membership(2, loaded.metadata) == "validation"
    assert xg.model_membership(3, loaded.metadata) == "test"
    assert xg.model_membership(99, loaded.metadata) == "unseen"
    assert "statsbomb_final_test_comparison" in loaded.metadata
    assert loaded.metadata["metrics"]["test"]["penalties"]["model"]["n"] == 4
    with pytest.raises(FileExistsError):
        xg.train_model(tmp_path / "dataset", output)
    (output / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        xg.load_model(output)


def test_test_labels_do_not_change_model_selection_or_penalties(tmp_path):
    frame = _write_dataset(tmp_path / "original")
    changed = _write_dataset(tmp_path / "changed", invert_test=True)
    original = xg.train_model(tmp_path / "original", tmp_path / "first", candidates=[0.1, 1])
    other = xg.train_model(tmp_path / "changed", tmp_path / "second", candidates=[0.1, 1])
    assert original.metadata["config"]["C"] == other.metadata["config"]["C"]
    assert original.metadata["selection"]["candidates"] == other.metadata["selection"]["candidates"]
    assert original.metadata["penalty_component"] == other.metadata["penalty_component"]
    np.testing.assert_array_equal(xg.predict_xg(frame, original)["xg_model"], xg.predict_xg(changed, other)["xg_model"])


def test_inference_ignores_non_shots_shootouts_and_post_shot_columns():
    train = pd.DataFrame({"x": [80, 100, 90, 110], "y": [40] * 4,
                          "body_part": ["Right Foot"] * 4, "shot_type": ["Open Play"] * 4})
    pipeline = xg.build_pipeline().fit(train, [0, 0, 0, 1])
    artifact = xg.ModelArtifact(pipeline, {"penalty_component": {"available": True, "conversion_rate": 0.8}})
    events = pd.DataFrame({"x": [100] * 5, "y": [40] * 5, "period": [1, 2, 5, 1, 4],
                           "event_type": ["Shot", "Shot", "Shot", "Pass", "Shot"],
                           "is_penalty": pd.Series([False, True, True, pd.NA, False], dtype="boolean")})
    predicted = xg.predict_xg(events, artifact)
    assert predicted.loc[1, "xg_model"] == 0.8
    assert predicted.loc[[2, 3], "xg_model"].isna().all()
    assert predicted.loc[0, "xg_model"] == predicted.loc[4, "xg_model"]
    modified = events.assign(is_goal=True, xg_statsbomb=1.0, outcome="Goal", end_x=120)
    pd.testing.assert_series_equal(predicted["xg_model"], xg.predict_xg(modified, artifact)["xg_model"])


def test_metrics_handle_empty_and_single_class_without_invalid_json():
    empty = xg.evaluate_probabilities([], [])
    assert empty["log_loss"] is None and empty["n"] == 0
    single = xg.evaluate_probabilities([0, 0], [0.1, 0.2])
    assert single["roc_auc"] is None and single["log_loss"] > 0
    missing = xg.evaluate_probabilities([0, 1], [np.nan, 0.7])
    assert missing["n_missing"] == 1 and missing["n"] == 1
    json.dumps([empty, single, missing], allow_nan=False)


def test_training_checks_completed_shot_table_hash(tmp_path):
    _write_dataset(tmp_path / "dataset")
    with (tmp_path / "dataset" / "shots.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        xg.train_model(tmp_path / "dataset", tmp_path / "model")
