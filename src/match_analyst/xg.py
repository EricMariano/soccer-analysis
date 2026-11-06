"""Reproducible non-penalty xG and a separate, train-only penalty estimate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ("distance_statsbomb", "angle_radians")
CATEGORICAL_FEATURES = ("body_part", "shot_type")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SPLITS = ("train", "validation", "test")
MODEL_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ModelArtifact:
    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]


def geometry_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Distance to (120,40) and visible angle of the (120,36)/(120,44) goal.

    Distances are in StatsBomb coordinate units, not metres. ``atan2`` avoids
    dividing by distance to the goal line. On the goal line between the posts
    the angle is pi; exactly at either post it is defined as zero because one
    ray has zero length. Missing, non-finite and out-of-pitch inputs become
    missing features for the pipeline's training-fitted imputer.
    """
    x = pd.to_numeric(shots.get("x", pd.Series(np.nan, index=shots.index)), errors="coerce")
    y = pd.to_numeric(shots.get("y", pd.Series(np.nan, index=shots.index)), errors="coerce")
    x = x.to_numpy(dtype=float, na_value=np.nan)
    y = y.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 120) & (y >= 0) & (y <= 80)
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    dx = 120.0 - x
    top_y, bottom_y = 36.0 - y, 44.0 - y
    distance = np.hypot(dx, 40.0 - y)
    angle = np.arctan2(np.abs(dx * bottom_y - dx * top_y), dx * dx + top_y * bottom_y)
    distance[~valid], angle[~valid] = np.nan, np.nan
    return pd.DataFrame(
        {"distance_statsbomb": distance, "angle_radians": angle}, index=shots.index
    )


def prepare_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Return only the pre-shot whitelist, ignoring all other input columns.

    Precomputed geometry is deliberately recalculated from shot origins.
    Outcomes, endpoints, keeper actions and the provider's xG can never become
    model inputs, even when present in the same event table.
    """
    features = geometry_features(shots)
    for column in CATEGORICAL_FEATURES:
        values = shots.get(column, pd.Series(np.nan, index=shots.index))
        features[column] = values.map(
            lambda value: value.strip() if isinstance(value, str) and value.strip() else np.nan
        ).astype(object)
    return features.loc[:, list(FEATURES)]


def build_pipeline(c: float = 1.0, seed: int = 42) -> Pipeline:
    """Include feature construction, imputation, encoding and prediction."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
    )
    return Pipeline([
        ("features", FunctionTransformer(prepare_features, validate=False)),
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(C=c, max_iter=2000, random_state=seed)),
    ])


def _boolean(values: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    def parse(value: Any) -> Any:
        if pd.isna(value):
            if allow_missing:
                return pd.NA
            raise ValueError(f"{name} contains missing values.")
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.lower() in {"true", "false", "1", "0"}:
            return value.lower() in {"true", "1"}
        raise ValueError(f"{name} must contain only boolean or zero/one values.")
    return values.map(parse).astype("boolean")


def _eligible_mask(shots: pd.DataFrame) -> pd.Series:
    periods = pd.to_numeric(shots.get("period", pd.Series(1, index=shots.index)), errors="coerce")
    eligible = periods.isin([1, 2, 3, 4])
    if "event_type" in shots:
        eligible &= shots["event_type"].eq("Shot").fillna(False)
    return eligible


def _target(shots: pd.DataFrame) -> np.ndarray:
    return _boolean(shots["is_goal"], "is_goal").to_numpy(dtype=int)


def estimate_penalties(train_shots: pd.DataFrame) -> dict[str, Any]:
    """Beta(1,1) posterior mean, estimated exclusively from training penalties."""
    eligible = train_shots.loc[_eligible_mask(train_shots)]
    if "split" in eligible and not eligible["split"].eq("train").all():
        raise ValueError("Penalty estimation accepts only the training split.")
    penalties = eligible.loc[_boolean(eligible["is_penalty"], "is_penalty").to_numpy(dtype=bool)]
    n, goals = len(penalties), int(_target(penalties).sum())
    return {
        "n": n, "goals": goals, "conversion_rate": (goals + 1) / (n + 2),
        "available": n > 0, "method": "Beta(1,1) posterior mean",
        "prior_alpha": 1.0, "prior_beta": 1.0,
        "posterior_alpha": goals + 1, "posterior_beta": n - goals + 1,
        "scope": "in-game penalties only; period 5 excluded", "training_only": True,
    }


def predict_xg(shots: pd.DataFrame, model: ModelArtifact) -> pd.DataFrame:
    """Copy events and attach one consistent own-model xG source.

    ``npxg_model`` is populated only for non-penalty shots. ``xg_model`` adds
    the separate in-game penalty component. Non-shots and shootouts keep NaN.
    No fitting occurs here. An unavailable penalty component keeps those
    probabilities missing rather than silently reporting a partial total.
    """
    result = shots.copy()
    result["xg_model"] = np.nan
    result["npxg_model"] = np.nan
    eligible = _eligible_mask(result)
    if not eligible.any():
        return result
    penalty = pd.Series(False, index=result.index)
    penalty.loc[eligible] = _boolean(result.loc[eligible, "is_penalty"], "is_penalty").to_numpy(dtype=bool)
    non_penalty = eligible & ~penalty
    if non_penalty.any():
        probabilities = model.pipeline.predict_proba(result.loc[non_penalty])[:, 1]
        result.loc[non_penalty, "npxg_model"] = probabilities
        result.loc[non_penalty, "xg_model"] = probabilities
    component = model.metadata["penalty_component"]
    if component["available"]:
        result.loc[eligible & penalty, "xg_model"] = component["conversion_rate"]
    return result


def evaluate_probabilities(
    y_true: Iterable[int], probabilities: Iterable[float], n_bins: int = 10,
) -> dict[str, Any]:
    """Probability metrics with coverage and explicit empty/single-class cases."""
    raw_y = np.asarray(list(y_true))
    p = np.asarray(list(probabilities), dtype=float)
    if raw_y.shape != p.shape or raw_y.ndim != 1:
        raise ValueError("Targets and probabilities must be equally sized vectors.")
    if not np.isin(raw_y, [0, 1]).all():
        raise ValueError("Targets must be zero or one.")
    y = raw_y.astype(int)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    metric: dict[str, Any] = {
        "n_total": len(y), "n": int(valid.sum()), "n_missing": int((~valid).sum()),
        "goals": int(y[valid].sum()), "log_loss": None, "brier_score": None,
        "roc_auc": None, "calibration": [],
    }
    if not valid.any():
        return metric
    y, p = y[valid], p[valid]
    metric["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    metric["brier_score"] = float(brier_score_loss(y, p, pos_label=1))
    if len(np.unique(y)) == 2:
        metric["roc_auc"] = float(roc_auc_score(y, p))
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="uniform", pos_label=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.searchsorted(bins[1:-1], p)
    occupied = np.unique(bin_ids)
    metric["calibration"] = [
        {"lower": float(bins[i]), "upper": float(bins[i + 1]),
         "count": int((bin_ids == i).sum()), "mean_predicted": float(mean),
         "observed_goal_rate": float(rate)}
        for i, mean, rate in zip(occupied, predicted, observed, strict=True)
    ]
    return metric


def validate_splits(shots: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, list[int]]:
    """Reject match leakage, undeclared matches and inconsistent row assignments."""
    declared = manifest.get("split_match_ids")
    if not isinstance(declared, dict) or set(declared) != set(SPLITS):
        raise ValueError("Manifest must declare train, validation and test match IDs.")
    result: dict[str, list[int]] = {}
    for split in SPLITS:
        values = declared[split]
        if not isinstance(values, list) or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values):
            raise ValueError(f"Manifest {split} match IDs must be positive integers.")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate match IDs in manifest {split}.")
        result[split] = sorted(values)
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if set(result[split]) & set(result[other]):
                raise ValueError(f"A match appears in both {split} and {other}.")
    if not shots["split"].isin(SPLITS).all():
        raise ValueError("Every shot must belong to train, validation or test.")
    numeric_ids = pd.to_numeric(shots["match_id"], errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.eq(np.floor(numeric_ids)).all():
        raise ValueError("Every shot needs a valid integer match_id.")
    if shots.groupby("match_id")["split"].nunique().gt(1).any():
        raise ValueError("Shots from a match cannot appear in different splits.")
    for split in SPLITS:
        actual = set(numeric_ids.loc[shots["split"].eq(split)].astype(int))
        if not actual.issubset(result[split]):
            raise ValueError(f"Shot table contains an undeclared match in {split}.")
    if "event_id" in shots and shots["event_id"].duplicated().any():
        raise ValueError("Shot table contains duplicate event IDs.")
    return result


def model_membership(match_id: int, metadata: dict[str, Any]) -> str:
    for split in SPLITS:
        if int(match_id) in metadata.get("split_match_ids", {}).get(split, []):
            return split
    return "unseen"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluate_split(frame: pd.DataFrame, artifact: ModelArtifact, base_rate: float) -> dict[str, Any]:
    predictions = predict_xg(frame, artifact)
    penalty = _boolean(frame["is_penalty"], "is_penalty").to_numpy(dtype=bool)
    base_penalty = artifact.metadata["penalty_component"]
    penalty_rate = base_penalty["goals"] / base_penalty["n"] if base_penalty["n"] else np.nan
    baseline = np.where(penalty, penalty_rate, base_rate)
    result = {}
    for name, mask in (("npxg", ~penalty), ("penalties", penalty), ("all_shots", np.ones(len(frame), dtype=bool))):
        target = _target(frame.loc[mask])
        result[name] = {
            "model": evaluate_probabilities(target, predictions.loc[mask, "xg_model"]),
            "baseline": evaluate_probabilities(target, baseline[mask]),
        }
    return result


def _calibration_plot(metrics: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axes, ("validation", "test"), strict=True):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#a4afa6", label="Perfect calibration")
        for model_key, label, color in (("model", "Own npxG", "#153345"), ("baseline", "Train-rate baseline", "#bd831e")):
            points = metrics[split]["npxg"][model_key]["calibration"]
            ax.plot([p["mean_predicted"] for p in points], [p["observed_goal_rate"] for p in points],
                    marker="o", label=label, color=color)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed goal rate",
               title=f"{split.title()} · non-penalty shots")
        ax.legend(fontsize=8)
    fig.suptitle("Calibration · uniform probability bins · StatsBomb Open Data")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def train_model(
    dataset_dir: Path | str,
    output_dir: Path | str,
    seed: int = 42,
    candidates: Iterable[float] = (0.1, 1.0, 10.0),
    model_version: str | None = None,
) -> ModelArtifact:
    """Select only on validation, freeze that choice, then evaluate test once.

    Existing model/selection artifacts are never overwritten. This keeps the
    final test report attached to the validation-selected model configuration.
    """
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    manifest_path, shots_path = dataset_dir / "manifest.json", dataset_dir / "shots.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Dataset manifest needs a nonempty dataset_id.")
    shot_table = manifest.get("shot_table")
    if shot_table is not None and (
        not isinstance(shot_table, dict) or shot_table.get("sha256") != _sha256(shots_path)
    ):
        raise ValueError("Shot table checksum does not match the completed dataset manifest.")
    shots = pd.read_csv(shots_path)
    required = {"match_id", "split", "is_goal", "is_penalty", "x", "y", "period"}
    if not required.issubset(shots):
        raise ValueError(f"Shot table is missing columns: {sorted(required - set(shots))}")
    split_match_ids = validate_splits(shots, manifest)
    eligible = shots.loc[_eligible_mask(shots)].copy()
    if any((output_dir / filename).exists() for filename in ("metadata.json", "model.joblib", "selection.json")):
        raise FileExistsError(f"A model or frozen selection already exists in {output_dir}; choose a new output directory.")
    c_values = sorted(set(float(c) for c in candidates))
    if not c_values or not all(np.isfinite(c) and c > 0 for c in c_values):
        raise ValueError("Regularization candidates must be finite positive values.")
    train = eligible.loc[eligible["split"].eq("train")]
    validation = eligible.loc[eligible["split"].eq("validation")]
    train_np = train.loc[~_boolean(train["is_penalty"], "is_penalty").to_numpy(dtype=bool)]
    validation_np = validation.loc[~_boolean(validation["is_penalty"], "is_penalty").to_numpy(dtype=bool)]
    if not len(train_np) or not len(validation_np):
        raise ValueError("Training and validation each need eligible non-penalty shots.")
    if not eligible["split"].eq("test").any():
        raise ValueError("The final test split needs eligible shots.")
    y_train, y_validation = _target(train_np), _target(validation_np)
    if len(np.unique(y_train)) != 2:
        raise ValueError("Logistic regression training requires goals and non-goals.")
    history, pipelines = [], {}
    for c in c_values:
        pipeline = build_pipeline(c=c, seed=seed)
        pipeline.fit(train_np, y_train)
        probabilities = pipeline.predict_proba(validation_np)[:, 1]
        history.append({"C": c, "validation_log_loss": float(log_loss(y_validation, probabilities, labels=[0, 1]))})
        pipelines[c] = pipeline
    selected = min(history, key=lambda item: (item["validation_log_loss"], item["C"]))
    selected_at = _now()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "dataset_id": dataset_id, "selected_at": selected_at, "seed": seed,
        "criterion": "minimum validation non-penalty log loss; lower C breaks ties",
        "candidates": history, "selected_C": selected["C"],
        "fit_split": "train", "selection_split": "validation", "test_evaluated_at_selection": False,
        "features": list(FEATURES), "shots_sha256": _sha256(shots_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    # The immutable selection is on disk before any test labels are evaluated.
    _write_json(output_dir / "selection.json", selection)
    metadata: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": model_version or f"logreg-xg-v1-{dataset_id.removeprefix('statsbomb-')[:12]}",
        "model_type": "LogisticRegression non-penalty xG + separate Beta(1,1) penalty component",
        "dataset_id": dataset_id, "source_revision": manifest.get("source_revision"),
        "features": list(FEATURES), "split_match_ids": split_match_ids,
        "selected_at": selected_at, "trained_at": _now(), "selection": selection,
        "penalty_component": estimate_penalties(train),
        "baseline": {"method": "observed training goal rate", "non_penalty_goal_rate": float(y_train.mean()),
                     "non_penalty_shots": len(y_train), "non_penalty_goals": int(y_train.sum())},
        "config": {"seed": seed, "C": selected["C"], "candidate_C": c_values,
                   "class_weight": None, "max_iter": 2000, "numeric_imputation": "median",
                   "categorical_imputation": "Unknown", "unknown_categories": "ignore",
                   "calibration_bins": 10, "calibration_strategy": "uniform"},
        "geometry": {"coordinate_system": "StatsBomb 120x80", "distance_unit": "StatsBomb units",
                     "goal_center": [120, 40], "goal_posts": [[120, 36], [120, 44]],
                     "angle_unit": "radians", "angle_at_post": 0.0},
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib")},
        "limitations": list(manifest.get("coverage_notes", [])) + (
            list(manifest.get("coverage", {}).get("limitations", []))
            if isinstance(manifest.get("coverage"), dict) else []
        ),
    }
    artifact = ModelArtifact(pipelines[selected["C"]], metadata)
    metadata["metrics"] = {}
    metadata["split_counts"] = {}
    for split in SPLITS:
        group = eligible.loc[eligible["split"].eq(split)]
        metadata["metrics"][split] = _evaluate_split(group, artifact, float(y_train.mean()))
        counts_by_competition = {}
        if "competition_name" in group:
            for name, part in group.groupby("competition_name", dropna=False):
                counts_by_competition[str(name)] = {
                    "matches": int(part["match_id"].nunique()), "shots": len(part), "goals": int(_target(part).sum()),
                }
        metadata["split_counts"][split] = {
            "manifest_matches": len(split_match_ids[split]), "matches_with_shots": int(group["match_id"].nunique()),
            "shots": len(group), "goals": int(_target(group).sum()), "by_competition": counts_by_competition,
        }
    test = eligible.loc[eligible["split"].eq("test")]
    if "xg_statsbomb" in test:
        source_prob = pd.to_numeric(test["xg_statsbomb"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        test_penalty = _boolean(test["is_penalty"], "is_penalty").to_numpy(dtype=bool)
        metadata["statsbomb_final_test_comparison"] = {
            name: evaluate_probabilities(_target(test.loc[mask]), source_prob[mask])
            for name, mask in (("npxg", ~test_penalty), ("penalties", test_penalty), ("all_shots", np.ones(len(test), dtype=bool)))
        }
    else:
        metadata["statsbomb_final_test_comparison"] = {"available": False, "reason": "xg_statsbomb column absent"}
    metadata["test_evaluated_at"] = _now()
    metadata["excluded_ineligible_rows"] = len(shots) - len(eligible)
    val_metric = metadata["metrics"]["validation"]["npxg"]
    metadata["beats_baseline_validation"] = val_metric["model"]["log_loss"] < val_metric["baseline"]["log_loss"]
    if not metadata["beats_baseline_validation"]:
        metadata["limitations"].append("Own npxG did not improve validation log loss over the training-rate baseline; investigate coverage and features before relying on it.")
    if not metadata["penalty_component"]["available"]:
        metadata["limitations"].append("No eligible training penalties; full own-model xG is unavailable for penalty shots.")
    _write_json(output_dir / "dataset_manifest.json", manifest)
    _write_json(output_dir / "split_manifests.json", split_match_ids)
    joblib.dump(artifact.pipeline, output_dir / "model.joblib")
    _calibration_plot(metadata["metrics"], output_dir / "calibration.png")
    metadata["checksums"] = {
        filename: _sha256(output_dir / filename)
        for filename in ("model.joblib", "selection.json", "dataset_manifest.json", "split_manifests.json", "calibration.png")
    }
    metadata["dataset_files"] = {"shots.csv": _sha256(shots_path), "manifest.json": _sha256(manifest_path)}
    _write_json(output_dir / "metadata.json", metadata)
    return artifact


def load_model(model_dir: Path | str) -> ModelArtifact:
    """Load a local, trusted artifact after checking its recorded file hashes.

    Joblib is pickle-based: this application loads only artifacts trained on
    this computer, never untrusted uploads. Hashes detect accidental changes;
    they are not a substitute for trusting the producer of a model.
    """
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != MODEL_SCHEMA_VERSION or metadata.get("features") != list(FEATURES):
        raise ValueError("Model metadata schema or feature contract is incompatible.")
    checksums = metadata.get("checksums", {})
    expected_files = {"model.joblib", "selection.json", "dataset_manifest.json", "split_manifests.json", "calibration.png"}
    if not expected_files.issubset(checksums):
        raise ValueError("Model metadata has an incomplete artifact checksum manifest.")
    for filename in sorted(expected_files):
        if _sha256(model_dir / filename) != checksums[filename]:
            raise ValueError(f"Model artifact checksum mismatch: {filename}.")
    pipeline = joblib.load(model_dir / "model.joblib")
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Saved artifact is not a scikit-learn pipeline.")
    return ModelArtifact(pipeline, metadata)
