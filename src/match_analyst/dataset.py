"""Reproduce a dated, match-grouped shot dataset from a pinned source revision."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from match_analyst.data import DEFAULT_CACHE_DIR, PROJECT_ROOT, DataError, StatsBombClient, _atomic_write
from match_analyst.normalize import normalize_shots

SPLITS = ("train", "validation", "test")
PROCESSING_VERSION = "shot-dataset-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "dataset.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


class DatasetError(DataError):
    """The selected dataset is unavailable, inconsistent, or would leak matches."""


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    contents = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _atomic_write(path, contents.encode("utf-8"))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _config_date(value: Any, description: str) -> date:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise DatasetError(f"Invalid ISO date for {description}: {value!r}.") from exc


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise DatasetError("Dataset configuration must use schema_version 1.")
    revision = config.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DatasetError("Dataset source_revision must pin a full lowercase commit SHA.")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise DatasetError("Dataset configuration needs a name.")
    scope = config.get("scope")
    if not isinstance(scope, dict) or any(
        key not in scope for key in (
            "competition_gender", "competition_youth", "competition_international"
        )
    ):
        raise DatasetError("Dataset scope must declare gender, youth and international flags.")
    if scope["competition_gender"] not in {"male", "female"} or any(
        not isinstance(scope[key], bool) for key in ("competition_youth", "competition_international")
    ):
        raise DatasetError("Dataset scope has invalid gender or boolean flags.")
    split_configs = config.get("splits")
    if not isinstance(split_configs, dict) or set(split_configs) != set(SPLITS):
        raise DatasetError("Dataset configuration must define train, validation and test splits.")
    previous_end: date | None = None
    for split in SPLITS:
        settings = split_configs[split]
        if not isinstance(settings, dict):
            raise DatasetError(f"Invalid configuration for split {split}.")
        start = _config_date(settings.get("start_date"), f"{split}.start_date")
        end = _config_date(settings.get("end_date"), f"{split}.end_date")
        if start > end or (previous_end is not None and start <= previous_end):
            raise DatasetError("Split date windows must be ordered and must not overlap.")
        previous_end = end
        selections = settings.get("competitions")
        if not isinstance(selections, list) or not selections:
            raise DatasetError(f"Split {split} must select at least one competition/season.")
        for selected in selections:
            if not isinstance(selected, dict) or not all(
                _positive_int(selected.get(key)) for key in ("competition_id", "season_id")
            ):
                raise DatasetError(f"Invalid competition/season selection in {split}.")
            if "expected_matches" in selected and not _positive_int(selected["expected_matches"]):
                raise DatasetError("expected_matches must be a positive integer when supplied.")
    demonstration_ids = config.get("demonstration_match_ids", [])
    if not isinstance(demonstration_ids, list) or not all(_positive_int(mid) for mid in demonstration_ids):
        raise DatasetError("demonstration_match_ids must contain positive integers.")
    return deepcopy(config)


def load_dataset_config(config_path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate editable cohort selection and chronological split rules."""
    try:
        with Path(config_path).open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"Could not read dataset configuration at {config_path}.") from exc
    return _validate_config(config)


def validate_match_splits(split_match_ids: dict[str, list[int]]) -> None:
    """Reject repeated IDs within or between splits before any training can occur."""
    if set(split_match_ids) != set(SPLITS):
        raise DatasetError("Match manifests must define train, validation and test.")
    seen: set[int] = set()
    for split in SPLITS:
        ids = split_match_ids[split]
        if not all(_positive_int(mid) for mid in ids):
            raise DatasetError(f"Split {split} contains an invalid match ID.")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise DatasetError("A match appears more than once in the split manifests.")
        seen.update(ids)


def build_catalog_manifest(config: dict[str, Any], client: StatsBombClient) -> dict[str, Any]:
    """Confirm cohort availability and dates using catalogues only; fetch no events."""
    config = _validate_config(config)
    if client.revision != config["source_revision"]:
        raise DatasetError("Client revision differs from the pinned dataset configuration.")
    competitions = client.competitions()
    lookup = {(row["competition_id"], row["season_id"]): row for row in competitions}
    source_files = {"competitions.json": client.file_provenance("competitions.json")}
    manifest_matches: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    split_ids: dict[str, list[int]] = {split: [] for split in SPLITS}
    holdout_ids = set(config.get("demonstration_match_ids", []))
    for split in SPLITS:
        settings = config["splits"][split]
        start = date.fromisoformat(settings["start_date"])
        end = date.fromisoformat(settings["end_date"])
        for selection in settings["competitions"]:
            cid, sid = selection["competition_id"], selection["season_id"]
            competition = lookup.get((cid, sid))
            if competition is None:
                raise DatasetError(f"Competition {cid}, season {sid} is absent from the pinned catalogue.")
            for key in ("competition_gender", "competition_youth", "competition_international"):
                if competition.get(key) != config["scope"][key]:
                    raise DatasetError(f"Competition {cid}/{sid} violates the configured scope ({key}).")
            matches = client.matches(cid, sid)
            relative_path = f"matches/{cid}/{sid}.json"
            source_files[relative_path] = client.file_provenance(relative_path)
            selected_matches = []
            excluded: Counter[str] = Counter()
            for match in matches:
                match_date = _config_date(match.get("match_date"), f"match {match.get('match_id')}")
                if not start <= match_date <= end:
                    excluded["outside_split_dates"] += 1
                    continue
                if match.get("match_status", "available") != "available":
                    excluded["events_not_available"] += 1
                    continue
                mid = match.get("match_id")
                if not _positive_int(mid):
                    raise DatasetError("The match catalogue contains an invalid match ID.")
                if split != "test" and mid in holdout_ids:
                    raise DatasetError(f"Demonstration match {mid} cannot enter {split}.")
                record = {
                    "match_id": mid,
                    "competition_id": cid,
                    "season_id": sid,
                    "competition_name": competition["competition_name"],
                    "season_name": competition["season_name"],
                    "match_date": match_date.isoformat(),
                    "split": split,
                    "home_team": match.get("home_team", {}).get("home_team_name"),
                    "away_team": match.get("away_team", {}).get("away_team_name"),
                    "home_score": match.get("home_score"),
                    "away_score": match.get("away_score"),
                }
                selected_matches.append(record)
                split_ids[split].append(mid)
            if not selected_matches:
                raise DatasetError(f"No available matches selected for {split}, competition {cid}/{sid}.")
            expected = selection.get("expected_matches")
            if expected is not None and len(selected_matches) != expected:
                raise DatasetError(
                    f"Expected {expected} eligible matches for {split} {cid}/{sid}; "
                    f"the pinned catalogue provides {len(selected_matches)}. Review the selection before downloading events."
                )
            dates = sorted(row["match_date"] for row in selected_matches)
            coverage.append({
                "split": split,
                "competition_id": cid,
                "season_id": sid,
                "competition_name": competition["competition_name"],
                "season_name": competition["season_name"],
                "catalog_matches": len(matches),
                "selected_matches": len(selected_matches),
                "expected_matches": expected,
                "first_match_date": dates[0],
                "last_match_date": dates[-1],
                "excluded_catalog_matches": dict(excluded),
                "coverage_basis": "Observed pinned catalogue records within configured dates; no claim about other seasons.",
            })
            manifest_matches.extend(selected_matches)
    validate_match_splits(split_ids)
    split_ids = {split: sorted(ids) for split, ids in split_ids.items()}
    manifest_matches.sort(key=lambda row: (SPLITS.index(row["split"]), row["match_date"], row["match_id"]))
    identity = {
        "processing_version": PROCESSING_VERSION,
        "config": config,
        "matches": manifest_matches,
        "catalog_checksums": {path: meta["sha256"] for path, meta in sorted(source_files.items())},
    }
    return {
        "schema_version": 1,
        "processing_version": PROCESSING_VERSION,
        "status": "catalog_only",
        "name": config["name"],
        "source_revision": client.revision,
        "selection_id": "selection-" + _digest(identity)[:20],
        "dataset_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "catalog_sources": [{"path": path, **meta} for path, meta in sorted(source_files.items())],
        "coverage": coverage,
        "split_match_ids": split_ids,
        "matches": manifest_matches,
        "coverage_notes": config.get("coverage_notes", []),
    }


def prepare_dataset(
    config_path: Path | str = DEFAULT_CONFIG,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    offline: bool = False,
    workers: int = 4,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Incrementally cache events and write deterministic eligible shot rows.

    All shots in a match inherit the split from its dated catalogue record.
    In-game penalties are retained and flagged for a separate model component;
    period-five shootouts are excluded by the normalizer. Download timestamps
    and machine-specific paths never influence ``dataset_id``.
    """
    if not _positive_int(workers) or workers > 16:
        raise DatasetError("workers must be an integer from 1 to 16.")
    config = load_dataset_config(config_path)
    client = StatsBombClient(cache_dir, revision=config["source_revision"], offline=offline)
    manifest = build_catalog_manifest(config, client)
    output_dir = Path(output_dir)
    _write_json(output_dir / "catalog_manifest.json", manifest)
    if progress:
        summary = ", ".join(f"{split}: {len(manifest['split_match_ids'][split])}" for split in SPLITS)
        progress(f"Catalogue confirmed ({summary}); loading requested event files.")

    # Import only for preparation: catalogue inspection needs no ML runtime.
    from match_analyst.xg import geometry_features

    def process_match(record: dict[str, Any]) -> tuple[dict, pd.DataFrame, dict]:
        loaded = client.load_match(record)
        normalized = normalize_shots(
            loaded.events,
            match_id=record["match_id"],
            competition_id=record["competition_id"],
            season_id=record["season_id"],
        )
        shots = normalized.shots.copy()
        for column in ("split", "match_date", "competition_name", "season_name"):
            shots[column] = record[column]
        geometry = geometry_features(shots)
        for column in ("distance_statsbomb", "angle_radians"):
            shots[column] = geometry[column]
        provenance = loaded.provenance
        complete_record = {
            **record,
            "events_sha256": provenance["sha256"],
            "events_source_url": provenance["source_url"],
            "events_downloaded_at": provenance["downloaded_at"],
        }
        audit = {
            "match_id": record["match_id"],
            "split": record["split"],
            **normalized.report.as_dict(),
            "shots": len(shots),
            "goals": int(shots["is_goal"].sum()),
            "penalties": int(shots["is_penalty"].sum()),
            "penalty_goals": int(shots.loc[shots["is_penalty"], "is_goal"].sum()),
        }
        return complete_record, shots, audit

    completed: dict[int, tuple[dict, pd.DataFrame, dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_match, record): record for record in manifest["matches"]}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            completed[record["match_id"]] = result
            if progress:
                progress(f"[{len(completed)}/{len(futures)}] {record['split']} match {record['match_id']} loaded")
    ordered = [completed[record["match_id"]] for record in manifest["matches"]]
    shots = pd.concat([item[1] for item in ordered], ignore_index=True)
    shots = shots.sort_values(["match_id", "period", "index"], kind="stable").reset_index(drop=True)
    if shots["event_id"].duplicated().any():
        raise DatasetError("An event ID appears in more than one selected match.")
    if len(shots) == 0:
        raise DatasetError("The selected matches contain no eligible shots.")
    event_checksums = [
        {"match_id": record["match_id"], "split": record["split"], "sha256": record["events_sha256"]}
        for record, _, _ in ordered
    ]
    dataset_id = "statsbomb-" + _digest({
        "selection_id": manifest["selection_id"],
        "source_revision": client.revision,
        "processing_version": PROCESSING_VERSION,
        "events": event_checksums,
    })[:20]
    manifest.update({"status": "complete", "dataset_id": dataset_id, "matches": [item[0] for item in ordered]})
    audits = [item[2] for item in ordered]
    excluded: Counter[str] = Counter()
    for audit in audits:
        excluded.update(audit["excluded_by_reason"])
    split_counts = {}
    competition_counts = []
    for split in SPLITS:
        frame = shots.loc[shots["split"] == split]
        penalties = frame["is_penalty"]
        split_counts[split] = {
            "matches": len(manifest["split_match_ids"][split]),
            "shots": len(frame),
            "goals": int(frame["is_goal"].sum()),
            "non_penalty_shots": int((~penalties).sum()),
            "non_penalty_goals": int(frame.loc[~penalties, "is_goal"].sum()),
            "penalties": int(penalties.sum()),
            "penalty_goals": int(frame.loc[penalties, "is_goal"].sum()),
        }
    for coverage in manifest["coverage"]:
        frame = shots.loc[
            (shots["split"] == coverage["split"])
            & (shots["competition_id"] == coverage["competition_id"])
            & (shots["season_id"] == coverage["season_id"])
        ]
        competition_counts.append({
            **coverage,
            "shots": len(frame),
            "goals": int(frame["is_goal"].sum()),
            "penalties": int(frame["is_penalty"].sum()),
        })
    diagnostics = {
        "dataset_id": dataset_id,
        "source_revision": client.revision,
        "split_counts": split_counts,
        "competition_counts": competition_counts,
        "total_events": sum(audit["total_events"] for audit in audits),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "shootout_shots_excluded": excluded.get("penalty_shootout", 0),
        "missing_values": {column: int(count) for column, count in shots.isna().sum().items()},
        "body_part_counts": {str(value): int(count) for value, count in shots["body_part"].fillna("<missing>").value_counts().items()},
        "shot_type_counts": {str(value): int(count) for value, count in shots["shot_type"].fillna("<missing>").value_counts().items()},
        "matches": audits,
        "coverage_notes": manifest["coverage_notes"],
    }
    manifest["split_counts"] = split_counts
    manifest["shot_table"] = {"filename": "shots.csv", "rows": len(shots), "columns": list(shots.columns)}
    csv_contents = shots.to_csv(index=False, lineterminator="\n").encode("utf-8")
    manifest["shot_table"]["sha256"] = hashlib.sha256(csv_contents).hexdigest()
    _atomic_write(output_dir / "shots.csv", csv_contents)
    _write_json(output_dir / "diagnostics.json", diagnostics)
    # Write the completion marker last: an interrupted run leaves no new complete manifest.
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "dataset_id": dataset_id,
        "manifest_path": str(output_dir / "manifest.json"),
        "shots_path": str(output_dir / "shots.csv"),
        "diagnostics_path": str(output_dir / "diagnostics.json"),
        "split_counts": split_counts,
    }
