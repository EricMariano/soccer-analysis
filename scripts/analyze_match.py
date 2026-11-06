"""Export one match analysis outside Streamlit, using the same calculation layers.

Without --match-id the demonstration is discovered from the real catalogue.
For an explicit match, --competition-id and --season-id identify its catalogue.
Every run creates a fresh output folder so existing human-review records survive.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from zipfile import ZipFile

import pandas as pd

from match_analyst.data import DEFAULT_CACHE_DIR, PROJECT_ROOT, DataError, StatsBombClient
from match_analyst.metrics import build_facts, filter_events
from match_analyst.narrative import generate_report
from match_analyst.normalize import normalize_events
from match_analyst.xg import load_model, model_membership, predict_xg

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "xg_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
PERIOD_LABELS = {1: "First half", 2: "Second half", 3: "Extra time first half", 4: "Extra time second half"}


def _positive_id(value: str) -> int:
    try:
        identifier = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a positive integer ID.") from exc
    if identifier <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer ID.")
    return identifier


def _fresh_directory(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = parent / (name if suffix == 1 else f"{name}-{suffix}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


def analyze_match(
    *,
    offline: bool = False,
    match_id: int | None = None,
    competition_id: int = 43,
    season_id: int = 106,
    team_id: int | None = None,
    period: int | None = None,
    player_id: int | None = None,
    xg_source: str = "xg_statsbomb",
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    """Return export paths and counts; never train or update a model.

    StatsBomb analysis works without a model. If a local model is available,
    both separately named probabilities and the comparison chart are exported.
    An explicitly requested own source requires a working saved artifact.
    """
    if xg_source not in {"xg_statsbomb", "xg_model"}:
        raise DataError("xg_source must be xg_statsbomb or xg_model.")
    if period is not None and period not in PERIOD_LABELS:
        raise DataError("period must be 1, 2, 3 or 4; penalty shootouts are excluded.")
    client = StatsBombClient(cache_dir, offline=offline)
    if match_id is None:
        selected = client.find_demo_match()
    else:
        matches = client.matches(competition_id, season_id)
        selected = next((match for match in matches if match["match_id"] == match_id), None)
        if selected is None:
            raise DataError(
                f"Match {match_id} is not in competition {competition_id}, season {season_id}."
            )
    loaded = client.load_match(selected)
    normalized = normalize_events(
        loaded.events,
        match_id=selected["match_id"],
        competition_id=selected["competition_id"],
        season_id=selected["season_id"],
    )
    events = normalized.events
    teams = {
        selected[f"{side}_team"][f"{side}_team_id"]: selected[f"{side}_team"][f"{side}_team_name"]
        for side in ("home", "away")
    }
    if team_id is not None and team_id not in teams:
        raise DataError(f"Team {team_id} does not belong to match {selected['match_id']}.")
    player_rows = events if team_id is None else events.loc[events.team_id.eq(team_id)]
    players = player_rows.dropna(subset=["player_id", "player_name"])[["player_id", "player_name"]].drop_duplicates("player_id")
    player_names = {int(row.player_id): str(row.player_name) for row in players.itertuples()}
    if player_id is not None and player_id not in player_names:
        raise DataError(f"Player {player_id} is not present in the selected match/team.")

    model = None
    warnings = []
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            model = load_model(model_dir)
            events = predict_xg(events, model)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            model = None
            if xg_source == "xg_model":
                raise DataError(f"The saved own model cannot be used: {exc}") from exc
            warnings.append(f"Own model unavailable: {exc}")
    elif xg_source == "xg_model":
        raise DataError(
            f"No saved own model at {model_dir}. Run the explicit training command first, "
            "or select xg_statsbomb. Analysis never trains a model."
        )
    revision = loaded.provenance["source_revision"]
    version = revision if xg_source == "xg_statsbomb" else model.metadata["model_version"]
    label = (f"xG StatsBomb · {revision[:12]}" if xg_source == "xg_statsbomb"
             else f"Own xG · {model.metadata['model_version']}")
    filters = {"team_id": team_id, "period": period, "player_id": player_id}
    scope = " · ".join([
        f"Match {selected['match_id']}", teams.get(team_id, "Both teams"),
        PERIOD_LABELS.get(period, "Full match"), player_names.get(player_id, "All players"),
    ])
    plot_scope = scope
    facts = build_facts(
        events, selected, filters=filters, xg_source=xg_source,
        xg_version=version, validation=normalized.report.as_dict(),
    )
    facts["provenance"] = loaded.provenance
    facts["scope_label"] = scope
    facts["model"] = None if model is None else {
        "version": model.metadata["model_version"],
        "dataset_id": model.metadata["dataset_id"],
        "match_membership": model_membership(selected["match_id"], model.metadata),
        "penalty_component": model.metadata["penalty_component"],
    }
    facts["limitations"].extend(warnings)
    if model is not None and xg_source == "xg_model":
        facts["limitations"].extend(model.metadata.get("limitations", []))
    # build_facts returns independent JSON-safe containers; keep evidence in sync
    # when adding source/model limitations before generating the report.
    facts["evidence"]["analysis.limitations"]["value"] = list(facts["limitations"])
    report = generate_report(facts)
    scoped = filter_events(events, **filters)
    shots = scoped.loc[scoped.event_type.eq("Shot")].copy()
    statistics = pd.DataFrame(facts["teams"])

    # Configure a writable noninteractive plotting environment only for the CLI.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "match-analyst-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    from match_analyst.exports import bundle_export
    from match_analyst.plots import (
        plot_cumulative_xg, plot_defensive_map, plot_pass_heatmap,
        plot_pass_map, plot_shot_map, plot_xg_comparison,
    )
    figures = {
        "shots": plot_shot_map(shots, selected, xg_source=xg_source, xg_label=label, scope=plot_scope),
        "passes": plot_pass_map(scoped, selected, scope=plot_scope),
        "defensive": plot_defensive_map(scoped, selected, scope=plot_scope),
        "pass_origins": plot_pass_heatmap(scoped, selected, scope=plot_scope),
        "cumulative_xg": plot_cumulative_xg(
            scoped, selected, xg_source=xg_source, xg_label=label,
            scope=plot_scope, clock_events=events,
        ),
    }
    if model is not None:
        figures["xg_comparison"] = plot_xg_comparison(
            scoped, scope=plot_scope, model_label=model.metadata["model_version"],
        )
    # A separate header keeps source versions on non-xG maps without crowding
    # the shared scope/legend footer used by the plotting layer.
    for figure in figures.values():
        figure.text(.97, .975, label, ha="right", va="top", fontsize=8, color="#153345")
    archive = bundle_export(facts, report, figures, stats=statistics, shots=shots)
    folder_name = (
        f"match-{selected['match_id']}-{xg_source}"
        f"-team-{team_id or 'all'}-period-{period or 'all'}-player-{player_id or 'all'}"
    )
    destination = _fresh_directory(Path(output_dir), folder_name)
    archive_path = destination / "analysis.zip"
    archive_path.write_bytes(archive)
    with ZipFile(BytesIO(archive)) as bundle:
        bundle.extractall(destination)
        exported_files = sorted(bundle.namelist())
    return {
        "match_id": selected["match_id"],
        "match_date": selected["match_date"],
        "teams": list(teams.values()),
        "final_score": {"home": selected["home_score"], "away": selected["away_score"]},
        "filters": facts["filters"],
        "xg_source": xg_source,
        "xg_version": version,
        "model": facts["model"],
        "shots": len(shots),
        "shots_by_team": {str(name): int(count) for name, count in shots.groupby("team_name").size().items()},
        "validation": normalized.report.as_dict(),
        "output_dir": str(destination.resolve()),
        "archive_path": str(archive_path.resolve()),
        "exported_files": exported_files,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use only the local cache; never access the network.")
    parser.add_argument("--match-id", type=_positive_id, help="Default: discover Brazil v Serbia 2022.")
    parser.add_argument("--competition-id", type=_positive_id, default=43)
    parser.add_argument("--season-id", type=_positive_id, default=106)
    parser.add_argument("--team-id", type=_positive_id)
    parser.add_argument("--period", type=int, choices=range(1, 5))
    parser.add_argument("--player-id", type=_positive_id)
    parser.add_argument("--xg-source", choices=("xg_statsbomb", "xg_model"), default="xg_statsbomb")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    try:
        result = analyze_match(**vars(args))
    except (DataError, OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Could not analyze match: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
