"""Portable analysis exports and separate review records for future reports."""

from io import BytesIO
import json
import re
from zipfile import ZIP_DEFLATED, ZipFile

from matplotlib.figure import Figure
import pandas as pd

from match_analyst.metrics import json_safe


def _json(value: object) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def bundle_export(
    facts: dict,
    report: dict,
    figures: dict[str, Figure],
    stats: pd.DataFrame | None = None,
    *,
    shots: pd.DataFrame | None = None,
) -> bytes:
    """Return a ZIP of exact displayed facts, statistics, text and figures.

    Facts and generated output are separate. ``human_review.json`` is an
    unapproved empty slot; filling it is an explicit future human action.
    When supplied, ``shots`` is exported unchanged as ``shots.csv`` so event
    identifiers and each separately named probability source remain visible.
    This function does not mutate inputs or write anywhere on disk.
    """
    destination = BytesIO()
    table = stats if stats is not None else pd.DataFrame(facts["teams"])
    table = table.drop(columns=["defensive_counts", "denominators"], errors="ignore")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("facts.json", _json(facts))
        archive.writestr("report.json", _json(report))
        archive.writestr("statistics.csv", table.to_csv(index=False, na_rep=""))
        if shots is not None:
            archive.writestr("shots.csv", shots.to_csv(index=False, na_rep=""))
        archive.writestr(
            "report.md",
            f"# Match report — {facts['match']['match_id']}\n\n"
            f"{report['text']}\n\n"
            f"Source: {facts['attribution']}\n\n"
            f"Metrics: {facts['metrics_version']} · Report: {report['report_version']}\n",
        )
        archive.writestr("human_review.json", _json({
            "match_id": facts["match"]["match_id"], "filters": facts["filters"],
            "metrics_version": facts["metrics_version"],
            "report_version": report["report_version"],
            "approved_correction": None, "approved_by": None, "approved_at": None,
        }))
        used_names = set()
        for index, (name, figure) in enumerate(figures.items()):
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or f"figure_{index + 1}"
            base_name, suffix = safe_name, 2
            while safe_name in used_names:
                safe_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(safe_name)
            image = BytesIO()
            figure.savefig(image, format="png", dpi=150, bbox_inches="tight")
            archive.writestr(f"figures/{safe_name}.png", image.getvalue())
    return destination.getvalue()
