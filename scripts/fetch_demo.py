"""Prepare the demonstration cache, or verify it with networking disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from match_analyst.data import DataError, StatsBombClient
from match_analyst.normalize import normalize_shots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Never access the network.")
    parser.add_argument("--cache-dir", type=Path, help="Override the local raw-data cache.")
    args = parser.parse_args()
    options = {"offline": args.offline}
    if args.cache_dir is not None:
        options["cache_dir"] = args.cache_dir
    try:
        client = StatsBombClient(**options)
        match = client.find_demo_match()
        loaded = client.load_match(match)
        result = normalize_shots(
            loaded.events,
            match_id=match["match_id"],
            competition_id=match["competition_id"],
            season_id=match["season_id"],
        )
    except DataError as exc:
        parser.exit(1, f"Could not load the demonstration: {exc}\n")

    print(json.dumps({
        "match_id": match["match_id"],
        "match_date": match["match_date"],
        "home_team": match["home_team"]["home_team_name"],
        "away_team": match["away_team"]["away_team_name"],
        "final_score": f"{match['home_score']}–{match['away_score']}",
        "shots_by_team": result.shots.groupby("team_name").size().to_dict(),
        "excluded_by_reason": result.report.excluded_by_reason,
        "event_path": str(loaded.event_path),
        "offline": args.offline,
        "provenance": loaded.provenance,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
