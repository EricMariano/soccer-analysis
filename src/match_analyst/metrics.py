"""Match facts calculated independently of the interface and text renderer.

Spatial definitions use native StatsBomb units: attacking penalty area
x >= 102 and 18 <= y <= 62; final third x >= 80. Corridor boundaries split
the destination y axis into equal thirds (left, centre, right, in the
attacking team's perspective). Missing pass locations never remove attempts
from the pass-completion denominator; spatial metrics report their coverage.
"""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import pandas as pd


METRICS_VERSION = "2.0.0"
XG_LABELS = {"xg_statsbomb": "xG StatsBomb", "xg_model": "xG próprio"}
STAT_COLUMNS = (
    "team_id", "team_name", "shots", "shots_inside_box", "shots_outside_box",
    "goals", "passes_attempted", "passes_completed", "pass_accuracy",
    "final_third_entries", "entries_left", "entries_center", "entries_right",
    "pressures", "interceptions", "tackles", "clearances", "xg", "npxg",
    "xg_per_shot", "xg_observed_shots", "xg_missing_shots",
    "npxg_observed_shots", "npxg_missing_shots", "passes_with_coordinates",
    "shots_with_coordinates",
)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values and missing values into strict JSON data."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _team_catalog(events: pd.DataFrame, match: dict | None) -> dict[int, str]:
    teams: dict[int, str] = {}
    if match:
        for side in ("home", "away"):
            team = match.get(f"{side}_team", {})
            identifier = team.get(f"{side}_team_id", team.get("id"))
            name = team.get(f"{side}_team_name", team.get("name"))
            if identifier is not None and name:
                teams[int(identifier)] = str(name)
    if "team_id" in events and "team_name" in events:
        for row in events[["team_id", "team_name"]].dropna().drop_duplicates().itertuples(index=False):
            teams.setdefault(int(row.team_id), str(row.team_name))
    return teams


def _same_own_goal(left: pd.Series, right: pd.Series) -> bool:
    if left["period"] != right["period"] or left["match_id"] != right["match_id"]:
        return False
    left_related = left.get("related_event_ids", [])
    right_related = right.get("related_event_ids", [])
    left_related = left_related if isinstance(left_related, list) else []
    right_related = right_related if isinstance(right_related, list) else []
    if right["event_id"] in left_related or left["event_id"] in right_related:
        return True
    left_time, right_time = left.get("timestamp"), right.get("timestamp")
    if pd.notna(left_time) and pd.notna(right_time):
        return left_time == right_time
    return left["minute"] == right["minute"] and left["second"] == right["second"]


def _with_goal_credits(events: pd.DataFrame, match: dict | None) -> pd.DataFrame:
    """Credit an own goal once, preferring Own Goal Against over its mirror.

    Linked Own Goal For records are a second representation of the same goal.
    When links are absent, equal period/clock timestamps identify the mirror.
    Keeping the credit on the canonical event lets filters use the scoring
    team without fabricating a Shot or changing the original acting team.
    """
    if "_goal_team_id" in events:
        return events.copy()
    result = events.copy()
    result["_goal_team_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if result.empty:
        return result
    kind = result["event_type"]
    goals = kind.eq("Shot") & result["is_goal"].fillna(False)
    result.loc[goals, "_goal_team_id"] = result.loc[goals, "team_id"]
    teams = _team_catalog(result, match)
    against = result[kind.eq("Own Goal Against")]
    credited_against: list[tuple[pd.Series, int]] = []
    for index, row in against.iterrows():
        opponents = [identifier for identifier in teams if identifier != int(row["team_id"])]
        mirrors = result[kind.eq("Own Goal For")]
        mirrors = mirrors[mirrors.apply(lambda other: _same_own_goal(row, other), axis=1)]
        if not mirrors.empty:
            credited = int(mirrors.iloc[0]["team_id"])
        elif len(opponents) == 1:
            credited = opponents[0]
        else:
            continue
        result.loc[index, "_goal_team_id"] = credited
        credited_against.append((row, credited))
    for index, row in result[kind.eq("Own Goal For")].iterrows():
        if not any(credited == int(row["team_id"]) and _same_own_goal(other, row) for other, credited in credited_against):
            result.loc[index, "_goal_team_id"] = int(row["team_id"])
    return result


def filter_events(
    events: pd.DataFrame,
    team_id: int | None = None,
    period: int | Sequence[int] | None = None,
    player_id: int | None = None,
) -> pd.DataFrame:
    """Select the same slice used by facts and charts, always excluding period 5.

    Normally ``team_id`` refers to the acting team. Internally credited own
    goals follow the scoring team, so the score for a team filter is correct.
    Player filters describe the player's actions and exclude other players'
    goal credits, including own goals. Inputs are never mutated.
    """
    mask = events["period"].isin([1, 2, 3, 4])
    if team_id is not None:
        team_mask = events["team_id"].eq(team_id)
        if "_goal_team_id" in events:
            own = events["event_type"].isin(["Own Goal For", "Own Goal Against"])
            team_mask = team_mask.where(~own, events["_goal_team_id"].eq(team_id))
        mask &= team_mask.fillna(False)
    if period is not None:
        periods = [period] if isinstance(period, int) else list(period)
        mask &= events["period"].isin(periods)
    if player_id is not None:
        mask &= events["player_id"].eq(player_id).fillna(False)
    return events.loc[mask].copy().sort_values(["period", "index"], kind="stable").reset_index(drop=True)


def _probabilities(shots: pd.DataFrame, source: str) -> pd.Series:
    if source not in shots:
        return pd.Series(float("nan"), index=shots.index, dtype=float)
    values = pd.to_numeric(shots[source], errors="coerce").astype(float)
    return values.where(values.between(0, 1))


def _complete_sum(values: pd.Series, source_available: bool) -> float | None:
    if not source_available or values.isna().any():
        return None
    return float(values.sum())


def team_statistics(
    events: pd.DataFrame,
    match: dict | None = None,
    xg_source: str = "xg_statsbomb",
) -> pd.DataFrame:
    """One row per match team, with explicit probability/spatial coverage.

    xG totals are unavailable if any eligible shot lacks the selected source.
    A known source has a zero total for zero shots; ratios without denominators
    remain unavailable. npxG always excludes in-game penalties as well.
    """
    if xg_source not in XG_LABELS:
        raise ValueError(f"Unsupported xG source: {xg_source}")
    events = _with_goal_credits(filter_events(events), match)
    teams = _team_catalog(events, match)
    records = []
    for team_id, team_name in teams.items():
        selected = events[events["team_id"].eq(team_id)]
        shots = selected[selected["event_type"].eq("Shot")]
        passes = selected[selected["event_type"].eq("Pass")]
        shot_xy = shots[["x", "y"]].notna().all(axis=1)
        inside = shot_xy & shots["x"].ge(102) & shots["y"].between(18, 62)
        complete = passes["is_complete"].fillna(False).astype(bool)
        pass_xy = passes[["x", "y", "end_x", "end_y"]].notna().all(axis=1)
        entries = passes.loc[complete & pass_xy & passes["x"].lt(80) & passes["end_x"].ge(80)]
        probabilities = _probabilities(shots, xg_source)
        non_penalty = ~shots["is_penalty"].fillna(False).astype(bool)
        npxg_values = probabilities[non_penalty]
        xg = _complete_sum(probabilities, xg_source in events)
        npxg = _complete_sum(npxg_values, xg_source in events)
        defensive = selected.get("defensive_action", pd.Series(dtype="string")).value_counts()
        records.append({
            "team_id": team_id, "team_name": team_name,
            "shots": len(shots), "shots_inside_box": int(inside.sum()),
            "shots_outside_box": int((shot_xy & ~inside).sum()),
            "goals": int(events["_goal_team_id"].eq(team_id).sum()),
            "passes_attempted": len(passes), "passes_completed": int(complete.sum()),
            "pass_accuracy": float(complete.sum() / len(passes)) if len(passes) else None,
            "final_third_entries": len(entries),
            "entries_left": int(entries["end_y"].lt(80 / 3).sum()),
            "entries_center": int((entries["end_y"].ge(80 / 3) & entries["end_y"].lt(160 / 3)).sum()),
            "entries_right": int(entries["end_y"].ge(160 / 3).sum()),
            "pressures": int(defensive.get("Pressure", 0)),
            "interceptions": int(defensive.get("Interception", 0)),
            "tackles": int(defensive.get("Tackle", 0)),
            "clearances": int(defensive.get("Clearance", 0)),
            "xg": xg, "npxg": npxg,
            "xg_per_shot": xg / len(shots) if xg is not None and len(shots) else None,
            "xg_observed_shots": int(probabilities.notna().sum()),
            "xg_missing_shots": int(probabilities.isna().sum()),
            "npxg_observed_shots": int(npxg_values.notna().sum()),
            "npxg_missing_shots": int(npxg_values.isna().sum()),
            "passes_with_coordinates": int(pass_xy.sum()),
            "shots_with_coordinates": int(shot_xy.sum()),
        })
    return pd.DataFrame(records, columns=STAT_COLUMNS)


def build_facts(
    events: pd.DataFrame,
    match: dict,
    filters: dict | None = None,
    xg_source: str = "xg_statsbomb",
    xg_version: str | None = None,
    validation: Any = None,
) -> dict:
    """Build JSON-safe evidence from complete events and an explicit slice.

    Pass unfiltered events here: own-goal mirrors are reconciled before applying
    the requested team/period/player filters. Final score stays separate from
    goals in the selected slice and never supplies a filtered goal count.
    """
    if xg_source not in XG_LABELS:
        raise ValueError(f"Unsupported xG source: {xg_source}")
    filters = {key: value for key, value in (filters or {}).items() if value is not None}
    unknown = set(filters) - {"team_id", "period", "player_id"}
    if unknown:
        raise ValueError(f"Unsupported filters: {sorted(unknown)}")
    full = _with_goal_credits(filter_events(events), match)
    selected = filter_events(full, **filters)
    stats = team_statistics(selected, match=match, xg_source=xg_source)
    if "team_id" in filters:
        stats = stats[stats["team_id"].eq(filters["team_id"])]
    teams = json_safe(stats.to_dict(orient="records"))
    limitations = [
        "Mapas de eventos mostram onde as ações ocorreram; não medem tempo de permanência ou posse de bola.",
        "A soma de xG descreve oportunidades registradas pelo modelo; não determina quem deveria vencer.",
    ]
    score_reconciliation = {}
    for side in ("home", "away"):
        team = match.get(f"{side}_team", {})
        identifier = team.get(f"{side}_team_id", team.get("id"))
        expected = match.get(f"{side}_score")
        if identifier is not None:
            calculated = int(full["_goal_team_id"].eq(identifier).sum())
            score_reconciliation[side] = {
                "team_id": identifier, "calculated_goals": calculated,
                "catalogue_goals": expected,
                "matches_catalogue": calculated == expected if expected is not None else None,
            }
            if expected is not None and calculated != expected:
                limitations.append(
                    f"A reconciliação do placar completo de {side} diverge do catálogo: "
                    f"{calculated} gol(s) nos eventos e {expected} no placar final."
                )
    if not len(selected[selected["event_type"].eq("Shot")]):
        limitations.append("O recorte selecionado não contém finalizações.")
    if xg_source not in events:
        limitations.append(f"{XG_LABELS[xg_source]} não está disponível para esta partida.")
    if filters:
        limitations.append("Os indicadores e o texto usam os filtros informados; o placar final é apenas contexto.")
    evidence: dict[str, Any] = {}
    for team in teams:
        identifier = team["team_id"]
        team["defensive_counts"] = {
            "Pressure": team["pressures"], "Interception": team["interceptions"],
            "Tackle": team["tackles"], "Clearance": team["clearances"],
        }
        team["denominators"] = {
            "pass_accuracy": team["passes_attempted"], "xg_per_shot": team["shots"],
            "xg": team["shots"],
            "npxg": team["npxg_observed_shots"] + team["npxg_missing_shots"],
            "xg_observed_shots": team["shots"], "xg_missing_shots": team["shots"],
            "entry_corridor_share": team["final_third_entries"],
            "xg_coverage": team["shots"],
            "npxg_coverage": team["npxg_observed_shots"] + team["npxg_missing_shots"],
            "spatial_pass_coverage": team["passes_attempted"],
        }
        for key, value in team.items():
            denominator = team["denominators"].get(key)
            if key.startswith("entries_"):
                denominator = team["final_third_entries"]
            evidence[f"teams.{identifier}.{key}"] = {"value": value, "denominator": denominator}
        if team["xg_missing_shots"]:
            limitations.append(
                f"{team['team_name']}: {team['xg_missing_shots']} finalização(ões) sem {XG_LABELS[xg_source]}; total de xG indisponível."
            )
        if team["passes_with_coordinates"] < team["passes_attempted"]:
            limitations.append(
                f"{team['team_name']}: entradas no terço final usam {team['passes_with_coordinates']} de "
                f"{team['passes_attempted']} passes com coordenadas completas."
            )
    comparisons: dict[str, Any] = {}
    if len(teams) == 2:
        first, second = teams
        comparisons = {
            "team_order": [first["team_id"], second["team_id"]],
            "shot_difference": first["shots"] - second["shots"],
            "xg_difference": first["xg"] - second["xg"]
            if first["xg"] is not None and second["xg"] is not None else None,
        }
    validation_data = validation.as_dict() if hasattr(validation, "as_dict") else validation
    if validation_data and validation_data.get("excluded_count", 0):
        limitations.append(f"A validação excluiu {validation_data['excluded_count']} evento(s); consulte o diagnóstico para os motivos.")
    match_identity = {
        "match_id": match.get("match_id"), "match_date": match.get("match_date"),
        "competition": match.get("competition"), "season": match.get("season"),
        "home_team": match.get("home_team"), "away_team": match.get("away_team"),
        "final_score": {"home": match.get("home_score"), "away": match.get("away_score")},
    }
    evidence["match.identity"] = {"value": match_identity, "denominator": None}
    evidence["match.score_reconciliation"] = {"value": score_reconciliation, "denominator": None}
    evidence["scope.filters"] = {"value": filters, "denominator": None}
    evidence["xg.source"] = {"value": {"source": xg_source, "version": xg_version}, "denominator": None}
    evidence["analysis.limitations"] = {"value": limitations, "denominator": None}
    return json_safe({
        "schema_version": "1.0.0", "metrics_version": METRICS_VERSION,
        "match": match_identity, "filters": filters, "teams": teams,
        "xg": {"source": xg_source, "label": XG_LABELS[xg_source], "version": xg_version,
               "available": xg_source in events, "scope": "all_shots"},
        "comparisons": comparisons, "limitations": limitations,
        "score_reconciliation": score_reconciliation,
        "evidence": evidence, "validation": validation_data,
        "attribution": "StatsBomb Open Data — https://github.com/statsbomb/open-data",
    })
