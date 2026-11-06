"""Deterministic Portuguese reports backed by machine-readable evidence."""

from typing import Any


REPORT_VERSION = "rules-1.0.0"


def _number(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def generate_report(
    facts: dict[str, Any],
    min_entries: int = 10,
    corridor_margin: float = 0.10,
) -> dict[str, Any]:
    """Render facts without calculating match metrics or using an LLM.

    Corridor concentration requires at least ``min_entries`` and a lead of
    ``corridor_margin`` (fraction of all entries) over the next corridor.
    These are descriptive thresholds, not a test of tactical dominance.
    """
    if min_entries < 1 or not 0 < corridor_margin <= 1:
        raise ValueError("min_entries must be positive and corridor_margin in (0, 1]")
    sentences: list[dict[str, Any]] = []

    def add(text: str, *keys: str) -> None:
        missing = set(keys) - set(facts["evidence"])
        if missing:
            raise ValueError(f"Report evidence is missing: {sorted(missing)}")
        sentences.append({"text": text, "evidence_keys": list(keys)})

    if facts.get("filters"):
        add("Este resumo considera o recorte selecionado na interface.", "scope.filters")
    for team in facts["teams"]:
        prefix = f"teams.{team['team_id']}."
        name = team["team_name"]
        if team["shots"]:
            add(
                f"{name} finalizou {team['shots']} vezes, sendo {team['shots_inside_box']} de dentro da área.",
                prefix + "shots", prefix + "shots_inside_box",
            )
        else:
            add(f"{name} não registrou finalizações neste recorte.", prefix + "shots")
        if team["passes_attempted"]:
            add(
                f"{name} completou {team['passes_completed']} de {team['passes_attempted']} passes "
                f"({_number(team['pass_accuracy'] * 100)}%).",
                prefix + "passes_completed", prefix + "passes_attempted", prefix + "pass_accuracy",
            )
        else:
            add(f"{name} não registrou passes; a taxa de acerto não está disponível.", prefix + "passes_attempted", prefix + "pass_accuracy")
        add(
            f"{name} registrou {team['final_third_entries']} entradas por passe no terço final: "
            f"{team['entries_left']} pela esquerda, {team['entries_center']} pelo centro e {team['entries_right']} pela direita.",
            prefix + "final_third_entries", prefix + "entries_left", prefix + "entries_center", prefix + "entries_right",
        )
        corridors = sorted(
            [(team["entries_left"], "esquerda", "entries_left"),
             (team["entries_center"], "centro", "entries_center"),
             (team["entries_right"], "direita", "entries_right")],
            reverse=True,
        )
        if team["final_third_entries"] >= min_entries and (
            corridors[0][0] - corridors[1][0] >= corridor_margin * team["final_third_entries"] - 1e-12
        ):
            add(
                f"{name} concentrou as entradas pelo corredor da {corridors[0][1]}."
                if corridors[0][1] != "centro" else f"{name} concentrou as entradas pelo corredor central.",
                prefix + "final_third_entries", prefix + corridors[0][2], prefix + corridors[1][2],
            )
        label = facts["xg"]["label"]
        if team["xg"] is not None:
            add(
                f"Segundo {label}, {name} somou {_number(team['xg'])} de xG neste recorte.",
                prefix + "xg", prefix + "xg_observed_shots", "xg.source",
            )
        elif team["npxg"] is not None:
            add(
                f"Segundo {label}, {name} somou {_number(team['npxg'])} sem pênaltis (npxG); "
                "o total com pênaltis não está disponível.",
                prefix + "npxg", prefix + "npxg_observed_shots", prefix + "xg_missing_shots", "xg.source",
            )
        else:
            add(
                f"O total de {label} de {name} não está disponível para este recorte.",
                prefix + "xg", prefix + "xg_missing_shots", "xg.source",
            )
    add(
        "A soma de xG descreve as oportunidades registradas segundo a fonte selecionada e não determina quem deveria vencer.",
        "analysis.limitations", "xg.source",
    )
    return {
        "text": "\n\n".join(sentence["text"] for sentence in sentences),
        "sentences": sentences,
        "report_version": REPORT_VERSION,
        "metrics_version": facts["metrics_version"],
        "match_id": facts["match"]["match_id"],
        "filters": facts["filters"],
        "rules": {"min_entries": min_entries, "corridor_margin": corridor_margin},
    }
