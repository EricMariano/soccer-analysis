"""Shot maps drawn in the original StatsBomb attacking coordinates."""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mplsoccer import Pitch
import numpy as np
import pandas as pd


BACKGROUND = "#f7f5ef"
INK = "#153345"
GOAL = "#bd831e"
PITCH_LINES = "#a4afa6"


def team_name(match: dict[str, Any], side: str) -> str:
    """Read the display name without using it as an identifier."""
    team = match[f"{side}_team"]
    return team[f"{side}_team_name"]


def plot_shot_map(
    shots: pd.DataFrame, match: dict[str, Any], *,
    xg_source: str | None = None, xg_label: str | None = None,
    scope: str = "Full match",
) -> Figure:
    """Return one pitch per team, with equal-size markers for every shot.

    StatsBomb expresses shots in the acting team's left-to-right attacking
    frame. Keeping x and y unchanged therefore gives the same direction for
    both teams and all periods. Period 5 must already have been excluded by
    normalization; it is also excluded here to keep this public plot safe.
    """
    fig = Figure(figsize=(13, 5.9), facecolor=BACKGROUND)
    axes = fig.subplots(1, 2)
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color=BACKGROUND,
        line_color=PITCH_LINES,
        linewidth=1,
        line_zorder=1,
        goal_type="box",
    )
    eligible = shots.loc[shots["period"].ne(5)]
    if "event_type" in eligible:
        eligible = eligible.loc[eligible["event_type"].eq("Shot")]

    for side, ax in zip(("home", "away"), axes, strict=True):
        pitch.draw(ax=ax)
        team_id = match[f"{side}_team"][f"{side}_team_id"]
        team_shots = eligible.loc[eligible["team_id"].eq(team_id)]
        for is_goal, face, edge in (
            (False, "none", INK),
            (True, GOAL, INK),
        ):
            selected = team_shots.loc[team_shots["is_goal"].eq(is_goal)]
            pitch.scatter(
                selected["x"],
                selected["y"],
                s=(np.maximum(12, selected[xg_source].fillna(0.17).to_numpy(dtype=float) * 500)
                   if xg_source and xg_source in selected else 85),
                marker="o",
                facecolors=face,
                edgecolors=edge,
                linewidths=1.5,
                alpha=0.95,
                ax=ax,
                zorder=3 if is_goal else 2,
            )
        shot_label = "shot" if len(team_shots) == 1 else "shots"
        ax.set_title(
            f"{team_name(match, side)}  /  {len(team_shots)} {shot_label}",
            loc="left",
            color=INK,
            fontsize=14,
            fontweight="bold",
            pad=16,
        )
        ax.annotate(
            "Direction of attack",
            xy=(0.95, -0.055),
            xytext=(0.55, -0.055),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="left",
            va="center",
            color=INK,
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1},
        )
        if team_shots.empty:
            ax.text(
                60, 40, "No eligible shots", ha="center", va="center",
                fontsize=12, color=INK,
                bbox={"facecolor": BACKGROUND, "edgecolor": "none", "pad": 8},
                zorder=4,
            )

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=INK, markeredgewidth=1.5, markersize=8,
               label="Other shot"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=GOAL,
               markeredgecolor=INK, markeredgewidth=1.5, markersize=8,
               label="Goal"),
    ]
    fig.suptitle("Shot locations" + (f" · {xg_label or xg_source}" if xg_source else ""), x=0.04, ha="left", fontsize=18,
                 color=INK, fontweight="bold")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.035, 0.075),
               ncol=2, frameon=False, labelcolor=INK)
    fig.text(
        0.04, 0.045,
        f"{scope} · Shootout excluded · " + (
            "Marker area follows xG (minimum visible size); missing xG: fixed size" if xg_source
            else "Equal marker sizes"
        ),
        color=INK, fontsize=9,
    )
    fig.text(
        0.04, 0.012,
        "Data: StatsBomb Open Data · Coordinates: StatsBomb units (120 × 80) · Both teams attack right",
        color=INK, fontsize=9,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.23, wspace=0.12)
    return fig


def _pitches(match: dict, title: str, scope: str) -> tuple[Figure, Any, Pitch]:
    figure = Figure(figsize=(13, 6.2), facecolor=BACKGROUND)
    axes = figure.subplots(1, 2)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BACKGROUND,
                  line_color=PITCH_LINES, linewidth=1, line_zorder=1)
    for side, ax in zip(("home", "away"), axes, strict=True):
        pitch.draw(ax=ax)
        ax.set_title(team_name(match, side), loc="left", fontsize=14, color=INK, pad=12)
    figure.suptitle(title, x=.04, ha="left", fontsize=18, color=INK)
    figure.text(.04, .075, scope + " · Penalty shootout excluded", fontsize=9, color=INK)
    figure.text(.04, .035, "Data: StatsBomb Open Data · StatsBomb units (120 × 80) · Both teams attack right",
                fontsize=9, color=INK)
    figure.subplots_adjust(left=.04, right=.97, bottom=.22, top=.83, wspace=.12)
    return figure, axes, pitch


def _team_events(events: pd.DataFrame, match: dict, side: str) -> pd.DataFrame:
    team_id = match[f"{side}_team"][f"{side}_team_id"]
    return events.loc[events["team_id"].eq(team_id) & events["period"].ne(5)]


def plot_pass_map(events: pd.DataFrame, match: dict, *, scope: str = "Full match") -> Figure:
    figure, axes, pitch = _pitches(match, "Pass locations and outcomes", scope)
    passes = events.loc[events["event_type"].eq("Pass")]
    for side, ax in zip(("home", "away"), axes, strict=True):
        team = _team_events(passes, match, side)
        located = team.dropna(subset=["x", "y", "end_x", "end_y"])
        for complete, color in ((True, "#237b83"), (False, "#b95442")):
            rows = located.loc[located["is_complete"].eq(complete)]
            if not rows.empty:
                pitch.arrows(rows.x, rows.y, rows.end_x, rows.end_y,
                             ax=ax, color=color, alpha=.55, width=1.3,
                             headwidth=3.5, headlength=4, zorder=2)
        ax.set_title(f"{team_name(match, side)} / {len(team)} passes", loc="left", color=INK)
        if located.empty:
            ax.text(60, 40, "No located passes", ha="center", color=INK)
        if len(located) != len(team):
            ax.text(.02, -.08, f"{len(team)-len(located)} passes without valid locations",
                    transform=ax.transAxes, fontsize=9, color=INK)
    figure.legend(handles=[Line2D([], [], color="#237b83", label="Complete"),
                           Line2D([], [], color="#b95442", label="Incomplete / other outcome")],
                  loc="lower left", bbox_to_anchor=(.03, .105), frameon=False, ncol=2)
    return figure


def plot_defensive_map(events: pd.DataFrame, match: dict, *, scope: str = "Full match") -> Figure:
    figure, axes, pitch = _pitches(match, "Defensive action locations", scope)
    styles = {
        "Pressure": ("o", "#527b5c"), "Interception": ("D", "#277b9c"),
        "Tackle": ("^", "#b45d28"), "Clearance": ("s", "#805c91"),
    }
    for side, ax in zip(("home", "away"), axes, strict=True):
        team = _team_events(events, match, side)
        count = 0
        for label, (marker, color) in styles.items():
            rows = team.loc[team["defensive_action"].eq(label)].dropna(subset=["x", "y"])
            count += len(rows)
            if not rows.empty:
                pitch.scatter(rows.x, rows.y, marker=marker, s=45, color=color, alpha=.7, ax=ax, zorder=3)
        if count == 0:
            ax.text(60, 40, "No located defensive actions", ha="center", color=INK)
    figure.legend(handles=[Line2D([], [], marker=m, color=c, linestyle="none", label=k)
                           for k, (m, c) in styles.items()],
                  loc="lower left", bbox_to_anchor=(.03, .105), ncol=4, frameon=False)
    return figure


def plot_pass_heatmap(events: pd.DataFrame, match: dict, *, scope: str = "Full match") -> Figure:
    figure, axes, pitch = _pitches(match, "Mapa de calor das origens dos passes", scope)
    passes = events.loc[events["event_type"].eq("Pass")]
    grids = []
    for side in ("home", "away"):
        rows = _team_events(passes, match, side).dropna(subset=["x", "y"])
        grids.append(pitch.bin_statistic(rows.x, rows.y, statistic="count", bins=(6, 5)))
    maximum = max(1., *(float(np.max(grid["statistic"])) for grid in grids))
    mesh = None
    for ax, grid in zip(axes, grids, strict=True):
        mesh = pitch.heatmap(grid, ax=ax, cmap="YlGnBu", vmin=0, vmax=maximum,
                             edgecolors=BACKGROUND, linewidth=.6, zorder=0)
    figure.colorbar(mesh, ax=list(axes), location="right", fraction=.025, pad=.02,
                    label="Pass origins per cell (shared count scale)")
    figure.text(.04, .12, "Counts of pass origins; this is not player residence time.", fontsize=9, color=INK)
    return figure


def _elapsed_clock(events: pd.DataFrame) -> tuple[pd.Series, dict[int, float]]:
    """Monotonic playing time, retaining each period's recorded added time."""
    clocks = pd.Series(index=events.index, dtype=float)
    boundaries = {}
    offset = 0.
    for period, base, duration in ((1, 0, 45), (2, 45, 45), (3, 90, 15), (4, 105, 15)):
        rows = events.loc[events.period.eq(period)]
        if not rows.empty:
            boundaries[period] = offset
            within = (rows.minute.astype(float) + rows.second.astype(float)/60 - base).clip(lower=0)
            clocks.loc[rows.index] = offset + within
            offset += max(float(duration), float(within.max()))
        else:
            offset += duration
    return clocks, boundaries


def _clock_label(row: pd.Series) -> str:
    period = int(row.period)
    end = {1:45, 2:90, 3:105, 4:120}[period]
    minute, second = int(row.minute), int(row.second)
    clock = f"{end}+{minute-end}:{second:02d}" if minute >= end else f"{minute}:{second:02d}"
    return f"P{period} {clock}"


def plot_cumulative_xg(
    events: pd.DataFrame, match: dict, *, xg_source: str = "xg_statsbomb",
    xg_label: str | None = None, scope: str = "Full match",
    clock_events: pd.DataFrame | None = None,
) -> Figure:
    figure = Figure(figsize=(11, 5.7), facecolor=BACKGROUND)
    ax = figure.subplots()
    ax.set_facecolor(BACKGROUND)
    ordered = events.loc[events.period.isin([1, 2, 3, 4])].sort_values(["period", "index"], kind="stable").copy()
    reference = ordered if clock_events is None else clock_events.loc[clock_events.period.isin([1, 2, 3, 4])]
    reference_clock, boundaries = _elapsed_clock(reference)
    if clock_events is None:
        ordered["plot_minute"] = reference_clock
    else:
        bases = ordered.period.map({1:0,2:45,3:90,4:105}).astype(float)
        ordered["plot_minute"] = ordered.period.map(boundaries).astype(float) + (
            ordered.minute.astype(float) + ordered.second.astype(float)/60 - bases
        ).clip(lower=0)
    end = float(reference_clock.max()) if not reference.empty else 90.
    for side, color in (("home", "#237b83"), ("away", "#bd831e")):
        rows = _team_events(ordered, match, side)
        rows = rows.loc[rows.event_type.eq("Shot")]
        if xg_source == "npxg_model":
            rows = rows.loc[~rows.is_penalty]
        if xg_source not in rows or rows[xg_source].isna().any():
            ax.text(.02, .9 if side == "home" else .82,
                    f"{team_name(match, side)}: xG unavailable for one or more shots",
                    transform=ax.transAxes, color=color)
            continue
        cumulative = rows[xg_source].astype(float).cumsum()
        total = float(cumulative.iloc[-1]) if len(cumulative) else 0.
        ax.step([0., *rows.plot_minute.tolist(), end], [0., *cumulative.tolist(), total],
                where="post", color=color, lw=2.2, label=f"{team_name(match, side)} · {total:.2f}")
        goals = rows.loc[rows.is_goal]
        if not goals.empty:
            ax.scatter(goals.plot_minute, cumulative.loc[goals.index], s=50, marker="*", color=color, zorder=4)
            for idx, goal in goals.iterrows():
                ax.annotate(_clock_label(goal), (goal.plot_minute, cumulative.loc[idx]),
                            xytext=(5, 9), textcoords="offset points", fontsize=8, color=color)
    for period, minute in boundaries.items():
        ax.axvline(minute, color=PITCH_LINES, ls=":", lw=.8)
        ax.text(minute, 1.02, {1:"1H",2:"2H",3:"ET1",4:"ET2"}[period],
                transform=ax.get_xaxis_transform(), fontsize=9, color=INK)
    ax.set(xlabel="Playing time (minutes; recorded added time retained, half-time breaks omitted)",
           ylabel=xg_label or xg_source, xlim=(0, max(end, 1)))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="upper left")
    figure.suptitle(f"Cumulative {xg_label or xg_source}", x=.09, ha="left", color=INK, fontsize=18)
    figure.text(.09, .055, scope + " · Shootout excluded · Stars: goals", fontsize=9, color=INK)
    figure.text(.09, .015, "Data: StatsBomb Open Data · Each curve uses one xG source", fontsize=9, color=INK)
    figure.subplots_adjust(left=.09, right=.97, bottom=.2, top=.83)
    return figure


def plot_xg_comparison(
    events: pd.DataFrame, *, scope: str = "Full match", model_label: str = "Own xG",
) -> Figure:
    figure = Figure(figsize=(8, 5.8), facecolor=BACKGROUND)
    ax = figure.subplots()
    rows = events.loc[events.event_type.eq("Shot") & events.period.ne(5)].dropna(subset=["xg_statsbomb", "xg_model"])
    for penalty, color, label in ((False, "#237b83", "Non-penalty shots"), (True, "#bd831e", "In-game penalties")):
        selected = rows.loc[rows.is_penalty.eq(penalty)]
        ax.scatter(selected.xg_statsbomb, selected.xg_model, color=color, alpha=.7, s=35, label=label)
    ax.plot([0, 1], [0, 1], ls=":", color=PITCH_LINES)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="xG StatsBomb", ylabel=model_label)
    ax.set_facecolor(BACKGROUND)
    ax.legend(frameon=False)
    figure.suptitle("Two estimates for the same shots", x=.12, ha="left", fontsize=17, color=INK)
    figure.text(.12, .07, scope + f" · {len(rows)} paired shots · Shootout excluded", fontsize=9, color=INK)
    figure.text(.12, .025, f"Data: StatsBomb Open Data · {model_label} · Agreement does not establish accuracy", fontsize=9, color=INK)
    figure.subplots_adjust(left=.12, right=.95, top=.88, bottom=.2)
    return figure
