"""Local football analysis; training remains an explicit command outside the UI."""
from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from match_analyst.data import DataError, StatsBombClient
from match_analyst.exports import bundle_export
from match_analyst.metrics import build_facts, filter_events
from match_analyst.narrative import generate_report
from match_analyst.normalize import normalize_events
from match_analyst.plots import (
    plot_cumulative_xg, plot_defensive_map, plot_pass_heatmap, plot_pass_map,
    plot_shot_map, plot_xg_comparison, team_name,
)
from match_analyst.xg import load_model, model_membership, predict_xg

ROOT = Path(__file__).resolve().parent
PERIODS = {1: "First half", 2: "Second half", 3: "Extra time · first half", 4: "Extra time · second half"}


def match_label(match: dict) -> str:
    return f"{match['match_date']} · {team_name(match, 'home')} vs {team_name(match, 'away')}"


@st.cache_resource(show_spinner=False)
def _saved_model(path: str, metadata_stamp: int):
    return load_model(path)


def _number(value, decimals=2):
    return "Not available" if value is None or pd.isna(value) else f"{value:.{decimals}f}"


def _show_figure(figure):
    st.pyplot(figure, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Match Analyst", page_icon="⚽", layout="wide")
    st.markdown("""<style>
        .stApp {background:#f7f5ef;color:#153345;}
        [data-testid="stSidebar"] {background:#eaece6;}
        h1,h2,h3 {color:#153345;font-family:Georgia,serif;}
        .block-container {max-width:1500px;padding-top:2.6rem;}
        .eyebrow {color:#526956;letter-spacing:.18em;font-size:.75rem;font-weight:700;}
        .scoreline {font-family:Georgia,serif;color:#153345;font-size:clamp(1.8rem,4vw,3rem);margin:.6rem 0;}
        .score {white-space:nowrap;padding:0 .7rem;color:#8b5c12;}
        [data-testid="stMetricValue"] {font-family:Georgia,serif;}
        </style>""", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">MATCH ANALYST / LOCAL FOOTBALL LAB</div>', unsafe_allow_html=True)
    st.title("Every action tells a little more.")
    st.caption("Explore the match. Check the evidence. Keep the analysis local.")

    with st.sidebar:
        st.header("Match selection")
        offline = st.checkbox("Offline mode", key="offline", value=os.getenv("MATCH_ANALYST_OFFLINE", "").lower() in {"1", "true", "yes"},
                              help="Use only files already downloaded. No network requests.")
    options = {"offline": offline}
    if os.getenv("MATCH_ANALYST_CACHE_DIR"):
        options["cache_dir"] = Path(os.environ["MATCH_ANALYST_CACHE_DIR"])
    try:
        client = StatsBombClient(**options)
        catalogue = client.competitions()
        if not catalogue:
            raise DataError("The competition catalogue is empty. Check the source or use a valid local cache.")
        competitions = {int(row["competition_id"]): row for row in catalogue}
        ids = sorted(competitions, key=lambda cid: competitions[cid]["competition_name"])
        preferred = next((cid for cid in ids if competitions[cid].get("competition_name") == "FIFA World Cup"
                          and competitions[cid].get("competition_gender") == "male"), ids[0])
        with st.sidebar:
            competition_id = st.selectbox("Competition", ids, index=ids.index(preferred), key="competition",
                format_func=lambda cid: f"{competitions[cid]['competition_name']} · {competitions[cid].get('competition_gender', '')}")
            seasons = {int(row["season_id"]): row["season_name"] for row in catalogue if row["competition_id"] == competition_id}
            season_ids = sorted(seasons, key=lambda sid: seasons[sid], reverse=True)
            if not season_ids:
                raise DataError("No seasons are available for this competition.")
            default_season = next((sid for sid in season_ids if seasons[sid] == "2022"), season_ids[0])
            season_id = st.selectbox("Season", season_ids, index=season_ids.index(default_season), key="season", format_func=seasons.get)
        matches = sorted(client.matches(competition_id, season_id), key=lambda m: (m["match_date"], m["match_id"]))
        if not matches:
            raise DataError("No matches are available in this competition and season.")
        default_match = next((i for i,m in enumerate(matches) if {team_name(m,"home"),team_name(m,"away")} == {"Brazil","Serbia"}), 0)
        with st.sidebar:
            selected_id = st.selectbox("Match", [m["match_id"] for m in matches], index=default_match, key="match",
                                      format_func={m["match_id"]:match_label(m) for m in matches}.get)
            st.caption("Events are downloaded only when a match is selected; cached files are reused.")
        selected = next(m for m in matches if m["match_id"] == selected_id)
        with st.spinner("Loading and validating events…"):
            loaded = client.load_match(selected)
            normalized = normalize_events(loaded.events, selected_id, competition_id, season_id)
    except (DataError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.info("Turn off Offline mode while connected and open the match once; then use the local cache."
                if offline else "Check your connection and the local cache, then reload to try again.")
        st.stop()

    events = normalized.events
    home_id = selected["home_team"]["home_team_id"]
    away_id = selected["away_team"]["away_team_id"]
    teams = {home_id:team_name(selected,"home"), away_id:team_name(selected,"away")}
    with st.sidebar:
        st.divider()
        st.header("Analysis scope")
        team_id = st.selectbox("Team", [None, home_id, away_id], key="team", format_func=lambda v: teams.get(v,"Both teams"))
        period = st.selectbox("Period", [None,*sorted(events.period.dropna().unique().astype(int).tolist())],
                              key="period", format_func=lambda v: PERIODS.get(v,"Full match"))
        player_rows = events if team_id is None else events.loc[events.team_id.eq(team_id)]
        players = player_rows.dropna(subset=["player_id", "player_name"])[["player_id","player_name"]].drop_duplicates("player_id")
        player_names = {int(row.player_id):str(row.player_name) for row in players.itertuples()}
        player_id = st.selectbox("Player (all views)", [None,*sorted(player_names,key=player_names.get)], key="player",
                                format_func=lambda v: player_names.get(v,"All players"))
    filters = {"team_id":team_id,"period":None if period is None else int(period),"player_id":player_id}
    scope = " · ".join([teams.get(team_id,"Both teams"), PERIODS.get(period,"Full match"), player_names.get(player_id,"All players")])

    model_dir = Path(os.getenv("MATCH_ANALYST_MODEL_DIR", str(ROOT / "models" / "xg_v1")))
    model = None
    metadata_path = model_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            model = _saved_model(str(model_dir), metadata_path.stat().st_mtime_ns)
            events = predict_xg(events, model)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            model = None
            st.warning(f"The saved own model could not be loaded: {exc}")
    source_labels = {"xg_statsbomb":"xG StatsBomb"}
    if model is not None:
        source_labels["xg_model"] = f"Own xG · {model.metadata['model_version']}"
    with st.sidebar:
        source = st.selectbox("xG source", list(source_labels), key="xg_source", format_func=source_labels.get)
        if model is None:
            st.caption("Own model unavailable. StatsBomb xG is available when supplied by the source.")
    xg_version = loaded.provenance["source_revision"] if source == "xg_statsbomb" else model.metadata["model_version"]
    facts = build_facts(events, selected, filters=filters, xg_source=source,
                       xg_version=xg_version, validation=normalized.report.as_dict())
    facts["provenance"] = loaded.provenance
    facts["scope_label"] = scope
    facts["model"] = None if model is None else {
        "version":model.metadata["model_version"],"dataset_id":model.metadata["dataset_id"],
        "match_membership":model_membership(selected_id,model.metadata),
        "penalty_component":model.metadata["penalty_component"],
    }
    if model is not None and source == "xg_model":
        facts["limitations"].append("Modelo próprio ajustado em seleções masculinas adultas da Copa de 2018; a generalização para outros recortes não foi estabelecida.")
    report = generate_report(facts)
    scoped = filter_events(events, **filters)
    shots = scoped.loc[scoped.event_type.eq("Shot")]

    st.markdown(f'<div class="scoreline">{escape(teams[home_id])}<span class="score">{selected["home_score"]} – {selected["away_score"]}</span>{escape(teams[away_id])}</div>', unsafe_allow_html=True)
    competition_name = competitions[competition_id]["competition_name"]
    st.caption(f"Final score · {selected['match_date']} · {competition_name} {seasons[season_id]} · Match ID {selected_id}")
    st.caption(f"Analysis scope: {scope} · {source_labels[source]} · Shootout excluded")
    if model is not None:
        membership = model_membership(selected_id, model.metadata)
        label = {"train":"Training match — in-sample predictions", "validation":"Validation match — used for model selection",
                 "test":"Held-out test match — excluded from training and selection", "unseen":"Outside the saved dataset"}.get(membership,membership)
        st.info(f"Own model: {label}. Training and evaluation metadata are available below.")
    if normalized.report.invalid_count or normalized.report.duplicate_count:
        st.warning(f"Validation excluded {normalized.report.invalid_count} invalid and {normalized.report.duplicate_count} duplicate events. See Data checks.")
    if normalized.report.shootout_count:
        st.info(f"{normalized.report.shootout_count} penalty shootout events are excluded from normal statistics, maps and xG.")
    if shots.empty:
        st.info("No eligible shots in this scope. Shot counts are zero; ratios without a denominator are not available.")
    st.subheader("The numbers in this scope")
    for column, team in zip(st.columns(max(1,len(facts["teams"]))), facts["teams"]):
        with column:
            st.markdown(f"**{team['team_name']}**")
            a,b,c = st.columns(3)
            a.metric("Goals in scope",team["goals"])
            b.metric("Shots",team["shots"])
            c.metric(source_labels[source],_number(team["xg"]))
            accuracy = "Not available" if team['pass_accuracy'] is None else f"{100*team['pass_accuracy']:.1f}%"
            st.caption(f"npxG: {_number(team['npxg'])} · xG / shot: {_number(team['xg_per_shot'])} · Pass completion: {accuracy}")
    statistics = pd.DataFrame(facts["teams"])
    display_columns = ["team_name","shots","shots_inside_box","shots_outside_box","goals","passes_attempted","passes_completed","pass_accuracy","final_third_entries","entries_left","entries_center","entries_right","pressures","interceptions","tackles","clearances","xg","npxg","xg_per_shot"]
    with st.expander("All statistics and definitions"):
        st.dataframe(statistics[[c for c in display_columns if c in statistics]], hide_index=True, width="stretch")
        st.markdown("Completed passes have no pass outcome in the StatsBomb JSON. Final-third entries are complete passes starting at x < 80 and ending at x ≥ 80; lanes use the destination y coordinate. Defensive types are counted separately. xG is unavailable if any eligible shot has no value. npxG excludes in-game penalties. Counts of events do not measure possession or time spent in a region.")

    figures = {
        "shots":plot_shot_map(shots,selected,xg_source=source,xg_label=source_labels[source],scope=scope),
        "passes":plot_pass_map(scoped,selected,scope=scope),
        "defensive":plot_defensive_map(scoped,selected,scope=scope),
        "pass_origins":plot_pass_heatmap(scoped,selected,scope=scope),
        "cumulative_xg":plot_cumulative_xg(scoped,selected,xg_source=source,xg_label=source_labels[source],scope=scope,clock_events=events),
    }
    tab_names = ["Shots", "Passes", "Defensive actions", "Pass origins", "Cumulative xG"]
    if model is not None:
        figures["xg_comparison"] = plot_xg_comparison(scoped,scope=scope,model_label=source_labels["xg_model"])
        tab_names.append("Compare xG")
    st.subheader("The match, on the pitch")
    for tab, (name, figure) in zip(st.tabs(tab_names), figures.items()):
        with tab:
            _show_figure(figure)
            if name == "shots":
                st.caption(f"Each row is one shot in the selected scope. Displayed xG source: {source_labels[source]}.")
                other_source = "xg_model" if source == "xg_statsbomb" else "xg_statsbomb"
                shot_columns = [source,other_source,"player_name","team_name","period","minute","second","outcome","is_penalty","x","y","event_id"]
                st.dataframe(shots[[column for column in shot_columns if column in shots]], hide_index=True, width="stretch",
                    column_config={"xg_statsbomb":st.column_config.NumberColumn("xG StatsBomb",format="%.3f"),
                                   "xg_model":st.column_config.NumberColumn("Own xG",format="%.3f")})
    st.caption("Coordinates retain each acting team's StatsBomb attacking frame. Both teams attack right. Goal markers do not include own-goal events. Shared heatmap scales compare event counts.")
    st.subheader("A summary grounded in the data")
    st.markdown(report["text"])
    with st.expander("Evidence and limitations"):
        st.json(report["sentences"])
        for limitation in facts["limitations"]:
            st.write("• " + limitation)
        st.json(facts["evidence"])
    st.subheader("Take this analysis with you")
    name = f"match-{selected_id}-{source}-team-{team_id or 'all'}-period-{period or 'all'}-player-{player_id or 'all'}"
    archive = bundle_export(facts,report,figures,stats=statistics,shots=shots)
    st.download_button("Download complete analysis (.zip)", archive, name+".zip", "application/zip", key="export_zip")
    a,b,c = st.columns(3)
    a.download_button("Facts · JSON",json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False),name+".json","application/json")
    b.download_button("Statistics · CSV",statistics.to_csv(index=False),name+".csv","text/csv")
    c.download_button("Summary · Markdown",report["text"],name+".md","text/markdown")
    st.download_button("Shot details · CSV",shots.to_csv(index=False),name+"-shots.csv","text/csv")
    st.caption("The ZIP also contains PNG charts, sentence evidence, metric versions and a separate human-review record.")
    with st.expander("Data checks and source"):
        st.write("Scope: full match metadata; analysis uses the filters shown above. Penalty shootout excluded.")
        st.json(normalized.report.as_dict())
        st.write("Raw event file",str(loaded.event_path))
        st.json(loaded.provenance)
        if model is not None:
            st.write("Own-model metadata (training runs only through the explicit training script)")
            st.json(model.metadata)
        logo = ROOT / "assets" / "statsbomb-logo.png"
        if logo.is_file():
            st.image(str(logo),width=180)
        st.markdown("Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). See the source's [terms and attribution requirements](https://github.com/statsbomb/open-data#terms--conditions) and [licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).")


if __name__ == "__main__":
    main()
