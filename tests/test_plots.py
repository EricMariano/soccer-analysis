"""The map must preserve the normalized coordinates and eligibility rules."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from match_analyst.plots import plot_shot_map, plot_cumulative_xg, plot_pass_heatmap, _elapsed_clock


MATCH = {
    "home_team": {"home_team_id": 1, "home_team_name": "Home"},
    "away_team": {"away_team_id": 2, "away_team_name": "Away"},
}


def test_map_keeps_away_and_second_half_coordinates_and_excludes_shootout():
    shots = pd.DataFrame([
        {"team_id": 1, "period": 1, "x": 100.0, "y": 30.0, "is_goal": False},
        {"team_id": 2, "period": 2, "x": 110.0, "y": 50.0, "is_goal": True},
        {"team_id": 2, "period": 5, "x": 108.0, "y": 40.0, "is_goal": True},
    ])
    figure = plot_shot_map(shots, MATCH)
    try:
        home_other, home_goal = figure.axes[0].collections[-2:]
        away_other, away_goal = figure.axes[1].collections[-2:]
        assert home_other.get_offsets().tolist() == [[100.0, 30.0]]
        assert home_goal.get_offsets().size == 0
        assert away_other.get_offsets().size == 0
        assert away_goal.get_offsets().tolist() == [[110.0, 50.0]]
        assert home_other.get_sizes().tolist() == away_goal.get_sizes().tolist()
        assert figure.axes[1].get_ylim()[0] > figure.axes[1].get_ylim()[1]
    finally:
        plt.close(figure)


def test_map_can_draw_a_match_with_no_shots():
    shots = pd.DataFrame(columns=["team_id", "period", "x", "y", "is_goal"])
    figure = plot_shot_map(shots, MATCH)
    try:
        assert all(any(text.get_text() == "No eligible shots" for text in ax.texts)
                   for ax in figure.axes)
    finally:
        plt.close(figure)


def test_added_time_is_retained_before_next_period_and_cumulative_uses_one_source():
    events = pd.DataFrame([
        {"team_id":1,"period":1,"index":1,"minute":47,"second":0,"event_type":"Shot", "is_goal":False,"is_penalty":False,"xg_statsbomb":.2,"xg_model":.9},
        {"team_id":1,"period":2,"index":2,"minute":46,"second":0,"event_type":"Shot", "is_goal":True,"is_penalty":False,"xg_statsbomb":.3,"xg_model":.9},
        {"team_id":2,"period":2,"index":3,"minute":94,"second":0,"event_type":"Shot", "is_goal":False,"is_penalty":False,"xg_statsbomb":.1,"xg_model":.9},
    ])
    clocks, boundaries = _elapsed_clock(events)
    assert clocks.tolist() == [47.,48.,96.]
    assert boundaries == {1:0.,2:47.}
    figure = plot_cumulative_xg(events,MATCH,xg_source="xg_statsbomb")
    assert list(figure.axes[0].lines[0].get_ydata()) == [0.,.2,.5,.5]
    assert any("P2 46:00" in item.get_text() for item in figure.axes[0].texts)


def test_pass_heatmaps_share_a_volume_scale():
    events = pd.DataFrame([
        {"team_id":team,"period":1,"event_type":"Pass","x":100.,"y":30.}
        for team in [1,1,1,2]
    ])
    figure = plot_pass_heatmap(events,MATCH)
    assert figure.axes[0].collections[-1].get_clim() == (0.,3.)
    assert figure.axes[1].collections[-1].get_clim() == (0.,3.)


def test_player_filter_keeps_full_match_period_offsets():
    full = pd.DataFrame([
        {"team_id":1,"period":1,"index":1,"minute":40,"second":0,"event_type":"Shot","is_goal":False,"is_penalty":False,"xg_statsbomb":.2},
        {"team_id":2,"period":1,"index":2,"minute":50,"second":0,"event_type":"Half End","is_goal":False,"is_penalty":False,"xg_statsbomb":None},
        {"team_id":1,"period":2,"index":3,"minute":46,"second":0,"event_type":"Shot","is_goal":False,"is_penalty":False,"xg_statsbomb":.3},
    ])
    scoped = full.loc[full.event_type.eq("Shot")].reset_index(drop=True)
    figure = plot_cumulative_xg(scoped,MATCH,clock_events=full)
    assert figure.axes[0].lines[0].get_xdata().tolist() == [0.,40.,51.,51.]
