"""2010 Chargers retrospective dashboard.

Reads only outputs/analysis.json and data/processed — no recomputation of the
statistics, except the What If tab's standings toggle, which reruns the same
division tiebreak function the analysis uses.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from models import division_winner

NAVY = "#07152B"
PANEL = "#0D2142"
POWDER = "#0080C6"
SKY = "#5BC2E7"
GOLD = "#FFC20E"
GREY = "#A5ACAF"
DIM = "#4A5A78"
RED = "#D6432D"
WHITE = "#F5F6F7"
GRID = "#1A2E52"
AFC_WEST = {"LAC", "KC", "LV", "DEN"}
TEAM_NAMES = {"LAC": "Chargers", "KC": "Chiefs", "LV": "Raiders", "DEN": "Broncos"}
PLAYOFF_ORDER = [
    "Won Super Bowl",
    "Lost Super Bowl",
    "Lost Conference",
    "Lost Divisional",
    "Lost Wild Card",
    "Missed playoffs",
]
PLAYOFF_COLORS = {
    "Won Super Bowl": GOLD,
    "Lost Super Bowl": "#E8A33D",
    "Lost Conference": SKY,
    "Lost Divisional": POWDER,
    "Lost Wild Card": "#3D6A9E",
    "Missed playoffs": DIM,
}

st.set_page_config(
    page_title="The Tragedy of the 2010 San Diego Chargers",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.2rem; }}
.hero-band {{
    background: radial-gradient(ellipse 800px 260px at 15% 0%, rgba(0,128,198,.28), transparent 65%), {PANEL};
    padding: 1.8rem 1.8rem 1.5rem; border-bottom: 4px solid {GOLD}; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 20px;
}}
.hero-band svg {{ width: 56px; height: 56px; flex-shrink: 0; }}
.hero-band h1 {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 2.4rem;
    color: {WHITE}; margin: 0 0 6px; line-height: 1.05; }}
.hero-band p {{ color: #C3C9CC; margin: 0; font-size: 1rem; }}
[data-testid="stMetric"] {{ background: {PANEL}; border-bottom: 4px solid {POWDER}; padding: 14px 14px 10px; }}
[data-testid="stMetricValue"] {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ color: #C3C9CC; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDelta"] {{ background: none !important; padding-left: 0 !important; color: {GREY};
    font-size: 0.82rem; }}
hr {{ height: 1px !important; background-color: #22385E !important; border: none !important;
    opacity: 1 !important; margin: 1.6rem 0 0.4rem !important; }}
h1, h2, h3 {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }}
h2 {{ font-size: 1.8rem !important; line-height: 1.15; margin-top: 2rem !important; }}
h3 {{ font-size: 1.25rem !important; font-weight: 600 !important; color: #E6E9EA; letter-spacing: 0.02em;
    margin-top: 1.9rem !important; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ max-width: 44rem;
    line-height: 1.62; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: #C3C9CC; }}
.callout {{ background: {PANEL}; border-left: 4px solid {GOLD}; padding: 0.9rem 1.1rem; margin: 0.8rem 0 1rem;
    max-width: 44rem; }}
@media (max-width: 640px) {{
    .hero-band {{ flex-direction: column; align-items: flex-start; gap: 12px; padding: 1.2rem 1.1rem 1rem; }}
    .hero-band svg {{ width: 40px; height: 40px; }}
    .hero-band h1 {{ font-size: 1.55rem; }}
    h2 {{ font-size: 1.45rem !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1.9rem !important; }}
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_analysis() -> dict:
    with open(OUTPUTS / "analysis.json") as f:
        return json.load(f)


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name)


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=NAVY,
        plot_bgcolor=PANEL,
        font={"family": "Inter, sans-serif", "color": WHITE, "size": 13},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
        },
        hoverlabel={"bgcolor": PANEL, "font_color": WHITE},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#2C4470", linecolor=GREY)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#2C4470", linecolor=GREY)
    return fig


def label(row) -> str:
    return f"{int(row['season'])} {row['team']}"


A = load_analysis()
seasons = load("team_season_modeled.csv")
games = load("team_game.csv")
plays = load("sd2010_plays.csv")
careers = load("qb_career.csv")
qb_seasons = load("qb_season.csv")
seasons["label"] = seasons.apply(label, axis=1)
sd_mask = (seasons["season"] == 2010) & (seasons["team"] == "LAC")
sd_games = (
    games[(games["season"] == 2010) & (games["team"] == "LAC")]
    .sort_values("week")
    .copy()
)
OFF = A["official_2010"]
SIM = A["simulation"]

st.markdown(
    """
<div class="hero-band">
<svg viewBox="0 0 64 64" aria-hidden="true"><polygon points="36,4 14,36 30,36 24,60 50,24 34,24 42,4"
 fill="#FFC20E"/></svg>
<div><h1>The Tragedy of the 2010 San Diego Chargers</h1>
<p>The 2010 San Diego Chargers had the NFL’s No. 1 offense and No. 1 defense, and went 9–7.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "The Story",
        "Best Offense, Best Defense",
        "The Anomaly",
        "Special Teams",
        "What If",
        "Philip Rivers",
        "Methodology",
    ]
)

# ---------------------------------------------------------------- The Story
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total offense",
        f"{OFF['yards_per_game']} yds/g",
        "No. 1 in the NFL",
        delta_color="off",
    )
    c2.metric(
        "Total defense",
        f"{OFF['yards_allowed_per_game']} yds/g",
        "No. 1 in the NFL",
        delta_color="off",
    )
    c3.metric(
        "Yardage differential",
        f"+{OFF['yard_differential']:,}",
        "~1.8× the next-best team",
        delta_color="off",
    )
    c4.metric("Record", "9–7", "Missed the playoffs", delta_color="off")

    st.markdown(
        f"""
Only five teams in the Super Bowl era finished No. 1 in both total offense and total defense. Four of
them had the best (or tied-best) record in football. The fifth is the 2010 Chargers, who started **2–5**
while outgaining six of their first seven opponents by 125+ yards, then won 7 of their last 9 and
missed the AFC West by one game.

Their special teams gave up **{A["special_teams"]["points_allowed_on_returns_and_blocks"]} points**
on returns and blocked punts, had **{A["special_teams"]["punts_blocked_against"]} punts blocked**, and allowed
**{OFF["punt_return_avg_allowed"]} yards per punt return**, the most since the NFL began tracking it in 1976.
"""
    )

    st.subheader("Winning the yardage battle, losing the games")
    sd_games["cum_diff"] = sd_games["yard_diff"].cumsum()
    sd_games["record"] = [
        f"{(sd_games['result'].iloc[: i + 1] == 'W').sum()}–{(sd_games['result'].iloc[: i + 1] == 'L').sum()}"
        for i in range(len(sd_games))
    ]
    fig = go.Figure()
    fig.add_bar(
        x=sd_games["week"],
        y=sd_games["yard_diff"],
        name="Game yardage margin",
        showlegend=False,
        marker_color=[POWDER if r == "W" else RED for r in sd_games["result"]],
        customdata=np.stack(
            [sd_games["opp"], sd_games["pts"], sd_games["opp_pts"], sd_games["result"]],
            axis=1,
        ),
        hovertemplate="Week %{x} vs %{customdata[0]}<br>%{customdata[3]} %{customdata[1]}–"
        "%{customdata[2]}<br>Yardage margin %{y:+}<extra></extra>",
    )
    fig.add_scatter(
        x=sd_games["week"],
        y=sd_games["cum_diff"],
        name="Season-to-date margin",
        yaxis="y2",
        mode="lines+markers+text",
        line={"color": GOLD, "width": 3},
        text=sd_games["record"],
        textposition="top center",
        textfont={"color": GOLD, "size": 11},
        hovertemplate="After week %{x}: %{y:+,} yds<extra></extra>",
    )
    fig.update_layout(
        yaxis={"title": "Yards, single game"},
        yaxis2={
            "title": "Yards, season to date",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        xaxis={"title": "Week", "dtick": 1},
        bargap=0.25,
    )
    for r in sd_games[sd_games["yard_diff"] == 0].itertuples():
        fig.add_annotation(
            x=r.week,
            y=0,
            text="Even",
            showarrow=False,
            yshift=10,
            font={"color": POWDER, "size": 11},
        )
    st.plotly_chart(style_fig(fig, 440), width="stretch")
    st.caption(
        "Bars: blue = win, red = loss. Gold labels are the record after each game. Yards from "
        "play-by-play (official season total: +1,984). Week 10 was the bye. Week 12 at Indianapolis "
        "was dead even: 303 scrimmage yards each."
    )

    st.subheader("Seven losses, five of them by one score")
    fig = go.Figure()
    fig.add_scatter(
        x=sd_games["yard_diff"],
        y=sd_games["margin"],
        mode="markers+text",
        text=sd_games["opp"],
        textposition="top center",
        marker={
            "size": 13,
            "color": [POWDER if r == "W" else RED for r in sd_games["result"]],
            "line": {"color": WHITE, "width": 1},
        },
        customdata=sd_games["week"],
        hovertemplate="Week %{customdata}: outgained by %{x:+} yds, margin %{y:+}<extra></extra>",
    )
    fig.add_hrect(y0=-8, y1=0, fillcolor=RED, opacity=0.12, line_width=0)
    fig.add_annotation(
        x=300, y=-4, text="One-score losses", showarrow=False, font={"color": RED}
    )
    fig.update_layout(
        xaxis_title="Yardage margin", yaxis_title="Point margin", showlegend=False
    )
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.caption(
        "The red band holds the five early losses: in four of them San Diego outgained the opponent by "
        "184 to 248 yards. Week 6 at St. Louis was the only one they lost on yards too."
    )

    rk = A["rankings"]
    st.markdown(
        f"""
<div class="callout"><b>Through five games</b> the Chargers had outgained opponents by
<b>{int(rk["top_first5"][0]["first5_yard_diff"]):+,} yards</b>, the best start of any team since 1999.
They were <b>2–3</b>.</div>
""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------- Best Offense, Best Defense
with tabs[1]:
    st.markdown(
        "Every season since 1999, the league’s top-3 total offenses and top-3 total defenses (yards "
        "per game), placed by where their *other* unit ranked. One team sits in the top-left corner."
    )
    sd_row = seasons[sd_mask].iloc[0]
    others = seasons[~sd_mask]
    top_o = others["off_rank"] <= 3
    top_d = others["def_rank"] <= 3
    both = others[top_o & top_d].assign(unit="Top 3 offense and defense")
    best_o = others[top_o & ~top_d].assign(unit="Top 3 offense")
    best_d = others[top_d & ~top_o].assign(unit="Top 3 defense")

    def jitter(season: pd.Series) -> pd.Series:
        # Spread same-rank seasons slightly off their rank line so they don't stack.
        return ((season - 1999) % 5 - 2) * 0.1

    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=0.5,
        x1=3.5,
        y0=0.5,
        y1=3.5,
        line={"color": GOLD, "width": 1, "dash": "dot"},
    )
    fig.add_annotation(
        x=3.5,
        y=3.5,
        text="Top 3 in both",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"color": GOLD, "size": 11},
    )
    shown = set()
    for df, sym in [(best_o, "circle"), (best_d, "diamond"), (both, "square")]:
        for result in PLAYOFF_ORDER:
            sub = df[df["playoff_result"] == result]
            if sub.empty:
                continue
            fig.add_scatter(
                x=sub["def_rank"] + (jitter(sub["season"]) if sym == "diamond" else 0),
                y=sub["off_rank"] + (jitter(sub["season"]) if sym == "circle" else 0),
                mode="markers",
                name=result,
                legendgroup=result,
                showlegend=result not in shown,
                visible=True if result == "Missed playoffs" else "legendonly",
                marker={
                    "symbol": sym,
                    "size": 13,
                    "color": PLAYOFF_COLORS[result],
                    "line": {"color": WHITE, "width": 1},
                },
                customdata=np.stack(
                    [
                        sub["season"],
                        sub["team"],
                        sub["unit"],
                        sub["off_rank"],
                        sub["def_rank"],
                        sub["wins"],
                        sub["losses"],
                        sub["playoff_result"],
                    ],
                    axis=1,
                ),
                hovertemplate="%{customdata[0]} %{customdata[1]} — %{customdata[2]}<br>"
                "Offense #%{customdata[3]}, defense #%{customdata[4]}<br>"
                "%{customdata[5]}–%{customdata[6]}, %{customdata[7]}<extra></extra>",
            )
            shown.add(result)
    fig.add_scatter(
        x=[1],
        y=[1],
        mode="markers",
        name="2010 Chargers (both)",
        marker={
            "symbol": "star",
            "size": 26,
            "color": GOLD,
            "line": {"color": WHITE, "width": 1},
        },
        hovertemplate="2010 LAC — No. 1 offense and No. 1 defense<br>9–7, missed playoffs<extra></extra>",
    )
    fig.add_annotation(
        x=1,
        y=1,
        text="2010 SD: No. 1 and No. 1, 9–7, no playoffs",
        showarrow=True,
        arrowcolor=GOLD,
        ax=150,
        ay=110,
        xanchor="left",
        font={"color": GOLD, "size": 13},
        bgcolor=PANEL,
    )
    rank_ticks = [1, 2, 3, 8, 16, 24, 32]
    fig.update_layout(
        xaxis={
            "title": "Defense rank (yards allowed)",
            "range": [0, 33],
            "tickvals": rank_ticks,
        },
        yaxis={
            "title": "Offense rank (yards gained)",
            "range": [33, 0],
            "tickvals": rank_ticks,
        },
        legend_title_text="",
    )
    style_fig(fig, 560).update_layout(
        legend={"y": -0.2, "yanchor": "top"}, margin={"b": 90}
    )
    st.plotly_chart(fig, width="stretch")
    both_list = ", ".join(
        f"{int(r.season)} {r.team} (#{int(r.off_rank)} offense, #{int(r.def_rank)} defense)"
        for r in both.sort_values("season").itertuples()
    )
    st.caption(
        "Circles across the top = top-3 offenses, placed by their defense rank. Diamonds down the left "
        "= top-3 defenses, placed by their offense rank. Squares = top 3 in both. Colored by playoff "
        "result; only teams that missed the playoffs show at first, so click the legend to add the "
        f"rest. Ties show every tied team. Besides 2010 San Diego, {len(both)} teams finished top 3 "
        f"in both: {both_list}. None was No. 1 in both."
    )

    st.subheader("Every team-season since 1999: yards gained vs. yards allowed")
    fig = go.Figure()
    for result in PLAYOFF_ORDER[::-1]:
        sub = seasons[(seasons["playoff_result"] == result) & ~sd_mask]
        fig.add_scatter(
            x=sub["yards_pg"],
            y=sub["yards_allowed_pg"],
            mode="markers",
            name=result,
            marker={"size": 7, "color": PLAYOFF_COLORS[result], "opacity": 0.75},
            text=sub["label"],
            hovertemplate="%{text}<br>%{x:.1f} gained, %{y:.1f} allowed<extra></extra>",
        )
    fig.add_scatter(
        x=[sd_row["yards_pg"]],
        y=[sd_row["yards_allowed_pg"]],
        mode="markers+text",
        name="2010 Chargers",
        text=["2010 SD"],
        textposition="middle left",
        textfont={"color": GOLD, "size": 14},
        marker={
            "size": 20,
            "color": GOLD,
            "symbol": "star",
            "line": {"color": WHITE, "width": 1},
        },
    )
    fig.update_layout(
        xaxis_title="Yards gained per game →",
        yaxis={"title": "← Yards allowed per game", "autorange": "reversed"},
    )
    st.plotly_chart(style_fig(fig, 520), width="stretch")
    st.caption(
        "Up and to the right is better on both sides of the ball. Missed-playoff teams are the darkest "
        "dots."
    )

    st.subheader("Playoff rate by offense and defense rank")
    s = seasons.assign(
        o=pd.cut(
            seasons["off_rank"],
            [0, 1, 5, 10, 16, 32],
            labels=["1", "2–5", "6–10", "11–16", "17–32"],
        ),
        d=pd.cut(
            seasons["def_rank"],
            [0, 1, 5, 10, 16, 32],
            labels=["1", "2–5", "6–10", "11–16", "17–32"],
        ),
    )
    grid = s.pivot_table(
        index="d", columns="o", values="made_playoffs", aggfunc="mean", observed=False
    )
    counts = s.pivot_table(
        index="d", columns="o", values="made_playoffs", aggfunc="size", observed=False
    )
    text = [
        [f"{v:.0%}<br>n={int(n)}" if n else "" for v, n in zip(rv, rn)]
        for rv, rn in zip(grid.values, counts.values)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[f"Off {c}" for c in grid.columns],
            y=[f"Def {i}" for i in grid.index],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, PANEL], [1, POWDER]],
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate="%{y}, %{x}: %{z:.0%} made playoffs<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    st.caption(
        "Ranks are by yards per game. The top-left cell is a sample of one: the 2010 Chargers, "
        "the only team since 1999 to rank No. 1 in both."
    )

# ------------------------------------------------------------- The Anomaly
with tabs[2]:
    ym = A["yardage_model"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Expected wins from yardage",
        f"{ym['sd_expected_wins']:.1f}",
        "actual: 9",
        delta_color="off",
    )
    c2.metric(
        "Expected wins from points",
        f"{A['pythagorean']['expected_wins']:.1f}",
        "Pythagorean, exponent 2.37",
        delta_color="off",
    )
    c3.metric(
        "Teams at +100 yds/game since 1999",
        ym["teams_with_yard_diff_pg_100_plus"],
        f"{len(ym['those_that_missed_playoffs'])} missed the playoffs",
        delta_color="off",
    )

    st.subheader("Yardage margin vs. wins, every team-season since 1999")
    fig = go.Figure()
    others = seasons[~sd_mask]
    fig.add_scatter(
        x=others["yard_diff_pg"],
        y=others["win_pct"],
        mode="markers",
        name="Team-season",
        marker={"size": 7, "color": POWDER, "opacity": 0.45},
        text=others["label"],
        hovertemplate="%{text}<br>%{x:+.1f} yds/g, %{y:.0%} win pct<extra></extra>",
    )
    xs = np.linspace(seasons["yard_diff_pg"].min(), seasons["yard_diff_pg"].max(), 50)
    fig.add_scatter(
        x=xs,
        y=ym["intercept"] + ym["slope"] * xs,
        mode="lines",
        name="Expected",
        line={"color": GREY, "dash": "dash"},
    )
    fig.add_scatter(
        x=[sd_row["yard_diff_pg"]],
        y=[sd_row["win_pct"]],
        mode="markers+text",
        name="2010 Chargers",
        text=["2010 SD"],
        textposition="bottom center",
        textfont={"color": GOLD, "size": 14},
        marker={
            "size": 20,
            "color": GOLD,
            "symbol": "star",
            "line": {"color": WHITE, "width": 1},
        },
    )
    for season, team, txt in [
        (2001, "LA", "2001 Rams"),
        (2007, "NE", "2007 Patriots"),
        (2019, "DAL", "2019 Cowboys"),
    ]:
        r = seasons[(seasons["season"] == season) & (seasons["team"] == team)].iloc[0]
        fig.add_annotation(
            x=r["yard_diff_pg"],
            y=r["win_pct"],
            text=txt,
            showarrow=True,
            arrowcolor=GREY,
            font={"color": WHITE, "size": 12},
            ax=-50,
            ay=-25,
        )
    fig.update_layout(
        xaxis_title="Yardage margin per game",
        yaxis={"title": "Win percentage", "tickformat": ".0%"},
        showlegend=True,
    )
    st.plotly_chart(style_fig(fig, 500), width="stretch")
    st.caption(
        f"Linear fit: r = {ym['r']:.2f}. San Diego’s margin projects to {ym['sd_expected_wins']:.1f} wins. "
        "The 2019 Cowboys are the only other +100 yds/game team to miss the playoffs."
    )

    st.subheader("The best yardage margins since 1999")
    top = pd.DataFrame(A["rankings"]["top_yard_diff"])
    top["Record"] = (
        top["wins"].astype(int).astype(str)
        + "–"
        + top["losses"].astype(int).astype(str)
    )
    top = top.rename(
        columns={
            "season": "Season",
            "team": "Team",
            "yard_diff": "Yardage margin",
            "playoff_result": "Postseason",
        }
    )[["Season", "Team", "Yardage margin", "Record", "Postseason"]]
    top["Yardage margin"] = top["Yardage margin"].map(lambda v: f"{int(v):+,}")
    st.dataframe(top, hide_index=True, width="stretch")
    st.caption(
        "Play-by-play yardage. The 16–0 2007 Patriots finished ten yards behind San Diego."
    )

    st.subheader("Hot starts: yardage margin after five games")
    f5 = seasons.dropna(subset=["first5_yard_diff"])
    fig = go.Figure()
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(f5))
    fig.add_scatter(
        x=f5["first5_wins"] + jitter,
        y=f5["first5_yard_diff"],
        mode="markers",
        name="Team-season",
        marker={"size": 6, "color": POWDER, "opacity": 0.4},
        text=f5["label"],
        hovertemplate="%{text}<br>%{y:+} yds through 5 games<extra></extra>",
    )
    fig.add_scatter(
        x=[2],
        y=[sd_row["first5_yard_diff"]],
        mode="markers+text",
        text=["2010 SD"],
        textposition="middle right",
        textfont={"color": GOLD, "size": 14},
        name="2010 Chargers",
        marker={
            "size": 20,
            "color": GOLD,
            "symbol": "star",
            "line": {"color": WHITE, "width": 1},
        },
    )
    fig.update_layout(
        xaxis={"title": "Wins in first five games", "dtick": 1},
        yaxis_title="Yardage margin, games 1–5",
    )
    st.plotly_chart(style_fig(fig, 440), width="stretch")
    six = pd.DataFrame(A["rankings"]["six_of_first_seven_outgained_125"])
    st.markdown(
        "**Outgained six of their first seven opponents by 125+ yards (since 1999):** "
        + ", ".join(
            f"{r.season} {r.team} (started {r.first7_wins}–{7 - r.first7_wins})"
            for r in six.itertuples()
        )
    )

# ------------------------------------------------------------ Special Teams
with tabs[3]:
    stA = A["special_teams"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Points allowed on returns & blocks",
        stA["points_allowed_on_returns_and_blocks"],
        "3 KR TD, 1 PR TD, blocked-punt TD, 2 safeties",
        delta_color="off",
    )
    c2.metric(
        "Punts blocked",
        stA["punts_blocked_against"],
        "of 12 in the whole NFL",
        delta_color="off",
    )
    c3.metric(
        "Punt return avg allowed",
        f"{OFF['punt_return_avg_allowed']}",
        "most since 1976",
        delta_color="off",
    )
    c4.metric(
        "Special teams EPA",
        f"{stA['st_epa']:.0f}",
        f"{stA['st_epa_rank_worst']}nd-worst of {A['n_team_seasons']} since 1999",
        delta_color="off",
    )

    st.subheader("Win probability, play by play")
    losses = sd_games[(sd_games["result"] == "L") & (sd_games["week"] <= 7)]
    options = {
        f"Week {int(r.week)} — {'vs' if r.home else '@'} {r.opp} ({int(r.pts)}–{int(r.opp_pts)})": int(
            r.week
        )
        for r in losses.itertuples()
    }
    choice = st.radio(
        "Game", list(options), horizontal=True, label_visibility="collapsed"
    )
    week = options[choice]
    # 2010 play_ids are not always chronological (e.g. play 1007 sits at 11:11 of Q1).
    g = (
        plays[plays["week"] == week]
        .sort_values(["minute", "play_id"])
        .reset_index(drop=True)
    )
    keys = [k for k in A["key_plays"] if k["week"] == week]
    fig = go.Figure()
    fig.add_scatter(
        x=g["minute"],
        y=g["sd_wp"],
        mode="lines",
        line={"color": POWDER, "width": 2.5, "shape": "hv"},
        name="Chargers win probability",
        customdata=g["desc"].str.slice(0, 110),
        hovertemplate="Min %{x:.1f}: %{y:.0%}<br>%{customdata}<extra></extra>",
    )
    stp = g[g["is_st"]]
    fig.add_scatter(
        x=stp["minute"],
        y=stp["sd_wp"],
        mode="markers",
        name="Special teams play",
        marker={"size": 6, "color": GREY},
        hoverinfo="skip",
    )
    for i, k in enumerate(keys):
        pos = g.index[g["play_id"] == k["play_id"]][0]
        y = float(g["sd_wp"].iloc[pos + 1]) if pos + 1 < len(g) else k["sd_wp_before"]
        fig.add_annotation(
            xanchor="right" if k["minute"] > 52 else "center",
            x=k["minute"],
            y=y,
            text=f"{k['label']}<br>{k['sd_wpa']:+.0%}"
            if k["sd_wpa"] <= -0.04
            else k["label"],
            showarrow=True,
            arrowcolor=GOLD,
            font={"color": GOLD, "size": 12},
            bgcolor=PANEL,
            ax=0,
            ay=-60 if i % 2 == 0 else 60,
        )
    for q in [15, 30, 45]:
        fig.add_vline(x=q, line={"color": GRID, "width": 1})
    fig.update_layout(
        yaxis={
            "title": "Chargers win probability",
            "tickformat": ".0%",
            "range": [0, 1],
        },
        xaxis={"title": "Game minute", "range": [0, 60.5]},
    )
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    game_wpa = next(x["st_wpa"] for x in A["games"] if x["week"] == week)
    st.caption(
        f"nflverse Vegas-adjusted win probability. Net win probability on all special teams plays "
        f"this game: {game_wpa:+.0%}. Grey dots mark every kick, punt and field goal."
    )
    if week == 3:
        st.markdown(
            '<div class="callout">Rivers tied it 20–20 on a two-point conversion. The tie held up for '
            "<b>15 seconds</b>: Leon Washington took the next kickoff 99 yards.</div>",
            unsafe_allow_html=True,
        )
    if week == 7:
        st.markdown(
            '<div class="callout">Down 23–20 with 27 seconds left, a false start turned a 45-yard tying '
            "try into a 50-yarder. Kris Brown, filling in for the injured Nate Kaeding, hit the right "
            "upright.</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Win probability given away on special teams, game by game")
    gw = pd.DataFrame(A["games"])
    fig = go.Figure(
        go.Bar(
            x=gw["week"],
            y=gw["st_wpa"],
            marker_color=[RED if r == "L" else POWDER for r in gw["result"]],
            customdata=np.stack(
                [gw["opp"], gw["result"], gw["pts"], gw["opp_pts"]], axis=1
            ),
            hovertemplate="Week %{x} vs %{customdata[0]} (%{customdata[1]} %{customdata[2]}–"
            "%{customdata[3]})<br>%{y:+.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Week", "dtick": 1},
        yaxis={"title": "Net win probability on ST plays", "tickformat": "+.0%"},
    )
    st.plotly_chart(style_fig(fig, 360), width="stretch")
    st.caption(
        "Red = loss. Summed over Weeks 1–7, special teams plays cost San Diego "
        f"{-sum(x['st_wpa'] for x in A['games'] if x['week'] <= 7):.1f} wins’ worth of win probability."
    )

    st.subheader("Special teams EPA, every team-season since 1999")
    fig = go.Figure()
    fig.add_histogram(
        x=seasons["st_epa"],
        nbinsx=50,
        marker_color=POWDER,
        opacity=0.75,
        name="Team-seasons",
        hovertemplate="%{x}: %{y} teams<extra></extra>",
    )
    fig.add_vline(x=stA["st_epa"], line={"color": GOLD, "width": 3})
    fig.add_annotation(
        x=stA["st_epa"],
        y=1,
        yref="paper",
        text=f"2010 SD: {stA['st_epa']:.0f}",
        showarrow=False,
        font={"color": GOLD, "size": 13},
        xanchor="left",
        xshift=6,
    )
    fig.update_layout(
        xaxis_title="Season special teams expected points added",
        yaxis_title="Team-seasons",
        showlegend=False,
        bargap=0.05,
    )
    st.plotly_chart(style_fig(fig, 360), width="stretch")
    worst = pd.DataFrame(stA["worst_st_epa"]).head(6)
    worst["Record"] = (
        worst["wins"].astype(int).astype(str)
        + "–"
        + worst["losses"].astype(int).astype(str)
    )
    st.dataframe(
        worst.rename(
            columns={
                "season": "Season",
                "team": "Team",
                "st_epa": "ST EPA",
                "playoff_result": "Postseason",
            }
        )[["Season", "Team", "ST EPA", "Record", "Postseason"]].round(1),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "EPA sums every kickoff, punt, field goal and extra point. About 4.8 points per game below "
        "average, roughly a field goal and a half every week."
    )

    st.subheader("Punt return yards allowed per return")
    pr = seasons.dropna(subset=["punt_ret_avg_allowed"])
    pr = (
        pr[pr["punt_rets_allowed"] >= 15]
        .nlargest(12, "punt_ret_avg_allowed")
        .iloc[::-1]
    )
    fig = go.Figure(
        go.Bar(
            y=pr["label"],
            x=pr["punt_ret_avg_allowed"],
            orientation="h",
            marker_color=[GOLD if lab == "2010 LAC" else DIM for lab in pr["label"]],
            hovertemplate="%{y}: %{x:.1f} yds/return<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Yards per return (play-by-play count)", showlegend=False
    )
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.caption(
        "Play-by-play counts 35 returns (15.1 avg); the NFL’s official count is 28 returns for 18.9. "
        "Either way, first since 1999. Minimum 15 returns."
    )

# ------------------------------------------------------------------ What If
with tabs[4]:
    st.markdown(
        f"""
Each scenario simulates every 2010 game involving an AFC West team {SIM["n_sims"]:,} times. A game's final
margin is drawn from a normal distribution centered on the rating gap plus home field
(+{SIM["home_edge"]:.1f}), with a standard deviation of {SIM["sigma"]:.1f} points (both fit on 1999–2025).
"""
    )
    scen = [
        (
            "Actual points (special teams included)",
            "points_based",
            SIM["sd_rating_points"],
            DIM,
        ),
        (
            "League-average special teams",
            "league_average_special_teams",
            SIM["sd_rating_st_neutral"],
            POWDER,
        ),
        ("What the yardage said", "yardage_based", SIM["sd_rating_yards"], GOLD),
    ]
    cols = st.columns(3)
    for col, (name, key, rating, _) in zip(cols, scen):
        col.metric(
            name,
            f"{SIM[key]['p_playoffs']:.0%} playoffs",
            f"{SIM[key]['p_division']:.0%} division · {rating:+.1f} pts/game rating",
            delta_color="off",
        )
    fig = go.Figure()
    for name, key, _, color in scen:
        dist = SIM[key]["wins_distribution"]
        xs = list(range(4, 17))
        fig.add_bar(
            x=xs, y=[dist.get(str(x), 0) for x in xs], name=name, marker_color=color
        )
    fig.add_vline(x=9, line={"color": RED, "dash": "dot", "width": 2})
    fig.add_annotation(
        x=9,
        y=0.97,
        yref="paper",
        yanchor="top",
        text="Actual: 9 wins",
        showarrow=False,
        font={"color": RED},
        xanchor="right",
        xshift=-6,
    )
    fig.update_layout(
        barmode="group",
        xaxis={"title": "Chargers wins", "dtick": 1},
        yaxis={"title": "Share of simulations", "tickformat": ".0%"},
        bargap=0.15,
    )
    st.plotly_chart(style_fig(fig, 440), width="stretch")
    st.caption(
        f"With league-average special teams, 9 or fewer wins happens in "
        f"{SIM['league_average_special_teams']['p_9_or_fewer_wins']:.1%} of seasons. The ST-neutral rating "
        "adds back the −76 special teams EPA (−4.8 points per game). It lands within 0.3 points of the "
        "yardage-based rating, built independently."
    )

    st.subheader("Flip one loss")
    st.markdown(
        "Pick any of the Chargers’ losses to turn into a win, and the AFC West standings update with "
        "the NFL tiebreakers (head-to-head, then division record)."
    )
    g10 = games[(games["season"] == 2010) & games["team"].isin(AFC_WEST)]
    loss_rows = sd_games[sd_games["result"] == "L"]
    loss_labels = {
        f"Week {int(r.week)} {'vs' if r.home else '@'} {r.opp} ({int(r.pts)}–{int(r.opp_pts)})": r.game_id
        for r in loss_rows.itertuples()
    }
    flipped = st.multiselect(
        "Losses to flip", list(loss_labels), default=[next(iter(loss_labels))]
    )
    flip_ids = {loss_labels[x] for x in flipped}
    seen = set()
    results = []
    for r in g10.itertuples():
        if r.game_id in seen and r.opp in AFC_WEST:
            continue
        seen.add(r.game_id)
        w, lo = (r.team, r.opp) if r.result == "W" else (r.opp, r.team)
        if r.game_id in flip_ids:
            w, lo = lo, w
        results.append((w, lo, 1.0))
    table = []
    for t in AFC_WEST:
        wins = sum(1 for w, _, _ in results if w == t)
        losses_ = sum(1 for _, lo, _ in results if lo == t)
        dw = sum(1 for w, lo, _ in results if w == t and lo in AFC_WEST)
        dl = sum(1 for w, lo, _ in results if lo == t and w in AFC_WEST)
        table.append(
            {"Team": TEAM_NAMES[t], "W": wins, "L": losses_, "Division": f"{dw}–{dl}"}
        )
    champ = division_winner(sorted(AFC_WEST), results, AFC_WEST)
    tdf = pd.DataFrame(table).sort_values(["W", "Team"], ascending=[False, True])
    tdf["Status"] = np.where(
        tdf["Team"] == TEAM_NAMES[champ], "🏆 Division champion", ""
    )
    st.dataframe(tdf, hide_index=True, width="stretch")
    if champ == "LAC":
        st.success("The Chargers win the AFC West and host a playoff game.")
    else:
        st.error("The Chiefs win the AFC West. The Chargers stay home.")
    st.caption(
        "Any single flipped loss is enough. The wild cards went to the 11–5 Jets and 12–4 Ravens, so the "
        "division was San Diego’s only way in."
    )

# ------------------------------------------------------------ Philip Rivers
with tabs[5]:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "2010 passing yards",
        f"{OFF['rivers_passing_yards']:,}",
        "led the NFL",
        delta_color="off",
    )
    c2.metric(
        "2010 passer rating",
        OFF["rivers_rating"],
        "3rd straight season over 100",
        delta_color="off",
    )
    c3.metric(
        "Vincent Jackson / Antonio Gates",
        "4 / 10 games",
        "holdout and injuries",
        delta_color="off",
    )

    st.subheader("Career passer rating vs. games played")
    min_att = st.slider("Minimum career pass attempts", 1000, 5000, 2000, 500)
    c = careers[
        careers["full_career_in_data"] & (careers["attempts"] >= min_att)
    ].copy()
    show_sb = st.toggle("Also show QBs who started a Super Bowl", value=True)
    fig = go.Figure()
    if show_sb:
        sb = c[c["started_super_bowl"]]
        fig.add_scatter(
            x=sb["games"],
            y=sb["rating"],
            mode="markers",
            name="Started a Super Bowl",
            marker={"size": 9, "color": DIM, "symbol": "diamond"},
            text=sb["player_display_name"],
            hovertemplate="%{text}<br>%{x} games, %{y:.1f} rating<extra></extra>",
        )
    no = c[~c["started_super_bowl"] & (c["player_display_name"] != "Philip Rivers")]
    # Label only the long or efficient careers; the crowded low-volume cluster stays hover-only.
    notable = (no["games"] >= 140) | (no["rating"] >= 93)
    fig.add_scatter(
        x=no["games"],
        y=no["rating"],
        mode="markers+text",
        name="Never started a Super Bowl",
        marker={"size": 10, "color": POWDER, "opacity": 0.8},
        text=np.where(notable, no["player_display_name"], ""),
        customdata=no["player_display_name"],
        textposition="top center",
        textfont={"size": 11, "color": GREY},
        hovertemplate="%{customdata}<br>%{x} games, %{y:.1f} rating<extra></extra>",
    )
    pr_ = careers[careers["player_display_name"] == "Philip Rivers"].iloc[0]
    fig.add_scatter(
        x=[pr_["games"]],
        y=[pr_["rating"]],
        mode="markers+text",
        name="Philip Rivers",
        text=["Philip Rivers"],
        textposition="bottom center",
        textfont={"color": GOLD, "size": 14},
        marker={
            "size": 22,
            "color": GOLD,
            "symbol": "star",
            "line": {"color": WHITE, "width": 1},
        },
    )
    fig.update_layout(
        xaxis_title="Career regular-season games", yaxis_title="Career passer rating"
    )
    st.plotly_chart(style_fig(fig, 540), width="stretch")
    st.caption(
        f"Rivers: {int(pr_['games'])} games, {pr_['rating']:.1f} career rating, {int(pr_['yards']):,} "
        "yards, including his three-game, 44-year-old comeback with the 2025 Colts. Limited to QBs whose careers began in 1999 or later, the start of the play-by-play data. "
        "Earlier careers (Marino, Fouts, Moon) would be truncated, so they're left out rather than "
        "shown wrong."
    )

    st.subheader("Three straight 100+ passer-rating seasons")
    streaks = A["rivers"]["pre_1999_streaks"] + A["rivers"]["streaks_3plus_since_1999"]
    fig = go.Figure()
    for i, s_ in enumerate(streaks):
        color = (
            GOLD
            if s_["qb"] == "Philip Rivers"
            else (POWDER if s_["end"] <= 2010 else DIM)
        )
        fig.add_bar(
            y=[f"{s_['qb']} ({s_['start']}–{s_['end']})"],
            x=[s_["end"] - s_["start"] + 1],
            base=[s_["start"] - 0.45],
            orientation="h",
            marker_color=color,
            showlegend=False,
            hovertemplate=f"{s_['qb']}: {s_['start']}–{s_['end']}<extra></extra>",
            width=0.7,
        )
    fig.add_vline(x=2010.55, line={"color": GOLD, "dash": "dot"})
    fig.add_annotation(
        x=2010.55,
        y=1,
        yref="paper",
        text="end of 2010",
        showarrow=False,
        xanchor="left",
        font={"color": GOLD},
    )
    fig.update_layout(
        xaxis={"title": "Season", "range": [1989, 2026.5], "dtick": 5},
        yaxis={"autorange": "reversed"},
        bargap=0.3,
    )
    st.plotly_chart(style_fig(fig, 440), width="stretch")
    st.caption(
        "Qualified seasons (14+ attempts per team game). Through 2010, Rivers was the third QB to do it. "
        "Steve Young’s run is from the Pro Football Hall of Fame list; everything since 1999 is computed "
        "from play-by-play. Nine more QBs have done it since."
    )

    st.subheader("Who Rivers threw to in 2010")
    rec = pd.DataFrame(
        [
            {"Player": "Antonio Gates", "Games": 10},
            {"Player": "Malcom Floyd", "Games": 11},
            {"Player": "Legedu Naanee", "Games": 9},
            {"Player": "Patrick Crayton", "Games": 8},
            {"Player": "Vincent Jackson", "Games": 4},
            {"Player": "Darren Sproles", "Games": 16},
        ]
    )
    ps = qb_seasons[
        (qb_seasons["season"] == 2010)
        & (qb_seasons["player_display_name"] == "Philip Rivers")
    ]
    fig = go.Figure(
        go.Bar(
            x=rec["Player"],
            y=rec["Games"],
            marker_color=[RED if g < 12 else POWDER for g in rec["Games"]],
            text=rec["Games"],
            textposition="outside",
        )
    )
    fig.add_hline(y=16, line={"color": GREY, "dash": "dash"})
    fig.update_layout(
        yaxis={"title": "Games played (of 16)", "range": [0, 18]}, showlegend=False
    )
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(
        "Games played per nflverse player stats. Jackson held out over his contract for most of "
        "the season. Rivers still led the NFL in yards (4,710), yards per attempt (8.7) and yards per game."
    )

# -------------------------------------------------------------- Methodology
with tabs[6]:
    st.markdown(
        """
**Data.** nflverse play-by-play, schedules and player stats, 1999–2025 regular seasons (861 team-seasons).
Team yardage is summed from play-by-play (pass and run plays, sacks included). It runs a few yards off the
official totals, so headline numbers quote the official figure and all rankings use play-by-play, keeping every
team on the same footing.

**Special teams EPA.** Expected points added on every kickoff, punt, field goal and extra point, netted for
each team (their EPA as the kicking or receiving side, minus their opponents').

**Expected wins.** (1) Pythagorean expectation from points scored and allowed, exponent 2.37. (2) A linear fit of
win percentage on yardage margin per game across all 861 team-seasons.

**Simulation.** Team ratings in expected points per game. *Actual points* = 2010 point differential per game.
*Yardage* = the fitted points-per-yard line applied to each team's yardage margin. *League-average special
teams* = San Diego's actual rating with its special teams EPA per game added back; everyone else unchanged.
Game margins ~ Normal(rating gap + home edge, σ), with σ estimated using leave-one-out ratings so a game
never informs its own prediction. Playoffs = win the AFC West (tiebreakers: record, head-to-head, division
record, coin flip), or 12+ wins as a wild card (11 wins = coin flip, since the last 2010 AFC wild card went
11–5). No strength-of-schedule adjustment.

**What the data can't say.** Play-by-play starts in 1999, so "Super Bowl era" claims that need earlier seasons
(the five No. 1/No. 1 teams, the 1976/1979 Steelers' yardage margin, 3-straight-100 QBs before 1999) come from
cited sources, listed in `REFERENCES.md`.
"""
    )
    claims = pd.DataFrame(
        [
            (
                "No. 1 offense and defense, missed playoffs (only team ever)",
                "Wikipedia; confirmed for 1999–2025 in data",
            ),
            (
                "+1,984 official yardage margin; ~1.8× next-best 2010 team",
                "Official totals; play-by-play ranking",
            ),
            (
                "Best five-game yardage start since 1999 (+1,087), record 2–3",
                "Play-by-play",
            ),
            (
                "Outgained 6 of first 7 by 125+: 2010 SD, 2019 NE, 2019 SF only",
                "Play-by-play, 1999–2025",
            ),
            (
                "4 of the NFL's 12 blocked punts; two in one game, back to back",
                "Play-by-play",
            ),
            (
                "18.9 punt return avg allowed, most since 1976",
                "Rotowire (2011); #1 since 1999 in play-by-play",
            ),
            ("2nd-worst special teams EPA of 861 team-seasons", "Play-by-play"),
            (
                "Any one early loss flipped wins the AFC West",
                "Schedules + NFL tiebreakers",
            ),
            (
                "Rivers 3rd QB with three straight 100+ ratings (through 2010)",
                "Play-by-play + HOF list for Young",
            ),
            (
                "KC 31–0, largest shutout of 2010; KC lost 30–7 in the Wild Card round",
                "Schedules",
            ),
        ],
        columns=["Claim", "Source"],
    )
    st.dataframe(claims, hide_index=True, width="stretch")
