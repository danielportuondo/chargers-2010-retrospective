"""Build the processed tables every later step reads (data/processed/*.csv).

This is the only module that opens raw play-by-play. Yardage here is summed
from play-by-play, so it runs a few yards off the NFL's official season totals
(e.g. 2010 SD offense: 6,343 here vs 6,329 official). Rankings and all-time
comparisons use these numbers so every team is measured the same way; the
headline official totals live in analysis.py with their sources.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from models import passer_rating

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Schedules keep each franchise's code at the time; play-by-play uses today's.
TEAM_CODE = {"SD": "LAC", "OAK": "LV", "STL": "LA"}
KICK_TYPES = ["kickoff", "punt", "field_goal", "extra_point"]
PBP_COLS = [
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "yards_gained",
    "special_teams_play",
    "epa",
    "vegas_wp",
    "vegas_wpa",
    "punt_blocked",
    "return_yards",
    "punt_returner_player_name",
    "touchdown",
    "td_team",
    "interception",
    "fumble_lost",
    "qtr",
    "time",
    "game_seconds_remaining",
    "total_home_score",
    "total_away_score",
    "desc",
    "field_goal_result",
]


def load_pbp() -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "pbp").glob("*.parquet")):
        df = pd.read_parquet(path, columns=PBP_COLS)
        frames.append(df[df["season_type"] == "REG"])
    return pd.concat(frames, ignore_index=True)


def load_schedules() -> pd.DataFrame:
    s = pd.read_parquet(RAW / "schedules.parquet")
    for col in ["home_team", "away_team"]:
        s[col] = s[col].replace(TEAM_CODE)
    return s


def team_games(sched: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    reg = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()]
    reg = reg[reg["season"] <= 2025]
    home = reg.assign(
        team=reg["home_team"],
        opp=reg["away_team"],
        home=1,
        pts=reg["home_score"],
        opp_pts=reg["away_score"],
    )
    away = reg.assign(
        team=reg["away_team"],
        opp=reg["home_team"],
        home=0,
        pts=reg["away_score"],
        opp_pts=reg["home_score"],
    )
    g = pd.concat([home, away])[
        [
            "season",
            "game_id",
            "week",
            "team",
            "opp",
            "home",
            "pts",
            "opp_pts",
            "div_game",
        ]
    ]

    scrimmage = pbp[pbp["play_type"].isin(["pass", "run"])]
    yards = (
        scrimmage.groupby(["game_id", "posteam"])
        .agg(yards=("yards_gained", "sum"), plays=("yards_gained", "size"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    g = g.merge(yards, on=["game_id", "team"], how="left")
    opp_yards = yards.rename(
        columns={"team": "opp", "yards": "opp_yards", "plays": "opp_plays"}
    )
    g = g.merge(opp_yards, on=["game_id", "opp"], how="left")
    g = g.sort_values(["season", "team", "week"])
    g["game_num"] = g.groupby(["season", "team"]).cumcount() + 1
    g["margin"] = g["pts"] - g["opp_pts"]
    g["yard_diff"] = g["yards"] - g["opp_yards"]
    g["result"] = np.sign(g["margin"]).map({1: "W", -1: "L", 0: "T"})
    return g


def playoff_results(sched: pd.DataFrame) -> pd.DataFrame:
    post = sched[(sched["game_type"] != "REG") & sched["home_score"].notna()]
    order = {"WC": 1, "DIV": 2, "CON": 3, "SB": 4}
    rows = []
    for _, g in post.iterrows():
        for team, won in [
            (g["home_team"], g["home_score"] > g["away_score"]),
            (g["away_team"], g["away_score"] > g["home_score"]),
        ]:
            rows.append(
                {
                    "season": g["season"],
                    "team": team,
                    "round": order[g["game_type"]],
                    "won": won,
                }
            )
    p = (
        pd.DataFrame(rows)
        .sort_values("round")
        .groupby(["season", "team"])
        .last()
        .reset_index()
    )
    labels = {
        1: "Lost Wild Card",
        2: "Lost Divisional",
        3: "Lost Conference",
        4: "Lost Super Bowl",
    }
    p["playoff_result"] = np.where(
        p["won"] & (p["round"] == 4), "Won Super Bowl", p["round"].map(labels)
    )
    return p[["season", "team", "playoff_result"]]


def special_teams(pbp: pd.DataFrame) -> pd.DataFrame:
    st = pbp[pbp["play_type"].isin(KICK_TYPES) & pbp["epa"].notna()]
    # nflverse credits EPA to `posteam`: the kicking team on punts/FGs, the
    # receiving team on kickoffs. Netting both sides gives each team's total.
    gained = st.groupby(["season", "posteam"])["epa"].sum()
    lost = st.groupby(["season", "defteam"])["epa"].sum()
    out = (
        gained.rename_axis(["season", "team"])
        .sub(lost.rename_axis(["season", "team"]), fill_value=0)
        .rename("st_epa")
        .reset_index()
    )

    punts = pbp[pbp["play_type"] == "punt"]
    returned = punts[
        punts["punt_returner_player_name"].notna() & (punts["punt_blocked"] != 1)
    ]
    pr = (
        returned.groupby(["season", "posteam"])
        .agg(
            punt_ret_yds_allowed=("return_yards", "sum"),
            punt_rets_allowed=("return_yards", "size"),
        )
        .rename_axis(["season", "team"])
        .reset_index()
    )
    blocked = (
        punts.groupby(["season", "posteam"])["punt_blocked"]
        .sum()
        .rename("punts_blocked_against")
        .rename_axis(["season", "team"])
        .reset_index()
    )

    kick_tds = pbp[
        pbp["play_type"].isin(["kickoff", "punt"]) & (pbp["touchdown"] == 1)
    ].copy()
    kick_tds["kicking_team"] = np.where(
        kick_tds["play_type"] == "punt", kick_tds["posteam"], kick_tds["defteam"]
    )
    kick_tds = kick_tds[kick_tds["td_team"] != kick_tds["kicking_team"]]
    ret_tds = (
        kick_tds.groupby(["season", "kicking_team"])
        .size()
        .rename("return_tds_allowed")
        .rename_axis(["season", "team"])
        .reset_index()
    )

    for extra in [pr, blocked, ret_tds]:
        out = out.merge(extra, on=["season", "team"], how="left")
    out["return_tds_allowed"] = out["return_tds_allowed"].fillna(0).astype(int)
    out["punt_ret_avg_allowed"] = out["punt_ret_yds_allowed"] / out["punt_rets_allowed"]
    return out


def team_seasons(
    games: pd.DataFrame, pbp: pd.DataFrame, playoffs: pd.DataFrame
) -> pd.DataFrame:
    s = (
        games.groupby(["season", "team"])
        .agg(
            games=("game_id", "size"),
            wins=("result", lambda r: (r == "W").sum()),
            losses=("result", lambda r: (r == "L").sum()),
            ties=("result", lambda r: (r == "T").sum()),
            pf=("pts", "sum"),
            pa=("opp_pts", "sum"),
            yards_for=("yards", "sum"),
            yards_against=("opp_yards", "sum"),
            plays_for=("plays", "sum"),
            plays_against=("opp_plays", "sum"),
        )
        .reset_index()
    )
    s["win_pct"] = (s["wins"] + 0.5 * s["ties"]) / s["games"]
    s["yards_pg"] = s["yards_for"] / s["games"]
    s["yards_allowed_pg"] = s["yards_against"] / s["games"]
    s["yard_diff"] = s["yards_for"] - s["yards_against"]
    s["yard_diff_pg"] = s["yard_diff"] / s["games"]
    s["point_diff_pg"] = (s["pf"] - s["pa"]) / s["games"]
    s["ypp_diff"] = (
        s["yards_for"] / s["plays_for"] - s["yards_against"] / s["plays_against"]
    )
    s["off_rank"] = (
        s.groupby("season")["yards_pg"].rank(ascending=False, method="min").astype(int)
    )
    s["def_rank"] = (
        s.groupby("season")["yards_allowed_pg"]
        .rank(ascending=True, method="min")
        .astype(int)
    )

    giveaways = (
        pbp.groupby(["season", "posteam"])[["interception", "fumble_lost"]]
        .sum()
        .sum(axis=1)
    )
    takeaways = (
        pbp.groupby(["season", "defteam"])[["interception", "fumble_lost"]]
        .sum()
        .sum(axis=1)
    )
    to = pd.DataFrame(
        {
            "giveaways": giveaways.rename_axis(["season", "team"]),
            "takeaways": takeaways.rename_axis(["season", "team"]),
        }
    ).reset_index()
    to["turnover_margin"] = to["takeaways"] - to["giveaways"]

    first5 = (
        games[games["game_num"] <= 5]
        .groupby(["season", "team"])
        .agg(
            first5_yard_diff=("yard_diff", "sum"),
            first5_wins=("result", lambda r: (r == "W").sum()),
        )
    )
    first7 = (
        games[games["game_num"] <= 7]
        .groupby(["season", "team"])
        .agg(
            first7_outgained_125=("yard_diff", lambda d: (d >= 125).sum()),
            first7_wins=("result", lambda r: (r == "W").sum()),
        )
    )

    s = (
        s.merge(to, on=["season", "team"], how="left")
        .merge(special_teams(pbp), on=["season", "team"], how="left")
        .merge(first5.reset_index(), on=["season", "team"], how="left")
        .merge(first7.reset_index(), on=["season", "team"], how="left")
        .merge(playoffs, on=["season", "team"], how="left")
    )
    s["playoff_result"] = s["playoff_result"].fillna("Missed playoffs")
    s["made_playoffs"] = s["playoff_result"] != "Missed playoffs"
    return s


def chargers_2010_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    sd = pbp[
        (pbp["season"] == 2010)
        & ((pbp["home_team"] == "LAC") | (pbp["away_team"] == "LAC"))
    ].copy()
    # Vegas-adjusted win probability: San Diego was favored in most of these
    # games, which a neutral prior would hide.
    # Timeouts and quarter breaks carry no posteam, so flipping their win
    # probability to San Diego's side would draw false spikes.
    sd = sd[sd["vegas_wp"].notna() & sd["posteam"].notna()]
    sd["sd_wp"] = np.where(sd["posteam"] == "LAC", sd["vegas_wp"], 1 - sd["vegas_wp"])
    sd["sd_wpa"] = np.where(sd["posteam"] == "LAC", sd["vegas_wpa"], -sd["vegas_wpa"])
    sd["is_st"] = sd["play_type"].isin(KICK_TYPES)
    sd["sd_home"] = sd["home_team"] == "LAC"
    sd["opp"] = np.where(sd["sd_home"], sd["away_team"], sd["home_team"])
    sd["sd_score"] = np.where(
        sd["sd_home"], sd["total_home_score"], sd["total_away_score"]
    )
    sd["opp_score"] = np.where(
        sd["sd_home"], sd["total_away_score"], sd["total_home_score"]
    )
    sd["minute"] = (3600 - sd["game_seconds_remaining"]) / 60
    keep = [
        "week",
        "game_id",
        "play_id",
        "opp",
        "qtr",
        "time",
        "minute",
        "play_type",
        "is_st",
        "sd_wp",
        "sd_wpa",
        "sd_score",
        "opp_score",
        "desc",
        "epa",
        "posteam",
    ]
    return sd[keep]


def qb_seasons(players: pd.DataFrame) -> pd.DataFrame:
    ps = pd.read_parquet(RAW / "player_stats_reg.parquet")
    q = ps[ps["attempts"] > 0].copy()
    q["rating"] = passer_rating(
        q["completions"],
        q["attempts"],
        q["passing_yards"],
        q["passing_tds"],
        q["passing_interceptions"],
    )
    # Qualifying threshold for rate stats is 14 attempts per team game.
    q["qualified"] = q["attempts"] >= 14 * np.where(q["season"] >= 2021, 17, 16)
    q = q.merge(
        players[["gsis_id", "rookie_season"]],
        left_on="player_id",
        right_on="gsis_id",
        how="left",
    )
    cols = [
        "player_id",
        "player_display_name",
        "season",
        "games",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rating",
        "qualified",
        "rookie_season",
    ]
    return q[cols]


def qb_careers(qbs: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    sb = sched[sched["game_type"] == "SB"]
    sb_ids = set(sb["home_qb_id"].dropna()) | set(sb["away_qb_id"].dropna())
    c = (
        qbs.groupby(["player_id", "player_display_name"])
        .agg(
            first_season=("season", "min"),
            last_season=("season", "max"),
            rookie_season=("rookie_season", "first"),
            games=("games", "sum"),
            completions=("completions", "sum"),
            attempts=("attempts", "sum"),
            yards=("passing_yards", "sum"),
            tds=("passing_tds", "sum"),
            ints=("passing_interceptions", "sum"),
        )
        .reset_index()
    )
    c["rating"] = passer_rating(
        c["completions"], c["attempts"], c["yards"], c["tds"], c["ints"]
    )
    c["started_super_bowl"] = c["player_id"].isin(sb_ids)
    # Careers that began before 1999 would be truncated by the data window, so
    # they're excluded rather than shown with a partial (and wrong) career line.
    c["full_career_in_data"] = c["rookie_season"].fillna(c["first_season"]) >= 1999
    return c


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pbp = load_pbp()
    sched = load_schedules()
    players = pd.read_parquet(RAW / "players.parquet")

    games = team_games(sched, pbp)
    seasons = team_seasons(games, pbp, playoff_results(sched))
    qbs = qb_seasons(players)
    tables = {
        "team_game.csv": games,
        "team_season.csv": seasons,
        "sd2010_plays.csv": chargers_2010_plays(pbp),
        "qb_season.csv": qbs,
        "qb_career.csv": qb_careers(qbs, sched),
        "schedule_2026.csv": sched[sched["season"] == 2026][
            [
                "season",
                "week",
                "game_type",
                "away_team",
                "away_score",
                "home_team",
                "home_score",
            ]
        ],
    }
    for name, df in tables.items():
        df.to_csv(PROCESSED / name, index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
