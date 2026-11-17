"""Every number the post and dashboard quote, written to outputs/analysis.json.

Reads only data/processed. Run build_tables.py first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import division_winner, linear_fit, pythagorean_win_pct, simulate_division

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
AFC_WEST = {"LAC", "KC", "LV", "DEN"}
N_SIMS = 20_000

# Official NFL season totals (play-by-play sums differ by a few yards).
# Sources: Rotowire 2011 Chargers preview; 3rd Drawer Down; Rick Gosselin 2010 ST rankings.
OFFICIAL_2010 = {
    "total_yards": 6329,
    "yards_per_game": 395.6,
    "yards_allowed": 4345,
    "yards_allowed_per_game": 271.6,
    "yard_differential": 6329 - 4345,
    "passing_yards_allowed": 3135,
    "first_downs_allowed": 247,
    "rivers_passing_yards": 4710,
    "rivers_rating": 101.8,
    "punt_return_avg_allowed": 18.9,
    "punt_return_yards_allowed": 528,
}

# The special teams plays the post walks through, located in play-by-play by week + text.
KEY_PLAYS = [
    (1, "J.Arenas to LAC 32", "Arenas 36-yd punt return"),
    (1, "J.Arenas pushed ob", "Arenas 24-yd punt return"),
    (1, "D.McCluster for 94 yards", "McCluster 94-yd punt return TD"),
    (3, "L.Washington for 101 yards", "Washington 101-yd kickoff return TD"),
    (3, "L.Washington for 99 yards", "Washington 99-yd kickoff return TD"),
    (5, "BLOCKED by 25-R.Cartwright", "Blocked punt → safety"),
    (5, "BLOCKED by 83-B.Myers", "Blocked punt → TD"),
    (6, "D.Amendola to LA 49", "Amendola 42-yd punt return"),
    (6, "field goal is BLOCKED", "Kaeding 49-yd FG blocked"),
    (7, "J.Edelman to NE 48", "Edelman 34-yd punt return"),
    (7, "50 yard field goal is No Good", "50-yd FG hits upright (after false start)"),
]


def rank_of(series: pd.Series, mask: pd.Series, ascending: bool) -> int:
    return int(series.rank(ascending=ascending, method="min")[mask].iloc[0])


def key_play_rows(plays: pd.DataFrame) -> list[dict]:
    rows = []
    for week, needle, label in KEY_PLAYS:
        hit = plays[
            (plays["week"] == week)
            & plays["desc"].str.contains(needle, regex=False, na=False)
        ]
        if len(hit) != 1:
            raise ValueError(
                f"expected exactly one play for {label!r}, found {len(hit)}"
            )
        p = hit.iloc[0]
        rows.append(
            {
                "week": week,
                "opp": p["opp"],
                "qtr": int(p["qtr"]),
                "time": p["time"],
                "minute": float(p["minute"]),
                "play_id": int(p["play_id"]),
                "label": label,
                "sd_wp_before": float(p["sd_wp"]),
                "sd_wpa": float(p["sd_wpa"]),
            }
        )
    return rows


def flip_scenarios(games: pd.DataFrame) -> list[dict]:
    g = games[(games["season"] == 2010) & games["team"].isin(AFC_WEST)]
    one_row = g[g["home"] == 1]  # one row per game, so no game counts twice
    base = []
    for _, r in one_row.iterrows():
        if r["result"] == "W":
            base.append((r["team"], r["opp"], 1.0, r["game_id"]))
        elif r["result"] == "L":
            base.append((r["opp"], r["team"], 1.0, r["game_id"]))
    # Non-division teams' games against division teams are where the AFC West
    # teams got wins/losses too; they're all in `base` via home rows of either side.
    away_rows = g[(g["home"] == 0) & ~g["opp"].isin(AFC_WEST)]
    for _, r in away_rows.iterrows():
        w, lo = (r["team"], r["opp"]) if r["result"] == "W" else (r["opp"], r["team"])
        base.append((w, lo, 1.0, r["game_id"]))

    sd_losses = g[(g["team"] == "LAC") & (g["result"] == "L")].sort_values("week")
    scenarios = [
        {
            "flipped": None,
            "winner": division_winner(
                sorted(AFC_WEST), [b[:3] for b in base], AFC_WEST
            ),
        }
    ]
    for _, loss in sd_losses.iterrows():
        res = [
            (lo, w, wt) if gid == loss["game_id"] else (w, lo, wt)
            for w, lo, wt, gid in base
        ]
        scenarios.append(
            {
                "flipped": f"Week {int(loss['week'])} vs {loss['opp']}",
                "winner": division_winner(sorted(AFC_WEST), res, AFC_WEST),
            }
        )
    return scenarios


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    seasons = pd.read_csv(PROCESSED / "team_season.csv")
    games = pd.read_csv(PROCESSED / "team_game.csv")
    plays = pd.read_csv(PROCESSED / "sd2010_plays.csv")
    qbs = pd.read_csv(PROCESSED / "qb_season.csv")
    is_sd = (seasons["season"] == 2010) & (seasons["team"] == "LAC")
    sd = seasons[is_sd].iloc[0]
    out: dict = {"official_2010": OFFICIAL_2010, "n_team_seasons": len(seasons)}

    out["pythagorean"] = {
        "expected_win_pct": float(pythagorean_win_pct(sd["pf"], sd["pa"])),
        "expected_wins": float(pythagorean_win_pct(sd["pf"], sd["pa"]) * 16),
        "actual_wins": int(sd["wins"]),
    }

    wins_fit = linear_fit(seasons["yard_diff_pg"], seasons["win_pct"])
    seasons["expected_win_pct"] = (
        wins_fit["intercept"] + wins_fit["slope"] * seasons["yard_diff_pg"]
    )
    seasons["win_pct_resid"] = seasons["win_pct"] - seasons["expected_win_pct"]
    seasons["expected_wins"] = seasons["expected_win_pct"] * seasons["games"]
    seasons["wins_below_expected"] = seasons["expected_wins"] - (
        seasons["wins"] + 0.5 * seasons["ties"]
    )
    out["yardage_model"] = {
        **wins_fit,
        "sd_expected_wins": float(seasons.loc[is_sd, "expected_wins"].iloc[0]),
        "sd_wins_below_expected": float(
            seasons.loc[is_sd, "wins_below_expected"].iloc[0]
        ),
        "sd_shortfall_rank": rank_of(
            seasons["wins_below_expected"], is_sd, ascending=False
        ),
        "teams_with_yard_diff_pg_100_plus": int((seasons["yard_diff_pg"] >= 100).sum()),
        "those_that_missed_playoffs": seasons[
            (seasons["yard_diff_pg"] >= 100) & ~seasons["made_playoffs"]
        ][["season", "team"]].to_dict("records"),
    }
    seasons.to_csv(PROCESSED / "team_season_modeled.csv", index=False)

    pts_fit = linear_fit(seasons["yard_diff_pg"], seasons["point_diff_pg"])
    s10 = seasons[seasons["season"] == 2010].set_index("team")
    yard_ratings = (
        pts_fit["intercept"] + pts_fit["slope"] * s10["yard_diff_pg"]
    ).to_dict()
    point_ratings = s10["point_diff_pg"].to_dict()

    # Leave-one-out ratings: a team's rating for a game excludes that game, so
    # the spread of actual-minus-predicted isn't shrunk by in-sample fitting.
    tot = seasons[["season", "team", "pf", "pa", "games"]].assign(
        pd_total=lambda d: d["pf"] - d["pa"]
    )
    reg = (
        games[games["home"] == 1]
        .merge(tot[["season", "team", "pd_total", "games"]], on=["season", "team"])
        .merge(
            tot[["season", "team", "pd_total", "games"]].rename(
                columns={"team": "opp", "pd_total": "opp_total", "games": "opp_games"}
            ),
            on=["season", "opp"],
        )
    )
    pred = (reg["pd_total"] - reg["margin"]) / (reg["games"] - 1) - (
        reg["opp_total"] + reg["margin"]
    ) / (reg["opp_games"] - 1)
    home_edge = float((reg["margin"] - pred).mean())
    sigma = float((reg["margin"] - pred - home_edge).std())

    g10 = games[(games["season"] == 2010) & (games["home"] == 1)]
    sched = [
        (r["team"], r["opp"])
        for _, r in g10.iterrows()
        if r["team"] in AFC_WEST or r["opp"] in AFC_WEST
    ]
    st_neutral = {
        **point_ratings,
        "LAC": point_ratings["LAC"] - sd["st_epa"] / sd["games"],
    }
    sim = {
        name: simulate_division(sched, ratings, AFC_WEST, home_edge, sigma, N_SIMS)
        for name, ratings in [
            ("points_based", point_ratings),
            ("yardage_based", yard_ratings),
            ("league_average_special_teams", st_neutral),
        ]
    }
    out["simulation"] = {
        "n_sims": N_SIMS,
        "home_edge": home_edge,
        "sigma": sigma,
        "points_per_yard_pg": pts_fit["slope"],
        "sd_rating_points": point_ratings["LAC"],
        "sd_rating_yards": yard_ratings["LAC"],
        "sd_rating_st_neutral": st_neutral["LAC"],
        **sim,
    }

    out["flip_scenarios"] = flip_scenarios(games)

    st_plays = plays[plays["is_st"]]
    sd_games = games[(games["season"] == 2010) & (games["team"] == "LAC")].sort_values(
        "week"
    )
    st_wpa = st_plays.groupby("week")["sd_wpa"].sum()
    out["games"] = [
        {
            "week": int(r["week"]),
            "opp": r["opp"],
            "home": bool(r["home"]),
            "pts": int(r["pts"]),
            "opp_pts": int(r["opp_pts"]),
            "result": r["result"],
            "yards": int(r["yards"]),
            "opp_yards": int(r["opp_yards"]),
            "st_wpa": float(st_wpa.get(r["week"], 0.0)),
        }
        for _, r in sd_games.iterrows()
    ]
    out["key_plays"] = key_play_rows(plays)

    out["special_teams"] = {
        "st_epa": float(sd["st_epa"]),
        "st_epa_rank_worst": rank_of(seasons["st_epa"], is_sd, ascending=True),
        "worst_st_epa": seasons.nsmallest(10, "st_epa")[
            ["season", "team", "st_epa", "wins", "losses", "playoff_result"]
        ].to_dict("records"),
        "punt_ret_avg_allowed_pbp": float(sd["punt_ret_avg_allowed"]),
        "punt_ret_avg_rank_worst": rank_of(
            seasons["punt_ret_avg_allowed"], is_sd, ascending=False
        ),
        "return_tds_allowed": int(sd["return_tds_allowed"]),
        "punts_blocked_against": int(sd["punts_blocked_against"]),
        "points_allowed_on_returns_and_blocks": 3 * 7 + 7 + 7 + 2 * 2,
        "st_share_of_plays": float(plays["is_st"].mean()),
    }

    out["rankings"] = {
        "yard_diff_rank": rank_of(seasons["yard_diff"], is_sd, ascending=False),
        "top_yard_diff": seasons.nlargest(5, "yard_diff")[
            ["season", "team", "yard_diff", "wins", "losses", "playoff_result"]
        ].to_dict("records"),
        "ypp_diff_rank_2010": rank_of(
            seasons.loc[seasons["season"] == 2010, "ypp_diff"],
            is_sd[seasons["season"] == 2010],
            ascending=False,
        ),
        "first5_yard_diff_rank": rank_of(
            seasons["first5_yard_diff"], is_sd, ascending=False
        ),
        "top_first5": seasons.nlargest(5, "first5_yard_diff")[
            ["season", "team", "first5_yard_diff", "first5_wins"]
        ].to_dict("records"),
        "six_of_first_seven_outgained_125": seasons[
            seasons["first7_outgained_125"] >= 6
        ][["season", "team", "first7_wins"]].to_dict("records"),
        "number_one_offense_and_defense_since_1999": seasons[
            (seasons["off_rank"] == 1) & (seasons["def_rank"] == 1)
        ][["season", "team"]].to_dict("records"),
    }

    q = qbs[qbs["qualified"] & (qbs["rating"] >= 100)].sort_values("season")
    streaks = []
    for name, grp in q.groupby("player_display_name"):
        yrs = sorted(grp["season"])
        run = [yrs[0]]
        for y in yrs[1:] + [None]:
            if y is not None and y == run[-1] + 1:
                run.append(y)
                continue
            if len(run) >= 3:
                streaks.append({"qb": name, "start": run[0], "end": run[-1]})
            run = [y]
    out["rivers"] = {
        "streaks_3plus_since_1999": sorted(streaks, key=lambda s: s["end"]),
        "pre_1999_streaks": [
            {
                "qb": "Steve Young",
                "start": 1991,
                "end": 1994,
                "source": "Pro Football Hall of Fame 100-point QB list",
            }
        ],
    }

    with open(OUTPUTS / "analysis.json", "w") as f:
        json.dump(
            out,
            f,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    print(
        json.dumps(
            {
                k: out[k]
                for k in [
                    "pythagorean",
                    "simulation",
                    "flip_scenarios",
                    "special_teams",
                    "rankings",
                ]
            },
            indent=1,
            default=str,
        )[:6000]
    )
    print(
        "yardage model:",
        {
            k: v
            for k, v in out["yardage_model"].items()
            if k != "those_that_missed_playoffs"
        },
        out["yardage_model"]["those_that_missed_playoffs"],
    )


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
