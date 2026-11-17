"""Pure calculation functions: no file I/O, so each one is unit-tested directly."""

from collections.abc import Iterable

import numpy as np
from scipy.stats import norm

PYTHAGOREAN_EXPONENT = 2.37


def passer_rating(cmp, att, yds, td, ints):
    att = np.asarray(att, dtype=float)

    def clip(x):
        return np.clip(x, 0, 2.375)

    a = clip((np.asarray(cmp) / att - 0.3) * 5)
    b = clip((np.asarray(yds) / att - 3) * 0.25)
    c = clip(np.asarray(td) / att * 20)
    d = clip(2.375 - np.asarray(ints) / att * 25)
    return (a + b + c + d) / 6 * 100


def pythagorean_win_pct(points_for, points_against, exponent=PYTHAGOREAN_EXPONENT):
    pf = np.asarray(points_for, dtype=float) ** exponent
    pa = np.asarray(points_against, dtype=float) ** exponent
    return pf / (pf + pa)


def linear_fit(x, y) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r = np.corrcoef(x, y)[0, 1]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "r2": float(r**2),
        "resid_sd": float(resid.std(ddof=2)),
    }


def game_win_prob(expected_margin, sigma):
    """P(win) when the final margin is Normal(expected_margin, sigma)."""
    return norm.cdf(np.asarray(expected_margin, dtype=float) / sigma)


def division_winner(
    teams: Iterable[str],
    results: list[tuple[str, str, float]],
    division: set[str],
    rng: np.random.Generator | None = None,
) -> str:
    """NFL division tiebreak, first steps: best record, then head-to-head among
    the tied teams, then division record, then a coin flip.

    `results` are (winner, loser, weight) tuples; a tie is two half-weight rows.
    """
    teams = list(teams)
    wins = {t: 0.0 for t in teams}
    for w, _, wt in results:
        if w in wins:
            wins[w] += wt
    best = max(wins.values())
    tied = [t for t in teams if np.isclose(wins[t], best)]
    for scope in ("h2h", "division"):
        if len(tied) == 1:
            return tied[0]
        pool = set(tied) if scope == "h2h" else division
        rec = {t: 0.0 for t in tied}
        games = {t: 0.0 for t in tied}
        for w, lo, wt in results:
            if w in rec and lo in pool:
                rec[w] += wt
                games[w] += wt
            if lo in rec and w in pool:
                games[lo] += wt
        pct = {t: rec[t] / games[t] if games[t] else 0.5 for t in tied}
        top = max(pct.values())
        tied = [t for t in tied if np.isclose(pct[t], top)]
    if len(tied) == 1:
        return tied[0]
    rng = rng or np.random.default_rng(0)
    return tied[rng.integers(len(tied))]


def simulate_division(
    schedule: list[tuple[str, str]],
    ratings: dict[str, float],
    division: set[str],
    home_edge: float,
    sigma: float,
    n_sims: int,
    seed: int = 2010,
    focus: str = "LAC",
    wildcard_wins: float = 11.0,
) -> dict:
    """Simulate every game involving a division team; report `focus` outcomes.

    Playoffs = win the division, or reach `wildcard_wins` as a non-winner. In
    2010 the last AFC wild card went 11-5, so 11 wins is treated as a coin flip
    and 12+ as safe — an approximation that avoids simulating all 32 teams.
    """
    rng = np.random.default_rng(seed)
    home = np.array([h for h, _ in schedule])
    away = np.array([a for _, a in schedule])
    mu = np.array([ratings[h] - ratings[a] + home_edge for h, a in schedule])
    p_home = game_win_prob(mu, sigma)
    focus_wins = np.zeros(n_sims)
    div_titles = 0
    playoffs = 0
    for i in range(n_sims):
        home_won = rng.random(len(schedule)) < p_home
        winners = np.where(home_won, home, away)
        losers = np.where(home_won, away, home)
        results = [(w, lo, 1.0) for w, lo in zip(winners, losers)]
        fw = float((winners == focus).sum())
        focus_wins[i] = fw
        champ = division_winner(sorted(division), results, division, rng)
        if champ == focus:
            div_titles += 1
            playoffs += 1
        elif fw > wildcard_wins or (fw == wildcard_wins and rng.random() < 0.5):
            playoffs += 1
    return {
        "p_division": div_titles / n_sims,
        "p_playoffs": playoffs / n_sims,
        "mean_wins": float(focus_wins.mean()),
        "p_9_or_fewer_wins": float((focus_wins <= 9).mean()),
        "wins_distribution": {
            int(k): float(v) / n_sims
            for k, v in zip(*np.unique(focus_wins, return_counts=True))
        },
    }
