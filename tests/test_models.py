import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import (
    division_winner,
    game_win_prob,
    linear_fit,
    passer_rating,
    pythagorean_win_pct,
    simulate_division,
)

DIV = {"A", "B", "C", "D"}


def test_passer_rating_matches_rivers_2010():
    # 357/541, 4,710 yds, 30 TD, 13 INT -> 101.8 official
    assert passer_rating(357, 541, 4710, 30, 13) == pytest.approx(101.8, abs=0.05)


def test_passer_rating_perfect_is_158_3():
    assert passer_rating(20, 20, 400, 5, 0) == pytest.approx(158.3, abs=0.05)


def test_pythagorean_even_teams_is_half():
    assert pythagorean_win_pct(300, 300) == pytest.approx(0.5)


def test_pythagorean_2010_chargers():
    assert pythagorean_win_pct(441, 322) * 16 == pytest.approx(10.85, abs=0.01)


def test_linear_fit_recovers_line():
    x = np.arange(10)
    fit = linear_fit(x, 2 * x + 1)
    assert fit["slope"] == pytest.approx(2)
    assert fit["intercept"] == pytest.approx(1)
    assert fit["r2"] == pytest.approx(1)


def test_game_win_prob_symmetry():
    assert game_win_prob(0, 13.5) == pytest.approx(0.5)
    assert game_win_prob(5, 13.5) + game_win_prob(-5, 13.5) == pytest.approx(1)


def test_division_winner_best_record():
    results = [("A", "X", 1), ("A", "Y", 1), ("B", "X", 1)]
    assert division_winner("ABCD", results, DIV) == "A"


def test_division_winner_head_to_head_breaks_tie():
    # A and B both 2-1; B won the head-to-head.
    results = [
        ("A", "X", 1),
        ("B", "Y", 1),
        ("B", "A", 1),
        ("A", "Z", 1),
        ("Q", "B", 1),
    ]
    assert division_winner("ABCD", results, DIV) == "B"


def test_division_winner_division_record_after_split_h2h():
    # A and B 2-2 overall, split head-to-head; A beat C, B lost to C.
    results = [
        ("A", "B", 1),
        ("B", "A", 1),
        ("A", "C", 1),
        ("C", "B", 1),
        ("B", "X", 1),
        ("X", "A", 1),
    ]
    assert division_winner("ABCD", results, DIV) == "A"


def test_simulation_heavy_favorite_wins_division():
    sched = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("A", "D"), ("D", "A")]
    ratings = {"A": 30.0, "B": 0.0, "C": 0.0, "D": 0.0}
    out = simulate_division(
        sched, ratings, DIV, home_edge=0, sigma=13.5, n_sims=2000, focus="A"
    )
    assert out["p_division"] > 0.95
    assert sum(out["wins_distribution"].values()) == pytest.approx(1)
