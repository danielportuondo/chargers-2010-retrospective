# The Tragedy of the 2010 San Diego Chargers

A statistical retrospective of the 2010 San Diego Chargers: the NFL's No. 1 total offense and No. 1 total
defense, a +1,984-yard differential, and a 9–7 record that missed the playoffs, mostly because of
historically bad special teams.

**Live dashboard:** https://chargers-2010-retrospective.streamlit.app/

## Headline findings

- **The only team since 1999 ranked No. 1 in both total offense and total defense.** (In the Super Bowl
  era, one of five; the other four made the playoffs.)
- **Expected wins:** 13.4 from yardage margin, 10.9 from points scored and allowed. Actual: 9.
- **Best five-game yardage start since 1999** (+1,087). Record after five games: 2–3.
- **Outgained six of their first seven opponents by 125+ yards.** Only the 2019 Patriots and 2019 49ers
  match that since 1999, and both started 7–0. San Diego started 2–5.
- **Special teams:** 2nd-worst special teams EPA of 861 team-seasons since 1999. 4 of the NFL's 12 blocked
  punts, 4 return touchdowns allowed (most in the NFL), 18.9 yards per punt return allowed (most since 1976).
- **Simulation (20,000 seasons):** with league-average special teams, the 2010 Chargers make the playoffs
  94% of the time and win 9 or fewer games 1.2% of the time.
- **Any one of their first five losses flipped** wins them the AFC West.

## Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Seven tabs: The Story · Best Offense, Best Defense · The Anomaly · Special Teams (play-by-play win
probability for every early loss) · What If (simulation + a flip-a-loss standings toggle) · Philip Rivers ·
Methodology. Reads only `outputs/` and `data/processed/`.

## Setup and pipeline

Python 3.12+.

```bash
uv venv && uv pip install -r requirements.txt
uv run python src/data_acquisition.py   # nflverse play-by-play, schedules, player stats (1999–2025), cached to data/raw/
uv run python src/build_tables.py       # team-season, team-game, Chargers play and QB tables -> data/processed/
uv run python src/analysis.py           # every quoted number -> outputs/analysis.json
```

The first run downloads 27 seasons of play-by-play (~470 MB). It caches per season and re-fetches any
truncated file.

## Layout

- `src/models.py`: pure functions (passer rating, Pythagorean expectation, linear fit, division tiebreak,
  season simulation), all unit-tested
- `src/build_tables.py`: the only module that reads raw play-by-play
- `src/analysis.py`: rankings, models, simulation, counterfactuals
- `dashboard/app.py`: Streamlit app
- `tests/`: `uv pip install -r requirements-dev.txt && uv run pytest`
- `REFERENCES.md`: data sources and the citation behind every claim that predates the 1999 data window

## Caveats

- Yardage is summed from play-by-play and runs a few yards off official totals (2010 SD offense: 6,343 vs
  6,329 official). Headline numbers quote the official figures; rankings use play-by-play so every team is
  measured the same way.
- Play-by-play starts in 1999. "Super Bowl era" claims that need earlier seasons come from cited sources.
- The simulation covers only games involving AFC West teams, uses season-level ratings with no
  strength-of-schedule adjustment, and approximates the wild-card cutoff at 11 wins (the 2010 AFC
  sixth seed went 11–5).
