"""Pull and cache raw nflverse data to data/raw so later steps never re-download.

Play-by-play is cached one file per season and validated on read, so an
interrupted run resumes cleanly and a truncated file is re-fetched instead of
poisoning every downstream step.
"""

from pathlib import Path

import nflreadpy as nfl
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PBP_DIR = RAW_DIR / "pbp"
SEASONS = list(range(1999, 2026))


def _is_valid_parquet(path: Path) -> bool:
    try:
        pq.ParquetFile(path)
        return True
    except (OSError, pa.ArrowException):
        return False


def _cache(path: Path, loader) -> None:
    if path.exists() and _is_valid_parquet(path):
        print(f"skip {path.name} (cached)")
        return
    df = loader().to_pandas()
    df.to_parquet(path, index=False)
    print(f"{path.name}: {len(df):,} rows")


def fetch_all() -> None:
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        _cache(
            PBP_DIR / f"{season}.parquet", lambda s=season: nfl.load_pbp(seasons=[s])
        )
    # 2026 is included only for the "Jets were also 2-3 after Week 5" comparison.
    _cache(
        RAW_DIR / "schedules.parquet",
        lambda: nfl.load_schedules(seasons=SEASONS + [2026]),
    )
    _cache(
        RAW_DIR / "player_stats_reg.parquet",
        lambda: nfl.load_player_stats(seasons=SEASONS, summary_level="reg"),
    )
    _cache(RAW_DIR / "players.parquet", nfl.load_players)


if __name__ == "__main__":
    fetch_all()
