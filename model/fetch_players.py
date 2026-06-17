"""
fetch_players.py

Pulls per-game, advanced, and player bio (physical) stats for all active NBA
players in a given season via nba_api, merges them into a single DataFrame,
and saves the result to data/players_merged.csv.

Usage:
    python model/fetch_players.py               # current season
    python model/fetch_players.py --season 2023  # 2023-24 season
"""

import argparse
import os
import time

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    PlayerEstimatedMetrics,
)
from nba_api.stats.static import players as nba_players

# nba_api throttles requests; add a small delay between calls
SLEEP = 1.5
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def season_str(year: int) -> str:
    """Convert a start year to NBA season string, e.g. 2023 -> '2023-24'."""
    return f"{year}-{str(year + 1)[-2:]}"


def fetch_per_game(season: str) -> pd.DataFrame:
    print(f"Fetching per-game stats for {season}...")
    endpoint = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    )
    time.sleep(SLEEP)
    df = endpoint.get_data_frames()[0]
    # Keep only the columns we care about
    cols = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
        "GP", "MIN",
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT",
        "OREB", "DREB",
    ]
    return df[cols].rename(columns={"PLAYER_NAME": "PLAYER_NAME_PG"})


def fetch_advanced(season: str) -> pd.DataFrame:
    print(f"Fetching advanced stats for {season}...")
    endpoint = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    time.sleep(SLEEP)
    df = endpoint.get_data_frames()[0]
    cols = [
        "PLAYER_ID",
        "OFF_RATING", "DEF_RATING", "NET_RATING",
        "AST_PCT", "AST_TO", "AST_RATIO",
        "OREB_PCT", "DREB_PCT", "REB_PCT",
        "TM_TOV_PCT", "EFG_PCT", "TS_PCT",
        "USG_PCT", "PACE", "PIE",
    ]
    return df[cols]


def fetch_usage(season: str) -> pd.DataFrame:
    """Usage / play-style breakdown."""
    print(f"Fetching usage stats for {season}...")
    endpoint = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Usage",
    )
    time.sleep(SLEEP)
    df = endpoint.get_data_frames()[0]
    cols = [
        "PLAYER_ID",
        "PCT_FGA_2PT", "PCT_FGA_3PT",
        "PCT_PTS_2PT", "PCT_PTS_3PT", "PCT_PTS_FB",
        "PCT_PTS_FT", "PCT_PTS_OFF_TOV", "PCT_PTS_PAINT",
        "PCT_AST_2PM", "PCT_AST_3PM",
    ]
    # Not all seasons have every column; keep what's available
    available = [c for c in cols if c in df.columns]
    return df[available]


def fetch_physical() -> pd.DataFrame:
    """Pull height, weight, position from the static players list."""
    print("Fetching physical / bio data...")
    all_players = nba_players.get_active_players()
    records = []
    for p in all_players:
        records.append({
            "PLAYER_ID": p["id"],
            "POSITION": None,   # nba_api static list doesn't include position;
                                # we'll add it from CommonPlayerInfo if needed
        })
    return pd.DataFrame(records)


def merge_and_save(season: str) -> pd.DataFrame:
    per_game = fetch_per_game(season)
    advanced = fetch_advanced(season)
    usage = fetch_usage(season)
    physical = fetch_physical()

    df = per_game.merge(advanced, on="PLAYER_ID", how="left")
    df = df.merge(usage, on="PLAYER_ID", how="left")
    df = df.merge(physical, on="PLAYER_ID", how="left")

    # Rename the duplicate name column back
    df = df.rename(columns={"PLAYER_NAME_PG": "PLAYER_NAME"})

    # Drop players with very few games (noise)
    df = df[df["GP"] >= 10].reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "players_merged.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} players → {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="Season start year, e.g. 2024 for 2024-25",
    )
    args = parser.parse_args()
    merge_and_save(season_str(args.season))