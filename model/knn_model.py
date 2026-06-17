"""
knn_model.py

Loads the merged player stats CSV, builds a KNN model on a normalized
feature matrix, and exposes find_similar_players() for use by the API.

Usage (standalone):
    python model/knn_model.py --player "LeBron James" --n 5
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "players_merged.csv")

# Features used for similarity — tweak as needed
FEATURE_COLS = [
    # Per-game production
    "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "OREB", "DREB",
    # Advanced / efficiency
    "TS_PCT", "EFG_PCT", "USG_PCT", "PIE",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "AST_TO", "OREB_PCT", "DREB_PCT",
    # Play style / usage
    "PCT_FGA_2PT", "PCT_FGA_3PT",
    "PCT_PTS_PAINT", "PCT_PTS_3PT", "PCT_PTS_FT",
    "MIN",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_model(df: pd.DataFrame):
    """
    Returns (scaler, knn, feature_matrix, valid_df) where valid_df is the
    subset of df that had enough non-null features to be included.
    """
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    sub = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE"] + available_cols].copy()

    # Drop rows with too many nulls (keep rows missing at most 20% of features)
    thresh = int(len(available_cols) * 0.8)
    sub = sub.dropna(subset=available_cols, thresh=thresh).reset_index(drop=True)

    # Fill remaining nulls with column median
    sub[available_cols] = sub[available_cols].fillna(sub[available_cols].median())

    X = sub[available_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = NearestNeighbors(metric="euclidean", algorithm="ball_tree")
    knn.fit(X_scaled)

    return scaler, knn, X_scaled, sub, available_cols


def find_similar_players(
    player_name: str,
    df: pd.DataFrame,
    scaler: StandardScaler,
    knn: NearestNeighbors,
    X_scaled: np.ndarray,
    valid_df: pd.DataFrame,
    feature_cols: list,
    n: int = 5,
) -> pd.DataFrame:
    """
    Returns a DataFrame of the n most similar players to player_name,
    with their distance score and key stats.
    """
    # Find player (case-insensitive partial match)
    mask = valid_df["PLAYER_NAME"].str.lower().str.contains(player_name.lower())
    matches = valid_df[mask]

    if matches.empty:
        raise ValueError(f"No player found matching '{player_name}'")
    if len(matches) > 1:
        print(f"Multiple matches: {matches['PLAYER_NAME'].tolist()} — using first.")

    idx = matches.index[0]
    query = X_scaled[idx].reshape(1, -1)

    # n+1 because the player themselves is always the closest match
    distances, indices = knn.kneighbors(query, n_neighbors=n + 1)

    results = valid_df.iloc[indices[0]].copy()
    results["similarity_distance"] = distances[0]

    # Drop the query player themselves
    results = results[results.index != idx].head(n)

    display_cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
                    "PTS", "REB", "AST", "TS_PCT", "USG_PCT", "similarity_distance"]
    display_cols = [c for c in display_cols if c in results.columns]

    return results[display_cols].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True, help="Player name to look up")
    parser.add_argument("--n", type=int, default=5, help="Number of comps to return")
    parser.add_argument("--data", default=DATA_PATH, help="Path to players_merged.csv")
    args = parser.parse_args()

    df = load_data(args.data)
    scaler, knn, X_scaled, valid_df, feature_cols = build_model(df)
    comps = find_similar_players(
        args.player, df, scaler, knn, X_scaled, valid_df, feature_cols, n=args.n
    )
    print(f"\nTop {args.n} comps for '{args.player}':\n")
    print(comps.to_string(index=False))


if __name__ == "__main__":
    main()
