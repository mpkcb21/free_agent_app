"""
app.py

Flask API for the free agent archetype finder.
Loads the KNN model once on startup, then serves queries.

Endpoints:
    GET  /players               — all players in the dataset
    GET  /players/search?q=     — autocomplete by name
    POST /comps                 — find similar players (KNN)
    POST /refresh               — re-fetch stats from NBA API

Run locally:
    cd free_agent_app
    source venv/bin/activate
    python api/app.py

Test:
    curl http://localhost:8000/players/search?q=luka
    curl -X POST http://localhost:8000/comps \
         -H "Content-Type: application/json" \
         -d '{"player": "Luka Doncic", "n": 5}'
"""

import os
import sys
import unicodedata
from flask_cors import CORS
from flask import Flask, jsonify, request


def normalize(s: str) -> str:
    """Strip accents and lowercase for fuzzy name matching."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


# Make sure model/ is on the path regardless of where the script is run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "model"))

from fetch_players import merge_and_save, season_str  # noqa: E402
from knn_model import build_model, find_similar_players  # noqa: E402

import pandas as pd  # noqa: E402

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# State — loaded once on startup
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(ROOT, "data", "players_merged.csv")
DEFAULT_SEASON = 2024

_df = None
_scaler = None
_knn = None
_X_scaled = None
_valid_df = None
_feature_cols = None


def load_model():
    """Load data and build KNN. Called on startup and after /refresh."""
    global _df, _scaler, _knn, _X_scaled, _valid_df, _feature_cols

    if not os.path.exists(DATA_PATH):
        print("No data found — fetching from NBA API...")
        _df = merge_and_save(season_str(DEFAULT_SEASON))
    else:
        print(f"Loading data from {DATA_PATH}")
        _df = pd.read_csv(DATA_PATH)

    _scaler, _knn, _X_scaled, _valid_df, _feature_cols = build_model(_df)
    print(f"Model ready — {len(_valid_df)} players, {len(_feature_cols)} features")


# Load model when module is imported (for gunicorn)
load_model()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/players", methods=["GET"])
def get_players():
    """Return all players with basic info."""
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "PTS", "REB", "AST"]
    available = [c for c in cols if c in _valid_df.columns]
    players = _valid_df[available].sort_values("PLAYER_NAME").to_dict(orient="records")
    return jsonify({"count": len(players), "players": players})


@app.route("/players/search", methods=["GET"])
def search_players():
    """
    Autocomplete search by player name.
    GET /players/search?q=luka
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"players": []})

    mask = _valid_df["PLAYER_NAME"].apply(normalize).str.contains(normalize(q), na=False)
    results = _valid_df[mask][["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]]\
        .sort_values("PLAYER_NAME")\
        .head(10)\
        .to_dict(orient="records")

    return jsonify({"players": results})


@app.route("/comps", methods=["POST"])
def get_comps():
    """
    Find similar players using KNN.

    Body (JSON):
        player  str   Player name (partial match OK)
        n       int   Number of comps to return (default 5, max 20)

    Example:
        {"player": "Luka Doncic", "n": 5}
    """
    body = request.get_json(silent=True) or {}
    player = body.get("player", "").strip()
    n = min(int(body.get("n", 5)), 20)

    if not player:
        return jsonify({"error": "Missing 'player' field"}), 400

    try:
        comps = find_similar_players(
            player, _df, _scaler, _knn, _X_scaled, _valid_df, _feature_cols, n=n
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    comps = comps.round(3)
    return jsonify({
        "query": player,
        "n": n,
        "comps": comps.to_dict(orient="records"),
    })


@app.route("/refresh", methods=["POST"])
def refresh():
    """
    Re-fetch player stats from the NBA API and rebuild the model.
    Accepts optional JSON body: {"season": 2024}
    """
    body = request.get_json(silent=True) or {}
    season_year = int(body.get("season", DEFAULT_SEASON))

    try:
        global _df
        _df = merge_and_save(season_str(season_year))
        load_model()
        return jsonify({"status": "ok", "season": season_str(season_year), "players": len(_valid_df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)