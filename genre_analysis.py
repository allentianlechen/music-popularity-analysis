"""
genre_analysis.py — Per-genre feature importance analysis
===========================================================
Standalone script — trains a separate RandomForest per top genre.
Output: genre_analysis.json
Run: python3 genre_analysis.py
"""

import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("cleaned.csv")

# Convert bool to int
if df["explicit"].dtype == bool:
    df["explicit"] = df["explicit"].astype(int)

# Must match the feature list in analyze.py
df["artist_avg_popularity"] = (
    df.groupby("artists")["popularity"].transform("mean").round(1)
)

FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "key", "mode", "time_signature", "explicit", "duration_min",
    "artist_avg_popularity",
]
TARGET = "popularity"

top_genres = df["track_genre"].value_counts().head(5).index
results = {}

for genre in top_genres:
    sub = df[df["track_genre"] == genre]
    if len(sub) < 200:
        continue
    X, y = sub[FEATURES], sub[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    results[genre] = {
        "r2":         round(r2_score(y_te, m.predict(X_te)), 3),
        "importance": dict(zip(FEATURES, m.feature_importances_.round(3).tolist())),
    }
    print(f"  {genre}: R² = {results[genre]['r2']}")

with open("genre_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved genre_analysis.json")
