# config.py — shared feature configuration
# Import in analyze.py and APP.py to keep feature lists in sync

SLIDER_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

EXTRA_FEATURES = [
    "key", "mode", "time_signature", "explicit", "duration_min",
]

# Full feature list used by the model (artist_avg_popularity appended at runtime)
BASE_FEATURES = SLIDER_FEATURES + EXTRA_FEATURES

TARGET = "popularity"
