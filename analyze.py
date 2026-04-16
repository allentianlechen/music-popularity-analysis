"""
analyze.py — Step 2: Train Popularity Prediction Model
=======================================================
Loads cleaned.csv, trains a Random Forest model, saves model.pkl
Run: python3 analyze.py
"""

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# These 9 features get UI sliders; users can control them directly
SLIDER_FEATURES: list[str] = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# These 5 features are used internally by the model but have no sliders
EXTRA_FEATURES: list[str] = [
    "key", "mode", "time_signature", "explicit", "duration_min",
]

FEATURES: list[str] = SLIDER_FEATURES + EXTRA_FEATURES
TARGET: str = "popularity"


def train() -> None:
    # ── 1. LOAD ───────────────────────────────────────────────────────────────
    df = pd.read_csv("cleaned.csv")
    logger.info("Loaded cleaned dataset: %d rows", df.shape[0])

    if df["explicit"].dtype == bool:
        df["explicit"] = df["explicit"].astype(int)

    # ── 2. ARTIST AVERAGE POPULARITY ─────────────────────────────────────────
    df["artist_avg_popularity"] = (
        df.groupby("artists")["popularity"].transform("mean").round(1)
    )
    all_features = FEATURES + ["artist_avg_popularity"]

    X = df[all_features]
    y = df[TARGET]

    # ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Training on %d tracks, testing on %d tracks", len(X_train), len(X_test))

    # ── 4. TRAIN MODEL ────────────────────────────────────────────────────────
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Regressor trained")

    # ── 5. EVALUATE ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    logger.info("R² = %.3f  |  MAE = %.1f popularity points", r2, mae)

    # ── 5b. CROSS-VALIDATION ─────────────────────────────────────────────────
    cv_scores = cross_val_score(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        X, y, cv=5, scoring="r2"
    )
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std  = float(cv_scores.std())
    logger.info("CV R²: %.3f ± %.3f", cv_r2_mean, cv_r2_std)

    # ── 5c. AUDIO-ONLY R² ─────────────────────────────────────────────────────
    X_base = df[SLIDER_FEATURES + EXTRA_FEATURES]
    X_tr_base, X_te_base, y_tr_base, y_te_base = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )
    m_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    m_base.fit(X_tr_base, y_tr_base)
    r2_base = r2_score(y_te_base, m_base.predict(X_te_base))
    logger.info("Audio-only R² (no artist): %.3f  |  Artist fame contribution: %.3f",
                r2_base, r2 - r2_base)
    audio_importance = pd.Series(
        m_base.feature_importances_, index=SLIDER_FEATURES + EXTRA_FEATURES
    )

    # ── 5c. SCORE RANGE ───────────────────────────────────────────────────────
    all_preds = model.predict(X)
    pred_min  = float(np.percentile(all_preds, 5))
    pred_max  = float(np.percentile(all_preds, 95))
    logger.info("Prediction range (p5–p95): %.1f → %.1f", pred_min, pred_max)

    # ── 5d. RECOMMENDED VALUES ────────────────────────────────────────────────
    all_preds_base = m_base.predict(X_base)
    top_mask_base  = all_preds_base >= np.percentile(all_preds_base, 99)
    recommended = {
        feat: round(float(X_base[top_mask_base][feat].mean()), 3)
        for feat in SLIDER_FEATURES
    }
    logger.info("Recommended audio profile (top-1%% audio-only model): %s", recommended)

    # ── 6. FEATURE IMPORTANCE ─────────────────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=all_features)
    importance = importance.sort_values(ascending=False)
    for feat, score in importance.items():
        bar = "█" * int(score * 100)
        logger.info("  %-25s %.3f  %s", feat, score, bar)

    # ── 7. GENRE MEANS ────────────────────────────────────────────────────────
    genre_means = (
        df.groupby("track_genre")[all_features].mean()
        .round(3)
        .to_dict(orient="index")
    )

    # ── 8. SAVE MODEL + METADATA ──────────────────────────────────────────────
    payload = {
        "model":           model,
        "features":        all_features,
        "slider_features": SLIDER_FEATURES,
        "importance":      importance.to_dict(),
        "r2":              round(r2, 3),
        "mae":             round(mae, 1),
        "pred_min":        round(pred_min, 3),
        "pred_max":        round(pred_max, 3),
        "recommended":     recommended,
        "ranges": {
            feat: {
                "min":  round(float(df[feat].min()), 3),
                "max":  round(float(df[feat].max()), 3),
                "mean": round(float(df[feat].mean()), 3),
            }
            for feat in all_features
        },
        "cv_r2_mean":             round(cv_r2_mean, 3),
        "cv_r2_std":              round(cv_r2_std,  3),
        "r2_base":                round(r2_base, 3),
        "audio_importance":       audio_importance.to_dict(),
        "artist_lookup":          df.groupby("artists")["popularity"].mean().round(1).to_dict(),
        "global_avg_popularity":  round(float(df["popularity"].mean()), 1),
        "genre_means":            genre_means,
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)

    logger.info("Saved model.pkl — ready to run: python3 APP.py")


if __name__ == "__main__":
    train()
