"""
analyze.py — Step 2: Train Popularity Prediction Model
=======================================================
Loads cleaned.csv, trains a Random Forest model, saves model.pkl
Run: python3 analyze.py
"""

import logging
import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

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


# ── ARTIST AVERAGE TRANSFORMER ───────────────────────────────────────────────

class ArtistAvgTransformer(BaseEstimator, TransformerMixin):
    """Compute artist_avg_popularity from training data only (no leakage).

    fit()  — learns per-artist mean popularity from the training split.
    transform() — appends artist_avg_popularity column; unseen artists get
                  the global training mean. Drops the 'artists' column so
                  only numeric features reach the downstream model.
    """

    def __init__(self) -> None:
        self.artist_means_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ArtistAvgTransformer":
        df = X.copy()
        df["_target"] = y.values
        self.artist_means_ = df.groupby("artists")["_target"].mean().round(1).to_dict()
        self.global_mean_ = round(float(y.mean()), 1)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["artist_avg_popularity"] = (
            df["artists"].map(self.artist_means_).fillna(self.global_mean_)
        )
        df = df.drop(columns=["artists"])
        return df


def train() -> None:
    # ── 1. LOAD ───────────────────────────────────────────────────────────────
    df = pd.read_csv("cleaned.csv")
    logger.info("Loaded cleaned dataset: %d rows", df.shape[0])

    if df["explicit"].dtype == bool:
        df["explicit"] = df["explicit"].astype(int)

    # ── 2. PREPARE X AND y ───────────────────────────────────────────────────
    # Include 'artists' column so the transformer can compute artist averages.
    # The transformer drops 'artists' after adding artist_avg_popularity.
    raw_features = FEATURES + ["artists"]
    X_raw = df[raw_features]
    y = df[TARGET]

    # ── 3. TRAIN / TEST SPLIT (before computing artist averages) ─────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    logger.info("Training on %d tracks, testing on %d tracks",
                len(X_train_raw), len(X_test_raw))

    # ── 4. FIT TRANSFORMER ON TRAINING DATA ONLY ─────────────────────────────
    artist_transformer = ArtistAvgTransformer()
    artist_transformer.fit(X_train_raw, y_train)
    X_train = artist_transformer.transform(X_train_raw)
    X_test = artist_transformer.transform(X_test_raw)

    all_features = list(X_train.columns)
    logger.info("Features after transform: %s", all_features)
    logger.info("Unseen artists in test set get global mean: %.1f",
                artist_transformer.global_mean_)

    # ── 5. TRAIN MODEL ────────────────────────────────────────────────────────
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Regressor trained")

    # ── 6. EVALUATE ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    logger.info("R² = %.3f  |  MAE = %.1f popularity points", r2, mae)

    # ── 6b. CROSS-VALIDATION (Pipeline recomputes artist avg per fold) ───────
    cv_pipeline = Pipeline([
        ("artist_avg", ArtistAvgTransformer()),
        ("rf", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipeline, X_raw, y, cv=cv, scoring="r2")
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std  = float(cv_scores.std())
    logger.info("CV R² (leakage-free pipeline): %.3f ± %.3f", cv_r2_mean, cv_r2_std)

    # ── 6c. AUDIO-ONLY R² (no artist feature) ────────────────────────────────
    X_base = df[SLIDER_FEATURES + EXTRA_FEATURES]
    X_tr_base, X_te_base, y_tr_base, y_te_base = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )
    m_base = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    m_base.fit(X_tr_base, y_tr_base)
    r2_base = r2_score(y_te_base, m_base.predict(X_te_base))
    logger.info("Audio-only R² (no artist): %.3f  |  Artist fame contribution: %.3f",
                r2_base, r2 - r2_base)
    audio_importance = pd.Series(
        m_base.feature_importances_, index=SLIDER_FEATURES + EXTRA_FEATURES
    )

    # ── 6d. SCORE RANGE ──────────────────────────────────────────────────────
    X_all_transformed = artist_transformer.transform(X_raw)
    all_preds = model.predict(X_all_transformed)
    pred_min  = float(np.percentile(all_preds, 5))
    pred_max  = float(np.percentile(all_preds, 95))
    logger.info("Prediction range (p5–p95): %.1f → %.1f", pred_min, pred_max)

    # ── 6e. RECOMMENDED VALUES ───────────────────────────────────────────────
    all_preds_base = m_base.predict(X_base)
    top_mask_base  = all_preds_base >= np.percentile(all_preds_base, 99)
    recommended = {
        feat: round(float(X_base[top_mask_base][feat].mean()), 3)
        for feat in SLIDER_FEATURES
    }
    logger.info("Recommended audio profile (top-1%% audio-only model): %s", recommended)

    # ── 7. FEATURE IMPORTANCE ────────────────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=all_features)
    importance = importance.sort_values(ascending=False)
    for feat, score in importance.items():
        bar = "█" * int(score * 100)
        logger.info("  %-25s %.3f  %s", feat, score, bar)

    # ── 8. GENRE MEANS ───────────────────────────────────────────────────────
    # Use base features (no artist_avg_popularity) since that column
    # doesn't exist in the original dataframe — it's computed per-split.
    genre_features = SLIDER_FEATURES + EXTRA_FEATURES
    genre_means = (
        df.groupby("track_genre")[genre_features].mean()
        .round(3)
        .to_dict(orient="index")
    )

    # ── 9. SAVE MODEL + METADATA ─────────────────────────────────────────────
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
            **{
                feat: {
                    "min":  round(float(df[feat].min()), 3),
                    "max":  round(float(df[feat].max()), 3),
                    "mean": round(float(df[feat].mean()), 3),
                }
                for feat in FEATURES  # base features from the original df
            },
            "artist_avg_popularity": {
                "min":  round(float(X_train["artist_avg_popularity"].min()), 3),
                "max":  round(float(X_train["artist_avg_popularity"].max()), 3),
                "mean": round(float(X_train["artist_avg_popularity"].mean()), 3),
            },
        },
        "cv_r2_mean":             round(cv_r2_mean, 3),
        "cv_r2_std":              round(cv_r2_std,  3),
        "r2_base":                round(r2_base, 3),
        "audio_importance":       audio_importance.to_dict(),
        # Training-data-only artist lookup (no leakage)
        "artist_lookup":          artist_transformer.artist_means_,
        "global_avg_popularity":  artist_transformer.global_mean_,
        "genre_means":            genre_means,
    }

    joblib_dump(payload, "model.pkl", compress=3)

    logger.info("Saved model.pkl — ready to run: python3 APP.py")


def _prewarm_numba() -> None:
    """Pre-compile numba guvectorize functions used by librosa.

    This runs during the build step so that compiled artifacts are cached
    in the deployed image. Without this, the first runtime call to
    librosa.load() triggers JIT compilation that OOMs on 512 MB instances.
    """
    try:
        import librosa.core.audio  # noqa: F401 — triggers @guvectorize compilation
        logger.info("numba pre-compiled for librosa")
    except Exception:
        logger.warning("numba pre-compilation skipped (librosa not installed)")


if __name__ == "__main__":
    train()
    _prewarm_numba()
