"""
analyze.py — Step 2: Train Popularity Prediction Model
=======================================================
Loads cleaned.csv, trains a Random Forest model, saves model.pkl
Run: python3 analyze.py
"""

import json
import logging
import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# The contextual analysis model can use Kaggle/Spotify-provided audio labels.
CONTEXT_AUDIO_FEATURES: list[str] = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# The upload model only uses features this app can defensibly approximate from
# local signal processing. Speechiness, instrumentalness, liveness, and valence
# are excluded because they require speech/vocal/live/mood recognition that this
# lightweight local pipeline does not have.
UPLOAD_FEATURES: list[str] = [
    "danceability", "energy", "loudness", "acousticness", "tempo",
]

# Backward-compatible name used by the UI/tests.
SLIDER_FEATURES: list[str] = UPLOAD_FEATURES

# These fields are useful for contextual analysis, but are not extracted from an
# uploaded audio file and are therefore excluded from upload inference.
EXTRA_FEATURES: list[str] = [
    "key", "mode", "time_signature", "explicit", "duration_min",
]

CONTEXT_BASE_FEATURES: list[str] = CONTEXT_AUDIO_FEATURES + EXTRA_FEATURES
TARGET: str = "popularity"

MODEL_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "n_estimators",
    "context_features",
    "upload_features",
    "features",
    "slider_features",
    "importance",
    "audio_importance",
    "ranges",
    "context_metrics",
    "upload_metrics",
    "r2",
    "mae",
    "pred_min",
    "pred_max",
    "recommended",
    "cv_r2_mean",
    "cv_r2_std",
    "r2_base",
    "global_avg_popularity",
    "genre_means",
)


def _json_ready(value):
    """Convert numpy/pandas scalar containers into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


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
    raw_features = CONTEXT_BASE_FEATURES + ["artists"]
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
    context_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    context_model.fit(X_train, y_train)
    logger.info("Contextual regressor trained")

    # ── 6. EVALUATE ───────────────────────────────────────────────────────────
    y_pred = context_model.predict(X_test)
    context_r2  = r2_score(y_test, y_pred)
    context_mae = mean_absolute_error(y_test, y_pred)
    logger.info("Context model R² = %.3f  |  MAE = %.1f popularity points",
                context_r2, context_mae)

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

    # ── 6c. UPLOAD AUDIO-ONLY MODEL ──────────────────────────────────────────
    X_upload = df[UPLOAD_FEATURES]
    X_tr_upload, X_te_upload, y_tr_upload, y_te_upload = train_test_split(
        X_upload, y, test_size=0.2, random_state=42
    )
    upload_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    upload_model.fit(X_tr_upload, y_tr_upload)
    upload_pred = upload_model.predict(X_te_upload)
    upload_r2 = r2_score(y_te_upload, upload_pred)
    upload_mae = mean_absolute_error(y_te_upload, upload_pred)
    logger.info("Upload audio-only R²: %.3f  |  MAE = %.1f  |  Context lift: %.3f",
                upload_r2, upload_mae, context_r2 - upload_r2)
    audio_importance = pd.Series(
        upload_model.feature_importances_, index=UPLOAD_FEATURES
    )

    # ── 6d. SCORE RANGE ──────────────────────────────────────────────────────
    X_all_transformed = artist_transformer.transform(X_raw)
    all_preds = context_model.predict(X_all_transformed)
    pred_min  = float(np.percentile(all_preds, 5))
    pred_max  = float(np.percentile(all_preds, 95))
    logger.info("Prediction range (p5–p95): %.1f → %.1f", pred_min, pred_max)

    # ── 6e. RECOMMENDED VALUES ───────────────────────────────────────────────
    all_preds_base = upload_model.predict(X_upload)
    top_mask_base  = all_preds_base >= np.percentile(all_preds_base, 99)
    recommended = {
        feat: round(float(X_upload[top_mask_base][feat].mean()), 3)
        for feat in UPLOAD_FEATURES
    }
    logger.info("Recommended audio profile (top-1%% audio-only model): %s", recommended)

    # ── 7. FEATURE IMPORTANCE ────────────────────────────────────────────────
    importance = pd.Series(context_model.feature_importances_, index=all_features)
    importance = importance.sort_values(ascending=False)
    for feat, score in importance.items():
        bar = "█" * int(score * 100)
        logger.info("  %-25s %.3f  %s", feat, score, bar)

    # ── 8. GENRE MEANS ───────────────────────────────────────────────────────
    # Use base features (no artist_avg_popularity) since that column
    # doesn't exist in the original dataframe — it's computed per-split.
    genre_features = UPLOAD_FEATURES + EXTRA_FEATURES
    genre_means = (
        df.groupby("track_genre")[genre_features].mean()
        .round(3)
        .to_dict(orient="index")
    )

    # ── 8b. REAL-USE EVALUATION SLICES ───────────────────────────────────────
    grouped_scores: list[float] = []
    grouped_cv = GroupKFold(n_splits=5)
    for tr_idx, te_idx in grouped_cv.split(X_upload, y, groups=df["artists"]):
        fold_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        fold_model.fit(X_upload.iloc[tr_idx], y.iloc[tr_idx])
        grouped_scores.append(float(r2_score(y.iloc[te_idx], fold_model.predict(X_upload.iloc[te_idx]))))

    genre_metrics = {}
    test_eval = pd.DataFrame({
        "actual": y_te_upload.values,
        "predicted": upload_pred,
        "genre": df.loc[y_te_upload.index, "track_genre"].values,
    })
    for genre, g in test_eval.groupby("genre"):
        if len(g) >= 20:
            genre_metrics[genre] = {
                "mae": round(float(mean_absolute_error(g["actual"], g["predicted"])), 2),
                "n": int(len(g)),
            }

    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    test_eval["popularity_range"] = pd.cut(test_eval["actual"], bins=bins, labels=labels, include_lowest=True)
    error_by_popularity = {
        str(label): {
            "mae": round(float(mean_absolute_error(g["actual"], g["predicted"])), 2),
            "n": int(len(g)),
        }
        for label, g in test_eval.groupby("popularity_range", observed=True)
        if len(g) > 0
    }

    test_eval["prediction_bucket"] = pd.cut(test_eval["predicted"], bins=bins, labels=labels, include_lowest=True)
    calibration = {
        str(label): {
            "actual_mean": round(float(g["actual"].mean()), 2),
            "predicted_mean": round(float(g["predicted"].mean()), 2),
            "n": int(len(g)),
        }
        for label, g in test_eval.groupby("prediction_bucket", observed=True)
        if len(g) > 0
    }

    # ── 9. SAVE MODEL + METADATA ─────────────────────────────────────────────
    payload = {
        "schema_version":   3,
        "context_model":    context_model,
        "upload_model":     upload_model,
        "n_estimators":     upload_model.n_estimators,
        "context_features": all_features,
        "upload_features":  UPLOAD_FEATURES,
        "context_metrics": {
            "random_split_r2": round(context_r2, 3),
            "random_split_mae": round(context_mae, 1),
            "cv_r2_mean": round(cv_r2_mean, 3),
            "cv_r2_std": round(cv_r2_std, 3),
        },
        "upload_metrics": {
            "random_split_r2": round(upload_r2, 3),
            "random_split_mae": round(upload_mae, 1),
            "artist_grouped_r2_mean": round(float(np.mean(grouped_scores)), 3),
            "artist_grouped_r2_std": round(float(np.std(grouped_scores)), 3),
            "genre_mae": genre_metrics,
            "error_by_popularity": error_by_popularity,
            "calibration": calibration,
        },
        # Backward-compatible aliases for older UI/tests.
        "model":           upload_model,
        "features":        UPLOAD_FEATURES,
        "slider_features": UPLOAD_FEATURES,
        "importance":      importance.to_dict(),
        "r2":              round(context_r2, 3),
        "mae":             round(upload_mae, 1),
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
                for feat in CONTEXT_BASE_FEATURES  # base features from the original df
            },
            "artist_avg_popularity": {
                "min":  round(float(X_train["artist_avg_popularity"].min()), 3),
                "max":  round(float(X_train["artist_avg_popularity"].max()), 3),
                "mean": round(float(X_train["artist_avg_popularity"].mean()), 3),
            },
        },
        "cv_r2_mean":             round(cv_r2_mean, 3),
        "cv_r2_std":              round(cv_r2_std,  3),
        "r2_base":                round(upload_r2, 3),
        "audio_importance":       audio_importance.to_dict(),
        # Training-data-only artist lookup (no leakage)
        "artist_lookup":          artist_transformer.artist_means_,
        "global_avg_popularity":  artist_transformer.global_mean_,
        "genre_means":            genre_means,
    }

    metadata = {
        key: _json_ready(payload[key])
        for key in MODEL_METADATA_KEYS
        if key in payload
    }
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    joblib_dump(payload, "model.pkl", compress=9)

    logger.info("Saved model.pkl and model_metadata.json — ready to run: python3 APP.py")


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
