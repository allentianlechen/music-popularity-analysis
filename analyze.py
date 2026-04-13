"""
analyze.py — Step 2: Train Popularity Prediction Model
=======================================================
Loads cleaned.csv, trains a Random Forest model, saves model.pkl
Run: python3 analyze.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, classification_report

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned.csv")
print(f"Loaded cleaned dataset: {df.shape[0]} rows")

# Convert bool columns to int for compatibility
if df["explicit"].dtype == bool:
    df["explicit"] = df["explicit"].astype(int)

# ── 2. DEFINE FEATURES ────────────────────────────────────────────────────────
# These 9 features get UI sliders; users can control them directly
SLIDER_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# These 5 features are used internally by the model but have no sliders
EXTRA_FEATURES = [
    "key", "mode", "time_signature", "explicit", "duration_min",
]

FEATURES = SLIDER_FEATURES + EXTRA_FEATURES
TARGET = "popularity"

# ── 2b. ARTIST AVERAGE POPULARITY ─────────────────────────────────────────────
# Artist fame is the strongest single predictor — proxy it with per-artist mean
df["artist_avg_popularity"] = (
    df.groupby("artists")["popularity"].transform("mean").round(1)
)
FEATURES = FEATURES + ["artist_avg_popularity"]

X = df[FEATURES]
y = df[TARGET]

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} tracks, testing on {len(X_test)} tracks")

# ── 4. TRAIN MODEL ────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Regressor trained")

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel performance:")
print(f"  R² score:  {r2:.3f}  (1.0 = perfect, 0.0 = no better than guessing mean)")
print(f"  Mean absolute error: {mae:.1f} popularity points")

# ── 5b. AUDIO-ONLY R² ─────────────────────────────────────────────────────────
# Train a separate model on just the 14 base features (no artist fame) to show
# how much audio features alone explain. More informative than full-model CV
# which would be inflated by artist_avg_popularity data leakage.
X_base = df[SLIDER_FEATURES + EXTRA_FEATURES]
X_tr_base, X_te_base, y_tr_base, y_te_base = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)
m_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
m_base.fit(X_tr_base, y_tr_base)
r2_base = r2_score(y_te_base, m_base.predict(X_te_base))
print(f"Audio-only R² (14 features, no artist): {r2_base:.3f}")
print(f"Full model R²  (+ artist fame):          {r2:.3f}")
print(f"Artist fame contribution:                {r2 - r2_base:.3f}")

# ── 5c. SCORE RANGE (for display scaling) ─────────────────────────────────────
all_preds = model.predict(X)
pred_min  = float(np.percentile(all_preds, 5))
pred_max  = float(np.percentile(all_preds, 95))
print(f"\nPrediction range (p5–p95): {pred_min:.1f} → {pred_max:.1f}")

# ── 5d. RECOMMENDED VALUES ────────────────────────────────────────────────────
# Recommended values: computed on SLIDER_FEATURES only, using an audio-only model
# so that artist fame does not bias what "optimal audio" looks like.
# We use the pre-trained audio-only model (m_base) and its top-1% predictions.
all_preds_base = m_base.predict(X_base)
top_mask_base  = all_preds_base >= np.percentile(all_preds_base, 99)
recommended = {
    feat: round(float(X_base[top_mask_base][feat].mean()), 3)
    for feat in SLIDER_FEATURES
}
print(f"\nRecommended values (avg of top 1% audio-only predicted tracks):")
for feat, val in recommended.items():
    print(f"  {feat:<25} {val}")

# ── 6. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=FEATURES)
importance = importance.sort_values(ascending=False)
print(f"\nFeature importance:")
for feat, score in importance.items():
    bar = "█" * int(score * 100)
    print(f"  {feat:<25} {score:.3f}  {bar}")

# ── 7. TIER CLASSIFIER ────────────────────────────────────────────────────────
# Popularity is skewed low; a low/medium/high classifier is more honest
df["popularity_tier"] = pd.cut(
    df["popularity"], bins=[-1, 30, 60, 100], labels=["low", "medium", "high"]
)
y_tier = df["popularity_tier"]
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y_tier, test_size=0.2, random_state=42, stratify=y_tier
)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_t, y_train_t)
print("\nTier classifier report:")
print(classification_report(y_test_t, clf.predict(X_test_t)))

# ── 8. GENRE MEANS ────────────────────────────────────────────────────────────
genre_means = (
    df.groupby("track_genre")[FEATURES].mean()
    .round(3)
    .to_dict(orient="index")
)

# ── 9. SAVE MODEL + METADATA ──────────────────────────────────────────────────
payload = {
    "model":           model,
    "features":        FEATURES,
    "slider_features": SLIDER_FEATURES,
    "importance":      importance.to_dict(),
    "r2":              round(r2, 3),
    "mae":             round(mae, 1),
    # p5–p95 range; used by the web app to rescale predictions to 0–100
    "pred_min":        round(pred_min, 3),
    "pred_max":        round(pred_max, 3),
    "recommended":     recommended,
    "ranges": {
        feat: {
            "min":  round(float(df[feat].min()), 3),
            "max":  round(float(df[feat].max()), 3),
            "mean": round(float(df[feat].mean()), 3),
        }
        for feat in FEATURES
    },
    # Audio-only R² (no artist fame) — shows audio feature contribution alone
    "r2_base": round(r2_base, 3),
    # Tier classifier
    "classifier":  clf,
    "tier_bins":   [-1, 30, 60, 100],
    "tier_labels": ["low", "medium", "high"],
    # Artist lookup for artist_avg_popularity
    "artist_lookup":          df.groupby("artists")["popularity"].mean().round(1).to_dict(),
    "global_avg_popularity":  round(float(df["popularity"].mean()), 1),
    # Genre means for genre-aware insight comparison
    "genre_means": genre_means,
}

with open("model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("\nSaved model.pkl")
print("Ready to run: python3 APP.py")
