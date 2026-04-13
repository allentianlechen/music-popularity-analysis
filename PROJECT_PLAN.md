# analysisProject — Project Plan

> **Goal:** Implement audio file upload with per-feature scoring, improve model accuracy, make the analysis more honest and educational, and enrich the UI — all without needing the Spotify API.

---

## Current File Map

| File | Role | Status |
|------|------|--------|
| `clean.py` | Loads `dataset.csv` → produces `cleaned.csv` | Done — do not modify |
| `analyze.py` | Loads `cleaned.csv` → trains RandomForest → saves `model.pkl` | Needs changes (Tasks 1.1, 1.2, 2.1, 2.3) |
| `APP.py` | Flask server: `/`, `/meta`, `/predict`, `/analyze-audio` | Mostly done; minor additions needed |
| `index.html` | Frontend: sliders, gauge, insights panel | Needs audio upload UI + several UI tasks |
| `model.pkl` | Serialised model + metadata | Regenerate after each `analyze.py` change |
| `cleaned.csv` | Cleaned Spotify dataset (~90k tracks) | Do not modify |
| `dataset.csv` | Raw source data | Do not modify |

**Current FEATURES in `analyze.py`** (9 of available columns):
```python
FEATURES = ["danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
```
**Unused columns available in `cleaned.csv`:** `key`, `mode`, `time_signature`, `explicit`, `duration_min`, `track_genre`, `artists`

---

## Important: What's Already Implemented

- `APP.py` already has a `/analyze-audio` POST endpoint that accepts a multipart file upload, runs librosa feature extraction, and returns JSON `{ features: { danceability, energy, ... } }`. **Do not rewrite this.**
- `/meta` endpoint already returns `features`, `importance`, `ranges`, `recommended`, `r2`, `mae`. All UI tasks can build on this without new endpoints unless noted.
- The `model.pkl` payload structure uses named keys. Always **extend** it (add new keys); never replace the whole structure.
- The p5–p95 rescaling in `/predict` is intentional — keeps the 0–100 display scale usable despite Spotify's skewed popularity distribution. Leave it.

---

## Phase 0 — Audio Upload UI (Core User Vision — highest priority)

> This is the most important missing piece. The backend (`/analyze-audio`) is done. Only the frontend is needed.

### Task 0.1 — Audio upload section in `index.html`
**File:** `index.html`

Add a full-width section **between** the `<header class="hero">` and the `<div class="layout">`. It should contain:

1. **Drag-and-drop zone** — accepts `.mp3`, `.wav`, `.flac`, `.m4a`. Use a `<input type="file" accept="audio/*">` inside a styled drop zone div. Handle both click-to-browse and drag-and-drop events.

2. **Loading state** — while waiting for `/analyze-audio` (~10–20s for librosa), show a spinner and "Analyzing audio…" text. Disable the drop zone to prevent double-uploads.

3. **Auto-fill sliders** — on a successful response, set each `<input id="slider-{feat}">` value to the returned feature value and update the displayed label. Then call the existing `predict()` function to trigger a fresh score.

4. **Per-feature score cards** — after analysis, render a grid of 9 cards (one per feature) showing:
   - Feature name
   - A **fitness score 0–100** (how close the extracted value is to the popularity-optimal recommended value)
   - The raw extracted value
   - A mini color-coded progress bar: green ≥ 70, amber 40–69, red < 40

   **Fitness score formula:**
   ```javascript
   function fitnessScore(feat, value) {
     const r = meta.ranges[feat];
     const rec = meta.recommended[feat];
     const span = r.max - r.min || 1;
     return Math.round(Math.max(0, Math.min(100,
       (1 - Math.abs(value - rec) / span) * 100
     )));
   }
   ```
   `meta.ranges` and `meta.recommended` are already loaded from `/meta` at page init.

5. **Actionable summary line** — e.g. "3 features need improvement: speechiness, liveness, tempo" (list features with fitness score < 50).

6. **Clear button** — resets sliders to dataset averages (use `meta.ranges[feat].mean`) and hides the score cards.

7. **Disclaimer line** (small, muted text): "Feature extraction is estimated via librosa — values may differ slightly from Spotify's own analysis."

**Design guidance:** Match the existing dark theme — `--bg: #080810`, `--surface: #0f0f1a`, `--accent: #c8f135`, `--accent2: #7c3aed`. Use `font-family: 'Azeret Mono'` for labels and `'Syne'` for large numbers. Animate the score cards in with the existing `slideIn` keyframe already defined in the CSS.

### Task 0.2 — Update hero steps
**File:** `index.html`

The hero currently shows one linear path. Update `.steps` to show two parallel paths:
- **Path A (manual):** Adjust sliders → Model runs → See score
- **Path B (upload):** Upload audio → Features extracted → See score

---

## Phase 1 — Quick Wins (low effort, high impact)

### Task 1.1 — Add unused features to the model
**File:** `analyze.py`

Extend `FEATURES` with the 5 unused columns. Random Forest handles integers and 0/1 booleans natively — no encoding needed.

```python
FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "key", "mode", "time_signature", "explicit", "duration_min",
]
```

After changing, run `python3 analyze.py` and compare R² before/after.

**Frontend note:** These new features will appear in `/meta` automatically. They should NOT get sliders in the UI (they are not audio-analysis features a user meaningfully controls). Handle this by maintaining a hardcoded `SLIDER_FEATURES` allowlist in `index.html` that only shows the original 9 features as sliders. The model still uses all 14 internally — the extras are set to dataset mean values when calling `/predict`.

### Task 1.2 — Add 5-fold cross-validation
**File:** `analyze.py`

After the existing training block, add:
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X, y, cv=5, scoring="r2"
)
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

Add to `model.pkl` payload:
```python
payload["cv_r2_mean"] = round(float(cv_scores.mean()), 3)
payload["cv_r2_std"]  = round(float(cv_scores.std()),  3)
```

The `/meta` endpoint will expose these automatically since it returns the full payload minus the model object.

### Task 1.3 — EDA script
**File:** new `eda.py`

Standalone script — run once for insight, output not used by the web app.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["popularity"], bins=50, edgecolor="black")
axes[0].set_title("Popularity Distribution (all tracks)")
axes[0].set_xlabel("Popularity Score (0–100)")
axes[0].set_ylabel("Number of Tracks")

genre_median = df.groupby("track_genre")["popularity"].median().sort_values(ascending=False).head(20)
genre_median.plot(kind="barh", ax=axes[1])
axes[1].set_title("Top 20 Genres by Median Popularity")
axes[1].set_xlabel("Median Popularity")

plt.tight_layout()
plt.savefig("eda_popularity.png", dpi=150)
print("Saved eda_popularity.png")
```

---

## Phase 2 — Model Improvements

### Task 2.1 — Add popularity tier classification
**Files:** `analyze.py`, `APP.py`

The raw popularity score is skewed heavily toward 0–40. A classifier that predicts "low / medium / high" is more honest.

In `analyze.py`, after the regression block:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df["popularity_tier"] = pd.cut(
    df["popularity"], bins=[-1, 30, 60, 100], labels=["low", "medium", "high"]
)
y_tier = df["popularity_tier"]
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X, y_tier, test_size=0.2, random_state=42, stratify=y_tier
)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_t, y_train_t)
print(classification_report(y_test_t, clf.predict(X_test_t)))

payload["classifier"] = clf
payload["tier_bins"]   = [-1, 30, 60, 100]
payload["tier_labels"] = ["low", "medium", "high"]
```

In `APP.py`'s `/predict` route, after computing `score`, add:
```python
tier_label = payload["classifier"].predict(values)[0]
return jsonify({"score": score, "tier": tier_label, "insights": insights})
```

The frontend can use `tier` to reinforce or replace the existing tier-pill logic.

### Task 2.2 — Per-genre feature analysis
**File:** new `genre_analysis.py`

```python
import pandas as pd, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("cleaned.csv")
FEATURES = [...]  # copy the full list from analyze.py after Task 1.1
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
        "r2": round(r2_score(y_te, m.predict(X_te)), 3),
        "importance": dict(zip(FEATURES, m.feature_importances_.round(3).tolist()))
    }

with open("genre_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved genre_analysis.json")
```

### Task 2.3 — Artist-level average popularity feature
**File:** `analyze.py`

Artist fame is the biggest single predictor of popularity. Proxy it with per-artist mean popularity already in the dataset.

Before the train/test split in `analyze.py`:
```python
df["artist_avg_popularity"] = df.groupby("artists")["popularity"].transform("mean").round(1)
FEATURES = FEATURES + ["artist_avg_popularity"]
X = df[FEATURES]
```

Add to `model.pkl` payload:
```python
payload["artist_lookup"]          = df.groupby("artists")["popularity"].mean().round(1).to_dict()
payload["global_avg_popularity"]  = round(float(df["popularity"].mean()), 1)
```

In `APP.py`'s `/predict`, accept an optional `artist` field:
```python
artist = data.get("artist", "").strip()
artist_avg = payload["artist_lookup"].get(artist, payload["global_avg_popularity"])
body_with_artist = {**data, "artist_avg_popularity": artist_avg}
values = [[body_with_artist[f] for f in features]]
```

In `index.html`, add an optional artist name text input above the sliders.

---

## Phase 3 — UI Improvements

### Task 3.1 — Show model accuracy honestly
**Files:** `index.html`

The `/meta` response already includes `r2`, `mae`, and (after Task 1.2) `cv_r2_mean`/`cv_r2_std`. Add an info card to the result panel with plain-English copy:

```
Model accuracy: R² = 0.XX  (CV: 0.XX ± 0.XX)
Audio features explain ~XX% of popularity variance.
The remaining ~XX% comes from artist fame, playlist placement,
and marketing — factors this tool cannot measure.
Mean prediction error: ±XX popularity points.
```

### Task 3.2 — Genre filter dropdown
**Files:** `index.html`, `APP.py`

Add a `/genres` endpoint to `APP.py`:
```python
@app.route("/genres")
def genres():
    return jsonify(sorted(genre_means.keys()))  # store genre_means in analyze.py → model.pkl
```

In `analyze.py`, compute and save genre-level feature means:
```python
payload["genre_means"] = df.groupby("track_genre")[FEATURES].mean().round(3).to_dict(orient="index")
```

In `APP.py`'s `/predict`, if `genre` is provided in the POST body, use genre-specific means for the insight comparison instead of the global means.

In `index.html`, add a `<select>` that fetches `/genres` on load. Send the selected genre with each predict call.

### Task 3.3 — Feature importance bar chart
**File:** `index.html`

The `/meta` response already has `importance` (a dict of feature → float). Render a horizontal bar chart — pure HTML/CSS is fine, no library needed. Place it in the result panel below the model stats card.

---

## Phase 4 — Code Quality

### Task 4.1 — Shared config
**File:** new `config.py`

```python
# config.py
FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "key", "mode", "time_signature", "explicit", "duration_min",
]
SLIDER_FEATURES = FEATURES[:9]  # only these get UI sliders
TARGET = "popularity"
```

Import in both `analyze.py` and `APP.py`. This eliminates the risk of the two files drifting out of sync on the feature list.

---

## Recommended Execution Order

| # | Task | Files | Notes |
|---|------|-------|-------|
| 1 | **Task 0.1** — audio upload UI | `index.html` | Core user vision; backend already done |
| 2 | **Task 0.2** — hero steps update | `index.html` | Quick, pairs with 0.1 |
| 3 | **Task 1.1** — add unused features | `analyze.py` | 1-line change; retrain model |
| 4 | **Task 1.2** — cross-validation | `analyze.py` | Makes accuracy numbers trustworthy |
| 5 | **Task 3.1** — show R²/MAE in UI | `index.html` | Pairs with CV results from Task 1.2 |
| 6 | **Task 1.3** — EDA script | `eda.py` (new) | Standalone; run once for insight |
| 7 | **Task 2.3** — artist avg feature | `analyze.py`, `APP.py`, `index.html` | Biggest single accuracy boost |
| 8 | **Task 2.1** — tier classifier | `analyze.py`, `APP.py` | More honest framing for skewed data |
| 9 | **Task 2.2** — genre analysis | `genre_analysis.py` (new) | Standalone; feeds Task 3.2 |
| 10 | **Task 3.2** — genre filter UI | `index.html`, `APP.py` | Depends on genre data from 2.2 |
| 11 | **Task 3.3** — importance chart | `index.html` | Polish |
| 12 | **Task 4.1** — config.py | `config.py`, `analyze.py`, `APP.py` | Optional cleanup, do last |

---

## Dependencies

Already in use: `flask`, `scikit-learn`, `pandas`, `numpy`, `librosa`

Additional:
```bash
pip install matplotlib   # Task 1.3 EDA charts
```

---

## Do NOT Change

- `clean.py` — deduplication and cleaning logic is correct.
- `APP.py`'s `/analyze-audio` route — librosa extraction is already implemented and working.
- The p5–p95 rescaling logic in `/predict` — intentional, keeps display scale useful.
- `model.pkl` payload structure — always extend with new keys, never replace the whole dict.
- `cleaned.csv` / `dataset.csv` — source data, read-only.
