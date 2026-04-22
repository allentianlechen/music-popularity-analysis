# analysisProject — Progress Tracker

> Consolidated from PROJECT_PLAN.md, IMPROVEMENTS.md, memory, and verified against current code.
> Last updated: 2026-04-21

---

## Project Goal

Web app that predicts Spotify track popularity from audio features. No Spotify API needed at runtime. Dataset: Kaggle "Spotify Tracks Dataset" (~90k tracks, ~114 genres) with pre-computed audio features + popularity scores.

**Stack:** Python, Flask, scikit-learn (RandomForest), librosa, HTML/CSS/JS (dark theme).

---

## File Map

| File | Role | Status |
|------|------|--------|
| `clean.py` | Loads `dataset.csv` → produces `cleaned.csv` | DONE — do not modify |
| `analyze.py` | Trains RandomForest regressor → saves `model.pkl` | DONE |
| `APP.py` | Flask server: `/`, `/meta`, `/predict`, `/analyze-audio`, `/genres` | DONE |
| `index.html` | Single-page UI: upload, gauge, insights, importance chart | DONE |
| `config.py` | Shared feature list used by `analyze.py` and `APP.py` | DONE |
| `eda.py` | Standalone EDA script — popularity distribution + genre chart | DONE |
| `genre_analysis.py` | Standalone per-genre model analysis → `genre_analysis.json` | DONE |
| `test_app.py` | Test suite for APP.py routes | DONE |
| `model.pkl` | Serialised model + metadata — regenerate after `analyze.py` changes | Current |
| `cleaned.csv` / `dataset.csv` | Source data — read-only | — |

---

## Phase 0 — Audio Upload UI

### Task 0.1 — Audio upload section `DONE`
- Drag-and-drop zone accepting `.mp3`, `.wav`, `.flac`, `.m4a`
- Loading spinner with "Analyzing audio… ~15 seconds" copy
- Auto-fills sliders from extracted features, then calls `predict()`
- Per-feature fitness score cards (grid, color-coded green/amber/red)
- Actionable summary line listing features with score < 50
- Clear button resets sliders to dataset means and hides cards
- Disclaimer: "Feature extraction is estimated via librosa…"

### Task 0.2 — Hero steps update `DONE`
- "Path A / Path B" labels replaced with "Manual" / "Upload"
- Hero description updated to reflect both interaction paths

---

## Phase 1 — Quick Wins

### Task 1.1 — Add unused features to model `DONE`
Extended `FEATURES` in `analyze.py` with 5 non-slider columns:
`key`, `mode`, `time_signature`, `explicit`, `duration_min`

These do not get UI sliders — sent as dataset mean values on each `/predict` call. `SLIDER_FEATURES` (9 features) and `EXTRA_FEATURES` (5 features) are maintained separately in `config.py`.

### Task 1.2 — 5-fold cross-validation `DONE`
`analyze.py` runs `cross_val_score` (5-fold, R²) after the train/test evaluation. Results logged and saved to `model.pkl` as `cv_r2_mean` / `cv_r2_std`. Accuracy card in `index.html` shows "Cross-validation R²: X.XXX ± X.XXX (5-fold)" when the fields are present in `/meta`.

### Task 1.3 — EDA script `DONE`
`eda.py` exists — outputs `eda_popularity.png` (popularity histogram + top-20 genres by median popularity). Standalone, not used by the web app.

---

## Phase 2 — Model Improvements

### Task 2.1 — Popularity tier classifier `INTENTIONAL PIVOT`
Original plan called for a RandomForest classifier saved in `model.pkl` and returned from `/predict`. Decision reversed: tier is now determined **client-side** from the 0–100 display score to avoid overconfidence and simplify the API. Tier labels were also softened ("High potential", "Good appeal", "Low signal"). The `classifier` key is not in `model.pkl`.

### Task 2.2 — Per-genre feature analysis `DONE`
`genre_analysis.py` exists — trains per-genre models for the top genres, saves importances + R² to `genre_analysis.json`. Standalone script, not used by the web app.

### Task 2.3 — Artist average popularity feature `DONE`
- `analyze.py` computes `artist_avg_popularity` as per-artist mean popularity, appended to `FEATURES` at training time
- `model.pkl` payload includes `artist_lookup` dict and `global_avg_popularity`
- `/predict` accepts optional `artist` field; falls back to global average if not found
- Response includes `artist_found` boolean so UI can show "✓ Found in dataset" or "Not in dataset — using global average"

---

## Phase 3 — UI Improvements

### Task 3.1 — Show model accuracy honestly `DONE`
Accuracy card in `index.html` shows:
- R² score and audio-only R² (from `meta.r2_base`)
- Plain-English: "Audio features alone explain ~X% — artist fame adds the remaining ~Y%"
- Unexplained variance attributed to playlist placement, release timing, marketing
- Mean prediction error ±MAE pts
- CV line will auto-appear when Task 1.2 is done (already wired via `meta.cv_r2_mean`)

### Task 3.2 — Genre filter dropdown `DONE`
- `analyze.py` computes `genre_means` (per-genre feature means) → saved in `model.pkl`
- `/genres` endpoint returns sorted genre list
- `/predict` uses genre-specific means for insight comparisons when `genre` is sent
- `index.html` has genre `<select>` that fetches `/genres` on load; genre sent with each predict call
- Insights title updates dynamically: "Your values vs. {genre} average"

### Task 3.3 — Feature importance bar chart `DONE`
- Horizontal bar chart rendered from `meta.importance` in the result panel
- Bars renormalized to audio-only features (excluding `artist_avg_popularity`) so values are readable (~5–25% per feature)
- Legend: "Share of audio impact"
- Bars are 6px tall with inline percentage label

---

## Phase 4 — Code Quality

### Task 4.1 — Shared `config.py` `DONE`
`config.py` defines `SLIDER_FEATURES`, `EXTRA_FEATURES`, `BASE_FEATURES`, `TARGET`. Both `analyze.py` and `APP.py` can import from it (currently `analyze.py` defines its own matching lists — full import hookup is optional cleanup).

---

## Security & Code Quality Audit (2026-04-13) — All Fixed

### Critical
| # | Issue | Status |
|---|-------|--------|
| C1 | Unsafe file extension — no allowlist | DONE — `ALLOWED_AUDIO_EXTENSIONS` frozenset |
| C2 | `debug=True` hardcoded — RCE risk | DONE — controlled via `FLASK_DEBUG` env var |
| C3 | Raw `str(exc)` in API responses — info leak | DONE — generic messages; exceptions logged server-side |

### High
| # | Issue | Status |
|---|-------|--------|
| H1 | `_extract_audio_features` 98 lines — over limit | DONE — split into 8 focused helper functions |
| H2 | No `MAX_CONTENT_LENGTH` — unbounded uploads | DONE — 50 MB cap on `app.config` |
| H3 | `pickle.load()` on unverified file | DONE — SHA-256 check via `MODEL_PKL_SHA256` env var |
| H4 | No `if __name__ == "__main__"` guard in `analyze.py` | DONE |
| H5 | `innerHTML` with server-derived feature names — XSS | DONE — all uses wrapped with `escapeHtml()` |
| H6 | `console.error` in production predict path | DONE — removed |

### Medium
| # | Issue | Status |
|---|-------|--------|
| M1 | `static_folder="."` serves source files over HTTP | DONE — `static_folder=None` |
| M2 | `major_p /= ...` mutates constant in-place | DONE — immutable tuples; new arrays per call |
| M3 | File `<input>` missing `<label>` — accessibility | DONE |
| M4 | "Clear" button missing `aria-label` | DONE |
| M5 | `renderFeatureCards` defined but never called | DONE — dead code removed |
| M6 | `data.error` silently swallowed | DONE — surfaces user-visible message |

### Low
| # | Issue | Status |
|---|-------|--------|
| L1 | 20+ `print()` calls in `analyze.py` | DONE — replaced with `logging` |
| L2 | Magic numbers without named constants | DONE — all extracted to module-level constants |
| L3 | `classifier` loaded from pickle but never used | DONE — not loaded; comment explains |
| L4 | No type annotations on function signatures | DONE |

---

## Audio Analysis Algorithm Improvements (2026-04-13)

All computed in `APP.py` via `librosa`; extracted features match Spotify's schema (0–1 for most, dB for loudness, BPM for tempo).

| Feature | Fix Applied | Status |
|---------|-------------|--------|
| **Tempo** | Replaced `beat_track` with tempogram autocorrelation; half/double-tempo disambiguation via tempogram support score; BPM folded into [60, 165] | DONE |
| **Speechiness** | Recalibrated divisor 3.0 → 9.0; added ZCR as secondary signal; combined MFCC delta + delta² + vocal-band flux + ZCR | DONE |
| **Instrumentalness** | Fixed mathematical bug (`mean(v_i/sum) ≡ 1/n`); switched to absolute MFCC variance of coefficients 1–4 + vibrato detection (4.5–7 Hz autocorrelation of spectral centroid, with rhythmic subdivisions excluded) | DONE |
| **Liveness** | Replaced reverb-decay approach (failed on sustained synth notes) with quiet-section noise floor (DR ratio) + mid-band spectral contrast variation | DONE |
| **Energy** | Active-frame loudness + HF energy ratio (>2 kHz) + spectral centroid — all shared STFT | DONE |
| **Acousticness** | HPSS harmonic ratio + harmonic spectral flatness | DONE |
| **Danceability** | IBI consistency (beat inter-onset variation) + PLP pulse strength | DONE |
| **Valence** | Krumhansl-Kessler key mode + tempo + spectral tilt + harmonic-to-total energy ratio | DONE |
| **Loudness** | Power-weighted active-frame mean; silence gaps below −50 dB excluded | DONE |

---

## UI Redesign Summary (Sections 3–6 of IMPROVEMENTS.md) `DONE`

- **Single-column layout** — `.sliders-panel` hidden; result panel revealed only after upload
- **Upload-only flow** — `uploadedFeatures` state; no startup `predict()` call; genre change guarded
- **Optimal Audio Profile card** — permanent card showing recommended vs. dataset mean for all 9 features
- **Collapsible sliders** — default collapsed; toggle button with chevron
- **Insights panel** — edge-to-edge, 20px feature names, larger badges, 7px bars, 14px values
- **Educational section** — model info, accuracy card, importance chart wrapped in "About the Model & Dataset"
- **Feature descriptions** — `.insight-desc` shown below each feature name in insights rows
- **Typography pass** — font sizes bumped across 26 elements for readability
- **Single tier badge** — replaces five always-dim pills; color/label driven by score range
- **Tier labels softened** — "High potential" / "Good appeal" / "Low signal"

---

## Remaining Work

### Task 6.8 — Processing Time Benchmarks  `TODO`

After installing ML packages, time the full pipeline on a sample file:

```
librosa (4 features):     target < 5s
Whisper tiny (60s clip):  target < 10s
Demucs (20s clip, CPU):   target < 60s
CLAP (30s clip):          target < 15s
Total:                    target < 90s
```

If Demucs on CPU exceeds 60s, reduce the analysis cap from 20s to 10s.
If total exceeds 90s, update the UI loading copy accordingly.

---

## Phase 5 — Audio Analysis & Impact Factor Fixes

> Added: 2026-04-15. Based on user-reported issues: (1) 72 BPM tracks analyzed as 144 BPM; (2) feature impact bars are misleadingly low while artist importance dominates despite never being user-input during audio analysis.

---

### Root Cause Analysis

#### Bug 1 — Tempo 2× Error (`APP.py: _compute_tempo`)

`librosa.beat.beat_track()` frequently locks onto the 8th-note subdivision rather than the quarter-note beat for slow (60–90 BPM) tracks — it returns 144 BPM when the true tempo is 72 BPM.

The existing half-tempo correction (`half_bt = tempo_bt / 2`) only fires when `_tg_score(half_bt) >= _tg_score(tempo_bt) * 0.65`. For slow tracks, the tempogram often has *more* energy at 144 (subdivision) than at 72 (beat), so the 65 % threshold is never met and the wrong value is kept. The fold-down loop (`while tempo_val > 165`) never triggers because 144 < 165, so 144 passes through uncorrected.

Additional weaknesses:
- `_tg_score` reads a single bin; at 72 BPM the energy can be smeared across neighbouring bins because tempogram resolution is coarser at lower BPM.
- The candidate set is limited to beat_track ± one octave; genuine 3:1 or other ratio errors are not handled.
- No independent cross-check (e.g. PLP or onset autocorrelation) validates the final pick.

#### Bug 2 — Feature Importance Misrepresents Audio-Only Context (`analyze.py` + `APP.py`)

`artist_avg_popularity` dominates `model.feature_importances_` (typically 30–55 %) because artist fame is the single strongest predictor of popularity in the dataset. All audio features share the remaining 45–70 %, so each individual feature appears to have low impact (~5–10 %).

This is doubly misleading in the audio-upload flow: when no artist is entered, `artist_avg_popularity` is fixed at `global_avg_pop` (a dataset-wide constant). A constant feature carries **zero** predictive signal at inference time — yet the importance chart still shows it as dominant. Users see audio features with 5–8 % bars and conclude audio characteristics "barely matter", which is incorrect.

The audio-only model `m_base` (trained without `artist_avg_popularity`) already exists inside `analyze.py` but its `feature_importances_` is never saved to `model.pkl`, so `APP.py` cannot use it.

---

### Task 5.1 — Fix Tempo Disambiguation  `DONE`

**File:** `APP.py: _compute_tempo()`

**Changes:**

1. **Lower the half-tempo threshold** from `0.65` → `0.50`. Equal tempogram energy at half-tempo is sufficient evidence to prefer the lower BPM (perceptual research shows listeners naturally gravitate to the slower beat when ambiguous).

2. **Switch from single-bin lookup to windowed-area score.** Replace `_tg_score` with a function that averages the tempogram over a ±8 % BPM window (`bpm * [0.92, 1.08]`) to handle spectral smearing at low frequencies.

3. **Add multi-candidate peak search as a tie-breaker.** After the 1× vs ½× check, extract the top-3 peaks from the mean tempogram. If the winning candidate from `beat_track` does not appear among the top-3 but a half or double does, prefer the top-3 winner that is closest to the fold-down range [60, 165].

4. **Add PLP pulse cross-check.** Compute `librosa.beat.plp()` and extract its dominant period. If the PLP-derived BPM differs from `tempo_bt` by more than 15 % but agrees with `half_bt` within 10 %, prefer `half_bt`.

5. **Keep the existing fold-down loop** as a final safety net, but extend it to also fold up values below `TEMPO_MIN_BPM` (already present — no change needed).

**Expected outcome:** 72 BPM tracks no longer reported as 144 BPM; the correction fires on genuine subdivision-locking cases while leaving correctly-detected tempos unchanged.

---

### Task 5.2 — Save Audio-Only Feature Importances to `model.pkl`  `DONE`

**File:** `analyze.py: train()`

`m_base` is already trained (the audio-only RandomForest used for `r2_base` and `recommended`). Currently its importances are discarded. Add one line to capture them:

```python
audio_importance = pd.Series(m_base.feature_importances_, index=SLIDER_FEATURES + EXTRA_FEATURES)
```

Add `"audio_importance": audio_importance.to_dict()` to the `payload` dict before pickling.

This is a training-time change only; `model.pkl` must be regenerated after.

---

### Task 5.3 — Use Audio-Only Importances in `/predict` When No Artist  `DONE`

**File:** `APP.py`

1. **Load `audio_importance`** from `payload` alongside `importance`:
   ```python
   audio_importance = payload.get("audio_importance", importance)
   ```

2. **In `/predict`:** choose which importance dict to use based on whether an artist was matched:
   - `artist_found = True` → use full-model `importance` (artist is a real signal)
   - `artist_found = False` (no artist entered, or not found in lookup) → use `audio_importance`, renormalized to sum to 1.0 across `slider_features`

3. **Renormalization helper** (a ~5-line pure function):
   - Extract importance values for `slider_features` from `audio_importance`
   - Divide each by their sum so they total 1.0
   - Return this as the importance sub-dict for the insights response

4. **Update `/meta`** to expose `audio_importance` so the frontend importance chart can also switch to audio-only values when appropriate.

**Expected outcome:** In audio-upload mode (no artist), impact bars correctly reflect each audio feature's share of the audio-predictive signal — bars will sum to 100 % and individual feature bars will be meaningfully sized (~8–20 % each).

---

### Task 5.4 — Secondary Audio Algorithm Improvements  `DONE`

Minor accuracy improvements identified during the review:

| Feature | Current Weakness | Proposed Fix |
|---------|-----------------|--------------|
| **Acousticness** | HPSS harmonic ratio is high for synthesizers (they produce tonal harmonics too) | Add spectral flatness of the *full* spectrum as a penalty term: synthesizers have flatter harmonic overtone structure than acoustic instruments. Weight: `harm_ratio * 0.5 + flatness_score * 0.3 + (1 - full_flatness * 10) * 0.2` |
| **Danceability** | `plp_score = mean(pulse)/max(pulse)` underrates polyrhythmic tracks because a sharp PLP peak depresses the ratio | Replace with `np.percentile(pulse, 75) / (np.max(pulse) + 1e-6)` — 75th percentile is more robust to sparse pulse responses |
| **Instrumentalness** | Vibrato detection applied to full `sr=22050` resolution — centroid autocorrelation at full resolution is noisy for short clips | Apply a median filter (kernel=3) to `centroid_frames` before autocorrelation to suppress frame-level jitter |
| **Liveness** | Heavily compressed studio tracks (DR ≈ 8) score falsely high liveness because their dynamic range resembles a live recording | Add a compression proxy: if `std(db_frames[active_mask]) < 4 dB`, apply a 0.3 penalty multiplier to `noise_floor_score` |

---

### Task 5.5 — Retrain and Verify  `DONE`

After Tasks 5.1–5.4:

1. Run `python3 analyze.py` to regenerate `model.pkl` with `audio_importance` included.
2. Verify the logged output shows `audio_importance` keys for all 14 audio features (9 slider + 5 extra).
3. Spot-check tempo on 3–4 files: one slow (≤80 BPM), one fast (≥140 BPM), one mid-range.
4. Confirm `/meta` response includes `audio_importance`.
5. Confirm `/predict` with no artist uses renormalized audio importance; with a known artist uses full importance.
6. Run `python3 test_app.py` — all existing tests must pass.

---

## Phase 6 — ML-based Feature Replacement (Speechiness, Instrumentalness)

> Added: 2026-04-15. Two audio features replaced with ML models. Acousticness, liveness, and valence retain their improved Phase 5 librosa implementations — CLAP (~900 MB) was evaluated but dropped as too heavyweight for a web app context.

### Design Decision: CLAP Dropped

CLAP would have added ~900 MB of model weights for acousticness/liveness/valence. Whisper tiny (~39 MB) + Demucs (~80 MB) = ~120 MB total, which is reasonable. The Phase 5 improved librosa heuristics for acousticness (HPSS + flatness penalty), liveness (DR ratio + compression proxy), and valence (KK key mode + spectral tilt + H/P ratio) are meaningfully better than the originals and add zero weight.

### Feature Strategy

| Feature | Implementation | Package required |
|---------|---------------|-----------------|
| speechiness | Whisper tiny — word rate from transcription | `openai-whisper` |
| instrumentalness | Demucs htdemucs_6s — vocal stem energy ratio | `demucs` + `torch` |
| acousticness | librosa — HPSS harmonic ratio + flatness penalty | none |
| liveness | librosa — DR ratio + compression proxy + mid contrast | none |
| valence | librosa — KK key mode + spectral tilt + H/P ratio | none |
| tempo, loudness, energy, danceability | librosa | none |

### New dependencies

```
openai-whisper    # Whisper speech recognition (~39 MB)
demucs            # Meta source separation (~80 MB)
```

All run **locally**; no API keys. Models are downloaded automatically on first use to `~/.cache/`. Total download: ~120 MB.

Each model is gated behind its own availability flag (`WHISPER_AVAILABLE`, `DEMUCS_AVAILABLE`). If a package is not installed, that feature is not returned by `/analyze-audio` — the UI slider stays at dataset mean and the prediction still runs.

---

### Task 6.1 — Model Loading Infrastructure  `DONE`

**File:** `APP.py`

Add three import blocks (each in a `try/except ImportError`) below the existing `librosa` import block, setting availability flags:

```python
try:
    import whisper as _whisper_lib
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import torch
    from demucs.apply import apply_model as _demucs_apply
    from demucs.pretrained import get_model as _demucs_get_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
```

Add three module-level `None` sentinels for the cached model objects:

```python
_whisper_model = None
_demucs_model  = None
_clap_model    = None
```

Add three lazy-loader functions — each loads once and stores in the sentinel:

```python
def _get_whisper_model():   # loads whisper.load_model("tiny")
def _get_demucs_model():    # loads get_model("htdemucs_6s")
def _get_clap_model():      # loads laion_clap.CLAP_Module; calls .load_ckpt()
```

Log a one-time `INFO` message when each model is loaded for the first time.

---

### Task 6.2 — Speechiness via Whisper  `DONE`

**File:** `APP.py`

**Approach:** Transcribe the first 60 seconds of audio with Whisper (tiny model). Word rate (words per second) is the primary signal. Spotify's scale anchors: `< 0.33` = music/no speech, `0.33–0.66` = mixed, `> 0.66` = mostly spoken word (rap, podcast, comedy).

**Implementation notes:**

- Resample `y` to 16 kHz (Whisper's required sample rate) using `librosa.resample` before passing to Whisper — do NOT write to disk; pass the numpy array directly via `whisper.transcribe(model, audio_array)`.
- Use `word_timestamps=False` for speed (only need segment-level output).
- Count total words across all segments; divide by clip duration to get words/second.
- Normalisation curve: 0 wps → 0.0; 1 wps → ~0.5; 3+ wps → ~1.0. Use `tanh`-style saturation: `min(wps / 3.0, 1.0)` maps the range naturally.
- Cap input at 60 seconds (not 90) — beyond that, speech detection accuracy doesn't improve but compute doubles.

**New constants to add:**

```python
WHISPER_MODEL_SIZE: str    = "tiny"
SPEECHINESS_WPS_SAT: float = 3.0   # words-per-second at which score saturates to 1.0
WHISPER_ANALYSIS_SEC: int  = 60
```

---

### Task 6.3 — Instrumentalness via Demucs  `DONE`

**File:** `APP.py`

**Approach:** Run the `htdemucs_6s` model (6-source: drums, bass, other, vocals, guitar, piano) on a capped clip. Measure the RMS energy of the `vocals` stem relative to the total mix.

```
instrumentalness = 1.0 - clip(vocals_rms / (total_rms + ε), 0, 1)
```

**Processing time concern:** Demucs on CPU runs at ~1–3× real-time for `htdemucs_6s`. At 90s input that's 30–270s — unacceptable. **Cap the analysis window at 20 seconds.** 20s → ~7–60s on CPU; ~2–5s on GPU. If GPU is not available, still acceptable for a web app context.

**Implementation notes:**

- Load the audio segment using librosa (already done), then convert to the tensor format Demucs expects: shape `[batch=1, channels=2, samples]` at 44100 Hz. If the loaded `y` is mono at 22050 Hz, upsample to 44100 and duplicate to stereo.
- Call `apply_model(model, tensor, device="cpu")` — returns a tensor of shape `[1, sources, channels, samples]`. Source order for htdemucs_6s: drums, bass, other, vocals, guitar, piano (index 3 = vocals).
- Compute RMS of `sources[0, 3]` (vocals) and `sources[0].sum(0)` (mix reconstruction) — use the reconstructed mix rather than the original to avoid scale mismatch.
- Clip result to `[0, 1]`.

**New constants:**

```python
DEMUCS_MODEL_NAME: str      = "htdemucs_6s"
DEMUCS_ANALYSIS_SEC: int    = 20
DEMUCS_VOCALS_SOURCE_IDX: int = 3
```

---

### Task 6.4 — Acousticness, Liveness, Valence via CLAP  `DROPPED`

**File:** `APP.py`

**Approach:** Load the LAION-CLAP model once. For each audio clip, compute one audio embedding and three pairs of text embeddings. The score for each feature is the softmax-normalised cosine similarity toward the "positive" prompt vs the "negative" prompt:

```
score = exp(sim_pos) / (exp(sim_pos) + exp(sim_neg))
```

This maps to `[0.5, 1.0)` when pos wins and `(0.0, 0.5]` when neg wins, giving the full `[0, 1]` range in practice.

**Prompt pairs (chosen to be genre-neutral):**

```python
CLAP_PROMPTS: dict[str, tuple[str, str]] = {
    "acousticness": (
        "acoustic instruments folk music natural sound guitar piano violin",
        "electronic music synthesizer digital EDM programmed drums",
    ),
    "liveness": (
        "live concert performance audience applause crowd noise reverb",
        "clean studio recording produced mixing no audience",
    ),
    "valence": (
        "happy cheerful upbeat joyful euphoric positive bright energy",
        "sad melancholic dark gloomy tense angry depressing somber",
    ),
}
```

**Implementation notes:**

- CLAP expects audio at 48 kHz. Resample the 20–30s clip once; reuse the same resampled array for all three features (one audio embedding, three text-pair comparisons).
- Use `get_audio_embedding_from_data` and `get_text_embedding` from the LAION-CLAP API.
- All three scores come from a **single model inference pass** — compute the audio embedding once, then dot-product with the 6 text embeddings (3 pos + 3 neg). This keeps the incremental cost of adding all three features to roughly that of one.
- Cap input at 30 seconds. Beyond 30s, CLAP's global pooling averages over time and additional context doesn't significantly change the embedding.

**New constants:**

```python
CLAP_ANALYSIS_SEC: int = 30
CLAP_SR: int           = 48_000
```

---

### Task 6.5 — Update `_extract_audio_features`  `DONE`

7 features always computed via librosa; 2 ML features conditionally added.
`y_harmonic` and `y_percussive` retained (needed for acousticness and valence).
CLAP gate removed; acousticness/liveness/valence restored as Phase 5 librosa implementations.

---

### Task 6.6 — UI Updates  `DONE`

**File:** `index.html`

1. **Loading copy:** Update the spinner text from `"Analyzing audio… ~15 seconds"` to `"Analyzing audio… up to 60 seconds depending on installed models"`.

2. **Auto-fill indicator:** When sliders are filled from audio, the current code sets their values silently. Add a small `(auto)` label or green dot next to each slider that was auto-filled from audio, so users can distinguish ML-extracted values from default means. Sliders not in the response remain at mean with no indicator.

3. **Feature availability note:** If fewer than 9 features come back from `/analyze-audio`, show a one-line note beneath the upload zone: `"X of 9 features were extracted from audio. Install optional packages for full extraction (see README)."` Count `Object.keys(data.features).length` to determine how many were returned.

---

### Task 6.7 — Tests  `DONE`

**File:** `test_app.py`

1. Remove tests for the 5 deleted librosa functions: `test_compute_speechiness_in_0_1`, `test_compute_instrumentalness_in_0_1`, `test_compute_instrumentalness_not_always_zero`, `test_compute_liveness_in_0_1`, `test_compute_valence_in_0_1`.

2. Add `test_analyze_audio_always_returns_four_base_features`: confirm that `/analyze-audio` always returns at least `tempo`, `loudness`, `energy`, `danceability` regardless of which ML packages are installed.

3. Add `test_analyze_audio_returns_no_deleted_librosa_features`: confirm that the response never contains keys `acousticness_hpss`, `liveness_dr` or any other removed heuristic — specifically that `speechiness`, `instrumentalness`, `liveness`, `acousticness`, `valence` are only present when the corresponding ML package is available (mock `WHISPER_AVAILABLE = False` etc. to test the absent case).

4. Add unit tests for each new compute function (gated with `pytest.importorskip`):
   - `test_compute_speechiness_whisper_returns_zero_for_sine` — a pure sine wave has no speech
   - `test_compute_instrumentalness_demucs_range` — output ∈ [0, 1]
   - `test_compute_clap_features_returns_three_keys` — returns `acousticness`, `liveness`, `valence`

---

### Task 6.8 — Processing Time Benchmarks  `DONE`

After implementation, time the full pipeline on a sample file:

```
librosa (4 features):     target < 5s
Whisper tiny (60s clip):  target < 10s
Demucs (20s clip, CPU):   target < 60s
CLAP (30s clip):          target < 15s
Total:                    target < 90s
```

If Demucs on CPU exceeds 60s, reduce the clip cap from 20s to 10s. If total exceeds 90s, update the UI loading copy accordingly. Document the benchmarks in a comment at the top of `_extract_audio_features`.

---

## Phase 7 — Accuracy & Frontend Cleanup (Code Review 2026-04-16)

> Findings from a focused review of the 9 audio-feature compute functions and the `index.html` UI after the librosa-only revert. Severity tags reflect impact on prediction accuracy or user-visible behaviour.

### Audio Function Fixes

#### Task 7.1 — Raise tempo cap and tighten half-tempo trigger  `CRITICAL` `DONE`

**File:** `APP.py:39-40, 146-148, 192-195`

Current `TEMPO_MAX_BPM = 165.0` plus the fold-down loop forcibly halves any tempo > 165 BPM. Spotify-trained tempo distribution extends to ~240 BPM, so DnB / hardcore / fast techno tracks are reported at half their true BPM and fed into the RF model with the wrong value.

- Raise `TEMPO_MAX_BPM` to `210.0`.
- Lift the windowed half-tempo trigger from `0.50` back to `0.65` (perceptual research level), and gate it behind `len(beat_frames) >= 8` to avoid noisy disambiguation on short or sparse-onset clips.

#### Task 7.2 — Fix `_compute_danceability` PLP scoring direction  `HIGH` `DONE`

**File:** `APP.py:247-249`

`np.percentile(pulse, 75) / np.max(pulse)` collapses to ~0 for tracks with sharp, well-spaced beats (the most danceable case) and inflates for smeared rhythms (least danceable). Replace with beat-frame onset strength:

```python
plp_score = float(np.clip(onset_env[beat_frames].mean() / (onset_env.max() + 1e-9), 0.0, 1.0))
```

#### Task 7.3 — Re-scale spectral centroid term in `_compute_energy`  `HIGH` `DONE`

**File:** `APP.py:225-227`

Spectral centroid divided by Nyquist tops out near 0.36 for typical music, so the `centroid_norm * 0.2` term contributes only 0.03–0.07. Either drop the term or normalize against `nyquist * 0.5` so the value spans [0, 1] as the clip implies.

#### Task 7.4 — Tighten harmonic-exclusion in `_compute_instrumentalness`  `HIGH` `DONE`

**File:** `APP.py:303-314`

For 100–140 BPM tracks, the ±1 Hz exclusion zone wipes out the entire 4.5–7 Hz vibrato band (8th-note rate at 120 BPM = 4 Hz; 16th-note rate = 8 Hz). Vibrato detection becomes effectively disabled.

- Tighten exclusion from `> 1.0` Hz away to `> 0.5` Hz.
- Add `if ac[0] < 1e-6: vibrato_score = 0.0` short-circuit to avoid divide-by-near-zero on near-DC centroids.

#### Task 7.5 — Re-tune `_compute_valence` mode-score mapping  `MEDIUM` `DONE`

**File:** `APP.py:387-388`

`(diff + 0.5) / 1.0` saturates the range — clear major maps to only 0.55–0.90 and ambiguous tracks pile up around 0.5. Use `np.clip((diff + 0.3) / 0.6, 0.0, 1.0)` so the typical signed difference range maps across the full [0, 1].

#### Task 7.6 — Promote `_compute_liveness` magic threshold + clarify mask intent  `MEDIUM` `DONE`

**File:** `APP.py:344-371`

- Promote hardcoded `-45.0` to module-level `LIVENESS_QUIET_DB: float = -45.0`.
- `quiet_mask` (db < -45) and `active_mask` (db > -50) overlap in (-50, -45). Clarify intent — quiet sections should likely be `(-50, -45)` band and active should be `db > -25` to make the DR ratio meaningful.

#### Task 7.7 — Switch `_compute_loudness` to energy-domain mean  `MEDIUM` `DONE`

**File:** `APP.py:200-206`

Arithmetic mean of dB under-weights loud frames vs Spotify's LUFS-style integration. Replace with `10 * np.log10(np.mean(rms[active_mask]**2) + 1e-12)` to better match the training-data convention.

#### Task 7.8 — Add tonality-stability check to `_compute_acousticness`  `MEDIUM` `DONE`

**File:** `APP.py:323-341`

Pure digital synths score ≥ 0.9 acousticness because of low spectral flatness + high HPSS harmonic ratio. Add a third term that distinguishes acoustic sustain from synthetic sustain (e.g. centroid std-dev or attack-time variability) and reduce `harm_ratio` weight accordingly.

#### Task 7.9 — Split `_compute_tempo` into helpers  `LOW` `DONE`

**File:** `APP.py:119-197`

79 lines, above the 50-line guideline. Extract `_score_tempo_with_tempogram`, `_apply_top3_tiebreaker`, `_apply_plp_check` into private helpers.

---

### Frontend Cleanup

#### Task 7.10 — Delete dead slider/feature-card code  `HIGH` `DONE`

**File:** `index.html:171-174, 944-945, 1243-1283, 1834`

`.sliders-panel { display: none; }` and `.feature-cards-section { display: none; }` are never toggled visible. ~250 lines of HTML/JS (`toggleSliders`, `applyRecommended`, `resetSliders`, `buildSliders`, optimize button, feature-cards grid) run with no user-visible effect. Either delete the dead code or restore visibility — pick one.

#### Task 7.11 — Remove non-functional auto-fill dot + 9-feature note  `HIGH` `DONE`

**File:** `index.html:1943-1971`

- `document.querySelector('label[for="slider-${feat}"]')` returns `null` because sliders have `aria-label` only — green dots never appear.
- After the librosa-only refactor `_extract_audio_features` always returns 9 features, so the `extractedCount < 9` branch never fires.

Delete both blocks from `processAudioFile`.

#### Task 7.12 — Use `meta.audio_importance` in slider builder + importance chart  `MEDIUM` `DONE`

**File:** `index.html:1540-1543, 1857`

Backend already exposes `meta.audio_importance` (renormalized to sum to 1 over slider features). Slider impact bars and the importance chart still read `meta.importance` and only filter `artist_avg_popularity` ad-hoc. Switch both to `meta.audio_importance` so percentages sum cleanly to 100 % and reflect the audio-only model.

#### Task 7.13 — Add `aria-live` on desktop score gauge  `MEDIUM` `DONE`

**File:** `index.html:1300-1304, 1673-1680`

Score number lives inside an SVG `<text>`; screen readers don't reliably announce SVG text changes. Mobile sticky bar already has `aria-live="polite"` (L1234) but the desktop gauge does not. Mirror the score into a hidden `sr-only` live region or add `aria-live="polite"` to `.score-card`.

#### Task 7.14 — Remove inline event handlers + add CSP meta tag  `LOW` `DONE`

**File:** `index.html:1218, 1246, 1259-1260, head`

Replace `onclick="..."` with `addEventListener` registrations in the script block, then add `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src fonts.gstatic.com">`.

#### Task 7.15 — Self-host Google Fonts with `font-display: swap`  `LOW` `SKIP`
> Requires downloading font files to disk — deferred; functional impact is minimal.

**File:** `index.html:7`

External CDN adds 3 round trips before first paint. Self-hosting also closes the privacy concern around Google's font CDN.

---

### Implementation Order (Recommended)

1. Critical/High audio fixes first (7.1 → 7.4): single PR, regenerate `model.pkl` is **not** required (these only affect audio extraction at inference time).
2. Frontend cleanup (7.10 → 7.13): single PR, delete-only changes for 7.10/7.11 reduce surface area.
3. Medium/Low items (7.5 → 7.9, 7.14 → 7.15): bundle as time allows.
4. Re-run `python3 -m pytest test_app.py -v` after each step.

---

## Phase 8 — Data Science Portfolio Upgrade

> Added: 2026-04-17. Transforms the project from a working web app into a portfolio-quality data science project. Fixes data leakage, adds comprehensive EDA notebook, model comparison, SHAP interpretability, residual analysis, and statistical rigor.

### New Dependencies

```
xgboost>=2.0,<3.0
shap>=0.44,<1.0
statsmodels>=0.14,<1.0
jupyter>=1.0             # dev only, not on Render
matplotlib>=3.7,<4.0
seaborn>=0.13,<1.0
```

---

### Phase 8A — Fix Data Leakage (CRITICAL)

> `artist_avg_popularity` is currently computed from the full dataset (including test rows) before the train/test split at `analyze.py:47-48`. This inflates R² because test-set artist averages include test-set popularity values. The cross-validation at line 73 also uses this leaked feature.

#### Task 8A.1 — Create `ArtistAvgTransformer` (sklearn TransformerMixin)  `DONE`

**File:** `analyze.py`

Create a class that inherits from `BaseEstimator` + `TransformerMixin`:
- `fit(self, X, y)` — computes mean popularity per artist from **training data only**, stores in `self.artist_means_` dict + `self.global_mean_` fallback.
- `transform(self, X)` — maps each row's artist to the learned mean (unseen artists get `global_mean_`), appends `artist_avg_popularity` column, drops the `artists` column.

Must be picklable for `model.pkl`.

#### Task 8A.2 — Rewrite train/test split to use the transformer  `DONE`

**File:** `analyze.py`

Replace lines 47-58: split FIRST, then fit transformer on training data only, transform both train and test sets. Update `artist_lookup` and `global_avg_popularity` in `model.pkl` to use training-only means.

#### Task 8A.3 — Fix cross-validation to use a Pipeline  `DONE`

**File:** `analyze.py`

Replace the `cross_val_score` call (lines 73-76) with `sklearn.pipeline.Pipeline([("artist_avg", ArtistAvgTransformer()), ("rf", RandomForestRegressor(...))])`. Pass raw X (with `artists` column, without `artist_avg_popularity`) so the transformer recomputes per fold.

#### Task 8A.4 — Add leakage-fix tests  `DONE`

**File:** `test_app.py`

- `test_artist_avg_transformer_unseen_artist_gets_global_mean`
- `test_artist_avg_transformer_no_leakage` — verify transform output matches training-set mean, not full-dataset mean.
- `test_pipeline_cv_runs_without_error`

#### Task 8A.5 — Regenerate model.pkl and verify  `DONE`

Run `python3 analyze.py`. Expect R² to drop (leakage was inflating it). Verify all tests pass and Flask app works.

---

### Phase 8B — Comprehensive EDA Notebook

#### Task 8B.1 — Create `analysis.ipynb` with EDA sections  `DONE`

**File:** `analysis.ipynb` (NEW)

Sections with narrative markdown + code + interpretation:
1. **Introduction & Dataset Overview** — shape, dtypes, describe(), target definition
2. **Target Variable Analysis** — popularity distribution, heavy left skew, implications
3. **Audio Feature Distributions** — 3x3 histogram grid, note skew patterns
4. **Correlation Analysis** — heatmap (seaborn), strongest correlations, note that NO audio feature correlates strongly with popularity
5. **Multicollinearity Check (VIF)** — `statsmodels` VIF table, flag VIF > 10 (energy/loudness), discuss tree vs. linear model implications
6. **Genre Analysis** — genre distribution, top/bottom by median popularity, boxplots
7. **Artist Analysis** — tracks per artist, artist prolificacy vs popularity, why artist identity dominates
8. **Outlier Investigation** — tracks with popularity 0, tracks > 90, shared characteristics
9. **Feature-Popularity Relationships** — scatter/hexbin plots, confirm weak audio→popularity signal
10. **Hypothesis Tests** — "Are explicit tracks more popular?" (t-test), major vs minor key (t-test), time signature differences (ANOVA)

#### Task 8B.2 — "Why audio-only R² = 0.08" investigation section  `DONE`

**File:** `analysis.ipynb`

The most important analytical section:
1. State finding: audio features alone explain ~8% of variance
2. Evidence: audio-only model, predicted vs actual scatter (looks like a cloud)
3. Investigate: univariate R² per feature (all near zero), show artist feature jumps R² to ~0.50+
4. Conclusion: popularity is driven by artist fame, marketing, playlists — not audio characteristics
5. Reference external research on streaming popularity drivers

---

### Phase 8C — Model Comparison & Tuning

#### Task 8C.1 — Baseline model  `DONE`

**File:** `analysis.ipynb`

`DummyRegressor(strategy="mean")` — R² = 0.0 by definition. All real models must beat this. Report R², MAE.

#### Task 8C.2 — Train and compare 4 models  `DONE`

**File:** `analysis.ipynb`

Using leakage-free Pipeline from Phase 8A:
1. DummyRegressor (baseline)
2. Ridge Regression (+ StandardScaler in Pipeline)
3. RandomForestRegressor
4. XGBRegressor

Report per model: R², MAE, 5-fold CV R² (mean ± std), training time. Present as comparison table + bar chart.

#### Task 8C.3 — Hyperparameter tuning  `DONE`

**File:** `analysis.ipynb`

`RandomizedSearchCV` with `n_iter=50, cv=5, scoring="r2"` on the best model.

RandomForest search space: `n_estimators` [100,200,500], `max_depth` [10,20,30,None], `min_samples_split` [2,5,10], `min_samples_leaf` [1,2,4], `max_features` ["sqrt","log2",0.5].

XGBoost search space: `n_estimators` [100,200,500], `max_depth` [3,5,7,10], `learning_rate` [0.01,0.05,0.1,0.2], `subsample` [0.6,0.8,1.0], `colsample_bytree` [0.6,0.8,1.0].

Report best params, best CV R², improvement over defaults.

#### Task 8C.4 — Update `analyze.py` with best model  `DEFERRED`

**File:** `analyze.py`

Update model class and hyperparameters to match the winner from 8C.3.

---

### Phase 8D — Model Interpretability

#### Task 8D.1 — SHAP analysis  `DONE`

**File:** `analysis.ipynb`

1. `shap.TreeExplainer` on best tree model
2. SHAP values on 1000-row test sample
3. Plots: summary plot, bar plot, dependence plots for top 3 features
4. Markdown interpretation of which features push predictions up/down

#### Task 8D.2 — Partial Dependence Plots  `DONE`

**File:** `analysis.ipynb`

`sklearn.inspection.PartialDependenceDisplay` — 2x2 grid for top 4 audio features. Shows marginal effect of each feature on prediction.

#### Task 8D.3 — Residual Analysis  `DONE`

**File:** `analysis.ipynb`

1. Predicted vs Actual scatter (hexbin for 18k points)
2. Residual distribution histogram (check normality, centering)
3. Residuals vs Predicted (check heteroscedasticity)
4. MAE by popularity range (0-20, 20-40, 40-60, 60-80, 80-100)
5. MAE by genre (top 10)

---

### Phase 8E — Statistical Rigor

#### Task 8E.1 — Correlation matrix with p-values  `DONE`

**File:** `analysis.ipynb`

`scipy.stats.pearsonr` for each feature pair. Highlight correlations that are statistically significant (p < 0.05) but practically weak (|r| < 0.3). Explain this large-sample-size phenomenon.

#### Task 8E.2 — VIF multicollinearity analysis  `DONE`

**File:** `analysis.ipynb`

`statsmodels.stats.outliers_influence.variance_inflation_factor` table. Flag VIF > 10. Discuss tree-model robustness vs linear-model sensitivity.

#### Task 8E.3 — Confidence intervals + paired t-test  `DONE`

**File:** `analysis.ipynb`

95% CI on 5-fold CV scores: `mean ± t(0.025, df=4) × std / sqrt(5)`. Paired t-test (`scipy.stats.ttest_rel`) between top-2 models' CV scores to test significance of the difference.

---

### Phase 8F — Documentation & Polish

#### Task 8F.1 — Update README with analysis findings  `DONE`

**File:** `README.md`

Add "Analysis & Findings" section: leakage fix, key EDA findings, model comparison results, honest R² numbers, link to `analysis.ipynb`.

#### Task 8F.2 — Notebook narrative cleanup  `DONE`

**File:** `analysis.ipynb`

Ensure every code cell has a preceding markdown explanation and following interpretation. Verify `Restart & Run All` succeeds. Target 15-20 sections, ~80-100 cells.

#### Task 8F.3 — Update projectProgress.md  `DONE`

**File:** `projectProgress.md`

Mark tasks complete as they are finished.

---

### Implementation Order

```
Phase 8A (leakage fix) ──── MUST be first
    │
    v
Phase 8B (EDA notebook) ───────────┐
    │                               │
    v                               v
Phase 8C (model comparison)     Phase 8E (statistical rigor)
    │
    v
Phase 8D (interpretability)
    │
    v
Phase 8F (documentation)
```

### Success Criteria

- [x] `artist_avg_popularity` computed from training data only (no leakage)
- [x] Cross-validation uses Pipeline (artist averages recomputed per fold)
- [x] `analysis.ipynb` has 12+ sections with narrative markdown (19 sections)
- [x] Correlation heatmap + VIF table with interpretation
- [x] 4+ models compared (baseline, Ridge, RF, XGBoost)
- [x] Hyperparameter tuning with documented best params
- [x] SHAP summary + dependence plots
- [x] Residual analysis (predicted vs actual, error by range, error by genre)
- [x] "Why R² = 0.08" section with evidence and reasoning
- [x] All tests pass (38/38), notebook runs top-to-bottom without errors

### Phase 8 Completion Notes (2026-04-21)

**Committed:** `078173a` — pushed to `main` on GitHub.

**Key metrics after leakage fix:**
- R² = 0.446 (full model), CV R² = 0.439 +/- 0.008
- Audio-only R² = 0.257, MAE = 9.7 pts
- artist_avg_popularity dominates at 79.7% importance

**Bugs fixed during implementation:**
- `cross_val_score` with default (non-shuffled) KFold on ordered data produced R² = -29; fixed with `KFold(shuffle=True)`
- `genre_means` and `ranges` dict referenced `artist_avg_popularity` column not in original df; fixed to use base features + training-only stats
- RandomizedSearchCV `n_iter=50` timed out at 600s; reduced to `n_iter=20` with max `n_estimators=200`

**Files added/changed:**
- `analyze.py` — ArtistAvgTransformer, Pipeline CV, shuffled KFold, fixed genre_means/ranges
- `test_app.py` — 3 new leakage tests (38 total)
- `analysis.ipynb` — 19-section EDA notebook (executed, outputs preserved)
- `requirements-analysis.txt` — notebook-only dependencies
- `README.md` — updated metrics, added Analysis & Findings section
- `.gitignore` — added `eda_*.png`

**Deployment:** Render service needs redeployment — previous service was replaced by another project. Render MCP server configured locally for future deploys.

---

## Phase 9 — Render Deployment Fix

> Added: 2026-04-21. The Render-hosted site at `music-popularity-predictor.onrender.com` is not serving our app. Instead it returns a JSON response from a different FastAPI project (`/predict`, `/retrain`, `/docs`, `/redoc`). The service was either replaced or misconfigured.

### Task 9.1 — Diagnose Render Build Failure  `DONE`

**Findings (2026-04-22):**

Service `music-popularity-analysis` (`srv-d7ji7bd7vvec738v8630`) exists and is connected to the correct repo (`allentianlechen/music-popularity-analysis`), branch `main`, with auto-deploy enabled.

Build failed at `python3 analyze.py` step with:
```
ModuleNotFoundError: No module named 'pandas'
```

**Root causes identified:**
1. `pandas` was missing from `requirements.txt` — `analyze.py` imports it but it was never listed
2. Render used Python 3.14.3 (its current default) instead of 3.11.9 — the `PYTHON_VERSION` env var was in `render.yaml` but the yaml wasn't linked to the service (service name mismatch: yaml said `music-popularity-predictor`, service is `music-popularity-analysis`)
3. No `.python-version` file existed as a fallback

The numpy `<2.0` constraint forced building from source on Python 3.14 (no prebuilt wheel for cp314), causing the metadata step to take 4+ minutes before eventually succeeding — but then `analyze.py` crashed on missing pandas.

### Task 9.2 — Fix and Redeploy  `DONE`

**Fixes applied:**
- Added `pandas>=2.0,<3.0` to `requirements.txt`
- Created `.python-version` file with `3.11.9`
- Fixed `render.yaml` service name from `music-popularity-predictor` → `music-popularity-analysis`
- Set `PYTHON_VERSION=3.11.9` env var directly on Render service via MCP

### Task 9.3 — Update README with new URL  `DONE`

Updated `README.md` live demo link from `music-popularity-predictor.onrender.com` → `music-popularity-analysis.onrender.com`.
