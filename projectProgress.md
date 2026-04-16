# analysisProject — Progress Tracker

> Consolidated from PROJECT_PLAN.md, IMPROVEMENTS.md, memory, and verified against current code.
> Last updated: 2026-04-15

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
