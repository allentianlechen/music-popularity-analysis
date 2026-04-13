# IMPROVEMENTS — Status Log

All fixes have been applied. Model retrained and server restarted.

---

## SECTION 1 — `analyze.py` fixes

### FIX 1.1 ✅ DONE
**Recommended values bias fix.**
`recommended` now computed from the audio-only model (`m_base`) top-1% predictions instead of the full model. Prevents artist fame from biasing what "optimal audio" looks like. Only covers `SLIDER_FEATURES` (the 9 sliders), not `artist_avg_popularity`.

---

## SECTION 2 — `APP.py` fixes

### FIX 2.1 ✅ DONE
**Remove dead classifier tier from `/predict`.**
`tier_label` is no longer predicted or returned in the API response. Tier is determined client-side from the 0–100 display score. Removed the `if tier_label is not None` branch from the response.

### FIX 2.2 ✅ DONE
**Artist lookup feedback in `/predict`.**
Added `artist_found` boolean to response when an artist name is provided. `True` if artist is in the lookup table, `False` if falling back to global average. Frontend uses this to display "✓ Found in dataset" or "Not in dataset — using global average."

---

## SECTION 3 — `index.html` fixes (round 1)

### FIX 3.1 ✅ DONE
**Hero description updated.** "Tune the sliders below" → "Adjust the sliders manually or upload your track — the model scores it either way."

### FIX 3.2 ✅ DONE
**Path tags renamed.** "Path A" → "Manual", "Path B" → "Upload". Upload section title stripped of "Path B —" prefix.

### FIX 3.3 ✅ DONE
**Button renamed + explanation note.** "Optimize for popularity" → "Apply hit-track profile". Added amber note below sliders that appears on click, explaining audio-only scope.

### FIX 3.4 ✅ DONE
**Reset button added.** "Reset" button placed next to "Apply hit-track profile". Calls `resetSliders()` which restores all sliders to dataset mean and hides the amber note.

### FIX 3.5 ✅ DONE
**Artist lookup feedback in UI.** Status `<div>` added below artist input. Shows green "✓ Found in dataset" or amber "Not in dataset — using global average" after each prediction.

### FIX 3.6 ✅ DONE
**Genre label clarified.** "Genre context (optional)" → "Genre — affects insights only" to communicate that genre does not change the score.

### FIX 3.7 ✅ DONE
**Five tier pills replaced with single badge.** The five always-visible dim pills removed. Single `#tier-badge` span now renders with color/label matching the current score tier. `updateTiers` rewritten to drive the badge.

### FIX 3.8 ✅ DONE
**Insights title updates dynamically.** Added `id="insights-title"`. On each predict, title updates to "Your values vs. {genre} average" when a genre is selected, or reverts to "dataset average."

### FIX 3.9 ✅ DONE
**Feature cards title plain English.** "Feature Fitness Scores" → "How close each feature is to optimal."

### FIX 3.10 ✅ DONE
**Feature cards disclaimer clarified.** Added note: "Feature scores show closeness to typical hit-track values — they do not directly predict popularity." Disclaimer block restyled with border and subtle background.

### FIX 3.11 ✅ DONE
**Upload spinner time estimate.** "Analyzing audio…" → "Analyzing audio… ~15 seconds" to set expectations during librosa extraction.

### FIX 3.12 ✅ DONE
**Impact bars taller + show percentage.** Bar height 3px → 6px. Each bar now has an inline `${impPct}%` label to the right, making bar values readable.

### FIX 3.13 ✅ DONE
**Section legend updated.** Replaced "Impact on score" dot-legend (not connected to bars) with "Sorted by model weight" text label.

### FIX 3.14 ✅ DONE
**Tier labels softened.** "Charting potential" → "High potential", "Solid mainstream appeal" → "Good appeal", "Very low popularity" → "Low signal." Reduces overconfidence relative to true Spotify popularity.

---

## SECTION 4 — `index.html` new features (round 2)

### FIX 4.1 ✅ DONE
**Impact bars renormalized to audio-only features.** `impPct` now divided by sum of audio-feature importances (excluding `artist_avg_popularity`), so bars show share of audio impact (~5–25%) rather than near-zero share of total model importance. Legend updated to "Share of audio impact."

### FIX 4.2 ✅ DONE
**Permanent "Optimal Audio Profile" card.** New card (`#rec-profile-card`) inserted between score card and insights card. Shows recommended value + dataset mean for all 9 slider features as a bar chart. Populated from `meta.recommended` at startup via `renderRecommendedProfile()`.

### FIX 4.3 ✅ DONE
**Sliders collapsible, default collapsed.** Slider container hidden by CSS (`max-height:0`), toggled open by `toggleSliders()`. "Audio Features" label replaced with a toggle button showing a chevron. Filter row (artist + genre) remains always visible.

### FIX 4.4 ✅ DONE
**Typography pass — font sizes increased for readability.** 26 sub-fixes (A–BB) bumping mono labels from 10→12px, body text from 11–12→13px across: hero eyebrow, step-num, section-label, btn-optimize, filter-label, filter-input, slider-desc, impact-label, score-label, insights-title, insight-feat, insight-tag, cmp-label, cmp-val, stat-key, loading, upload-title, drop-zone-text, drop-zone-sub, upload-spinner, upload-error, feature-cards-title, btn-clear, fc-name, fc-raw, imp-title, imp-feat-name, imp-pct, path-tag, artist-status, optimize-note, tier-badge, impact-pct span.

---

## SECTION 5 — Full page redesign (upload-only, single-column)

### FIX 5.1 ✅ DONE
**Hero updated for upload-only flow.** Description updated. Manual steps-path row removed; only Upload path shown. Mobile sticky bar and score card initial text updated to "Upload a track to predict/get a prediction."

### FIX 5.2 ✅ DONE
**Single-column layout.** `.layout` changed from 2-column grid to `max-width:860px` block. `.sliders-panel` hidden (`display:none`) but kept in DOM for upload flow data. `.result-panel` made static (non-sticky). `.score-card` centered with `max-width:480px`.

### FIX 5.3 ✅ DONE
**Result panel hidden by default.** `id="result-panel"` added, `style="display:none"`. Panel revealed only after audio upload + predict completes.

### FIX 5.4 ✅ DONE
**`uploadedFeatures` state variable.** Added module-level `uploadedFeatures = null`. Startup `predict()` call removed. Genre change listener guarded to only run `predict()` if `uploadedFeatures` is set.

### FIX 5.5 ✅ DONE
**Upload flow stores features and reveals result panel.** After successful extraction, `uploadedFeatures = feats` set, `predict()` awaited, then result panel shown and scrolled into view. Clear upload resets `uploadedFeatures` to `null` and hides result panel without calling predict.

### FIX 5.6 ✅ DONE
**Insights panel redesigned (Image #2 style).** Insights card background/border removed (edge-to-edge). Feature names enlarged to 20px DM Sans. ABOVE/BELOW AVERAGE badge larger (5px 14px padding, 6px radius). Comparison bars taller (7px). Value text increased to 14px. Generous row padding (22px vertical).

### FIX 5.7 ✅ DONE
**Educational section wrapper.** `model-info`, `accuracy-info`, `importance-chart` wrapped in `<div class="edu-section">` with title "About the Model & Dataset." Separated from insights content by a top border.

---

## SECTION 6 — UI polish + audio analysis accuracy

### FIX 6.1 ✅ DONE
**Remove redundant `renderFeatureCards` call.** Removed `if (meta) renderFeatureCards(feats)` from `processAudioFile`. Feature-cards grid stays hidden; results panel (score + insights) is the only breakdown shown after upload.

### FIX 6.2 ✅ DONE
**Feature descriptions in insights rows.** Added `.insight-desc` CSS (13px DM Sans, light weight). Insight-top template updated to wrap feature name + description in a column div, showing `DESCRIPTIONS[feat]` below the feature name in each insight row.

### FIX 6.3 ✅ DONE
**Tempo octave error fix.** Replaced `librosa.beat.beat_track` with `librosa.feature.tempo` (tempogram autocorrelation, `start_bpm=100`) for more stable BPM estimation on slow/ambient/non-4/4 music. Post-process folds BPM into [60, 165] to suppress half/double-tempo errors. `onset_env` now computed once and reused by danceability (removed duplicate computation).

### FIX 6.4 ✅ DONE
**Speechiness calibration fix.** Recalibrated divisor 3.0 → 9.0 (pure instruments produce delta_mean ~0.5–3; speech ~5–9). Added Zero Crossing Rate as secondary signal (ZCR higher for speech than sustained instruments). Combined: `delta_mean/9.0 * 0.6 + zcr_mean/0.15 * 0.4`.

### FIX 6.5 ✅ DONE
**Instrumentalness mathematical bug fix.** Previous formula `mean(v_i / sum(v_i))` always evaluates to `1/n = 0.2`, making `instrumentalness ≡ 0` for every track. Fixed to use absolute temporal MFCC variance of MFCCs 1–4: `1.0 - mfcc_abs_var / 300.0`. Empirical scale: pure instrument ~20–200, vocal ~200–800.

---

## Post-execution

- `python3 analyze.py` — ✅ ran, model retrained. Recommended values from audio-only model: danceability 0.640, energy 0.653, loudness -6.532 dB, speechiness 0.080, acousticness 0.221, instrumentalness 0.028, liveness 0.162, valence 0.469, tempo 120 BPM.
- `python3 APP.py` — ✅ server running on port 8080.

---

## CODE REVIEW — Security & Quality Audit (2026-04-13)

> Automated review of APP.py, analyze.py, index.html.
> CRITICAL and HIGH issues fixed in the same session (see status below).

### CRITICAL

| # | File | Location | Issue | Status |
|---|------|----------|-------|--------|
| C1 | `APP.py` | L168 | **Unsafe file extension from `upload.filename`** — no allowlist; `.py`/`.pkl` files pass through to librosa | ✅ Fixed |
| C2 | `APP.py` | L271 | **`debug=True` in `app.run()`** — Werkzeug interactive debugger = RCE if server reachable | ✅ Fixed |
| C3 | `APP.py` | L178, 266 | **Raw `str(exc)` in API responses** — leaks file paths, library versions, temp file names | ✅ Fixed |

### HIGH

| # | File | Location | Issue | Status |
|---|------|----------|-------|--------|
| H1 | `APP.py` | L49–147 | `_extract_audio_features` is 98 lines — exceeds 50-line limit | ✅ Fixed |
| H2 | `APP.py` | — | No `MAX_CONTENT_LENGTH` — server buffers full upload before size check fires | ✅ Fixed |
| H3 | `APP.py` | L27 | `pickle.load()` on unverified file — tampered model executes arbitrary code at startup | ✅ Fixed |
| H4 | `analyze.py` | L16 | No `if __name__ == "__main__"` guard — full training runs on import | ✅ Fixed |
| H5 | `index.html` | L1749, 1916 | `innerHTML` with server-derived feature names — XSS if model.pkl is replaced | ✅ Fixed |
| H6 | `index.html` | L1768 | `console.error` in production predict path — leaks stack traces to browser console | ✅ Fixed |

### MEDIUM

| # | File | Issue | Status |
|---|------|-------|--------|
| M1 | `APP.py` | `static_folder="."` serves model.pkl, CSVs, source files at `/static/` | ✅ Fixed |
| M2 | `APP.py` | `major_p /= ...` mutates constant in-place — data race if threaded | ✅ Fixed |
| M3 | `index.html` | File `<input>` has no `<label>` — accessibility | ✅ Fixed |
| M4 | `index.html` | "Clear" button missing `aria-label` | ✅ Fixed |
| M5 | `index.html` | `renderFeatureCards` defined but never called — dead code | ✅ Fixed |
| M6 | `index.html` | `if (data.error) return;` swallows errors silently — no user feedback | ✅ Fixed |

### LOW

| # | File | Issue | Status |
|---|------|-------|--------|
| L1 | `analyze.py` | 20+ `print()` calls — should use `logging` | ✅ Fixed |
| L2 | `APP.py` | Magic numbers without named constants | ✅ Fixed |
| L3 | `APP.py` | `classifier` loaded from pickle but never used | ✅ Fixed |
| L4 | `APP.py` | No type annotations on function signatures | ✅ Fixed |
