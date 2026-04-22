"""
app.py — Step 3: Web Server
============================
Serves the webpage and handles popularity predictions.
Install: pip3 install flask scikit-learn
Run:     python3 APP.py
Then open: http://127.0.0.1:8080
"""

import hashlib
import hmac
import logging
import os
import sys
import tempfile
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from joblib import load as joblib_load

# Lazy-load librosa (and numpy/scipy) only when audio analysis is requested.
# This keeps startup RAM under Render's 512 MB free-tier limit.
LIBROSA_AVAILABLE: bool = False
librosa: Any = None
np: Any = None
median_filter: Any = None
soundfile: Any = None
soxr: Any = None


def _ensure_librosa() -> bool:
    """Import librosa on first use. Returns True if available."""
    global LIBROSA_AVAILABLE, librosa, np, median_filter, soundfile, soxr  # noqa: PLW0603
    if LIBROSA_AVAILABLE:
        return True
    try:
        # Disable numba JIT at runtime to avoid OOM from compilation on 512 MB
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        import librosa as _librosa
        import numpy as _np
        import soundfile as _sf
        import soxr as _soxr
        from scipy.ndimage import median_filter as _mf
        librosa = _librosa
        np = _np
        median_filter = _mf
        soundfile = _sf
        soxr = _soxr
        LIBROSA_AVAILABLE = True
        logger.info("librosa loaded on first audio request")
        return True
    except ImportError:
        return False


# Target sample rate for audio analysis
_TARGET_SR: int = 22050
# Max audio duration in seconds (30s is enough for feature extraction)
_MAX_DURATION_SEC: float = 30.0


# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
UPLOAD_MAX_BYTES: int        = 50 * 1024 * 1024          # 50 MB
TEMPO_MIN_BPM: float         = 60.0
TEMPO_MAX_BPM: float         = 210.0
TEMPO_START_BPM: int         = 100
LOUDNESS_MIN_DB: float       = -60.0
LOUDNESS_MAX_DB: float       = 0.0
SILENCE_THRESHOLD_DB: float  = -50.0
INSTR_VAR_DIVISOR: float     = 300.0
MFCC_DELTA_NORM: float       = 5.0
MFCC_DELTA2_NORM: float      = 8.0
ZCR_NORM: float              = 0.15
LIVENESS_QUIET_DB: float     = -45.0   # frames below this are considered quiet
LIVENESS_ACTIVE_DB: float    = -25.0   # frames above this are considered active

ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
)


# Krumhansl-Kessler key profiles (immutable tuples — never mutated in-place)
_MAJOR_PROFILE: tuple[float, ...] = (
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
)
_MINOR_PROFILE: tuple[float, ...] = (
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
)

# ── FLASK APP ─────────────────────────────────────────────────────────────────
# static_folder=None disables the automatic /static/ route so that source files,
# model.pkl, and CSVs are not inadvertently served over HTTP.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_BYTES  # 413 before route handler

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
def _load_model(path: str = "model.pkl") -> dict[str, Any]:
    """Load model.pkl via joblib, optionally verifying SHA-256."""
    expected_hash = os.environ.get("MODEL_PKL_SHA256")
    if expected_hash:
        with open(path, "rb") as f:
            raw = f.read()
        actual = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual, expected_hash):
            logger.error("model.pkl SHA-256 mismatch — refusing to load")
            sys.exit(1)
        logger.info("model.pkl integrity verified")
    return joblib_load(path)


try:
    payload = _load_model()
except FileNotFoundError:
    logger.error("model.pkl not found. Run 'python3 analyze.py' first.")
    sys.exit(1)

model            = payload["model"]
features         = payload["features"]
slider_features  = payload.get("slider_features", features[:9])
importance       = payload["importance"]
audio_importance = payload.get("audio_importance", importance)
ranges           = payload["ranges"]
r2               = payload["r2"]
mae              = payload["mae"]
pred_min         = payload.get("pred_min", 0)
pred_max         = payload.get("pred_max", 100)
recommended      = payload.get("recommended", {})
global_avg_pop   = payload.get("global_avg_popularity", 0)
genre_means      = payload.get("genre_means", {})
# classifier is intentionally not extracted — tier is determined client-side


def _audio_importance_normalized() -> dict[str, float]:
    """Return audio_importance for slider_features only, renormalized to sum to 1.0."""
    raw   = {f: audio_importance.get(f, 0.0) for f in slider_features}
    total = sum(raw.values()) or 1.0
    return {f: v / total for f, v in raw.items()}


# ── ML MODEL LAZY LOADERS ─────────────────────────────────────────────────────



# ── AUDIO FEATURE HELPERS ─────────────────────────────────────────────────────

def _score_tempo_bpm(bpm: float, tg_freqs: "np.ndarray", mean_tg: "np.ndarray") -> float:
    """Mean tempogram energy in a ±8 % BPM window (robust to spectral smearing at low BPM)."""
    lo   = bpm * 0.92
    hi   = bpm * 1.08
    mask = (tg_freqs >= lo) & (tg_freqs <= hi)
    if not np.any(mask):
        return float(mean_tg[int(np.argmin(np.abs(tg_freqs - bpm)))])
    return float(np.mean(mean_tg[mask]))


def _tempo_top3_tiebreaker(
    tempo_val: float, tempo_bt: float, half_bt: float,
    tg_freqs: "np.ndarray", mean_tg: "np.ndarray",
) -> float:
    """Refine tempo using top-3 tempogram peaks as a tie-breaker."""
    valid_mask  = (tg_freqs >= TEMPO_MIN_BPM) & (tg_freqs <= TEMPO_MAX_BPM)
    valid_freqs = tg_freqs[valid_mask]
    valid_tg    = mean_tg[valid_mask]
    if len(valid_tg) < 3:
        return tempo_val
    top3_bpms = valid_freqs[np.argsort(valid_tg)[-3:][::-1]]
    in_top3   = any(abs(tempo_val - p) / (tempo_val + 1e-6) < 0.08 for p in top3_bpms)
    if not in_top3:
        for cand in (half_bt, tempo_bt * 2.0):
            if (TEMPO_MIN_BPM <= cand <= TEMPO_MAX_BPM
                    and any(abs(cand - p) / (cand + 1e-6) < 0.08 for p in top3_bpms)):
                return cand
        return float(top3_bpms[0])
    return tempo_val


def _tempo_plp_check(
    tempo_val: float, tempo_bt: float, half_bt: float,
    onset_env: "np.ndarray", sr: int,
) -> float:
    """If onset-envelope periodicity agrees with half-tempo but tempo does not, prefer half.

    Uses autocorrelation of onset envelope instead of librosa.beat.plp
    to avoid numba @guvectorize which OOMs on Render 512 MB.
    """
    hop        = 512
    fps        = sr / hop
    ac         = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    if len(ac) < 2:
        return tempo_val
    plp_period = float(np.argmax(ac[1:]) + 1)
    plp_bpm    = 60.0 * fps / (plp_period + 1e-6)
    while plp_bpm > TEMPO_MAX_BPM:
        plp_bpm /= 2.0
    while plp_bpm < TEMPO_MIN_BPM:
        plp_bpm *= 2.0
    if (abs(plp_bpm - tempo_bt) / (tempo_bt + 1e-6) > 0.15
            and TEMPO_MIN_BPM <= half_bt <= TEMPO_MAX_BPM
            and abs(plp_bpm - half_bt) / (half_bt + 1e-6) < 0.10):
        return half_bt
    return tempo_val


def _estimate_beat_frames(
    onset_env: "np.ndarray", sr: int, tempo_bpm: float,
) -> "np.ndarray":
    """Estimate beat-aligned frames from onset envelope and tempo.

    librosa.beat.beat_track uses numba @guvectorize which OOMs on Render's
    512 MB free tier.  This replacement uses onset peaks snapped to the
    estimated beat grid — good enough for IBI consistency and onset-strength
    scoring in _compute_danceability.
    """
    hop = 512
    beat_period_frames = (60.0 * sr) / (tempo_bpm * hop + 1e-9)
    if beat_period_frames < 1:
        return np.array([], dtype=int)
    # Generate a regular grid at the estimated tempo
    n_frames = len(onset_env)
    grid = np.arange(0, n_frames, beat_period_frames).astype(int)
    grid = grid[grid < n_frames]
    # Snap each grid point to the nearest onset peak within ±half a beat
    half = max(1, int(beat_period_frames * 0.4))
    snapped: list[int] = []
    for g in grid:
        lo = max(0, g - half)
        hi = min(n_frames, g + half + 1)
        snapped.append(int(lo + np.argmax(onset_env[lo:hi])))
    return np.unique(np.array(snapped, dtype=int))


def _compute_tempo(y: "np.ndarray", sr: int) -> tuple[float, "np.ndarray"]:
    """Return (tempo_bpm, beat_frames). Multi-stage disambiguation:
    1. Windowed tempogram score (±8 % BPM window) via _score_tempo_bpm.
    2. Half-tempo preferred at ≥65 % relative support, gated on ≥8 beat frames.
    3. Top-3 tempogram peaks as tie-breaker via _tempo_top3_tiebreaker.
    4. PLP cross-check via _tempo_plp_check.

    Uses librosa.feature.tempo + onset envelope instead of librosa.beat.beat_track
    to avoid numba @guvectorize which OOMs on Render 512 MB.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_bt = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr, start_bpm=TEMPO_START_BPM)
    )[0])
    beat_frames = _estimate_beat_frames(onset_env, sr, tempo_bt)

    tg        = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tg_freqs  = librosa.tempo_frequencies(tg.shape[0], sr=sr)
    mean_tg   = np.mean(np.abs(tg), axis=1)

    half_bt = tempo_bt / 2.0

    # Step 1: windowed half-tempo check — gated on ≥8 beat frames to avoid
    # noisy disambiguation on short or sparse-onset clips.
    if (len(beat_frames) >= 8
            and TEMPO_MIN_BPM <= half_bt <= TEMPO_MAX_BPM
            and _score_tempo_bpm(half_bt, tg_freqs, mean_tg)
                >= _score_tempo_bpm(tempo_bt, tg_freqs, mean_tg) * 0.65):
        tempo_val = half_bt
    elif len(beat_frames) < 4:
        # tempo_bt already comes from librosa.feature.tempo — use it directly
        tempo_val = tempo_bt
    else:
        tempo_val = tempo_bt

    # Step 2: top-3 peak tie-breaker
    tempo_val = _tempo_top3_tiebreaker(tempo_val, tempo_bt, half_bt, tg_freqs, mean_tg)

    # Step 3: PLP cross-check
    tempo_val = _tempo_plp_check(tempo_val, tempo_bt, half_bt, onset_env, sr)

    # Final fold-down safety net
    while tempo_val > TEMPO_MAX_BPM:
        tempo_val /= 2.0
    while tempo_val < TEMPO_MIN_BPM:
        tempo_val *= 2.0

    # Re-estimate beat frames if tempo changed after disambiguation
    if abs(tempo_val - tempo_bt) / (tempo_bt + 1e-6) > 0.05:
        beat_frames = _estimate_beat_frames(onset_env, sr, tempo_val)

    return tempo_val, beat_frames


def _compute_loudness(y: "np.ndarray") -> float:
    """Power-weighted active-frame loudness (energy-domain mean, LUFS-style).

    Arithmetic mean of dB under-weights loud frames vs Spotify's LUFS convention.
    10*log10(mean(rms²)) better matches the training-data distribution.
    """
    rms         = librosa.feature.rms(y=y)[0]
    db          = librosa.amplitude_to_db(rms + 1e-9)
    active_mask = db > SILENCE_THRESHOLD_DB
    if active_mask.sum() > 0:
        loudness = float(10.0 * np.log10(np.mean(rms[active_mask] ** 2) + 1e-12))
    else:
        loudness = float(10.0 * np.log10(np.mean(rms ** 2) + 1e-12))
    return float(np.clip(loudness, LOUDNESS_MIN_DB, LOUDNESS_MAX_DB))


def _compute_energy(
    y: "np.ndarray", stft: "np.ndarray", freqs: "np.ndarray"
) -> float:
    """Energy [0,1]: active loudness + HF energy ratio + spectral centroid."""
    rms = librosa.feature.rms(y=y)[0]
    db  = librosa.amplitude_to_db(rms + 1e-9)
    active = db[db > SILENCE_THRESHOLD_DB]
    loudness_db   = float(np.mean(active)) if len(active) > 0 else float(np.mean(db))
    loudness_norm = float(np.clip((loudness_db - LOUDNESS_MIN_DB) / (-LOUDNESS_MIN_DB), 0.0, 1.0))

    # High-frequency energy ratio (above 2 kHz), scaled so a typical loud track ≈ 0.5
    hf_mask  = freqs >= 2000.0
    total_e  = float(np.sum(stft ** 2)) + 1e-9
    hf_ratio = float(np.clip(np.sum(stft[hf_mask] ** 2) / total_e * 4.0, 0.0, 1.0))

    # Spectral centroid normalized to half-Nyquist so typical music spans [0, 1]
    # (dividing by full Nyquist tops out near 0.36 and the term barely contributes)
    nyquist       = float(freqs[-1])
    spec_centroid = float(np.sum(freqs[:, np.newaxis] * stft ** 2) / (total_e * nyquist * 0.5))
    centroid_norm = float(np.clip(spec_centroid, 0.0, 1.0))

    return float(np.clip(loudness_norm * 0.5 + hf_ratio * 0.3 + centroid_norm * 0.2, 0.0, 1.0))


def _compute_danceability(
    y: "np.ndarray", sr: int, beat_frames: "np.ndarray"
) -> float:
    """Danceability [0,1]: IBI consistency + beat-frame onset strength.

    Previous plp_score = percentile/max collapsed to ~0 for sharp, well-spaced beats
    (the most danceable case). Replaced with mean onset strength at beat positions,
    which correctly peaks for rhythmically strong tracks.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    if len(beat_frames) >= 4:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        ibis       = np.diff(beat_times)
        ibi_cv     = float(np.std(ibis) / (np.mean(ibis) + 1e-6))
        ibi_score  = float(np.clip(1.0 - ibi_cv, 0.0, 1.0))
        safe_frames = beat_frames[beat_frames < len(onset_env)]
        plp_score  = float(np.clip(
            onset_env[safe_frames].mean() / (onset_env.max() + 1e-9), 0.0, 1.0
        )) if len(safe_frames) > 0 else 0.3
    else:
        ibi_score = 0.3
        plp_score = 0.3

    return float(np.clip(ibi_score * 0.5 + plp_score * 0.5, 0.0, 1.0))


def _compute_speechiness(
    y: "np.ndarray", sr: int, mfccs: "np.ndarray",
    stft: "np.ndarray", freqs: "np.ndarray",
) -> float:
    """Speechiness [0,1]: MFCC delta + delta² dynamics + vocal-band flux + ZCR."""
    mfcc_delta  = librosa.feature.delta(mfccs[:13])
    mfcc_delta2 = librosa.feature.delta(mfccs[:13], order=2)
    delta_norm  = float(np.clip(np.mean(np.abs(mfcc_delta))  / MFCC_DELTA_NORM,  0.0, 1.0))
    delta2_norm = float(np.clip(np.mean(np.abs(mfcc_delta2)) / MFCC_DELTA2_NORM, 0.0, 1.0))

    vocal_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    vocal_stft = stft[vocal_mask]
    if vocal_stft.shape[1] > 1:
        flux     = float(np.mean(np.abs(np.diff(vocal_stft, axis=1))))
        flux_rel = float(np.clip(flux / (np.mean(vocal_stft) + 1e-9), 0.0, 2.0)) / 2.0
    else:
        flux_rel = 0.0

    zcr_norm = float(np.clip(
        np.mean(librosa.feature.zero_crossing_rate(y)[0]) / ZCR_NORM, 0.0, 1.0
    ))

    return float(np.clip(
        delta_norm * 0.35 + delta2_norm * 0.20 + flux_rel * 0.30 + zcr_norm * 0.15,
        0.0, 1.0,
    ))


def _compute_instrumentalness(
    mfccs: "np.ndarray", stft: "np.ndarray", freqs: "np.ndarray",
    sr: int, tempo_val: float = 120.0,
) -> float:
    """Instrumentalness [0,1]: inverse of vocal presence (MFCC var + vibrato detection).

    Vibrato detection: 4.5–7 Hz periodic modulation in spectral centroid indicates singing.
    Median filter suppresses frame jitter; rhythmic subdivisions excluded to avoid false positives.
    """
    mfcc_var      = float(np.mean(np.var(mfccs[1:5], axis=1)))
    mfcc_var_norm = float(np.clip(mfcc_var / INSTR_VAR_DIVISOR, 0.0, 1.0))

    centroid_frames = np.sum(freqs[:, np.newaxis] * stft, axis=0) / (np.sum(stft, axis=0) + 1e-9)
    if len(centroid_frames) >= 3:
        centroid_frames = median_filter(centroid_frames, size=3).astype(centroid_frames.dtype)
    hop             = 512
    fps             = sr / hop
    centroid_normed = centroid_frames - float(np.mean(centroid_frames))
    ac              = librosa.autocorrelate(centroid_normed, max_size=len(centroid_normed) // 2)
    ac_normed       = ac / (ac[0] + 1e-9)

    lag_lo = max(1, int(fps / 7.0))
    lag_hi = max(lag_lo + 1, int(fps / 4.5))
    excluded_hz: tuple[float, ...] = (
        tempo_val * 4.0 / 60.0,
        tempo_val * 2.0 / 60.0,
        tempo_val * 1.0 / 60.0,
    )
    if ac[0] < 1e-6:
        # Near-DC centroid: avoid divide-by-near-zero; no meaningful vibrato
        vibrato_score = 0.0
    else:
        # Tighter ±0.5 Hz exclusion zone (was ±1.0 Hz) so 100–140 BPM tracks
        # don't accidentally mask the entire 4.5–7 Hz vibrato band.
        valid_lags = [
            lag for lag in range(lag_lo, lag_hi + 1)
            if lag < len(ac_normed)
            and all(abs(fps / lag - exc) > 0.5 for exc in excluded_hz)
        ]
        vibrato_score = float(np.clip(
            float(np.max(ac_normed[valid_lags])) if valid_lags else 0.0, 0.0, 1.0
        ))
    vocal_presence = float(np.clip(mfcc_var_norm * 0.40 + vibrato_score * 0.60, 0.0, 1.0))
    return float(np.clip(1.0 - vocal_presence, 0.0, 1.0))


def _compute_acousticness(y: "np.ndarray", sr: int, y_harmonic: "np.ndarray") -> float:
    """Acousticness [0,1]: HPSS harmonic ratio + flatness penalty + centroid variability.

    Synthesizers produce tonal harmonics (high harm_ratio) but have stable, flat overtone
    structure. Acoustic instruments have both lower flatness AND higher centroid variability
    (attack/sustain/decay differences). The centroid CV term distinguishes them.
    """
    harm_e  = float(np.mean(y_harmonic ** 2)) + 1e-9
    total_e = float(np.mean(y ** 2)) + 1e-9
    harm_ratio = float(np.clip(harm_e / total_e, 0.0, 1.0))

    flatness_harm  = float(np.mean(librosa.feature.spectral_flatness(y=y_harmonic)))
    flatness_score = float(np.clip(1.0 - flatness_harm * 20.0, 0.0, 1.0))

    full_flatness      = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    full_flatness_term = float(np.clip(1.0 - full_flatness * 10.0, 0.0, 1.0))

    centroid    = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_cv = float(np.clip(np.std(centroid) / (np.mean(centroid) + 1e-9), 0.0, 1.0))

    return float(np.clip(
        harm_ratio * 0.4 + flatness_score * 0.3 + full_flatness_term * 0.2 + centroid_cv * 0.1,
        0.0, 1.0,
    ))


def _compute_liveness(y: "np.ndarray", sr: int) -> float:
    """Liveness [0,1]: quiet-section noise floor (primary) + mid-band contrast (secondary).

    Studio recordings have near-silence between events (high DR); live recordings
    have persistent crowd/room noise in quiet sections (lower DR).
    A compression proxy penalises heavily compressed studio tracks that mimic low DR.
    """
    rms_frames = librosa.feature.rms(y=y)[0]
    db_frames  = librosa.amplitude_to_db(rms_frames + 1e-9)

    # quiet = (-50, -45) band: frames low enough to be near-silent but above true noise floor
    # active = db > -25: loud frames only; avoids near-silence biasing the DR ratio
    quiet_mask  = (db_frames >= SILENCE_THRESHOLD_DB) & (db_frames < LIVENESS_QUIET_DB)
    active_mask = db_frames > LIVENESS_ACTIVE_DB

    if quiet_mask.sum() >= 5 and active_mask.sum() >= 5:
        q_rms = float(np.mean(rms_frames[quiet_mask]))
        a_rms = float(np.mean(rms_frames[active_mask]))
        dr    = a_rms / (q_rms + 1e-9)
        noise_floor_score = float(np.clip(1.0 - np.log10(dr + 1.0) / 3.0, 0.0, 1.0))
        if float(np.std(db_frames[active_mask])) < 4.0:
            noise_floor_score *= 0.3
    else:
        noise_floor_score = 0.15

    spec_contrast    = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    mid_contrast_var = float(np.mean(np.std(spec_contrast[2:5], axis=1)))
    mid_score        = float(np.clip(mid_contrast_var / 10.0, 0.0, 1.0))

    return float(np.clip(noise_floor_score * 0.70 + mid_score * 0.30, 0.0, 1.0))


def _compute_valence(
    y: "np.ndarray", sr: int,
    y_harmonic: "np.ndarray", y_percussive: "np.ndarray",
    tempo_val: float, stft: "np.ndarray", freqs: "np.ndarray",
) -> float:
    """Valence [0,1]: Krumhansl-Kessler key mode + tempo + spectral tilt + H/P ratio."""
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-6)
    major_p = np.array(_MAJOR_PROFILE); major_p = major_p / major_p.sum()
    minor_p = np.array(_MINOR_PROFILE); minor_p = minor_p / minor_p.sum()
    cors_maj = [float(np.corrcoef(np.roll(chroma_mean, i), major_p)[0, 1]) for i in range(12)]
    cors_min = [float(np.corrcoef(np.roll(chroma_mean, i), minor_p)[0, 1]) for i in range(12)]
    diff     = max(cors_maj) - max(cors_min)
    # (diff + 0.5) / 1.0 saturated at 0.55–0.90 for clear major — replace with
    # tighter window so the full [0, 1] range is used in practice.
    mode_score = float(np.clip((diff + 0.3) / 0.6, 0.0, 1.0))

    tempo_norm = float(np.clip(
        (tempo_val - TEMPO_MIN_BPM) / (TEMPO_MAX_BPM - TEMPO_MIN_BPM), 0.0, 1.0
    ))

    mean_spec = np.mean(stft, axis=1) + 1e-9
    log_freqs = np.log(freqs[1:] + 1.0)
    log_spec  = np.log(mean_spec[1:])
    slope     = float(np.polyfit(log_freqs, log_spec, 1)[0]) if len(log_freqs) > 2 else -2.5
    spectral_tilt = float(np.clip((slope + 5.0) / 5.0, 0.0, 1.0))

    h_e     = float(np.mean(y_harmonic ** 2)) + 1e-9
    p_e     = float(np.mean(y_percussive ** 2)) + 1e-9
    h_ratio = float(np.clip(h_e / (h_e + p_e), 0.0, 1.0))

    return float(np.clip(
        mode_score * 0.45 + tempo_norm * 0.15 + spectral_tilt * 0.25 + h_ratio * 0.15,
        0.0, 1.0,
    ))


def _extract_audio_features(file_path: str) -> dict[str, float]:
    """Orchestrate per-feature helpers; return Spotify-like audio feature dict.

    All 9 features computed entirely with librosa — no external ML packages required.
    """
    # Load with soundfile + soxr to avoid numba @guvectorize in librosa.load()
    # which OOMs on Render's 512 MB free tier during JIT compilation.
    data, native_sr = soundfile.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)                    # stereo → mono
    max_samples = int(_MAX_DURATION_SEC * native_sr)
    if len(data) > max_samples:
        data = data[:max_samples]                    # trim to 30s
    if native_sr != _TARGET_SR:
        data = soxr.resample(data, native_sr, _TARGET_SR)
    y, sr = data, _TARGET_SR

    stft             = np.abs(librosa.stft(y))
    freqs            = librosa.fft_frequencies(sr=sr)
    mfccs            = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo_val, beat_frames = _compute_tempo(y, sr)

    return {
        "tempo":            round(tempo_val, 1),
        "loudness":         round(_compute_loudness(y), 1),
        "energy":           round(_compute_energy(y, stft, freqs), 3),
        "danceability":     round(_compute_danceability(y, sr, beat_frames), 3),
        "speechiness":      round(_compute_speechiness(y, sr, mfccs, stft, freqs), 3),
        "instrumentalness": round(_compute_instrumentalness(mfccs, stft, freqs, sr, tempo_val), 3),
        "acousticness":     round(_compute_acousticness(y, sr, y_harmonic), 3),
        "liveness":         round(_compute_liveness(y, sr), 3),
        "valence":          round(_compute_valence(y, sr, y_harmonic, y_percussive, tempo_val, stft, freqs), 3),
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/analyze-audio", methods=["POST"])
def analyze_audio() -> tuple[Any, int]:
    """Accept an audio file, extract Spotify-like features, return JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    upload = request.files["file"]

    # Validate extension against allowlist before touching the filesystem
    raw_suffix = os.path.splitext(upload.filename or "")[1].lower()
    if raw_suffix not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{raw_suffix or '(none)'}'. "
                     f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        }), 415

    # Lazy-load librosa on first audio request (keeps startup RAM low)
    if not _ensure_librosa():
        return jsonify({"error": "librosa is not installed. Run: pip install librosa"}), 503

    # Belt-and-suspenders size check (Flask MAX_CONTENT_LENGTH is the first gate)
    upload.seek(0, 2)
    if upload.tell() > UPLOAD_MAX_BYTES:
        return jsonify({"error": "File too large (max 50 MB)"}), 413
    upload.seek(0)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=raw_suffix, delete=False) as tmp:
            tmp_path = tmp.name
            upload.save(tmp_path)
        feat = _extract_audio_features(tmp_path)
        return jsonify({"features": feat}), 200
    except Exception:
        logger.exception("Audio analysis failed for upload '%s'", upload.filename)
        return jsonify({"error": "Audio analysis failed. Check that the file is a valid audio format."}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/")
def index() -> Any:
    return send_from_directory(".", "index.html")


@app.route("/meta")
def meta() -> Any:
    """Send feature metadata to the frontend."""
    response: dict[str, Any] = {
        "features":         features,
        "slider_features":  slider_features,
        "importance":       importance,
        "audio_importance": _audio_importance_normalized(),
        "ranges":           ranges,
        "r2":               r2,
        "mae":              mae,
        "recommended":      recommended,
    }
    if "r2_base" in payload:
        response["r2_base"] = payload["r2_base"]
    if "cv_r2_mean" in payload:
        response["cv_r2_mean"] = payload["cv_r2_mean"]
        response["cv_r2_std"]  = payload["cv_r2_std"]
    return jsonify(response)


@app.route("/genres")
def genres() -> Any:
    """Return sorted list of available genres."""
    return jsonify(sorted(genre_means.keys()))


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int]:
    """Receive feature values, return predicted popularity score + insights."""
    data = request.json or {}
    try:
        body   = {**data, "artist_avg_popularity": global_avg_pop}
        values = [[body.get(f, ranges[f]["mean"]) for f in features]]

        raw_score = model.predict(values)[0]

        span  = pred_max - pred_min if pred_max != pred_min else 1
        score = (raw_score - pred_min) / span * 100
        score = round(max(0.0, min(100.0, score)), 1)

        genre          = data.get("genre", "").strip()
        genre_specific = genre_means.get(genre, {}) if genre else {}

        imp_dict = _audio_importance_normalized()

        insights: dict[str, Any] = {}
        for f in slider_features:
            user_val = body.get(f, ranges[f]["mean"])
            avg      = genre_specific.get(f, ranges[f]["mean"])
            diff     = user_val - avg
            span_f   = ranges[f]["max"] - ranges[f]["min"]
            if abs(diff) < span_f * 0.05:
                direction = "average"
            elif diff > 0:
                direction = "above average"
            else:
                direction = "below average"
            insights[f] = {
                "value":      round(user_val, 3),
                "avg":        round(avg, 3),
                "direction":  direction,
                "importance": round(imp_dict.get(f, 0), 3),
            }

        return jsonify({"score": score, "insights": insights}), 200

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check input values."}), 400


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting server at http://127.0.0.1:%d  (debug=%s)", port, debug_mode)
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
