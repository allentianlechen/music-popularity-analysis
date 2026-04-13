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
import pickle
import sys
import tempfile
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
UPLOAD_MAX_BYTES: int        = 50 * 1024 * 1024          # 50 MB
TEMPO_MIN_BPM: float         = 60.0
TEMPO_MAX_BPM: float         = 165.0
TEMPO_START_BPM: int         = 100
LOUDNESS_MIN_DB: float       = -60.0
LOUDNESS_MAX_DB: float       = 0.0
CENTROID_NORM_DIVISOR: float = 4000.0
SPEECHINESS_DELTA_DIV: float = 9.0
SPEECHINESS_ZCR_DIV: float   = 0.15
INSTR_VAR_DIVISOR: float     = 300.0
LIVENESS_CONTRAST_DIV: float = 15.0

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
    """Load model.pkl, optionally verifying SHA-256 when MODEL_PKL_SHA256 is set."""
    with open(path, "rb") as f:
        raw = f.read()
    expected_hash = os.environ.get("MODEL_PKL_SHA256")
    if expected_hash:
        actual = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual, expected_hash):
            logger.error("model.pkl SHA-256 mismatch — refusing to load")
            sys.exit(1)
        logger.info("model.pkl integrity verified")
    return pickle.loads(raw)  # noqa: S301 — local model file, hash-checked when env var set


try:
    payload = _load_model()
except FileNotFoundError:
    logger.error("model.pkl not found. Run 'python3 analyze.py' first.")
    sys.exit(1)

model           = payload["model"]
features        = payload["features"]
slider_features = payload.get("slider_features", features[:9])
importance      = payload["importance"]
ranges          = payload["ranges"]
r2              = payload["r2"]
mae             = payload["mae"]
pred_min        = payload.get("pred_min", 0)
pred_max        = payload.get("pred_max", 100)
recommended     = payload.get("recommended", {})
artist_lookup   = payload.get("artist_lookup", {})
global_avg_pop  = payload.get("global_avg_popularity", 0)
genre_means     = payload.get("genre_means", {})
# classifier is intentionally not extracted — tier is determined client-side


# ── AUDIO FEATURE HELPERS ─────────────────────────────────────────────────────

def _compute_tempo(y: "np.ndarray", sr: int) -> tuple[float, "np.ndarray"]:
    """Return (tempo_bpm, onset_env). Uses tempogram autocorrelation; folds into [60,165]."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr, start_bpm=TEMPO_START_BPM)
    )[0])
    while tempo_val > TEMPO_MAX_BPM:
        tempo_val /= 2.0
    while tempo_val < TEMPO_MIN_BPM:
        tempo_val *= 2.0
    return tempo_val, onset_env


def _compute_loudness(y: "np.ndarray") -> float:
    """Loudness in dB, clipped to [-60, 0]."""
    rms = librosa.feature.rms(y=y)[0]
    loudness = float(librosa.amplitude_to_db(np.mean(rms) + 1e-9))
    return float(np.clip(loudness, LOUDNESS_MIN_DB, LOUDNESS_MAX_DB))


def _compute_energy(y: "np.ndarray") -> float:
    """Energy [0,1]: RMS + spectral flatness, sqrt-stretched."""
    rms = librosa.feature.rms(y=y)[0]
    rms_mean      = float(np.mean(rms))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    energy_raw    = float(np.clip(rms_mean * 6 + spec_flatness * 0.5, 0, 1))
    return float(np.clip(energy_raw ** 0.5, 0, 1))


def _compute_acousticness(y: "np.ndarray", sr: int) -> float:
    """Acousticness [0,1]: centroid + rolloff + flatness; low values → electronic."""
    spec_centroids  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff         = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness_frames = librosa.feature.spectral_flatness(y=y)[0]
    centroid_norm   = float(np.mean(spec_centroids)) / CENTROID_NORM_DIVISOR
    rolloff_norm    = float(np.mean(rolloff)) / (sr / 2)
    flatness_mean   = float(np.mean(flatness_frames))
    return float(np.clip(
        1 - (centroid_norm * 0.4 + rolloff_norm * 0.4 + flatness_mean * 10 * 0.2), 0, 1
    ))


def _compute_danceability(
    onset_env: "np.ndarray", sr: int, tempo_val: float
) -> float:
    """Danceability [0,1]: beat-period autocorrelation + single-frame regularity."""
    tempo_hz     = tempo_val / 60.0
    beat_lag     = int(sr / (512 * tempo_hz + 1e-6))
    beat_lag     = max(1, min(beat_lag, len(onset_env) - 1))
    ac_full      = librosa.autocorrelate(onset_env, max_size=beat_lag + 1)
    beat_strength = float(ac_full[beat_lag] / (ac_full[0] + 1e-6)) if beat_lag < len(ac_full) else 0.0
    regularity    = float(ac_full[1] / (ac_full[0] + 1e-6)) if len(ac_full) > 1 else 0.0
    return float(np.clip(beat_strength * 0.7 + regularity * 0.3, 0, 1))


def _compute_speechiness(
    y: "np.ndarray", sr: int, mfccs: "np.ndarray"
) -> float:
    """Speechiness [0,1]: recalibrated MFCC delta + ZCR."""
    mfcc_delta = librosa.feature.delta(mfccs[:5])
    delta_mean = float(np.mean(np.abs(mfcc_delta)))
    zcr_mean   = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    return float(np.clip(
        delta_mean / SPEECHINESS_DELTA_DIV * 0.6
        + zcr_mean / SPEECHINESS_ZCR_DIV * 0.4,
        0, 1
    ))


def _compute_instrumentalness(mfccs: "np.ndarray") -> float:
    """Instrumentalness [0,1]: absolute temporal variance of MFCCs 1–4."""
    mfcc_abs_var = float(np.mean(np.var(mfccs[1:5], axis=1)))
    return float(np.clip(1.0 - mfcc_abs_var / INSTR_VAR_DIVISOR, 0, 1))


def _compute_liveness(y: "np.ndarray", sr: int) -> float:
    """Liveness [0,1]: spectral contrast variation + harmonic noise floor."""
    spec_contrast   = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_var    = float(np.mean(np.std(spec_contrast, axis=1)))
    harmonic        = librosa.effects.harmonic(y)
    high_band_noise = float(np.mean(librosa.feature.spectral_flatness(y=harmonic)))
    return float(np.clip(
        contrast_var / LIVENESS_CONTRAST_DIV * 0.6 + high_band_noise * 5 * 0.4, 0, 1
    ))


def _compute_valence(y: "np.ndarray", sr: int) -> float:
    """Valence [0,1]: Krumhansl-Kessler major/minor key estimation."""
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-6)
    # Use normalised copies (immutable profiles, no in-place mutation)
    major_p = np.array(_MAJOR_PROFILE)
    minor_p = np.array(_MINOR_PROFILE)
    major_p = major_p / major_p.sum()
    minor_p = minor_p / minor_p.sum()
    cors_maj = [float(np.corrcoef(np.roll(chroma_mean, i), major_p)[0, 1]) for i in range(12)]
    cors_min = [float(np.corrcoef(np.roll(chroma_mean, i), minor_p)[0, 1]) for i in range(12)]
    diff    = max(cors_maj) - max(cors_min)
    return float(np.clip((diff * 2.5 + 1) / 2, 0, 1))


def _extract_audio_features(file_path: str) -> dict[str, float]:
    """Orchestrate per-feature helpers; return Spotify-like audio feature dict."""
    y, sr = librosa.load(file_path, duration=90, mono=True)

    tempo_val, onset_env = _compute_tempo(y, sr)
    loudness             = _compute_loudness(y)
    energy               = _compute_energy(y)
    acousticness         = _compute_acousticness(y, sr)
    danceability         = _compute_danceability(onset_env, sr, tempo_val)

    mfccs                = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    speechiness          = _compute_speechiness(y, sr, mfccs)
    instrumentalness     = _compute_instrumentalness(mfccs)

    liveness             = _compute_liveness(y, sr)
    valence              = _compute_valence(y, sr)

    return {
        "danceability":     round(danceability,     3),
        "energy":           round(energy,           3),
        "loudness":         round(loudness,          1),
        "speechiness":      round(speechiness,       3),
        "acousticness":     round(acousticness,      3),
        "instrumentalness": round(instrumentalness,  3),
        "liveness":         round(liveness,          3),
        "valence":          round(valence,           3),
        "tempo":            round(tempo_val,         1),
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

    # Check librosa availability after validation (so ext/size checks always work)
    if not LIBROSA_AVAILABLE:
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
        "features":        features,
        "slider_features": slider_features,
        "importance":      importance,
        "ranges":          ranges,
        "r2":              r2,
        "mae":             mae,
        "recommended":     recommended,
    }
    if "r2_base" in payload:
        response["r2_base"] = payload["r2_base"]
    return jsonify(response)


@app.route("/genres")
def genres() -> Any:
    """Return sorted list of available genres."""
    return jsonify(sorted(genre_means.keys()))


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int]:
    """Receive feature values, return predicted popularity score + insights."""
    data = request.json
    try:
        artist       = data.get("artist", "").strip()
        artist_found = artist in artist_lookup
        artist_avg   = artist_lookup.get(artist, global_avg_pop)

        body   = {**data, "artist_avg_popularity": artist_avg}
        values = [[body.get(f, ranges[f]["mean"]) for f in features]]

        raw_score = model.predict(values)[0]

        span  = pred_max - pred_min if pred_max != pred_min else 1
        score = (raw_score - pred_min) / span * 100
        score = round(max(0.0, min(100.0, score)), 1)

        genre          = data.get("genre", "").strip()
        genre_specific = genre_means.get(genre, {}) if genre else {}

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
                "importance": round(importance.get(f, 0), 3),
            }

        result: dict[str, Any] = {"score": score, "insights": insights}
        if artist:
            result["artist_found"] = artist_found
        return jsonify(result), 200

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check input values."}), 400


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting server at http://127.0.0.1:8080  (debug=%s)", debug_mode)
    app.run(debug=debug_mode, port=8080)
