"""
app.py — Step 3: Web Server
============================
Serves the webpage and handles popularity predictions.
Install: pip3 install flask scikit-learn
Run:     python3 app.py
Then open: http://localhost:8080
"""

import pickle
import sys
import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

app = Flask(__name__, static_folder=".")

# Load model once at startup
try:
    with open("model.pkl", "rb") as f:
        payload = pickle.load(f)
except FileNotFoundError:
    print("ERROR: model.pkl not found. Run 'python3 analyze.py' first.", file=sys.stderr)
    sys.exit(1)

model              = payload["model"]
features           = payload["features"]
slider_features    = payload.get("slider_features", features[:9])
importance         = payload["importance"]
ranges             = payload["ranges"]
r2                 = payload["r2"]
mae                = payload["mae"]
pred_min           = payload.get("pred_min", 0)
pred_max           = payload.get("pred_max", 100)
recommended        = payload.get("recommended", {})
classifier         = payload.get("classifier")
artist_lookup      = payload.get("artist_lookup", {})
global_avg_pop     = payload.get("global_avg_popularity", 0)
genre_means        = payload.get("genre_means", {})


def _extract_audio_features(file_path: str) -> dict:
    """Extract Spotify-like audio features from an audio file using librosa."""
    y, sr = librosa.load(file_path, duration=90, mono=True)

    # Tempo (BPM) — onset envelope computed once; reused by danceability
    # librosa.feature.tempo (tempogram autocorrelation) is more stable than beat_track
    # for slow, ambient, or non-4/4 music.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr, start_bpm=100)
    )[0])
    # Fold into [60, 165] BPM — suppresses half-tempo and double-tempo octave errors
    while tempo_val > 165:
        tempo_val /= 2.0
    while tempo_val < 60:
        tempo_val *= 2.0

    # Loudness (dB, typically -60 to 0)
    rms = librosa.feature.rms(y=y)[0]
    loudness = float(librosa.amplitude_to_db(np.mean(rms) + 1e-9))
    loudness = float(np.clip(loudness, -60, 0))

    # Energy (0-1): RMS + spectral flatness, sqrt-stretched to spread low-energy tracks
    rms_mean     = float(np.mean(rms))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    energy_raw   = float(np.clip(rms_mean * 6 + spec_flatness * 0.5, 0, 1))
    energy       = float(np.clip(energy_raw ** 0.5, 0, 1))

    # Acousticness (0-1): centroid + rolloff + flatness combined; low values → acoustic
    spec_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff        = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness_frames = librosa.feature.spectral_flatness(y=y)[0]
    centroid_norm  = float(np.mean(spec_centroids)) / 4000.0
    rolloff_norm   = float(np.mean(rolloff)) / (sr / 2)
    flatness_mean  = float(np.mean(flatness_frames))
    acousticness   = float(np.clip(
        1 - (centroid_norm * 0.4 + rolloff_norm * 0.4 + flatness_mean * 10 * 0.2), 0, 1
    ))

    # Danceability (0-1): beat-period autocorrelation + single-frame regularity
    # onset_env already computed above for tempo
    tempo_hz    = tempo_val / 60.0
    beat_lag    = int(sr / (512 * tempo_hz + 1e-6))
    beat_lag    = max(1, min(beat_lag, len(onset_env) - 1))  # clamp to valid range
    ac_full     = librosa.autocorrelate(onset_env, max_size=beat_lag + 1)
    beat_strength = float(ac_full[beat_lag] / (ac_full[0] + 1e-6)) if beat_lag < len(ac_full) else 0.0
    regularity    = float(ac_full[1] / (ac_full[0] + 1e-6)) if len(ac_full) > 1 else 0.0
    danceability  = float(np.clip(beat_strength * 0.7 + regularity * 0.3, 0, 1))

    # Speechiness (0-1): MFCC delta + ZCR
    # Recalibrated divisor: pure instruments ~0.5–3 delta_mean; speech ~5–9.
    # ZCR gates the signal: speech 0.08–0.18; sustained instruments 0.02–0.07.
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfccs[:5])
    delta_mean = float(np.mean(np.abs(mfcc_delta)))
    zcr_mean   = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    speechiness = float(np.clip(
        delta_mean / 9.0 * 0.6 + zcr_mean / 0.15 * 0.4, 0, 1
    ))

    # Instrumentalness (0-1): absolute temporal MFCC variance
    # BUG FIX: the previous formula computed mean(v_i/sum(v_i)) = 1/n = 0.2 always,
    # making instrumentalness ≡ 0 for every track regardless of content.
    # FIX: absolute variance of MFCCs 1–4 over time frames — higher = more spectral
    # change = more likely vocals. Scale: pure instrument ~20–200, vocal ~200–800.
    mfcc_abs_var     = float(np.mean(np.var(mfccs[1:5], axis=1)))
    instrumentalness = float(np.clip(1.0 - mfcc_abs_var / 300.0, 0, 1))

    # Liveness (0-1): contrast variation + harmonic noise floor (room/audience proxy)
    spec_contrast    = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_var     = float(np.mean(np.std(spec_contrast, axis=1)))
    harmonic         = librosa.effects.harmonic(y)
    high_band_noise  = float(np.mean(librosa.feature.spectral_flatness(y=harmonic)))
    liveness = float(np.clip(contrast_var / 15.0 * 0.6 + high_band_noise * 5 * 0.4, 0, 1))

    # Valence (0-1): Krumhansl-Kessler major/minor key estimation, difference stretched
    chroma      = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-6)
    major_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_p /= major_p.sum()
    minor_p /= minor_p.sum()
    cors_maj = [float(np.corrcoef(np.roll(chroma_mean, i), major_p)[0, 1]) for i in range(12)]
    cors_min = [float(np.corrcoef(np.roll(chroma_mean, i), minor_p)[0, 1]) for i in range(12)]
    diff    = max(cors_maj) - max(cors_min)
    valence = float(np.clip((diff * 2.5 + 1) / 2, 0, 1))

    return {
        "danceability":     round(float(np.clip(danceability, 0, 1)), 3),
        "energy":           round(float(np.clip(energy,       0, 1)), 3),
        "loudness":         round(loudness, 1),
        "speechiness":      round(float(np.clip(speechiness,  0, 1)), 3),
        "acousticness":     round(float(np.clip(acousticness, 0, 1)), 3),
        "instrumentalness": round(float(np.clip(instrumentalness, 0, 1)), 3),
        "liveness":         round(float(np.clip(liveness,     0, 1)), 3),
        "valence":          round(valence, 3),
        "tempo":            round(tempo_val, 1),
    }


@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    """Accept an audio file, extract Spotify-like features, return JSON."""
    if not LIBROSA_AVAILABLE:
        return jsonify({"error": "librosa is not installed. Run: pip install librosa"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    upload = request.files["file"]

    # Guard: reject files larger than 50 MB before loading into librosa
    MAX_BYTES = 50 * 1024 * 1024
    upload.seek(0, 2)
    if upload.tell() > MAX_BYTES:
        return jsonify({"error": "File too large (max 50 MB)"}), 413
    upload.seek(0)

    suffix = os.path.splitext(upload.filename)[1] if upload.filename else ".tmp"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        upload.save(tmp_path)

    try:
        feat = _extract_audio_features(tmp_path)
        return jsonify({"features": feat})
    except Exception as exc:
        return jsonify({"error": f"Audio analysis failed: {exc}"}), 400
    finally:
        os.unlink(tmp_path)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/meta")
def meta():
    """Send feature metadata to the frontend."""
    response = {
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
def genres():
    """Return sorted list of available genres."""
    return jsonify(sorted(genre_means.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    """Receive feature values, return predicted popularity score + tier."""
    data = request.json
    try:
        # Resolve artist_avg_popularity via lookup; fall back to global mean
        artist = data.get("artist", "").strip()
        artist_found = artist in artist_lookup
        artist_avg = artist_lookup.get(artist, global_avg_pop)

        # Build complete feature vector (all features the model expects)
        body = {**data, "artist_avg_popularity": artist_avg}
        values = [[body.get(f, ranges[f]["mean"]) for f in features]]

        raw_score = model.predict(values)[0]

        # Rescale from the model's natural p5–p95 range to 0–100.
        span  = pred_max - pred_min if pred_max != pred_min else 1
        score = (raw_score - pred_min) / span * 100
        score = round(max(0, min(100, score)), 1)

        # Tier is determined client-side from the 0-100 display score.
        # The backend classifier is not used in the response.
        tier_label = None

        # Genre-specific comparison means (fall back to global means)
        genre = data.get("genre", "").strip()
        genre_specific = genre_means.get(genre, {}) if genre else {}

        # Insights: only for slider features (the ones users actually control)
        insights = {}
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

        result = {"score": score, "insights": insights}
        if artist:
            result["artist_found"] = artist_found
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8080")
    app.run(debug=True, port=8080)
