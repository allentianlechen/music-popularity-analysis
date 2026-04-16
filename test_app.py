"""
test_app.py — Unit + integration tests for APP.py
Run: python3 -m pytest test_app.py -v
"""

import io
import json
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Flask test client, loaded once per session."""
    import APP  # noqa: PLC0415  (import inside function is intentional here)
    APP.app.config["TESTING"] = True
    APP.app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    with APP.app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def meta_data(client):
    """Cached /meta response."""
    res = client.get("/meta")
    assert res.status_code == 200
    return res.get_json()


# ── /meta ─────────────────────────────────────────────────────────────────────

class TestMeta:
    def test_status_200(self, client):
        assert client.get("/meta").status_code == 200

    def test_has_required_keys(self, meta_data):
        for key in ("features", "slider_features", "importance", "ranges", "r2", "mae", "recommended"):
            assert key in meta_data, f"Missing key: {key}"

    def test_slider_features_count(self, meta_data):
        assert len(meta_data["slider_features"]) == 9

    def test_recommended_covers_all_slider_features(self, meta_data):
        for feat in meta_data["slider_features"]:
            assert feat in meta_data["recommended"], f"recommended missing: {feat}"

    def test_ranges_have_min_max_mean(self, meta_data):
        for feat, rng in meta_data["ranges"].items():
            assert "min" in rng and "max" in rng and "mean" in rng


# ── /genres ───────────────────────────────────────────────────────────────────

class TestGenres:
    def test_status_200(self, client):
        assert client.get("/genres").status_code == 200

    def test_returns_list(self, client):
        data = client.get("/genres").get_json()
        assert isinstance(data, list)
        assert len(data) > 0


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def _post(self, client, body: dict):
        return client.post(
            "/predict",
            data=json.dumps(body),
            content_type="application/json",
        )

    def test_basic_predict_returns_score(self, client):
        res = self._post(client, {})
        assert res.status_code == 200
        data = res.get_json()
        assert "score" in data
        assert 0 <= data["score"] <= 100

    def test_score_always_in_0_100(self, client, meta_data):
        """Score must stay in [0, 100] even for extreme slider values."""
        extremes = {feat: meta_data["ranges"][feat]["max"] for feat in meta_data["slider_features"]}
        data = self._post(client, extremes).get_json()
        assert 0 <= data["score"] <= 100

        zeroes = {feat: meta_data["ranges"][feat]["min"] for feat in meta_data["slider_features"]}
        data = self._post(client, zeroes).get_json()
        assert 0 <= data["score"] <= 100

    def test_insights_contain_all_slider_features(self, client, meta_data):
        data = self._post(client, {}).get_json()
        assert "insights" in data
        for feat in meta_data["slider_features"]:
            assert feat in data["insights"], f"insights missing: {feat}"

    def test_insight_direction_values(self, client):
        data = self._post(client, {}).get_json()
        valid = {"above average", "below average", "average"}
        for feat, info in data["insights"].items():
            assert info["direction"] in valid, f"{feat}: unexpected direction '{info['direction']}'"

    def test_artist_found_never_in_response(self, client):
        # artist_found was removed — verify it is absent regardless of input
        for body in [{}, {"artist": "Taylor Swift"}, {"artist": "__unknown__"}]:
            data = self._post(client, body).get_json()
            assert "artist_found" not in data, f"artist_found leaked into response for body={body}"

    def test_genre_in_body_does_not_crash(self, client):
        res = self._post(client, {"genre": "pop"})
        assert res.status_code == 200

    def test_pred_min_equals_pred_max_still_returns_score(self, client):
        """Degenerate case: span == 0 should not divide by zero."""
        import APP
        orig_min, orig_max = APP.pred_min, APP.pred_max
        APP.pred_min = APP.pred_max = 30.0
        try:
            data = self._post(client, {}).get_json()
            assert "score" in data
            assert 0 <= data["score"] <= 100
        finally:
            APP.pred_min, APP.pred_max = orig_min, orig_max


# ── /analyze-audio ────────────────────────────────────────────────────────────

class TestAnalyzeAudio:
    def test_no_file_returns_400(self, client):
        res = client.post("/analyze-audio")
        assert res.status_code == 400
        assert "error" in res.get_json()

    def test_disallowed_extension_returns_415(self, client):
        data = {"file": (io.BytesIO(b"fake content"), "malicious.py")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 415
        assert "error" in res.get_json()

    def test_pkl_extension_rejected(self, client):
        data = {"file": (io.BytesIO(b"fake pickle"), "evil.pkl")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 415

    def test_exe_extension_rejected(self, client):
        data = {"file": (io.BytesIO(b"MZ"), "virus.exe")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 415

    def test_mp3_extension_passes_validation(self, client):
        """A valid extension should pass the 415 gate (may fail with 400 for bad audio content)."""
        data = {"file": (io.BytesIO(b"not real mp3"), "test.mp3")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        # Should NOT be 415 (extension allowed); will be 400 because content is invalid
        assert res.status_code != 415

    def test_wav_extension_passes_validation(self, client):
        data = {"file": (io.BytesIO(b"RIFF"), "test.wav")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code != 415

    def test_no_filename_returns_415(self, client):
        """Empty filename has no extension → not in allowlist → 415."""
        data = {"file": (io.BytesIO(b"data"), "")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 415


# ── /analyze-audio size guard ─────────────────────────────────────────────────

class TestUploadSizeGuard:
    def test_oversized_upload_returns_413(self, client):
        big = io.BytesIO(b"x" * (51 * 1024 * 1024))  # 51 MB
        data = {"file": (big, "big.mp3")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code in (413, 400)  # Flask 413 or manual check


# ── static_folder=None (security) ─────────────────────────────────────────────

class TestStaticFolderDisabled:
    def test_model_pkl_not_served(self, client):
        """model.pkl must not be accessible at /static/model.pkl."""
        res = client.get("/static/model.pkl")
        assert res.status_code == 404

    def test_app_py_not_served(self, client):
        res = client.get("/static/APP.py")
        assert res.status_code == 404


# ── Audio feature helper unit tests ───────────────────────────────────────────

class TestAudioFeatureHelpers:
    """Unit tests for each audio feature helper using a synthetic sine wave."""

    @pytest.fixture(scope="class")
    def sine_wave(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        sr = 22050
        duration = 5  # seconds — short enough for fast tests
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return y, sr

    @pytest.fixture(scope="class")
    def shared_features(self, sine_wave):
        """Pre-compute shared librosa features used by multiple helpers."""
        try:
            import librosa
            import numpy as np
        except ImportError:
            pytest.skip("librosa not available")
        y, sr = sine_wave
        stft  = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        return {"stft": stft, "freqs": freqs}

    def test_compute_tempo_in_valid_range(self, sine_wave):
        from APP import _compute_tempo, TEMPO_MIN_BPM, TEMPO_MAX_BPM
        y, sr = sine_wave
        tempo_val, beat_frames = _compute_tempo(y, sr)
        assert TEMPO_MIN_BPM <= tempo_val <= TEMPO_MAX_BPM

    def test_compute_loudness_in_db_range(self, sine_wave):
        from APP import _compute_loudness, LOUDNESS_MIN_DB, LOUDNESS_MAX_DB
        y, sr = sine_wave
        loudness = _compute_loudness(y)
        assert LOUDNESS_MIN_DB <= loudness <= LOUDNESS_MAX_DB

    def test_compute_energy_in_0_1(self, sine_wave, shared_features):
        from APP import _compute_energy
        y, sr = sine_wave
        val = _compute_energy(y, shared_features["stft"], shared_features["freqs"])
        assert 0.0 <= val <= 1.0

    def test_compute_danceability_in_0_1(self, sine_wave):
        from APP import _compute_danceability, _compute_tempo
        y, sr = sine_wave
        _, beat_frames = _compute_tempo(y, sr)
        val = _compute_danceability(y, sr, beat_frames)
        assert 0.0 <= val <= 1.0

    # ── /analyze-audio integration tests ──────────────────────────────────────

    def test_analyze_audio_always_returns_four_base_features(self, client):
        """Four librosa features must always be present regardless of ML packages."""
        import io
        data = {"file": (io.BytesIO(b"not real mp3"), "test.mp3")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        # If audio analysis succeeds, check the four base features are present.
        # If it fails (bad content), that's expected — we only check structure on success.
        if res.status_code == 200:
            feats = res.get_json().get("features", {})
            for key in ("tempo", "loudness", "energy", "danceability"):
                assert key in feats, f"base feature missing from /analyze-audio response: {key}"

    def test_analyze_audio_returns_no_deleted_librosa_features(self, client):
        """Legacy heuristic feature keys must never appear in the response."""
        import io
        data = {"file": (io.BytesIO(b"not real mp3"), "test.mp3")}
        res = client.post(
            "/analyze-audio",
            data=data,
            content_type="multipart/form-data",
        )
        if res.status_code == 200:
            feats = res.get_json().get("features", {})
            for bad_key in ("acousticness_hpss", "liveness_dr"):
                assert bad_key not in feats, f"deleted heuristic key found: {bad_key}"

    # ── ML feature unit tests (gated with pytest.importorskip) ────────────────

    def test_compute_speechiness_whisper_returns_low_for_sine(self, sine_wave):
        """A pure sine wave has no speech; speechiness should be near 0."""
        pytest.importorskip("whisper")
        from APP import _compute_speechiness_whisper
        y, sr = sine_wave
        val = _compute_speechiness_whisper(y, sr)
        assert 0.0 <= val <= 1.0
        assert val < 0.5, f"sine wave should have low speechiness, got {val}"

    def test_compute_instrumentalness_demucs_range(self, sine_wave):
        """Output must be in [0, 1]."""
        pytest.importorskip("demucs")
        pytest.importorskip("torch")
        from APP import _compute_instrumentalness_demucs
        y, sr = sine_wave
        val = _compute_instrumentalness_demucs(y, sr)
        assert 0.0 <= val <= 1.0

    def test_compute_acousticness_in_0_1(self, sine_wave):
        from APP import _compute_acousticness
        import librosa
        y, sr = sine_wave
        y_harmonic, _ = librosa.effects.hpss(y)
        assert 0.0 <= _compute_acousticness(y, y_harmonic) <= 1.0

    def test_compute_liveness_in_0_1(self, sine_wave):
        from APP import _compute_liveness
        y, sr = sine_wave
        assert 0.0 <= _compute_liveness(y, sr) <= 1.0

    def test_compute_valence_in_0_1(self, sine_wave, shared_features):
        from APP import _compute_valence, _compute_tempo
        import librosa
        y, sr = sine_wave
        tempo_val, _ = _compute_tempo(y, sr)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        val = _compute_valence(
            y, sr, y_harmonic, y_percussive, tempo_val,
            shared_features["stft"], shared_features["freqs"],
        )
        assert 0.0 <= val <= 1.0
