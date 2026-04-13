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

    def test_artist_found_true_for_known_artist(self, client):
        import APP
        if not APP.artist_lookup:
            pytest.skip("artist_lookup is empty")
        known = next(iter(APP.artist_lookup))
        data = self._post(client, {"artist": known}).get_json()
        assert data.get("artist_found") is True

    def test_artist_found_false_for_unknown_artist(self, client):
        data = self._post(client, {"artist": "__no_such_artist_xyzzy__"}).get_json()
        assert data.get("artist_found") is False

    def test_no_artist_field_omits_artist_found(self, client):
        data = self._post(client, {}).get_json()
        assert "artist_found" not in data

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

    def test_compute_tempo_in_valid_range(self, sine_wave):
        from APP import _compute_tempo, TEMPO_MIN_BPM, TEMPO_MAX_BPM
        y, sr = sine_wave
        tempo_val, onset_env = _compute_tempo(y, sr)
        assert TEMPO_MIN_BPM <= tempo_val <= TEMPO_MAX_BPM
        assert len(onset_env) > 0

    def test_compute_loudness_in_db_range(self, sine_wave):
        from APP import _compute_loudness, LOUDNESS_MIN_DB, LOUDNESS_MAX_DB
        y, sr = sine_wave
        loudness = _compute_loudness(y)
        assert LOUDNESS_MIN_DB <= loudness <= LOUDNESS_MAX_DB

    def test_compute_energy_in_0_1(self, sine_wave):
        from APP import _compute_energy
        y, sr = sine_wave
        assert 0.0 <= _compute_energy(y) <= 1.0

    def test_compute_acousticness_in_0_1(self, sine_wave):
        from APP import _compute_acousticness
        y, sr = sine_wave
        assert 0.0 <= _compute_acousticness(y, sr) <= 1.0

    def test_compute_speechiness_in_0_1(self, sine_wave):
        from APP import _compute_speechiness
        import numpy as np
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not available")
        y, sr = sine_wave
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        assert 0.0 <= _compute_speechiness(y, sr, mfccs) <= 1.0

    def test_compute_instrumentalness_in_0_1(self, sine_wave):
        from APP import _compute_instrumentalness
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not available")
        y, sr = sine_wave
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        val = _compute_instrumentalness(mfccs)
        assert 0.0 <= val <= 1.0

    def test_compute_instrumentalness_not_always_zero(self, sine_wave):
        """Regression test: previous bug returned 0.0 for every track."""
        from APP import _compute_instrumentalness
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa not available")
        y, sr = sine_wave
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        val = _compute_instrumentalness(mfccs)
        assert val != 0.0, "instrumentalness should not always be zero (regression test)"

    def test_compute_liveness_in_0_1(self, sine_wave):
        from APP import _compute_liveness
        y, sr = sine_wave
        assert 0.0 <= _compute_liveness(y, sr) <= 1.0

    def test_compute_valence_in_0_1(self, sine_wave):
        from APP import _compute_valence
        y, sr = sine_wave
        assert 0.0 <= _compute_valence(y, sr) <= 1.0
