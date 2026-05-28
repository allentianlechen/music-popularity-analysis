# Audio Popularity Signal

A web app that estimates a song's audio-based popularity signal from its sound. Upload any audio file and get a 0-100 score with a breakdown of the extracted audio features.

The full trained model includes historical artist popularity and has stronger metrics, but upload users do not provide artist, playlist, release, or marketing context. The upload experience is therefore best read as an audio-only appeal estimate, not a direct Spotify popularity forecast.

**Live demo:** https://music-popularity-analysis.onrender.com

## How It Works

1. **Upload** an MP3, WAV, FLAC, or M4A file
2. The server extracts 5 defensible local audio features using [librosa](https://librosa.org/) signal processing (`tempo`, `energy`, `danceability`, `loudness`, `acousticness`)
3. The upload-only Random Forest uses only those 5 extractable features and returns a score interval from tree prediction spread
4. The UI shows the score, uncertainty, extraction confidence, and feature-by-feature insights comparing your track to the dataset average

No Spotify API key is needed. Everything runs locally.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| ML Model | scikit-learn RandomForestRegressor (50 trees, joblib-compressed) |
| Audio Analysis | librosa (STFT, tempogram, HPSS, chroma, MFCCs) |
| Frontend | Vanilla HTML/CSS/JS (single-page, dark theme) |
| Dataset | [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~90k tracks, 114 genres) |
| Deployment | Render (gunicorn) |

## Project Structure

```
analysisProject/
  app.py              Flask server and local entrypoint
  index.html          Single-page UI
  requirements.txt    Runtime Python dependencies
  Makefile            Short commands for setup, start, test, and analyze
  render.yaml         Render deployment config
  cleaned.csv         Preprocessed dataset used by analyze.py
  model.pkl           Trained Random Forest models
  model_metadata.json Lightweight metadata loaded at server startup
  analyze.py          Trains model.pkl and model_metadata.json
  clean.py            Preprocesses dataset.csv into cleaned.csv
  config.py           Shared feature-list constants
  test_app.py         pytest test suite
  notebooks/          Research notebooks
  docs/               Planning notes, implementation logs, and preview files
```

## Running Locally

### Prerequisites

- Python 3.9+
- The Kaggle dataset (`dataset.csv`) if you want to regenerate `cleaned.csv` from scratch

### Setup

```bash
# Clone the repo
git clone https://github.com/allentianlechen/music-popularity-analysis.git
cd music-popularity-analysis

# Install dependencies once
pip install -r requirements.txt

# Start the server
python3 app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

You can also use the included `Makefile`:

```bash
make setup
make start
make test
```

Regenerate `model.pkl` and `model_metadata.json` only when you need to retrain
from `cleaned.csv`:

```bash
make analyze
```

### Running Tests

```bash
python3 -m pytest test_app.py -v
```

## Audio Feature Extraction

The upload flow now reports only the features this local pipeline can defend. Earlier heuristic outputs for `speechiness`, `instrumentalness`, `liveness`, and `valence` were removed from upload analysis because they were too easy to misclassify without speech recognition, vocal/source separation, live-room detection, or mood modeling.

| Feature | Method |
|---------|--------|
| **Tempo** | Tempogram autocorrelation with half/double disambiguation + PLP cross-check |
| **Loudness** | `pyloudnorm` integrated LUFS when available; otherwise energy-domain mean of active frames above -50 dB |
| **Energy** | Active loudness + HF energy ratio (>2 kHz) + spectral centroid |
| **Danceability** | Inter-beat-interval consistency + beat-frame onset strength |
| **Acousticness** | HPSS harmonic ratio + spectral flatness penalty + centroid variability |

## Model Performance

| Metric | Value | What it means |
|--------|-------|---------------|
| Context model R² (random split) | 0.445 | Includes `artist_avg_popularity`, so it reflects historical/contextual artist signal. |
| Upload audio-only R² (random split) | 0.183 | Relevant metric for uploaded tracks, using only the retained defensible local features. |
| Upload audio-only grouped R² by artist | 0.006 | Harder validation showing audio features barely generalize without artist context. |
| Upload MAE | 14.4 popularity points | Upload-model mean absolute error on the held-out split. |
| CV R² (5-fold, pipeline) | 0.436 +/- 0.008 | Leakage-free cross-validation of the full model pipeline. |

The gap between context and audio-only R² reflects reality: song popularity is driven more by artist fame, marketing, release timing, and playlist placement than audio characteristics alone. Upload scoring does not use artist popularity or default hidden context fields; it reports a limited audio-based signal.

## Analysis & Findings

See [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) for the full exploratory data analysis and model comparison.

**Key findings:**

1. **Audio features alone provide limited signal**: random split R² is about 0.18 after removing unreliable upload features, and artist-grouped R² is near zero because artist/context effects dominate.
2. **Artist fame is the dominant predictor.** Adding `artist_avg_popularity` (computed from training data only, no leakage) jumps R² from ~0.25 to ~0.45.
3. **The remaining ~55% is unexplained** — driven by playlist placement, marketing, release timing, and algorithmic recommendations.
4. **Data leakage was identified and fixed.** The original implementation computed artist averages from the full dataset (including test rows). The corrected `ArtistAvgTransformer` computes per-artist means from training data only and is used in a sklearn Pipeline for proper cross-validation.
5. **Statistical significance != practical significance.** With 90k observations, every audio feature correlates "significantly" with popularity (p < 0.05), but all are practically weak (|r| < 0.15).
6. **Model comparison:** Dummy, Ridge, RandomForest, and XGBoost were compared with hyperparameter tuning. Results, SHAP interpretability, and residual analysis are in the notebook.

## Deployment

The app is deployed on [Render](https://render.com) free tier (512 MB RAM).

**Key deployment details:**
- `render.yaml` installs dependencies during build and starts the app with `gunicorn`
- `.python-version` pins Python 3.11.9 (Render defaults to 3.14 which causes compatibility issues)
- `model_metadata.json` is loaded at startup so `/`, `/meta`, and `/health` don't unpickle the 80 MB model
- `model.pkl` is compressed with `joblib`; schema v3 stores both context and upload models, and `/predict` loads it lazily on first use
- `librosa` is lazy-loaded on first audio upload request to keep startup memory low
- `soundfile` and `soxr` are pinned explicitly because the app imports them for low-memory audio loading and resampling
- Env vars: `PYTHON_VERSION=3.11.9`, `NUMBA_CACHE_DIR=/tmp/numba-cache`, `NUMBA_NUM_THREADS=1`

### Self-hosting

```bash
pip install -r requirements.txt
# Run this only when regenerating artifacts from cleaned.csv
python3 analyze.py                  # generates model.pkl and model_metadata.json
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

The `--timeout 120` flag is needed because audio analysis can take up to 60 seconds for long files.

## Security

- File uploads validated against an extension allowlist
- 50 MB upload size limit
- No secrets in source code
- model.pkl integrity can be verified via `MODEL_PKL_SHA256` env var
- Content Security Policy header set by Flask while allowing the current single-file inline CSS/JS frontend
- Generic error messages (no stack traces exposed to users)

## License

This project was built as a learning exercise in ML + web development.
Dataset source: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) on Kaggle.
