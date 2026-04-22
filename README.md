# Song Popularity Predictor

A web app that predicts how popular a song could be on Spotify based on its audio characteristics. Upload any audio file and get an instant popularity score (0-100) with a breakdown of how each audio feature contributes.

**Live demo:** https://music-popularity-analysis.onrender.com

## How It Works

1. **Upload** an MP3, WAV, FLAC, or M4A file
2. The server extracts 9 audio features using [librosa](https://librosa.org/) signal processing (tempo, energy, danceability, loudness, speechiness, instrumentalness, acousticness, liveness, valence)
3. A Random Forest model trained on ~90,000 Spotify tracks predicts the popularity score
4. The UI shows the score, a tier label, and feature-by-feature insights comparing your track to the dataset average

No Spotify API key is needed. Everything runs locally.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| ML Model | scikit-learn RandomForestRegressor (100 trees) |
| Audio Analysis | librosa (STFT, tempogram, HPSS, chroma, MFCCs) |
| Frontend | Vanilla HTML/CSS/JS (single-page, dark theme) |
| Dataset | [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~90k tracks, 114 genres) |
| Deployment | Render (gunicorn) |

## Project Structure

```
analysisProject/
  APP.py            Flask server — routes + audio feature extraction
  analyze.py        Trains the Random Forest model, saves model.pkl
  clean.py          Preprocesses dataset.csv into cleaned.csv
  config.py         Shared feature lists (SLIDER_FEATURES, EXTRA_FEATURES)
  index.html        Single-page UI (upload, gauge, insights, importance chart)
  test_app.py       pytest test suite (38 tests)
  analysis.ipynb    Comprehensive EDA, model comparison, SHAP, residual analysis
  eda.py            Standalone EDA script (popularity distribution charts)
  genre_analysis.py Per-genre model analysis script
  cleaned.csv       Preprocessed dataset (tracked in git, 15 MB)
  model.pkl         Trained model (gitignored, regenerate with analyze.py)
  requirements.txt  Python dependencies
  render.yaml       Render deployment config
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

# Install dependencies
pip install -r requirements.txt

# Train the model (uses cleaned.csv already in the repo)
python3 analyze.py

# Start the server
python3 APP.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

### Running Tests

```bash
python3 -m pytest test_app.py -v
```

## Audio Feature Extraction

All 9 features are computed with librosa to approximate Spotify's audio analysis:

| Feature | Method |
|---------|--------|
| **Tempo** | Tempogram autocorrelation with half/double disambiguation + PLP cross-check |
| **Loudness** | Energy-domain mean (LUFS-style) of active frames above -50 dB |
| **Energy** | Active loudness + HF energy ratio (>2 kHz) + spectral centroid |
| **Danceability** | Inter-beat-interval consistency + beat-frame onset strength |
| **Speechiness** | MFCC delta dynamics + vocal-band spectral flux + zero-crossing rate |
| **Instrumentalness** | Inverse of vocal presence (MFCC variance + vibrato detection at 4.5-7 Hz) |
| **Acousticness** | HPSS harmonic ratio + spectral flatness penalty + centroid variability |
| **Liveness** | Quiet-section noise floor DR ratio + mid-band spectral contrast variation |
| **Valence** | Krumhansl-Kessler key mode + tempo + spectral tilt + harmonic/percussive ratio |

## Model Performance

| Metric | Value |
|--------|-------|
| R² (full model, leakage-free) | 0.446 |
| R² (audio-only, no artist) | 0.257 |
| MAE | 9.7 popularity points |
| CV R² (5-fold, pipeline) | 0.439 +/- 0.008 |

The gap between full and audio-only R² reflects reality: song popularity is driven more by artist fame, marketing, and playlist placement than audio characteristics alone. The model is honest about this.

## Analysis & Findings

See [`analysis.ipynb`](analysis.ipynb) for the full exploratory data analysis and model comparison.

**Key findings:**

1. **Audio features alone explain ~8% of popularity variance** (univariate R² near zero for each feature). Popularity is not about how a song sounds.
2. **Artist fame is the dominant predictor.** Adding `artist_avg_popularity` (computed from training data only, no leakage) jumps R² from ~0.08 to ~0.45.
3. **The remaining ~55% is unexplained** — driven by playlist placement, marketing, release timing, and algorithmic recommendations.
4. **Data leakage was identified and fixed.** The original implementation computed artist averages from the full dataset (including test rows). The corrected `ArtistAvgTransformer` computes per-artist means from training data only and is used in a sklearn Pipeline for proper cross-validation.
5. **Statistical significance != practical significance.** With 90k observations, every audio feature correlates "significantly" with popularity (p < 0.05), but all are practically weak (|r| < 0.15).
6. **Model comparison:** Dummy, Ridge, RandomForest, and XGBoost were compared with hyperparameter tuning. Results, SHAP interpretability, and residual analysis are in the notebook.

## Deployment

The app is configured for [Render](https://render.com):

1. Push this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com) > **New** > **Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and configures everything
5. Click **Create Web Service** and wait for the build (~5 min)

The build command runs `analyze.py` to generate `model.pkl` from `cleaned.csv`, then gunicorn serves the Flask app.

### Manual Deployment

If deploying elsewhere, the key steps are:

```bash
pip install -r requirements.txt
python3 analyze.py                  # generates model.pkl
gunicorn APP:app --bind 0.0.0.0:$PORT --timeout 120
```

The `--timeout 120` flag is needed because audio analysis can take up to 60 seconds for long files.

## Security

- File uploads validated against an extension allowlist
- 50 MB upload size limit
- No secrets in source code
- model.pkl integrity can be verified via `MODEL_PKL_SHA256` env var
- Content Security Policy header set
- Generic error messages (no stack traces exposed to users)

## License

This project was built as a learning exercise in ML + web development.
Dataset source: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) on Kaggle.
