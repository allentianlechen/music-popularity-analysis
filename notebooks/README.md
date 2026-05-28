# Research notebooks

This folder contains the executed research notebooks used to compare modeling
approaches for Spotify popularity prediction.

## Files

- `analysis.ipynb`: Main exploratory analysis and model comparison notebook.
- `popularity_random_forest_research.ipynb`: Random Forest comparison run used
  by the current website benchmark.
- `popularity_knn_research.ipynb`: Original OLS and kNN regression research.
- `popularity_knn_research_filtered.ipynb`: kNN/OLS rerun after filtering
  features with very low correlation to popularity.
- `requirements-analysis.txt`: Extra notebook dependencies that aren't needed
  to run the web app.

The website does not need these notebooks to start. They are included so users
can inspect the research behind the deployed model.
