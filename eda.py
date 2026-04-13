"""
eda.py — Exploratory Data Analysis
====================================
Standalone script — run once for insight.
Output: eda_popularity.png
Run: python3 eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["popularity"], bins=50, edgecolor="black")
axes[0].set_title("Popularity Distribution (all tracks)")
axes[0].set_xlabel("Popularity Score (0–100)")
axes[0].set_ylabel("Number of Tracks")

genre_median = (
    df.groupby("track_genre")["popularity"]
    .median()
    .sort_values(ascending=False)
    .head(20)
)
genre_median.plot(kind="barh", ax=axes[1])
axes[1].set_title("Top 20 Genres by Median Popularity")
axes[1].set_xlabel("Median Popularity")

plt.tight_layout()
plt.savefig("eda_popularity.png", dpi=150)
print("Saved eda_popularity.png")
