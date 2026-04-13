"""
clean.py — Step 1: Data Cleaning
=================================
Loads dataset.csv, cleans it, and saves cleaned.csv
Run: python3 clean.py
"""

import pandas as pd

# Load
df = pd.read_csv("dataset.csv")
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# Drop the leftover index column
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

# Drop rows with missing track name / artist
before = len(df)
df.dropna(subset=["track_name", "artists", "album_name"], inplace=True)
print(f"Dropped {before - len(df)} rows with missing name/artist")

# Handle duplicates — same track appears multiple times across genres
# Keep the row with the highest popularity score for each track_id
before = len(df)
df = df.sort_values("popularity", ascending=False)
df = df.drop_duplicates(subset="track_id", keep="first")
print(f"Dropped {before - len(df)} duplicate tracks (kept highest popularity per track)")

# Drop tracks with zero duration (clearly bad data)
before = len(df)
df = df[df["duration_ms"] > 0]
print(f"Dropped {before - len(df)} tracks with zero duration")

# Convert duration from ms to minutes for readability
df["duration_min"] = (df["duration_ms"] / 60_000).round(2)

# Reset index
df.reset_index(drop=True, inplace=True)

print(f"\nFinal clean dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(df[["track_name", "artists", "popularity", "track_genre"]].head(5).to_string())

# Save
df.to_csv("cleaned.csv", index=False, encoding="utf-8-sig")
print("\nSaved to cleaned.csv")