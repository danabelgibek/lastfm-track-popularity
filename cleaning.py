"""
cleaning.py - data cleaning for SDS307 HW1 (Option 2, Last.fm tracks).

Input : lastfm_tracks_raw_full.csv  (raw dataset produced by scraper.py, never modified)
Output: lastfm_tracks_cleaned.csv   (cleaned, with track/artist names - for analysis)
        lastfm_tracks_public.csv    (cleaned, names replaced by numeric ids - for sharing)

The script prints a cleaning log that is used in Sec. 3 of the report and
checks the course requirements (>= 10,000 rows and >= 20 features after cleaning).
Missing values are only reported here; they are imputed with the TRAIN median
after the train/test split (see the analysis notebook) to avoid data leakage.
"""
import numpy as np
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
RAW_FILE    = "lastfm_tracks_raw_full.csv"   # raw dataset (never overwritten)
CLEAN_FILE  = "lastfm_tracks_cleaned.csv"    # cleaned, with names
PUBLIC_FILE = "lastfm_tracks_public.csv"     # cleaned, no names (Google Drive)

TARGET       = "playcount"
NON_FEATURES = ["track_name", "artist_name", TARGET, "collected_at"]
NEAR_CONST   = 0.995     # drop feature if one value covers >= 99.5% of rows
MAX_NONE     = 0.80      # drop feature if >80% of values are the placeholder "none"
MIN_ROWS, MIN_FEATURES = 10000, 20       # course requirements

log = []   # cleaning log for the report

# ==========================================
# 1. Load raw data
# ==========================================
df = pd.read_csv(RAW_FILE)
n_rows_raw = len(df)
feat0 = [c for c in df.columns if c not in NON_FEATURES]
print(f"Raw dataset: {df.shape[0]} rows, {len(feat0)} features")

# ==========================================
# 2. Sanity checks on rows (duplicates, invalid target)
# ==========================================
before = len(df)
df = df.drop_duplicates(subset=["artist_name", "track_name"])
log.append(("duplicate tracks removed", before - len(df)))
before = len(df)
df = df[df[TARGET] > 0]
log.append(("rows with playcount <= 0 removed", before - len(df)))

# ==========================================
# 3. Remove constant / near-constant / mostly-"none" features
# ==========================================
drop_cols = []
for col in feat0:
    n_unique = df[col].nunique(dropna=False)
    top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
    if n_unique <= 1:
        drop_cols.append(col); log.append((f"drop '{col}'", "constant"))
    elif top_freq >= NEAR_CONST:
        drop_cols.append(col); log.append((f"drop '{col}'", f"near-constant ({top_freq:.2%})"))
    elif df[col].dtype == object and (df[col] == "none").mean() > MAX_NONE:
        drop_cols.append(col); log.append((f"drop '{col}'", f"{(df[col]=='none').mean():.1%} values are 'none'"))
df = df.drop(columns=drop_cols)

# ==========================================
# 4. Categorical feature: normalize text, group rare categories
# ==========================================
def reduce_categories(s, min_freq=0.01):
    freqs = s.value_counts(normalize=True)
    rare = freqs[freqs < min_freq].index
    return s.where(~s.isin(rare), "other")

df["genre_tag"] = df["genre_tag"].astype(str).str.lower().str.strip()
n_before = df["genre_tag"].nunique()
df["genre_tag"] = reduce_categories(df["genre_tag"])
log.append(("genre_tag categories", f"{n_before} -> {df['genre_tag'].nunique()}"))

# ==========================================
# 5. Missing values: reported here, imputed AFTER the train/test split
# ==========================================
na = df.isna().sum(); na = na[na > 0]
for c, k in na.items():
    log.append((f"missing in '{c}'", f"{k} ({k/len(df):.1%}), train-median imputation after split"))

# ==========================================
# 6. Final verification against the requirements
# ==========================================
features = [c for c in df.columns if c not in NON_FEATURES]
assert len(df) >= MIN_ROWS, f"Only {len(df)} rows!"
assert len(features) >= MIN_FEATURES, f"Only {len(features)} features!"

print("\nCLEANING LOG")
for k, v in log:
    print(f"  - {k}: {v}")
print(f"\nFinal: {len(df)} rows (>= {MIN_ROWS}) | {len(features)} features (>= {MIN_FEATURES})")
print("Features:", features)
print(f"Target '{TARGET}': median={df[TARGET].median():,.0f}, max={df[TARGET].max():,.0f}")

# ==========================================
# 7. Save cleaned and public files
# ==========================================
df.to_csv(CLEAN_FILE, index=False)
pub = df.copy()
pub["artist_id"] = pub["artist_name"].astype("category").cat.codes
pub["track_id"] = range(len(pub))
pub.drop(columns=["artist_name", "track_name"]).to_csv(PUBLIC_FILE, index=False)
print(f"Saved: {CLEAN_FILE}, {PUBLIC_FILE}")
