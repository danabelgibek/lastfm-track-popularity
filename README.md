# Predicting Track Popularity on Last.fm

What makes a music track popular? In this project I collected data about 12,035 tracks from the public Last.fm API
and used linear regression to find which factors are linked to the number of plays of a track.

**Main result:** a linear regression for the logarithm of the play count explains about **71%** of the variation
(test R² = 0.71) on artists the model has not seen. The size of the artist's audience and the track information on
Last.fm (tags, description, album) matter most; the title of a track adds almost nothing.

This is a course project for *SDS307 Modern Regression Analysis* (SDU University). The full report is in
[`report.pdf`](report.pdf).

## Data

- **Source:** Last.fm public API (`tag.getTopTracks`, `track.getInfo`, `artist.getInfo`), collected on October 7, 2026.
- **Size:** 12,035 tracks by 2,514 artists from 14 genre tags; 23 features after cleaning.
- **Target:** `playcount`, the number of plays of a track. It is very skewed, so the models use `log(1 + playcount)`.
- **Features:** title (length, flags), artist name, track information (duration, album, tags, description),
  artist statistics (listeners, plays), and the genre tag.
- **Dataset:** [Google Drive](https://drive.google.com/file/d/136-UX_rI5N2WoEPHlo5aL3l2i0R0QwI5/view?usp=sharing)
  (CSV; track and artist names are replaced by numeric ids).
- The data come from Last.fm and were used only for a university course project. Please read the
  [Last.fm API Terms of Service](https://www.last.fm/api/tos) before you reuse them.

## What I did

1. **Collection** (`scraper.py`): three API methods, one request per second, results saved after every request so the
   collection can continue after a break.
2. **Cleaning** (`cleaning.py`): removed 9 features that carry no information (constant or almost constant), filled the
   321 missing durations with the training median, kept the raw data unchanged.
3. **Analysis** (`analysis.ipynb`):
   - log transformation of the target and of two skewed features;
   - split 80/20 **by artist**, so no artist is in both the training and the test set;
   - linear regression (OLS) with `statsmodels` and `scikit-learn`, genre as dummy variables;
   - comparison of feature groups, a check for leakage, and diagnostics (residual plot, Q-Q plot, VIF,
     Breusch–Pagan test).

## Results

| Model | Features | Test R² |
|---|---|---|
| Baseline (training mean) | 0 | -0.006 |
| A: title and artist name | 10 | 0.012 |
| B: A + artist statistics | 15 | 0.421 |
| C: B + track information | 20 | 0.651 |
| D: C + genre and rank (full model) | 34 | 0.714 |
| Final model (D without two repeated features) | 32 | 0.711 |

![Strongest coefficients of the final model](figures/fig7_final_coefficients.png)

### Key findings

- **The artist matters most.** Artist statistics raise the test R² from 0.01 to 0.42. An artist audience that is 10% larger
  goes with about 4% more plays of a track.
- **Track information on Last.fm matters next.** Tracks with tags have about 3.3 times more plays (R² rises from 0.42
  to 0.65). This is an association, not a cause: tags exist mostly for tracks that are already popular.
- **The title explains almost nothing.** Title and artist-name features alone give R² = 0.012.
- **A random split would be too optimistic.** It gives a test R² of 0.751 instead of 0.714, because the same artist
  appears in both parts.
- A typical prediction is off by a factor of about 2.2, so the model is good for a rough comparison of tracks, not for
  exact forecasts.

### Things that did not work

- Some artist features (number of similar artists and tags, biography) were the same for almost all artists and were
  removed.
- The variable `rank_in_genre` is the position on a page of the tag list (1 to 50), not the real rank in the genre.
- Two features repeated each other (`has_wiki` and `log_artist_playcount`); one of them even got a wrong sign. After
  removing them the largest VIF fell from 70.6 to 6.4 at almost the same accuracy.
- The error variance is not constant and the residuals are not normal, so p-values should be read with care.

## Limitations

- The tracks come from top lists of tags, so popular tracks are over-represented and the results do not describe all
  tracks. Genre sizes depend on how the data were collected.
- The effects are associations, not causes.
- The data are one snapshot (October 2026) without release dates.
- Rows of the same artist are not fully independent (about 4.8 tracks per artist).

## How to run

```bash
pip install -r requirements.txt
# download lastfm_tracks_public.csv (link above) and put it next to analysis.ipynb
jupyter notebook analysis.ipynb
```

`scraper.py` needs a Last.fm API key in the environment variable `LASTFM_API_KEY` and takes several hours, so it is not
needed to run the notebook. `cleaning.py` reads `lastfm_tracks_raw_full.csv`, the output of `scraper.py`.

## Repository structure

```
├── README.md
├── report.pdf          full report of the project
├── scraper.py          data collection with the Last.fm API
├── cleaning.py         data cleaning
├── analysis.ipynb      EDA, models, diagnostics
├── requirements.txt
└── figures/            figures used in this README
```

## Notes

I used an AI assistant (Claude) for help with code and wording. I checked the results myself and I can explain the
methods used in this project.

Author: Dana Belgibek
