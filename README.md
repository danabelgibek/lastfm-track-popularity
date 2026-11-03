# Predicting Track Popularity on Last.fm

[One or two sentences in your own words: what you predict, with what data, and the main result.
Example: "Linear regression on 12,035 Last.fm tracks explains about 71% of the variation in the log of
play counts for artists the model has not seen."]

## Data
- Source: Last.fm public API (`tag.getTopTracks`, `track.getInfo`, `artist.getInfo`), collected on October 7, 2026.
- Size: 12,035 tracks by 2,514 artists, 23 features after cleaning. Target: `playcount` (number of plays of a track).
- Dataset (CSV, track and artist names replaced by numeric ids): [paste your Google Drive link here]
- The data come from Last.fm and were used only for a university course project (SDS307 Modern Regression Analysis, SDU).

## What I did
1. Collected the data with the API (`scraper.py`): one request per second, results saved after every request.
2. Cleaned the data (`cleaning.py`): removed 9 uninformative features, handled missing values, grouped categories.
3. Analysis (`analysis.ipynb`): log transformation of the target, split by artist (no artist in both train and test),
   linear regression (OLS), model comparison, diagnostics (residual plots, Q-Q plot, VIF, Breusch-Pagan test).

## Main results
| Model | Features | Test R² |
|---|---|---|
| Baseline (training mean) | 0 | -0.006 |
| A: title and artist name | 10 | 0.012 |
| B: A + artist statistics | 15 | 0.421 |
| C: B + track metadata | 20 | 0.651 |
| D: C + genre and rank (full model) | 34 | 0.714 |
| Final model (D without two redundant features) | 32 | 0.711 |

![Strongest coefficients of the final model](figures/fig7_final_coefficients.png)

### Key findings
- [Finding 1 in your own words, e.g. what matters most]
- [Finding 2]
- [Finding 3]

## Limitations
- [Write 2-3 points in your own words: sample from top lists of tags, associations not causes, non-constant error variance, ...]

## How to run
```bash
pip install -r requirements.txt
# download lastfm_tracks_public.csv (link above) and put it next to analysis.ipynb
jupyter notebook analysis.ipynb
```
`scraper.py` needs a Last.fm API key in the environment variable `LASTFM_API_KEY` and takes several hours; it is not
needed to run the notebook. `report.pdf` is the full report of the project.

## Notes
[Optional: one honest line about how you worked, e.g. that you used an AI assistant for help with code and wording
and checked the results yourself.]
