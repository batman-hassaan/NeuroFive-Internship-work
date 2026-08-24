# 🎧 Spotify Hit Song Predictor

Predicting whether a song has "hit" potential from its audio characteristics alone — an end-to-end ML
capstone project from raw data to a deployed, interactive app.

> 🔗 **Live demo:** _add your Streamlit Community Cloud URL here after deploying_

## Problem Statement

A&R teams, playlist curators, and independent artists constantly have to guess which unreleased or
under-the-radar tracks are worth pushing. This project asks: **can a song's audio DNA (tempo, energy,
danceability, loudness, etc.) — independent of the artist's existing fame or marketing spend — predict
whether it becomes a Spotify hit?**

This is not meant to replace human judgment. It's a **triage tool**: a fast, data-driven first pass to
surface tracks worth a closer listen out of a large catalog.

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114,000
tracks collected via the Spotify Web API, spanning 114 genres and 31,000+ artists, with audio features
computed by Spotify's audio analysis engine (danceability, energy, valence, tempo, loudness, acousticness,
instrumentalness, liveness, speechiness) plus a 0–100 popularity score.

After deduplicating tracks that appear under multiple genre tags, the working dataset is **89,740 unique
tracks**. `is_hit` is defined as `popularity >= 65`, capturing the top ~6.4% most popular tracks — a
deliberately imbalanced, realistic cutoff.

## Approach

1. **Clean** — dedupe by `track_id`, drop the handful of null rows, convert duration to minutes.
2. **EDA** — class balance, feature distributions (hit vs. not-hit), correlation with popularity, hit rate
   by genre.
3. **Feature engineering** — collapsed the 114-genre long tail to the top 30 + "other"; one-hot encoded
   genre/key/time-signature/mode/explicit; standardized numeric audio features.
4. **Modeling** — trained and compared **Logistic Regression, Decision Tree, Random Forest, and XGBoost**,
   all with class-imbalance handling (`class_weight='balanced'` / `scale_pos_weight`).
5. **Evaluation** — Precision/Recall/F1/ROC-AUC (not accuracy — see case study below for why).
6. **Deployment** — best model (XGBoost) saved as a single `joblib` pipeline (preprocessing + model) and
   served through a Streamlit app.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.624 | 0.126 | **0.817** | **0.218** | **0.779** |
| Random Forest | 0.645 | 0.125 | 0.757 | 0.215 | 0.762 |
| Logistic Regression | 0.563 | 0.111 | 0.831 | 0.196 | 0.754 |
| Decision Tree | 0.552 | 0.112 | 0.866 | 0.199 | 0.736 |

**XGBoost was deployed** — best ROC-AUC and F1. All models were deliberately tuned toward Recall (catching
~77–87% of true hits) at the cost of Precision, the right trade-off for a scouting tool. Genre dummy
variables dominate feature importance, followed by instrumentalness, loudness, acousticness, danceability,
and energy — hits skew louder, more danceable, more energetic, and less purely acoustic/instrumental.

## How to Run

**Notebook (full analysis):**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook Spotify_Hit_Predictor.ipynb
```

**Streamlit app (live demo):**
```bash
pip install -r requirements.txt
streamlit run spotify_app.py
```
Requires `spotify_hit_model.joblib` and `spotify_hit_model_meta.joblib` in the same folder (produced by the
notebook's final cells, already included in this repo).

## Files

- `Spotify_Hit_Predictor.ipynb` — full workflow: cleaning → EDA → feature engineering → 4-model comparison → deployment export
- `spotify_app.py` — Streamlit app
- `spotify_hit_model.joblib` / `spotify_hit_model_meta.joblib` — deployed model + metadata
- `requirements.txt` — app dependencies
- `CASE_STUDY.md` — half-page business write-up
