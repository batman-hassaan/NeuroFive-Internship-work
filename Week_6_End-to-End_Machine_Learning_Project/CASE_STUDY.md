# Case Study: Spotify Hit Song Predictor

**The problem.** Every year, labels, playlist curators, and independent artists sift through an overwhelming
volume of new music trying to guess which tracks are worth their limited attention — marketing budget,
playlist slots, promotional pushes. That triage is usually done by ear, by a small number of overworked
human curators, which doesn't scale and is inherently subjective. This project asks a narrower, testable
question: does a track's raw audio signature — tempo, energy, danceability, loudness, and similar Spotify-
computed features — carry any real signal about whether it becomes a hit, separate from an artist's existing
fame, label backing, or marketing spend?

**The approach.** Using 89,740 unique tracks across 114 genres from the Spotify Web API, I defined a "hit"
as a track in the top ~6.4% by Spotify's popularity score and trained four classification models —
Logistic Regression, Decision Tree, Random Forest, and XGBoost — on audio features plus genre. Because hits
are rare (a ~14.6:1 imbalance), every model used class-weighting or `scale_pos_weight` correction, and was
evaluated on Precision/Recall/F1/ROC-AUC rather than raw accuracy, which would have been trivially gamed by
just predicting "not a hit" every time.

**The finding.** XGBoost was the strongest model (ROC-AUC 0.78), catching 82% of true hits in the test set.
Genre turned out to be the single strongest predictor — unsurprising, since genre is really a proxy for a
track's whole production style and target audience — but audio features still mattered on their own:
hits skew louder, more danceable, more energetic, and less acoustic/instrumental than average tracks, a
pattern visible in the EDA well before any model was trained.

**The business value.** A tool like this isn't meant to replace a curator's ear — it's meant to save their
time. Deployed as the accompanying Streamlit app, an A&R scout or playlist editor could paste in a batch of
new submissions' audio features and instantly get a shortlist of tracks worth a first listen, cutting a
100-song pile down to the 15–20 most promising candidates before a human ever presses play. Because the
model is tuned toward high recall, it errs on the side of over-including borderline tracks rather than
silently filtering out a future hit — the cost of one extra listen is far lower than the cost of missing a
real one. The same underlying pipeline could be repurposed by independent artists deciding which of their
own demos to lead a release with, or by playlist algorithms looking for an early, catalog-agnostic signal
before a track has accumulated any streaming history of its own.
