import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

# ------------------------------------------------------------------ #
#  Page config                                                        #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Spotify Hit Song Predictor",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Theme + custom CSS                                                 #
# ------------------------------------------------------------------ #
SPOTIFY_BLACK   = "#121212"
SPOTIFY_DARK    = "#181818"
SPOTIFY_GRAY    = "#282828"
SPOTIFY_GREEN   = "#1DB954"
SPOTIFY_GREEN_D = "#1aa34a"
SPOTIFY_RED     = "#e22134"
SPOTIFY_TEXT    = "#FFFFFF"
SPOTIFY_MUTED   = "#B3B3B3"

st.markdown(f"""
<style>
    /* ---------- Base ---------- */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {SPOTIFY_BLACK};
        color: {SPOTIFY_TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #000000 0%, #121212 100%);
        border-right: 1px solid #2a2a2a;
    }}
    section[data-testid="stSidebar"] * {{
        color: {SPOTIFY_TEXT} !important;
    }}
    .stApp {{
        background: radial-gradient(1200px 600px at 80% -10%, rgba(29,185,84,0.10), transparent 60%),
                    radial-gradient(900px 500px at 0% 100%, rgba(29,185,84,0.05), transparent 60%),
                    {SPOTIFY_BLACK};
    }}

    /* ---------- Typography ---------- */
    h1, h2, h3, h4 {{
        color: {SPOTIFY_TEXT} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    p, span, li, label {{
        color: {SPOTIFY_MUTED};
    }}

    /* ---------- Cards ---------- */
    .card {{
        background: {SPOTIFY_DARK};
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }}
    .metric-card {{
        background: linear-gradient(135deg, {SPOTIFY_DARK} 0%, #1f1f1f 100%);
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 18px 20px;
    }}
    .hit-card {{
        background: linear-gradient(135deg, rgba(29,185,84,0.18) 0%, rgba(29,185,84,0.05) 100%);
        border: 1px solid rgba(29,185,84,0.55);
        border-radius: 16px;
        padding: 26px;
        box-shadow: 0 10px 40px rgba(29,185,84,0.18);
    }}
    .miss-card {{
        background: linear-gradient(135deg, rgba(226,33,52,0.15) 0%, rgba(226,33,52,0.04) 100%);
        border: 1px solid rgba(226,33,52,0.45);
        border-radius: 16px;
        padding: 26px;
        box-shadow: 0 10px 40px rgba(226,33,52,0.14);
    }}

    /* ---------- Buttons ---------- */
    div.stButton > button {{
        background: {SPOTIFY_GREEN};
        color: #000;
        font-weight: 800;
        border-radius: 30px;
        border: none;
        padding: 12px 28px;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }}
    div.stButton > button:hover {{
        background: {SPOTIFY_GREEN_D};
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(29,185,84,0.4);
    }}
    div.stButton > button:active {{
        transform: scale(0.99);
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {SPOTIFY_GRAY};
        border-radius: 10px 10px 0 0;
        padding: 10px 22px;
        color: {SPOTIFY_MUTED} !important;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: {SPOTIFY_GREEN} !important;
        color: #000 !important;
    }}

    /* ---------- Inputs ---------- */
    .stSlider > div > div > div > div {{
        background: {SPOTIFY_GREEN} !important;
    }}
    div[data-testid="stNumberInput"] > div > div,
    .stSelectbox > div > div {{
        background: {SPOTIFY_GRAY};
        border: 1px solid #3a3a3a;
        border-radius: 10px;
    }}

    /* ---------- Banner ---------- */
    .banner {{
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 60%, rgba(29,185,84,0.25) 100%);
        border-radius: 20px;
        padding: 30px 36px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        margin-bottom: 24px;
    }}
    .banner h1 {{
        font-size: 2.6rem !important;
        margin-bottom: 4px !important;
    }}
    .banner .sub {{
        font-size: 1rem;
        color: {SPOTIFY_MUTED};
    }}
    .pill {{
        display: inline-block;
        background: rgba(29,185,84,0.15);
        color: {SPOTIFY_GREEN};
        font-weight: 600;
        border: 1px solid rgba(29,185,84,0.4);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        margin-right: 8px;
    }}
    .pill-gray {{
        background: rgba(179,179,179,0.08);
        color: {SPOTIFY_MUTED};
        border: 1px solid #3a3a3a;
    }}

    /* ---------- Tooltips ---------- */
    .tip {{
        display: inline-block;
        width: 18px; height: 18px;
        line-height: 18px;
        text-align: center;
        border-radius: 50%;
        background: {SPOTIFY_GRAY};
        color: {SPOTIFY_MUTED};
        font-size: 11px;
        font-weight: bold;
        cursor: help;
        margin-left: 6px;
        border: 1px solid #3a3a3a;
    }}

    /* ---------- Badges ---------- */
    .badge-best {{
        display: inline-block;
        background: {SPOTIFY_GREEN};
        color: #000;
        font-weight: 800;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.72rem;
        letter-spacing: 0.5px;
    }}

    /* ---------- Footer ---------- */
    .footer {{
        border-top: 1px solid #2a2a2a;
        margin-top: 40px;
        padding-top: 18px;
        color: {SPOTIFY_MUTED};
        font-size: 0.85rem;
        text-align: center;
    }}

    /* ---------- Metric ---------- */
    [data-testid="stMetric"] > label {{
        color: {SPOTIFY_MUTED} !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stMetric"] > div {{
        color: {SPOTIFY_TEXT} !important;
        font-weight: 700 !important;
    }}

    /* spinner */
    .stSpinner > div {{
        border-top-color: {SPOTIFY_GREEN} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Artifact loading                                                   #
# ------------------------------------------------------------------ #
@st.cache_resource
def load_artifacts():
    """Load the model and any extra training artifacts saved from the notebook."""
    pipeline = joblib.load("spotify_hit_model.joblib")
    meta     = joblib.load("spotify_hit_model_meta.joblib")

    extras = {}
    for name in [
        "model_comparison.joblib",     # DataFrame of all model metrics
        "roc_curves.joblib",           # dict: model -> (fpr, tpr, auc)
        "confusion_matrices.joblib",   # dict: model -> 2x2 array
        "feature_importances.joblib",  # Series
        "eda_summary.joblib",          # dict with class balance / corr matrix / genre hit rates
    ]:
        try:
            extras[name.replace(".joblib", "")] = joblib.load(name)
        except FileNotFoundError:
            extras[name.replace(".joblib", "")] = None
    return pipeline, meta, extras

pipeline, meta, extras = load_artifacts()

# Friendly references
comp_df      = extras.get("model_comparison")
roc_curves   = extras.get("roc_curves")
cms          = extras.get("confusion_matrices")
importances  = extras.get("feature_importances")
eda_summary  = extras.get("eda_summary")


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def tooltip(text: str):
    return f'<span class="tip" title="{text}">i</span>'

def fmt_pct(x, digits=1):
    try:
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "—"

KEY_NAMES = ["C","C♯/D♭","D","D♯/E♭","E","F","F♯/G♭","G","G♯/A♭","A","A♯/B♭","B"]
genre_choices = sorted(meta.get("genre_choices", ["other"]))


# ------------------------------------------------------------------ #
#  Banner / Header                                                    #
# ------------------------------------------------------------------ #
st.markdown("""
<div class="banner">
    <h1>🎧 Spotify Hit Song Predictor</h1>
    <div class="sub">
        Predicts whether a track has <b>hit-like audio DNA</b> — trained on 90K+ Spotify tracks
        across 114 genres. A triage tool for A&R scouting and playlist curation, not a verdict.
    </div>
    <div style="margin-top:14px;">
        <span class="pill">XGBoost Model</span>
        <span class="pill pill-gray">Target: popularity ≥ 65</span>
        <span class="pill pill-gray">Top ~6% tracks flagged as hits</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Sidebar — Prediction inputs                                        #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("### 🎚️ Track Inputs")
    st.caption("Adjust the audio features below. Predictions update in real time on the **Predict** tab.")

    st.markdown(f"#### Genre {tooltip('Collapsed into top-30 genres + \"other\"')}", unsafe_allow_html=True)
    genre = st.selectbox(
        "Genre",
        genre_choices,
        index=genre_choices.index("other") if "other" in genre_choices else 0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"#### Audio Features {tooltip('Computed by Spotify audio analysis API')}", unsafe_allow_html=True)

    danceability   = st.slider("Danceability",     0.0, 1.0, 0.65, 0.01, help="How suitable a track is for dancing based on tempo regularity, beat strength, and overall stability.")
    energy         = st.slider("Energy",           0.0, 1.0, 0.70, 0.01, help="Perceptual measure of intensity and activity (loud, fast, noisy).")
    valence        = st.slider("Valence",          0.0, 1.0, 0.50, 0.01, help="Musical positiveness — happy/cheerful vs. sad/angry.")
    acousticness   = st.slider("Acousticness",     0.0, 1.0, 0.10, 0.01, help="Confidence the track is acoustic (non-electronic).")
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01, help="Predicts whether a track contains no vocals.")
    liveness       = st.slider("Liveness",         0.0, 1.0, 0.15, 0.01, help="Presence of an audience in the recording (live vs. studio).")
    speechiness    = st.slider("Speechiness",       0.0, 1.0, 0.05, 0.01, help="Spoken-word content (talk-show vs. music).")

    st.divider()
    st.markdown(f"#### Production {tooltip('Physical / structural properties')}", unsafe_allow_html=True)

    tempo          = st.number_input("Tempo (BPM)",        0.0, 250.0, 120.0, 1.0)
    loudness       = st.number_input("Loudness (dB)",     -60.0, 5.0, -6.0, 0.5)
    duration_min   = st.number_input("Duration (min)",    0.5, 15.0,  3.5, 0.1)
    key            = st.selectbox("Key", [str(k) for k in range(12)], index=0,
                                  format_func=lambda k: KEY_NAMES[int(k)])
    mode           = st.selectbox("Mode", ["1", "0"],
                                  format_func=lambda m: "Major" if m == "1" else "Minor")
    time_signature = st.selectbox("Time Signature", ["3", "4", "5"], index=1)
    explicit       = st.selectbox("Explicit", ["0", "1"],
                                  format_func=lambda x: "Yes" if x == "1" else "No")

    st.divider()
    predict_btn = st.button("🔮 Predict Hit Potential", type="primary", use_container_width=True)


# ------------------------------------------------------------------ #
#  Tabs                                                               #
# ------------------------------------------------------------------ #
tab_overview, tab_models, tab_viz, tab_predict = st.tabs([
    "📊 Overview", "🏆 Model Comparison", "📈 Visualizations", "🎯 Predict",
])


# ===================  OVERVIEW  =================== #
with tab_overview:
    if meta and meta.get("test_metrics"):
        m = meta["test_metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC",   f"{m['ROC_AUC']:.3f}",     "primary metric")
        c2.metric("Recall",    fmt_pct(m['Recall']),       "share of hits caught")
        c3.metric("Precision", fmt_pct(m['Precision']),    "of predicted hits, how many real")
        c4.metric("F1",        f"{m['F1']:.3f}",            "precision/recall balance")
        c5.metric("Accuracy",  fmt_pct(m['Accuracy']),     "overall correctness")

    st.markdown("<div class='card' style='margin-top:22px;'>", unsafe_allow_html=True)
    st.markdown("### 🧠 About the model")
    st.markdown(
        f"""
        <p>This dashboard wraps a <b>{meta.get('model_name','XGBoost')}</b> classifier trained on the
        <a href='https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset' target='_blank'>Spotify Tracks Dataset</a>
        (114,000 tracks, 114 genres). The target <code>is_hit = 1 if popularity ≥ 65</code> captures roughly
        the top 6% most popular tracks — a deliberately imbalanced, realistic "hit or not" cutoff.</p>
        <p>Because hits are rare (~6.4%), the model is tuned for <b>recall</b>: it flags most real hits
        but at the cost of precision. Treat its output as a <b>scouting shortlist</b> — a human should still
        review the candidates — not as a final verdict on chart success.</p>
        """, unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 How to use this dashboard")
        st.markdown(
            """
            1. Open the **🎯 Predict** tab (or use the sidebar inputs).
            2. Adjust sliders for the track's audio features.
            3. Click **Predict Hit Potential**.
            4. Read the confidence gauge + verdict card.
            5. Optionally download a JSON report of the prediction.
            """)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### ⚠️ Interpreting results")
        st.markdown(
            """
            - <b>High probability ≠ guaranteed hit.</b> Marketing, artist fame, and playlist placement are not in the model.
            - <b>Low probability ≠ guaranteed flop.</b> Sonic DNA is just one signal.
            - <b>Genre dominates</b> feature importance — production style and audience matter more than any single number.
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ===================  MODEL COMPARISON  =================== #
with tab_models:
    if comp_df is None:
        st.info("No `model_comparison.joblib` found. Re-run the notebook's artifact-saving cell (see footer) to populate this tab.")
    else:
        comp_df = comp_df.copy()
        best_name = meta.get("model_name", comp_df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"])

        st.markdown("### 🏆 Trained models — performance comparison")

        # Highlight best in the table
        def _row_style(row):
            return [f"background-color: rgba(29,185,84,0.15); color: white" if row["Model"] == best_name
                    else "background-color: #181818; color: #b3b3b3" for _ in row]
        styled = (comp_df.style
                  .apply(_row_style, axis=1)
                  .format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1","ROC_AUC"]})
                  .set_properties(**{"text-align": "center", "padding": "8px", "border": "1px solid #2a2a2a"}))
        st.write(styled.to_html(escape=False), unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top:6px;'><span class='badge-best'>★ BEST</span> &nbsp; selected by ROC-AUC: <b>{best_name}</b></p>", unsafe_allow_html=True)

        # Bar chart comparison
        st.markdown("#### 📊 Metric comparison across models")
        bar_df = comp_df.melt(id_vars="Model",
                              value_vars=["Precision","Recall","F1","ROC_AUC","Accuracy"],
                              var_name="Metric", value_name="Score")
        fig_bar = px.bar(bar_df, x="Model", y="Score", color="Metric", barmode="group",
                         color_discrete_sequence=["#1DB954","#e22134","#f9c74f","#577590","#90be6d"],
                         template="plotly_dark")
        fig_bar.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"),
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart
        st.markdown("#### 🕸️ Radar — best model vs. the field")
        radar_metrics = ["Precision","Recall","F1","ROC_AUC"]
        fig_radar = go.Figure()
        for _, row in comp_df.iterrows():
            vals = row[radar_metrics].tolist()
            vals += vals[:1]  # close the loop
            is_best = row["Model"] == best_name
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=radar_metrics + [radar_metrics[0]],
                fill="toself" if is_best else None,
                name=row["Model"],
                line=dict(width=3 if is_best else 1.5,
                          color=SPOTIFY_GREEN if is_best else "#737373"),
                fillcolor="rgba(29,185,84,0.18)" if is_best else None,
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                height=420, template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_radar, use_container_width=True)

        # ROC curves
        if roc_curves:
            st.markdown("#### 📈 ROC curves — all models on one plot")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                         line=dict(dash="dash", color="#555555", width=1)))
            colors = {"Logistic Regression":"#f9c74f",
                      "Decision Tree":"#f94144",
                      "Random Forest":"#577590",
                      "XGBoost":SPOTIFY_GREEN}
            for name, (fpr, tpr, auc) in roc_curves.items():
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                    line=dict(width=3 if name==best_name else 1.5,
                              color=colors.get(name,"#888888"))
                ))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  height=450, template="plotly_dark",
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion matrix for best model
        if cms and best_name in cms:
            st.markdown(f"#### 🧮 Confusion matrix — {best_name}")
            cm = np.array(cms[best_name])
            fig_cm = ff.create_annotated_heatmap(
                cm.tolist(),
                x=["Not Hit","Hit"], y=["Not Hit","Hit"],
                colorscale=[[0,"#1a1a1a"],[0.5,"#535353"],[1,SPOTIFY_GREEN]],
                showscale=True,
            )
            fig_cm.update_layout(height=380, template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 xaxis_title="Predicted", yaxis_title="Actual",
                                 font=dict(color="white"))
            st.plotly_chart(fig_cm, use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("True Negatives",  f"{tn:,}")
            c2.metric("False Positives", f"{fp:,}")
            c3.metric("False Negatives", f"{fn:,}")
            c4.metric("True Positives",  f"{tp:,}")


# ===================  VISUALIZATIONS  =================== #
with tab_viz:
    if importances is None and eda_summary is None:
        st.info("No EDA artifacts found. Re-run the notebook's artifact-saving cell to populate this tab.")

    if importances is not None:
        st.markdown(f"#### 🎯 Top 15 feature importances — {meta.get('model_name','Best model')}")
        imp_df = importances.head(15).sort_values().reset_index()
        imp_df.columns = ["Feature","Importance"]
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[SPOTIFY_GREEN],
                         template="plotly_dark")
        fig_imp.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="white"),
                              xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Genre dummies typically dominate — genre is a strong proxy for production style and audience.")

    if eda_summary is not None:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 📊 Class balance")
            counts = eda_summary.get("class_counts")
            if counts is not None:
                fig_cb = go.Figure()
                fig_cb.add_trace(go.Bar(x=["Not Hit","Hit"], y=[counts[0], counts[1]],
                                       marker_color=[SPOTIFY_GRAY, SPOTIFY_GREEN],
                                       text=[f"{counts[0]:,}", f"{counts[1]:,}"],
                                       textposition="outside"))
                fig_cb.update_layout(height=380, template="plotly_dark",
                                     paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)",
                                     yaxis_title="Track count", font=dict(color="white"))
                st.plotly_chart(fig_cb, use_container_width=True)
                st.caption("Severe class imbalance (~6% hits) — handled via class_weight/scale_pos_weight.")
        with right:
            st.markdown("#### 🎤 Top 10 genres by hit rate")
            ghr = eda_summary.get("genre_hit_rate")
            if ghr is not None:
                top = ghr.head(10).reset_index()
                top.columns = ["Genre","Hit Rate"]
                fig_g = px.bar(top, x="Hit Rate", y="Genre", orientation="h",
                               color_discrete_sequence=[SPOTIFY_GREEN], template="plotly_dark")
                fig_g.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_g, use_container_width=True)

        # Correlation heatmap
        corr = eda_summary.get("correlation")
        if corr is not None:
            st.markdown("#### 🔥 Correlation heatmap — audio features & popularity")
            fig_corr = px.imshow(corr, text_auto=".2f",
                                 color_continuous_scale="RdGy_r", template="plotly_dark")
            fig_corr.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="white"))
            st.plotly_chart(fig_corr, use_container_width=True)


# ===================  PREDICT  =================== #
with tab_predict:
    st.markdown("### 🎯 Hit-potential prediction")
    st.caption("Inputs come from the sidebar. Adjust the sliders, then click below.")

    input_df = pd.DataFrame([{
        "danceability":      danceability,
        "energy":            energy,
        "loudness":          loudness,
        "speechiness":       speechiness,
        "acousticness":      acousticness,
        "instrumentalness":  instrumentalness,
        "liveness":          liveness,
        "valence":           valence,
        "tempo":             tempo,
        "duration_min":      duration_min,
        "explicit":          int(explicit),
        "mode":              int(mode),
        "key":               key,
        "time_signature":    time_signature,
        "genre_grouped":     genre,
    }])

    col_form, col_result = st.columns([1, 1.3])
    with col_form:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Current inputs")

        # Build a clean 2-column preview with uniform types so Arrow doesn't choke
        display_df = input_df.T.copy()
        display_df.columns = ["Value"]
        display_df.index.name = "Feature"
        display_df = display_df.reset_index()
        # Explicit string cast — kills the "Could not convert '0' with type str" warning
        display_df["Value"] = display_df["Value"].astype(str)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=420,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature",  width="small"),
                "Value":   st.column_config.TextColumn("Value",    width="large"),
            },
        )
        btn_predict_page = st.button(
            "🔮 Predict Hit Potential",
            key="predict_page",
            type="primary",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_result:
        if predict_btn or btn_predict_page:
            with st.spinner("Running model..."):
                prediction   = pipeline.predict(input_df)[0]
                probability = float(pipeline.predict_proba(input_df)[0][1])

            # Verdict card
            if prediction == 1:
                st.markdown(f"""
                <div class="hit-card">
                    <h2 style="color:{SPOTIFY_GREEN}; margin-bottom:4px;">🎵 Hit potential detected</h2>
                    <p style="color:white; font-size:1.1rem;">
                        Estimated probability: <b>{probability:.1%}</b>
                    </p>
                    <p style="color:{SPOTIFY_MUTED}; font-size:0.9rem;">
                        Flagged as a candidate worth a second listen — but remember this is a triage tool,
                        not a final verdict.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="miss-card">
                    <h2 style="color:{SPOTIFY_RED}; margin-bottom:4px;">❌ Not flagged as a likely hit</h2>
                    <p style="color:white; font-size:1.1rem;">
                        Estimated probability: <b>{probability:.1%}</b>
                    </p>
                    <p style="color:{SPOTIFY_MUTED}; font-size:0.9rem;">
                        The model doesn't see hit-like audio DNA in this track — but sonic features are only
                        part of the story (marketing & artist fame aren't in the model).
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability*100,
                number={"suffix":"%", "font":{"size":48, "color":"white"}},
                title={"text":"Hit probability", "font":{"color":SPOTIFY_MUTED, "size":14}},
                gauge={
                    "axis": {"range":[0,100], "tickcolor":"#555"},
                    "bar": {"color": SPOTIFY_GREEN if prediction==1 else SPOTIFY_RED},
                    "bgcolor":"rgba(0,0,0,0)",
                    "bordercolor":"#2a2a2a",
                    "steps": [
                        {"range":[0,50],  "color":"rgba(226,33,52,0.25)"},
                        {"range":[50,75], "color":"rgba(249,199,79,0.25)"},
                        {"range":[75,100],"color":"rgba(29,185,84,0.30)"},
                    ],
                    "threshold": {
                        "line":{"color":"white","width":2},
                        "thickness":0.85,
                        "value":50,
                    },
                },
            ))
            fig_gauge.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(t=20,b=10,l=30,r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Threshold context
            st.caption(
                "Hits are only ~6% of tracks in the training data, so this model is tuned to catch "
                "candidates worth a second listen, not to make a final call. Decision threshold: **0.50**."
            )

            # Download report
            report = {
                "timestamp":   datetime.utcnow().isoformat(),
                "model_name":  meta.get("model_name"),
                "prediction":  "hit" if prediction==1 else "not_hit",
                "probability": probability,
                "inputs":      input_df.iloc[0].to_dict(),
                "model_metrics": meta.get("test_metrics"),
            }
            st.download_button(
                "⬇️ Download prediction report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"spotify_hit_prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=False,
            )
        else:
            st.info("👈 Adjust the inputs in the sidebar (or here), then click **Predict Hit Potential**.")


# ------------------------------------------------------------------ #
#  Footer                                                             #
# ------------------------------------------------------------------ #
st.markdown("""
<div class="footer">
    <p>🎧 <b>Spotify Hit Song Predictor</b> — capstone project · trained on the
    <a href="https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset" target="_blank">Spotify Tracks Dataset</a>
    (Kaggle).<br>
    Built with Streamlit · Plotly · scikit-learn · XGBoost. Class imbalance handled via
    <code>scale_pos_weight</code> / <code>class_weight='balanced'</code>.
    <br><br>
    For A&R scouting & playlist curation — <b>not</b> a guarantee of chart success.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Friendly hint if extra artifacts are missing                       #
# ------------------------------------------------------------------ #
missing = [k for k,v in extras.items() if v is None]
if missing:
    with st.expander(f"ℹ️ {len(missing)} optional artifact(s) not found — click to see how to generate them"):
        st.code("""
# Drop this cell into your notebook (after Section 8: Model Comparison) to persist
# everything the enhanced dashboard needs.

import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_curve

# 1) Comparison table — make sure 'results' is the list of dicts from eval_model
joblib.dump(pd.DataFrame(results), 'model_comparison.joblib')

# 2) ROC curves for each fitted pipeline
fpr_tpr = {}
for name, pipe in fitted_pipelines.items():
    proba = pipe.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    fpr_tpr[name] = (fpr.tolist(), tpr.tolist(), float(roc_auc_score(y_test, proba)))
joblib.dump(fpr_tpr, 'roc_curves.joblib')

# 3) Confusion matrices
cms = {name: confusion_matrix(y_test, pipe.predict(X_test)).tolist()
       for name, pipe in fitted_pipelines.items()}
joblib.dump(cms, 'confusion_matrices.joblib')

# 4) Feature importances from best model
joblib.dump(importances, 'feature_importances.joblib')

# 5) EDA summary
joblib.dump({
    'class_counts':   df['is_hit'].value_counts().to_dict(),
    'genre_hit_rate': genre_hit_rate['mean'].head(10),
    'correlation':    corr,
}, 'eda_summary.joblib')
""", language="python")