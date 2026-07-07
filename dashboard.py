"""
Credit Score Classification – Streamlit Dashboard
Run with:   streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import sys
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark gradient background */
  .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(15, 52, 96, 0.9) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Make sidebar radio texts explicitly highly visible and white */
  div[role="radiogroup"] label, div[role="radiogroup"] label p {
    color: #ffffff !important;
    font-size: 0.98rem !important;
    font-weight: 500 !important;
  }
  
  /* Hover effect for navbar links */
  div[role="radiogroup"] label:hover, div[role="radiogroup"] label:hover p {
    color: #ff758f !important;
    cursor: pointer;
  }

  /* Global light text overrides for all widget labels and markdown texts */
  label, .stWidgetLabel, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
    color: #eaeaea !important;
    font-weight: 500 !important;
  }

  /* Markdown copy text */
  div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] span {
    color: #eaeaea !important;
  }

  /* Target bold headers in form lists */
  strong, b {
    color: #ff758f !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
  }

  /* Slider bounds and ticks labels override */
  div[data-testid="stSlider"] div {
    color: #a0a0c0 !important;
  }

  /* Metric cards */
  .metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 18px 22px;
    text-align: center;
    backdrop-filter: blur(10px);
    color: #eaeaea !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
  .metric-value   { font-size: 2.2rem; font-weight: 700; margin: 0; }
  .metric-label   { font-size: 0.82rem; color: #a0a0c0 !important; font-weight: 600; margin-top: 6px; letter-spacing: 0.08em; text-transform: uppercase; }

  /* Section headers */
  .section-header {
    font-size: 1.15rem; font-weight: 600;
    border-left: 4px solid #e94560;
    padding-left: 10px; margin: 24px 0 12px 0;
    color: #eaeaea;
  }

  /* Score badge */
  .badge-good     { background:#27ae60; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
  .badge-standard { background:#f39c12; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
  .badge-poor     { background:#e74c3c; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }

  /* Plot background override */
  .stPlotlyChart, .stpyplot { background: transparent !important; }

  /* Divider */
  hr { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants / loaders (cached)
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {"Poor": "#e74c3c", "Standard": "#f39c12", "Good": "#27ae60"}
CS_ORDER = ["Poor", "Standard", "Good"]
DARK_BG  = "#1a1a2e"
CARD_BG  = "#16213e"
TEXT_COL = "#eaeaea"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#444",    "axes.labelcolor": TEXT_COL,
    "xtick.color": TEXT_COL,     "ytick.color": TEXT_COL,
    "text.color": TEXT_COL,      "font.family": "DejaVu Sans",
    "grid.color": "#333",        "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})


@st.cache_resource
def load_artifacts():
    model    = joblib.load("model_artifacts/lightgbm_model.joblib")
    profiles = pd.read_csv("model_artifacts/customer_profiles.csv")
    feats, cats = joblib.load("model_artifacts/features_info.joblib")
    return model, profiles, feats, cats


@st.cache_data
def load_train_data():
    return pd.read_csv("Data/train.csv", low_memory=False)


@st.cache_data
def load_submission():
    return pd.read_csv("Data/submission.csv")


@st.cache_data
def load_val_results():
    from src.preprocessing import preprocess_data
    from src.features import compute_lags_and_diffs
    model, profiles, features_list, cat_features = load_artifacts()
    train_raw = load_train_data()
    df_clean  = preprocess_data(train_raw)
    df_lags   = compute_lags_and_diffs(df_clean)
    target_map = {"Poor": 0, "Standard": 1, "Good": 2}
    df_lags["target"]     = df_lags["Credit_Score"].map(target_map)
    df_lags["target_lag1"] = (
        df_lags.groupby("Customer_ID")["target"].shift(1).astype("category")
    )
    df_feat = df_lags.merge(profiles, on="Customer_ID", how="left")
    num_cols = df_feat.select_dtypes(exclude=["category"]).columns
    df_feat[num_cols] = df_feat[num_cols].fillna(0)

    val_df = (
        df_feat[df_feat["Month_Val"] > 6]
        .copy()
        .sort_values(["Customer_ID", "Month_Val"])
    )
    val_df["pred"] = np.nan
    for month in [7, 8]:
        if month > 7:
            prev = val_df[val_df["Month_Val"] == month - 1][
                ["Customer_ID", "pred"]
            ]
            pred_map = dict(zip(prev["Customer_ID"], prev["pred"]))
            idx = val_df.index[val_df["Month_Val"] == month]
            val_df.loc[idx, "target_lag1"] = val_df.loc[
                idx, "Customer_ID"
            ].map(pred_map)
            val_df["target_lag1"] = val_df["target_lag1"].astype("category")
        curr  = val_df[val_df["Month_Val"] == month]
        probs = model.predict(curr[features_list])
        preds = np.argmax(probs, axis=1)
        val_df.loc[curr.index, "pred"]           = preds
        val_df.loc[curr.index, "probs_Poor"]     = probs[:, 0]
        val_df.loc[curr.index, "probs_Standard"] = probs[:, 1]
        val_df.loc[curr.index, "probs_Good"]     = probs[:, 2]
    return val_df


# ── Helper ─────────────────────────────────────────────────────────────────────
def make_fig(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    return fig, ax


def clean_float(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace("_", "")
    if "__" in s: s = s.replace("__", "")
    try: return float(s)
    except: return np.nan


def clean_int(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace("_", "")
    try: return int(s)
    except: return np.nan


def parse_history_age(val):
    if pd.isna(val) or not isinstance(val, str): return np.nan
    m = re.match(r"(\d+)\s+Years\s+and\s+(\d+)\s+Months", val.strip())
    if m: return int(m.group(1)) * 12 + int(m.group(2))
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h1 style='color:#ff758f;font-size:1.5rem;font-weight:700;'>💳 Credit Score AI</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='color:#c0c0d8;font-size:0.85rem;margin-top:2px;'>Powered by LightGBM + Sequential Prediction</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊 Overview", "🔍 EDA", "🤖 Model Performance", "🔮 Live Predictor", "📁 Predictions"],
        label_visibility="collapsed",
    )

model, profiles, features_list, cat_features = load_artifacts()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown(
        "<h1 style='color:#eaeaea;font-size:2rem;'>Credit Score Intelligence Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#b5b5c9;font-size:1rem;'>Automated credit score classification using LightGBM with auto-regressive sequential prediction</p>",
        unsafe_allow_html=True,
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("#ff758f", "83.1%", "Model Accuracy"),
        ("#27ae60", "0.831", "Macro F1 Score"),
        ("#f39c12", "100K", "Training Samples"),
        ("#00d2fc", "86", "Feature Dimensions"),
    ]
    for col, (color, value, label) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color};">{value}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Data overview
    c_left, c_right = st.columns([1.3, 1])
    with c_left:
        st.markdown('<div class="section-header">📋 Dataset Summary</div>', unsafe_allow_html=True)
        train_df = load_train_data()
        summary = pd.DataFrame({
            "Attribute": [
                "Training samples", "Test samples", "Unique customers",
                "Timeline (train)", "Timeline (test)", "Features", "Target classes"
            ],
            "Value": [
                "100,000 rows", "50,000 rows", "12,500 customers",
                "Jan – Aug (8 months)", "Sep – Dec (4 months)",
                "27 raw → 86 engineered", "Poor • Standard • Good"
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with c_right:
        st.markdown('<div class="section-header">🎯 Class Distribution</div>', unsafe_allow_html=True)
        counts = train_df["Credit_Score"].value_counts().reindex(CS_ORDER)
        fig, ax = make_fig((5, 4))
        wedges, _, autotexts = ax.pie(
            counts, labels=CS_ORDER,
            colors=[PALETTE[c] for c in CS_ORDER],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
            textprops={"color": TEXT_COL, "fontsize": 10},
        )
        for at in autotexts: at.set_fontweight("bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Architecture description
    st.markdown('<div class="section-header">🏗️ Model Architecture & Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("🧹 Data Cleaning", "Remove trailing underscores, impossible outliers, junk category values"),
        ("📐 Outlier Capping", "Hard bounds: Age 18-80, Loans 0-15, Inquiries 0-20, etc."),
        ("🔄 Smart Imputation", "Customer-level median imputation → global fallback. Credit history age reconstructed linearly."),
        ("⚙️ Feature Engineering", "Lag-1/Lag-2 temporal features, diff features, 9 loan-type indicators, 6-column profile stats (mean/std/min/max)"),
        ("🤖 LightGBM + Categoricals", "230 boost rounds, 127 leaves, native category encoding for 5 categorical features"),
        ("🔮 Auto-Regressive Prediction", "September → October → November → December. Each month's prediction feeds the next month's target_lag1."),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(steps):
        cols[i%3].markdown(f"""
        <div class="metric-card" style="text-align:left;margin-bottom:10px;">
          <div style="font-weight:700;margin-bottom:6px;color:#e94560;">{title}</div>
          <div style="font-size:0.85rem;opacity:0.8;">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.markdown("<h2 style='color:#eaeaea;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    df = load_train_data()
    df["Age"]               = df["Age"].apply(clean_int)
    df["Annual_Income"]     = df["Annual_Income"].apply(clean_float)
    df["Outstanding_Debt"]  = df["Outstanding_Debt"].apply(clean_float)
    df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].apply(clean_float)
    df["History_Age_Months"] = df["Credit_History_Age"].apply(parse_history_age)
    df.loc[(df["Age"] < 18) | (df["Age"] > 80), "Age"] = np.nan
    df.loc[df["Num_of_Delayed_Payment"] > 28, "Num_of_Delayed_Payment"] = np.nan
    df.loc[df["Monthly_Balance"].apply(clean_float) < 0, "Monthly_Balance"] = np.nan
    df["Occupation"] = df["Occupation"].replace("_______", np.nan).fillna("Unknown")
    df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan).fillna("Unknown")

    num_cols_eda = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "History_Age_Months", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distributions", "🔗 Correlations", "🏢 Categorical", "🔄 Feature Explorer", "📈 Customer Timelines"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        # Age distribution
        with c1:
            st.markdown('<div class="section-header">Age by Credit Score</div>', unsafe_allow_html=True)
            fig, ax = make_fig((6.5, 4))
            for cs, col in PALETTE.items():
                s = df[df["Credit_Score"] == cs]["Age"].dropna()
                ax.hist(s, bins=28, alpha=0.65, color=col, label=cs, edgecolor="none")
                ax.axvline(s.median(), color=col, linestyle="--", linewidth=1.5)
            ax.set_xlabel("Age"); ax.set_ylabel("Count")
            ax.legend(facecolor=CARD_BG, edgecolor="#555", labelcolor=TEXT_COL)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Delayed payments violin
        with c2:
            st.markdown('<div class="section-header">Delayed Payments Distribution</div>', unsafe_allow_html=True)
            plot_df = df[["Credit_Score", "Num_of_Delayed_Payment"]].dropna()
            fig, ax = make_fig((6.5, 4))
            parts = ax.violinplot(
                [plot_df[plot_df["Credit_Score"] == cs]["Num_of_Delayed_Payment"].values for cs in CS_ORDER],
                positions=[1, 2, 3], showmedians=True, showextrema=False,
            )
            for body, col in zip(parts["bodies"], [PALETTE[c] for c in CS_ORDER]):
                body.set_facecolor(col); body.set_alpha(0.6)
            parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
            ax.set_xticks([1, 2, 3]); ax.set_xticklabels(CS_ORDER)
            ax.set_ylabel("# Delayed Payments"); ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Annual income boxplot
        st.markdown('<div class="section-header">Annual Income by Credit Score  (log scale)</div>', unsafe_allow_html=True)
        fig, ax = make_fig((10, 4))
        data = [df[df["Credit_Score"] == cs]["Annual_Income"].dropna() for cs in CS_ORDER]
        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="#999"), capprops=dict(color="#999"),
                        flierprops=dict(marker="o", markersize=2, alpha=0.2,
                                        markerfacecolor="#aaa", markeredgecolor="none"))
        for patch, col in zip(bp["boxes"], [PALETTE[c] for c in CS_ORDER]):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(CS_ORDER)
        ax.set_ylabel("Annual Income (USD)"); ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        num_cols_eda = [
            "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
            "Num_Credit_Card", "Interest_Rate", "Delay_from_due_date",
            "Num_of_Delayed_Payment", "Outstanding_Debt", "Credit_Utilization_Ratio",
            "History_Age_Months", "Total_EMI_per_month",
        ]
        target_num = df["Credit_Score"].map({"Poor": 0, "Standard": 1, "Good": 2})
        corr_df = df[num_cols_eda].copy(); corr_df["Credit_Score_num"] = target_num
        corr = corr_df.corr()
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(10, 130, as_cmap=True),
                    vmax=0.6, vmin=-0.6, center=0, square=True, linewidths=0.4,
                    linecolor="#333", annot=True, fmt=".2f", annot_kws={"size": 7.5},
                    cbar_kws={"shrink": 0.7}, ax=ax)
        ax.tick_params(labelsize=8); plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        c1, c2 = st.columns(2)
        # Credit mix
        with c1:
            st.markdown('<div class="section-header">Credit Mix Breakdown</div>', unsafe_allow_html=True)
            cm_df = df[df["Credit_Mix"].isin(["Good", "Standard", "Bad"])].groupby(
                ["Credit_Mix", "Credit_Score"]
            ).size().unstack(fill_value=0).reindex(columns=CS_ORDER, fill_value=0)
            cm_pct = cm_df.div(cm_df.sum(axis=1), axis=0) * 100
            fig, ax = make_fig((6, 4))
            bot = np.zeros(len(cm_pct))
            for cs, col in zip(CS_ORDER, [PALETTE[c] for c in CS_ORDER]):
                ax.bar(cm_pct.index, cm_pct[cs], bottom=bot, color=col,
                       label=cs, edgecolor=DARK_BG, linewidth=0.5)
                bot += cm_pct[cs].values
            ax.set_ylabel("%"); ax.legend(facecolor=CARD_BG, edgecolor="#555", labelcolor=TEXT_COL)
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Occupation good rate
        with c2:
            st.markdown('<div class="section-header">Good Score Rate by Occupation</div>', unsafe_allow_html=True)
            occ_df = df[df["Occupation"] != "Unknown"]
            occ_rate = occ_df.groupby("Occupation").apply(
                lambda x: (x["Credit_Score"] == "Good").mean()
            ).sort_values(ascending=True)
            fig, ax = make_fig((6, 4.5))
            ax.barh(occ_rate.index, occ_rate.values * 100,
                    color=[plt.cm.RdYlGn(v) for v in occ_rate.values], edgecolor="none")
            ax.set_xlabel("% Good Credit Score")
            ax.axvline(occ_rate.mean() * 100, color="white", linestyle="--", linewidth=1.2)
            ax.grid(True, axis="x", alpha=0.3)
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab4:
        st.markdown('<div class="section-header">🔄 Interactive Numerical Feature Explorer</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.95rem;color:#b5b5c9;'>Compare distribution density and notches across any feature split by target credit score brackets.</p>", unsafe_allow_html=True)
        
        # User dropdown selectbox for numerical columns
        feat_to_explore = st.selectbox("Select target variable to visualize:", num_cols_eda, index=8)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown(f'<div style="font-weight:600;margin-bottom:8px;color:#00d2fc;">KDE Density Distribution for {feat_to_explore}</div>', unsafe_allow_html=True)
            fig, ax = make_fig((6.5, 4.5))
            for cs, col in PALETTE.items():
                s = df[df["Credit_Score"] == cs][feat_to_explore].dropna()
                if not s.empty:
                    sns.kdeplot(s, label=cs, color=col, fill=True, alpha=0.15, ax=ax, linewidth=2)
            ax.set_xlabel(feat_to_explore, color=TEXT_COL)
            ax.set_ylabel("Probability Density", color=TEXT_COL)
            ax.legend(facecolor=CARD_BG, edgecolor="#555", labelcolor=TEXT_COL)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True); plt.close()
            
        with c_p2:
            st.markdown(f'<div style="font-weight:600;margin-bottom:8px;color:#ff758f;">Boxplot Range and Outliers for {feat_to_explore}</div>', unsafe_allow_html=True)
            fig, ax = make_fig((6.5, 4.5))
            clean_lists = [df[df["Credit_Score"] == cs][feat_to_explore].dropna().values for cs in CS_ORDER]
            # Ensure none are empty to avoid matplotlib exceptions
            clean_lists_filtered = [lst if len(lst) > 0 else np.array([0]) for lst in clean_lists]
            box = ax.boxplot(clean_lists_filtered, patch_artist=True, notch=True,
                             medianprops=dict(color="white", linewidth=2),
                             whiskerprops=dict(color="#999"), capprops=dict(color="#999"),
                             flierprops=dict(marker="o", markersize=2.5, alpha=0.15,
                                             markerfacecolor="#aaa", markeredgecolor="none"))
            for patch, col in zip(box['boxes'], [PALETTE[c] for c in CS_ORDER]):
                patch.set_facecolor(col); patch.set_alpha(0.7)
            ax.set_xticks([1, 2, 3]); ax.set_xticklabels(CS_ORDER, color=TEXT_COL)
            ax.set_ylabel(feat_to_explore, color=TEXT_COL)
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab5:
        st.markdown('<div class="section-header">📈 Customer Longitudinal Sequence Visualizer</div>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.95rem;color:#b5b5c9;'>Customer-level longitudinal trajectories across months 1-8. Select a Customer ID to view temporal trend paths.</p>", unsafe_allow_html=True)
        
        # Sample of 10 customer IDs from train dataset
        SAMPLE_CUS_IDS = ['CUS_0xd40', 'CUS_0x44b0', 'CUS_0x82cb', 'CUS_0x7523', 'CUS_0x2009', 'CUS_0xc497', 'CUS_0x44b4', 'CUS_0xbbd3', 'CUS_0x21b1', 'CUS_0x2dbc']
        
        c_select, _ = st.columns([1.5, 2.5])
        with c_select:
            selected_cus = st.selectbox("Select Customer to Track:", SAMPLE_CUS_IDS, index=0)
        
        MONTH_ORDER_MAP = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8}
        cus_df = df[df['Customer_ID'] == selected_cus].copy()
        
        if cus_df.empty:
            st.warning(f"No records found for Customer: {selected_cus}")
        else:
            cus_df['Month_Idx'] = cus_df['Month'].map(MONTH_ORDER_MAP)
            cus_df = cus_df.sort_values('Month_Idx')
            
            # Print Score Progression Badges
            st.markdown("**Longitudinal Credit Score Progression timeline:**")
            badges_prog = []
            for _, row in cus_df.iterrows():
                badge_style = f"badge-{row['Credit_Score'].lower()}"
                badges_prog.append(f"<span style='font-size:0.9rem; margin-right:8px;'>{row['Month']}: <span class='{badge_style}' style='font-size:0.8rem;padding:2px 8px;'>{row['Credit_Score']}</span></span>")
            
            st.markdown(f"<div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); padding:16px; border-radius:12px; margin-bottom:20px; line-height:2.2;'>{' ➔ '.join(badges_prog)}</div>", unsafe_allow_html=True)
            
            # Grid of plots for dynamic variables
            fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
            fig.patch.set_facecolor(DARK_BG)
            
            metrics_settings = [
                ("Outstanding_Debt", "Outstanding Debt ($)", "#ff758f"),
                ("Monthly_Balance", "Monthly Balance ($)", "#00d2fc"),
                ("Num_of_Delayed_Payment", "Delayed Payments Count", "#f39c12")
            ]
            
            for arg_idx, (m_col, label, color_code) in enumerate(metrics_settings):
                ax = axes[arg_idx]
                ax.set_facecolor(CARD_BG)
                
                y_vals = cus_df[m_col].apply(clean_float).values
                x_vals = cus_df['Month'].values
                
                ax.plot(x_vals, y_vals, marker='o', linewidth=2.5, color=color_code, label=label)
                
                # Annotate values on the points
                for x_val, y_val in zip(x_vals, y_vals):
                    if not pd.isna(y_val):
                        ax.annotate(f"{y_val:,.1f}", (x_val, y_val), textcoords="offset points", 
                                    xytext=(0,6), ha='center', fontsize=8, color=TEXT_COL, fontweight="bold")
                                    
                ax.set_ylabel(label, color=TEXT_COL)
                ax.grid(True, alpha=0.25, linestyle="--")
                ax.tick_params(colors=TEXT_COL)
                
            axes[-1].set_xlabel("Timeline Month", color=TEXT_COL)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("<h2 style='color:#eaeaea;'>Model Performance</h2>", unsafe_allow_html=True)

    with st.spinner("Running sequential validation …"):
        val_df = load_val_results()

    y_true  = val_df["target"].astype(int)
    y_pred  = val_df["pred"].astype(int)
    y_probs = val_df[["probs_Poor", "probs_Standard", "probs_Good"]].values

    acc   = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("#ff758f", f"{acc:.1%}",  "Validation Accuracy"),
        ("#27ae60", f"{macro:.3f}", "Macro F1"),
        ("#f39c12", f"{len(val_df):,}", "Validation Rows"),
        ("#00d2fc", "Months 7-8",  "Evaluation Window"),
    ]
    for col, (color, value, label) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{color};">{value}</div>
          <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # Classification report table
    st.markdown('<div class="section-header">📄 Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_true, y_pred, target_names=CS_ORDER, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    c1, c2 = st.columns(2)

    # Confusion matrix
    with c1:
        st.markdown('<div class="section-header">🧩 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
        sns.heatmap(cm_norm, annot=True, fmt=".1f",
                    cmap=sns.color_palette("crest", as_cmap=True),
                    xticklabels=CS_ORDER, yticklabels=CS_ORDER,
                    linewidths=1, linecolor=DARK_BG,
                    annot_kws={"size": 13, "fontweight": "bold"}, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig, use_container_width=True); plt.close()

    # Feature importance
    with c2:
        st.markdown('<div class="section-header">🏆 Top Features (Gain)</div>', unsafe_allow_html=True)
        imp = pd.DataFrame({
            "feature": features_list,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False).head(20)
        fig, ax = make_fig((6, 5.5))
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(imp)))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1],
                color=colors[::-1], edgecolor="none")
        ax.set_xlabel("Feature Gain"); ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(labelsize=7.5)
        st.pyplot(fig, use_container_width=True); plt.close()

    # ROC curves
    st.markdown('<div class="section-header">📉 ROC Curves — One vs. Rest</div>', unsafe_allow_html=True)
    from sklearn.metrics import roc_curve, auc
    fig, ax = make_fig((10, 4))
    for i, (cs, col) in enumerate(zip(CS_ORDER, [PALETTE[c] for c in CS_ORDER])):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{cs} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(facecolor=CARD_BG, edgecolor="#555", labelcolor=TEXT_COL)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Live Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.markdown("<h2 style='color:#eaeaea;'>Live Credit Score Predictor</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='opacity:0.7;'>Enter a customer's financial profile below to predict their credit score bracket.</p>",
        unsafe_allow_html=True,
    )

    # Form inputs
    with st.form("predictor_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Personal Info**")
            age              = st.slider("Age", 18, 80, 35)
            occupation       = st.selectbox("Occupation", ["Scientist","Engineer","Doctor","Lawyer","Teacher","Accountant","Manager","Journalist","Developer","Architect","Musician","Writer","Entrepreneur","Mechanic","Media_Manager","Unknown"])
            annual_income    = st.number_input("Annual Income ($)", 5000, 500000, 50000, step=1000)
            monthly_salary   = st.number_input("Monthly In-Hand Salary ($)", 300, 20000, 4000, step=100)

        with c2:
            st.markdown("**Credit Profile**")
            num_bank_acc  = st.slider("# Bank Accounts", 0, 15, 4)
            num_cc        = st.slider("# Credit Cards", 0, 15, 3)
            interest_rate = st.slider("Interest Rate (%)", 1, 34, 14)
            num_loans     = st.slider("# Loans", 0, 15, 3)
            outstanding_debt = st.number_input("Outstanding Debt ($)", 0, 5000, 1200, step=50)
            credit_util   = st.slider("Credit Utilisation Ratio (%)", 0.0, 100.0, 30.0)
            history_months = st.slider("Credit History Age (months)", 0, 400, 200)

        with c3:
            st.markdown("**Payment Behaviour**")
            delay_days    = st.slider("Avg Days Past Due", 0, 67, 12)
            num_delayed   = st.slider("# Delayed Payments", 0, 28, 5)
            credit_mix    = st.selectbox("Credit Mix", ["Good","Standard","Bad"])
            payment_min   = st.selectbox("Pays Minimum Amount?", ["Yes","No"])
            payment_beh   = st.selectbox("Payment Behaviour", ["High_spent_Large_value_payments","High_spent_Medium_value_payments","High_spent_Small_value_payments","Low_spent_Large_value_payments","Low_spent_Medium_value_payments","Low_spent_Small_value_payments"])
            last_score    = st.selectbox("Last Month Credit Score", ["Poor","Standard","Good"])
            emi           = st.number_input("Total EMI / month ($)", 0.0, 10000.0, 100.0, step=10.0)
            invested      = st.number_input("Monthly Investment ($)", 0.0, 5000.0, 150.0, step=10.0)
            balance       = st.number_input("Monthly Balance ($)", 0.0, 2000.0, 350.0, step=10.0)

        submit = st.form_submit_button("🔮 Predict Credit Score", use_container_width=True)

    if submit:
        last_map = {"Poor": 0, "Standard": 1, "Good": 2}
        # All features expected by the model
        # We build a single row aligned to features_list
        PROFILE_COLS_BASE = [
            "Outstanding_Debt_clean","Delay_from_due_date_clean","Num_of_Delayed_Payment_clean",
            "Changed_Credit_Limit_clean","Num_Credit_Inquiries_clean","Monthly_Balance_clean",
        ]
        LOAN_TYPES = [
            "Payday_Loan","Credit_Builder_Loan","Not_Specified","Home_Equity_Loan",
            "Student_Loan","Mortgage_Loan","Personal_Loan","Debt_Consolidation_Loan","Auto_Loan",
        ]
        CAT_COLS = ["Occupation","Credit_Mix","Payment_of_Min_Amount","Payment_Behaviour"]

        row = {f: 0 for f in features_list}
        # Numeric
        row["Age_clean"] = age
        row["Annual_Income_clean"] = annual_income
        row["Monthly_Inhand_Salary_clean"] = monthly_salary
        row["Num_Bank_Accounts_clean"] = num_bank_acc
        row["Num_Credit_Card_clean"] = num_cc
        row["Interest_Rate_clean"] = interest_rate
        row["Num_of_Loan_clean"] = num_loans
        row["Delay_from_due_date_clean"] = delay_days
        row["Num_of_Delayed_Payment_clean"] = num_delayed
        row["Outstanding_Debt_clean"] = outstanding_debt
        row["Total_EMI_per_month"] = emi
        row["Amount_invested_monthly_clean"] = invested
        row["Monthly_Balance_clean"] = balance
        row["History_Age_Months_Imputed"] = history_months
        row["Credit_Utilization_Ratio"] = credit_util  # may not be a feature
        row["Changed_Credit_Limit_clean"] = 0  # default
        row["Num_Credit_Inquiries_clean"] = 0  # default
        # Profile stats – set to current values as approximations
        for stat in ["mean","std","min","max"]:
            row[f"Outstanding_Debt_clean_{stat}"] = outstanding_debt
            row[f"Delay_from_due_date_clean_{stat}"] = delay_days
            row[f"Num_of_Delayed_Payment_clean_{stat}"] = num_delayed
            row[f"Monthly_Balance_clean_{stat}"] = balance
            row[f"Changed_Credit_Limit_clean_{stat}"] = 0
            row[f"Num_Credit_Inquiries_clean_{stat}"] = 0
        # Categoricals
        row["Occupation"] = occupation
        row["Credit_Mix"] = credit_mix
        row["Payment_of_Min_Amount"] = payment_min
        row["Payment_Behaviour"] = payment_beh
        row["target_lag1"] = last_map[last_score]

        df_input = pd.DataFrame([row])
        for col in CAT_COLS + ["target_lag1"]:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype("category")

        # Align to model features
        for f in features_list:
            if f not in df_input.columns:
                df_input[f] = 0
        df_input = df_input[features_list]

        probs = model.predict(df_input)[0]
        pred_class = CS_ORDER[int(np.argmax(probs))]
        badge_cls  = f"badge-{pred_class.lower()}"
        conf = probs.max() * 100

        st.markdown("<br/>", unsafe_allow_html=True)
        c_res, c_probs = st.columns([1, 1.5])
        with c_res:
            color = PALETTE[pred_class]
            st.markdown(f"""
            <div class="metric-card" style="padding:30px;text-align:center;">
              <div style="font-size:0.9rem;opacity:0.7;margin-bottom:10px;">PREDICTED CREDIT SCORE</div>
              <div class="{badge_cls}" style="font-size:1.8rem;padding:10px 30px;">{pred_class}</div>
              <div style="margin-top:14px;font-size:0.85rem;opacity:0.7;">Confidence: {conf:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        with c_probs:
            st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
            fig, ax = make_fig((6, 3))
            bar_colors = [PALETTE[c] for c in CS_ORDER]
            bars = ax.bar(CS_ORDER, probs * 100, color=bar_colors, edgecolor=DARK_BG, linewidth=1)
            ax.set_ylabel("Probability (%)"); ax.set_ylim(0, 105)
            ax.grid(True, axis="y", alpha=0.3)
            for bar, v in zip(bars, probs * 100):
                ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                        f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold", color=TEXT_COL)
            st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Predictions
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Predictions":
    st.markdown("<h2 style='color:#eaeaea;'>Test Set Predictions</h2>", unsafe_allow_html=True)

    sub = load_submission()

    # Stats
    counts = sub["Credit_Score"].value_counts().reindex(CS_ORDER)
    c1, c2, c3 = st.columns(3)
    for col, cs in zip([c1, c2, c3], CS_ORDER):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value" style="color:{PALETTE[cs]};">{counts[cs]:,}</div>
          <div class="metric-label">{cs}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown('<div class="section-header">📋 Submission Preview</div>', unsafe_allow_html=True)
        st.dataframe(sub.head(50), use_container_width=True, hide_index=True)

    with c_right:
        st.markdown('<div class="section-header">📊 Prediction Distribution</div>', unsafe_allow_html=True)
        fig, ax = make_fig((5.5, 4))
        bar_colors = [PALETTE[c] for c in CS_ORDER]
        bars = ax.bar(CS_ORDER, counts.values, color=bar_colors, edgecolor=DARK_BG, linewidth=1, width=0.55)
        ax.set_ylabel("Count"); ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 80,
                    f"{v:,}", ha="center", fontsize=10, fontweight="bold", color=TEXT_COL)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Download button
    st.markdown("---")
    st.download_button(
        label="⬇️  Download submission.csv",
        data=sub.to_csv(index=False),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )
