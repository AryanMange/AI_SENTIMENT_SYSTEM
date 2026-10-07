"""
app.py
------
Streamlit Dashboard — AI-Based Public Sentiment Monitoring for Governance

Run:
    streamlit run app.py

Requirements:
    pip install -r requirements.txt
    python data_gen.py        # generates data/comments.csv
    python processor.py       # generates data/analyzed.csv
"""

import os
import subprocess
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GovSentinel — Public Sentiment Monitor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f0f2f6; }

    /* Metric card */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Header strip */
    .header-strip {
        background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .header-strip h1 { color: white; margin: 0; font-size: 2rem; }
    .header-strip p  { color: #bbdefb; margin: 4px 0 0 0; font-size: 1rem; }

    /* Alert table rows */
    .critical-row { background-color: #ffebee; }
    .high-row     { background-color: #fff3e0; }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a237e;
        border-left: 4px solid #1565c0;
        padding-left: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

ANALYZED_CSV = os.path.join("data", "analyzed.csv")
RAW_CSV      = os.path.join("data", "comments.csv")


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    if os.path.exists(ANALYZED_CSV):
        df = pd.read_csv(ANALYZED_CSV)
    elif os.path.exists(RAW_CSV):
        # Fallback: load raw + use true_sentiment as sentiment
        df = pd.read_csv(RAW_CSV)
        df["sentiment"]        = df["true_sentiment"]
        df["sentiment_score"]  = 0.85
        df["department"]       = df["topic"].map({
            "Health":    "Public Health Services",
            "Education": "Education and Schools",
            "Transport": "Infrastructure and Transport",
            "Taxes":     "Taxation and Finance",
        })
        df["dept_confidence"]  = 0.90
        df["urgency"] = df["sentiment"].map({
            "NEGATIVE": "CRITICAL",
            "NEUTRAL":  "MEDIUM",
            "POSITIVE": "LOW",
        })
        total    = len(df)
        pos      = (df["sentiment"] == "POSITIVE").sum()
        df["approval_rating"] = round(pos / total * 100, 1)
    else:
        st.error("No data found. Please run `python data_gen.py` first.")
        st.stop()
    return df


def needs_processing() -> bool:
    return os.path.exists(RAW_CSV) and not os.path.exists(ANALYZED_CSV)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg",
             width=80)
    st.markdown("## 🏛️ GovSentinel")
    st.markdown("*Public Sentiment Monitoring System*")
    st.divider()

    st.markdown("### ⚙️ Controls")

    # Data generation
    if st.button("🔄 Generate Fresh Data", use_container_width=True):
        with st.spinner("Generating 500 synthetic comments..."):
            subprocess.run(["python", "data_gen.py"], check=True)
        st.cache_data.clear()
        st.success("Data generated!")
        st.rerun()

    # Processing
    if needs_processing():
        st.warning("Raw data found but not analyzed.")
        if st.button("🤖 Run AI Analysis", use_container_width=True, type="primary"):
            with st.spinner("Running NLP models... (this takes a few minutes)"):
                subprocess.run(["python", "processor.py"], check=True)
            st.cache_data.clear()
            st.success("Analysis complete!")
            st.rerun()
    else:
        if os.path.exists(ANALYZED_CSV):
            st.success("✅ AI Analysis Complete")
            if st.button("🤖 Re-run Analysis", use_container_width=True):
                with st.spinner("Re-running NLP models..."):
                    subprocess.run(["python", "processor.py"], check=True)
                st.cache_data.clear()
                st.rerun()

    st.divider()

    st.markdown("### 🔍 Filters")
    df_raw = load_data()

    selected_topics = st.multiselect(
        "Filter by Topic",
        options=df_raw["topic"].unique().tolist(),
        default=df_raw["topic"].unique().tolist(),
    )

    selected_sentiments = st.multiselect(
        "Filter by Sentiment",
        options=["POSITIVE", "NEUTRAL", "NEGATIVE"],
        default=["POSITIVE", "NEUTRAL", "NEGATIVE"],
    )

    selected_platforms = st.multiselect(
        "Filter by Platform",
        options=df_raw["platform"].unique().tolist() if "platform" in df_raw.columns else ["All"],
        default=df_raw["platform"].unique().tolist() if "platform" in df_raw.columns else ["All"],
    )

    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")


# ── Load + Filter Data ────────────────────────────────────────────────────────

df = load_data()

mask = (
    df["topic"].isin(selected_topics) &
    df["sentiment"].isin(selected_sentiments)
)
if "platform" in df.columns and selected_platforms:
    mask = mask & df["platform"].isin(selected_platforms)

df_filtered = df[mask].copy()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-strip">
    <h1>🏛️ GovSentinel — Public Sentiment Monitor</h1>
    <p>Real-time AI analysis of citizen opinions on government services</p>
</div>
""", unsafe_allow_html=True)


# ── KPI Metrics ──────────────────────────────────────────────────────────────

total         = len(df_filtered)
positive_pct  = round((df_filtered["sentiment"] == "POSITIVE").sum() / max(total, 1) * 100, 1)
negative_pct  = round((df_filtered["sentiment"] == "NEGATIVE").sum() / max(total, 1) * 100, 1)
neutral_pct   = round((df_filtered["sentiment"] == "NEUTRAL").sum()  / max(total, 1) * 100, 1)
critical_cnt  = (df_filtered["urgency"] == "CRITICAL").sum() if "urgency" in df_filtered.columns else 0

approval = positive_pct  # approval = % positive comments

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta_color = "normal" if approval >= 30 else "inverse"
    st.metric(
        label="🗳️ Public Approval Rating",
        value=f"{approval}%",
        delta=f"{'Above' if approval >= 30 else 'Below'} 30% threshold",
        delta_color=delta_color,
    )
with col2:
    st.metric("😊 Positive Sentiment", f"{positive_pct}%",
              delta=f"{(df_filtered['sentiment']=='POSITIVE').sum()} comments")
with col3:
    st.metric("😐 Neutral Sentiment", f"{neutral_pct}%",
              delta=f"{(df_filtered['sentiment']=='NEUTRAL').sum()} comments")
with col4:
    st.metric("😡 Negative Sentiment", f"{negative_pct}%",
              delta=f"{(df_filtered['sentiment']=='NEGATIVE').sum()} comments",
              delta_color="inverse")
with col5:
    st.metric("🚨 Critical Alerts", critical_cnt,
              delta="Need immediate attention",
              delta_color="inverse" if critical_cnt > 10 else "normal")

st.divider()


# ── Row 1: Bar Chart + Pie Chart ──────────────────────────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="section-title">📊 Sentiment by Government Topic</div>',
                unsafe_allow_html=True)

    topic_sent = (
        df_filtered.groupby(["topic", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    color_map = {
        "POSITIVE": "#4CAF50",
        "NEUTRAL":  "#FF9800",
        "NEGATIVE": "#F44336",
    }

    fig_bar = px.bar(
        topic_sent,
        x="topic",
        y="count",
        color="sentiment",
        color_discrete_map=color_map,
        barmode="group",
        labels={"topic": "Government Sector", "count": "Number of Comments", "sentiment": "Sentiment"},
        text_auto=True,
    )
    fig_bar.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Sentiment",
        margin=dict(t=10, b=10),
        height=340,
        xaxis=dict(tickfont=dict(size=13)),
        bargap=0.2,
    )
    fig_bar.update_traces(textposition="outside", textfont_size=11)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">🥧 Overall Sentiment Distribution</div>',
                unsafe_allow_html=True)

    sent_counts = df_filtered["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment", "count"]

    fig_pie = px.pie(
        sent_counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map=color_map,
        hole=0.45,
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.05 if s == "NEGATIVE" else 0 for s in sent_counts["sentiment"]],
    )
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10),
        height=340,
        paper_bgcolor="white",
    )
    # Center annotation
    fig_pie.add_annotation(
        text=f"{approval}%<br>Approval",
        x=0.5, y=0.5,
        font=dict(size=18, color="#1a237e", family="Arial"),
        showarrow=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ── Row 2: Department Breakdown ───────────────────────────────────────────────

if "department" in df_filtered.columns:
    st.markdown('<div class="section-title">🏢 AI-Detected Department Classification</div>',
                unsafe_allow_html=True)

    dept_sent = (
        df_filtered.groupby(["department", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig_dept = px.bar(
        dept_sent,
        x="count",
        y="department",
        color="sentiment",
        color_discrete_map=color_map,
        orientation="h",
        barmode="stack",
        labels={"department": "", "count": "Number of Comments"},
        text_auto=True,
    )
    fig_dept.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=320,
        margin=dict(t=10, b=10, l=200),
        yaxis=dict(tickfont=dict(size=11)),
        legend_title="Sentiment",
    )
    fig_dept.update_traces(textposition="inside", textfont_size=10)
    st.plotly_chart(fig_dept, use_container_width=True)

st.divider()


# ── Row 3: Prophet Forecast + BERTopic ──────────────────────────────────────

FORECAST_CSV = os.path.join("data", "forecast.csv")
TOPIC_CSV    = os.path.join("data", "bertopic_topics.csv")
SHAP_CSV     = os.path.join("data", "shap_top_words.csv")

st.markdown('<div class="section-title">📈 Sentiment Trend + 30-Day Prophet Forecast</div>',
            unsafe_allow_html=True)

if os.path.exists(FORECAST_CSV):
    forecast_df = pd.read_csv(FORECAST_CSV)
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
    history_df  = forecast_df[~forecast_df["is_forecast"]]
    future_df   = forecast_df[forecast_df["is_forecast"]]
    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(
        x=history_df["ds"], y=history_df["yhat"], mode="lines+markers",
        name="Historical Trend", line=dict(color="#1565C0", width=2), marker=dict(size=4),
    ))
    fig_prophet.add_trace(go.Scatter(
        x=future_df["ds"], y=future_df["yhat"], mode="lines",
        name="30-Day Forecast", line=dict(color="#FF6F00", width=2, dash="dash"),
    ))
    fig_prophet.add_trace(go.Scatter(
        x=pd.concat([future_df["ds"], future_df["ds"][::-1]]),
        y=pd.concat([future_df["yhat_upper"], future_df["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(255,111,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="80% Confidence",
    ))
    fig_prophet.add_hline(y=0, line_dash="dot", line_color="grey",
                          annotation_text="Neutral baseline")
    fig_prophet.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", height=320,
        margin=dict(t=10, b=10),
        xaxis=dict(tickangle=-30, title="Date"),
        yaxis=dict(title="Avg Sentiment (+1=Positive, -1=Negative)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_prophet, use_container_width=True)
    st.caption("Orange dashed = Prophet 30-day forecast with 80% confidence band.")
else:
    if "timestamp" in df_filtered.columns:
        df_time = df_filtered.copy()
        df_time["date"] = pd.to_datetime(df_time["timestamp"]).dt.to_period("M").astype(str)
        trend = df_time.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig_t = px.line(trend, x="date", y="count", color="sentiment",
                        color_discrete_map=color_map, markers=True)
        fig_t.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig_t, use_container_width=True)
    st.caption("Run `python processor.py` to enable the Prophet 30-day forecast.")

st.divider()

# ── BERTopic ──────────────────────────────────────────────────────────────────

if "bert_topic_name" in df_filtered.columns:
    st.markdown('<div class="section-title">🧠 BERTopic — Auto-Discovered Governance Topics</div>',
                unsafe_allow_html=True)
    col_t1, col_t2 = st.columns([2, 3])
    with col_t1:
        tfreq = (df_filtered[df_filtered["bert_topic_id"] != -1]
                 .groupby("bert_topic_name").size().reset_index(name="Posts")
                 .sort_values("Posts", ascending=False).head(10))
        st.dataframe(tfreq.rename(columns={"bert_topic_name": "Discovered Topic"}),
                     use_container_width=True, height=260)
    with col_t2:
        tsent = (df_filtered[df_filtered["bert_topic_id"] != -1]
                 .groupby(["bert_topic_name", "sentiment"]).size().reset_index(name="count"))
        fig_bt = px.bar(tsent, x="count", y="bert_topic_name", color="sentiment",
                        color_discrete_map=color_map, orientation="h", barmode="stack",
                        labels={"bert_topic_name": "", "count": "Posts"})
        fig_bt.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=260, margin=dict(t=5, b=5, l=160))
        st.plotly_chart(fig_bt, use_container_width=True)
    st.caption("Topics auto-discovered by BERTopic. Labels = top 3 keywords. Topic -1 = outliers.")
    st.divider()

st.divider()


# ── Row 4: Critical Alerts Table ─────────────────────────────────────────────

st.markdown('<div class="section-title">🚨 Critical Negative Alerts — Immediate Government Attention Required</div>',
            unsafe_allow_html=True)

if "urgency" in df_filtered.columns:
    df_alerts = df_filtered[df_filtered["urgency"].isin(["CRITICAL", "HIGH"])].copy()
else:
    df_alerts = df_filtered[df_filtered["sentiment"] == "NEGATIVE"].copy()
    df_alerts["urgency"] = "CRITICAL"

df_alerts = df_alerts.sort_values(
    by="sentiment_score" if "sentiment_score" in df_alerts.columns else "topic",
    ascending=False
)

# Search box
search_query = st.text_input(
    "🔍 Search alerts by keyword",
    placeholder="e.g. hospital, road, tax, teacher...",
)
if search_query.strip():
    df_alerts = df_alerts[
        df_alerts["text"].str.contains(search_query.strip(), case=False, na=False)
    ]

# Topic filter for alerts
alert_topics = st.multiselect(
    "Filter alerts by topic",
    options=df_filtered["topic"].unique().tolist(),
    default=df_filtered["topic"].unique().tolist(),
    key="alert_topic_filter",
)
df_alerts = df_alerts[df_alerts["topic"].isin(alert_topics)]

st.caption(f"Showing **{len(df_alerts)}** alerts")

# Display columns
show_cols = ["urgency", "topic", "text", "sentiment_score", "platform", "city"]
show_cols = [c for c in show_cols if c in df_alerts.columns]

if len(df_alerts) == 0:
    st.info("No critical alerts match the current filters.")
else:
    # Color-code urgency
    def color_urgency(val):
        if val == "CRITICAL":
            return "background-color: #ffcdd2; color: #b71c1c; font-weight: bold;"
        elif val == "HIGH":
            return "background-color: #ffe0b2; color: #e65100; font-weight: bold;"
        return ""

    styled = (
        df_alerts[show_cols]
        .reset_index(drop=True)
        .rename(columns={
            "urgency":         "⚠️ Urgency",
            "topic":           "Department",
            "text":            "Citizen Comment",
            "sentiment_score": "Confidence",
            "platform":        "Platform",
            "city":            "City",
        })
    )

    # Paginate
    PAGE_SIZE = 15
    total_alerts = len(styled)
    max_pages = max(1, (total_alerts - 1) // PAGE_SIZE + 1)
    page = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
    start = (page - 1) * PAGE_SIZE
    end   = start + PAGE_SIZE

    st.dataframe(
        styled.iloc[start:end],
        use_container_width=True,
        height=420,
        column_config={
            "⚠️ Urgency":      st.column_config.TextColumn(width="small"),
            "Department":      st.column_config.TextColumn(width="medium"),
            "Citizen Comment": st.column_config.TextColumn(width="large"),
            "Confidence":      st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.2f", width="small"
            ),
            "Platform":        st.column_config.TextColumn(width="small"),
            "City":            st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(f"Page {page} of {max_pages}  |  {total_alerts} total alerts")

st.divider()


# ── Row 5: City Heatmap ───────────────────────────────────────────────────────

if "city" in df_filtered.columns:
    st.markdown('<div class="section-title">🗺️ Negativity by City</div>',
                unsafe_allow_html=True)

    city_neg = (
        df_filtered[df_filtered["sentiment"] == "NEGATIVE"]
        .groupby("city")
        .size()
        .reset_index(name="negative_count")
        .sort_values("negative_count", ascending=True)
    )

    fig_city = px.bar(
        city_neg,
        x="negative_count",
        y="city",
        orientation="h",
        color="negative_count",
        color_continuous_scale="Reds",
        labels={"negative_count": "Negative Comments", "city": "City"},
        text_auto=True,
    )
    fig_city.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
        margin=dict(t=10, b=10, l=120),
        coloraxis_showscale=False,
    )
    fig_city.update_traces(textposition="outside")
    st.plotly_chart(fig_city, use_container_width=True)

st.divider()


# ── SHAP Explainability Section ──────────────────────────────────────────────

if os.path.exists(SHAP_CSV):
    st.markdown('<div class="section-title">🔍 SHAP Explainability — Why Did the Model Predict This?</div>',
                unsafe_allow_html=True)
    st.caption("SHAP (SHapley Additive Explanations) shows which words pushed each prediction "
               "toward positive or negative. This is how policymakers can trust the AI output.")
    shap_df = pd.read_csv(SHAP_CSV)
    shap_show = shap_df[["text", "sentiment", "top_positive_words", "top_negative_words"]].copy()
    shap_show = shap_show.rename(columns={
        "text":               "Comment (sample)",
        "sentiment":          "AI Prediction",
        "top_positive_words": "Words → Positive",
        "top_negative_words": "Words → Negative",
    })
    st.dataframe(shap_show, use_container_width=True, height=320,
                 column_config={
                     "Comment (sample)": st.column_config.TextColumn(width="large"),
                     "AI Prediction":    st.column_config.TextColumn(width="small"),
                     "Words → Positive": st.column_config.TextColumn(width="medium"),
                     "Words → Negative": st.column_config.TextColumn(width="medium"),
                 })
    st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='text-align:center; color:#9e9e9e; font-size:0.85rem; padding:16px 0;'>
    GovSentinel  |  AI-Based Public Sentiment Monitoring System for Governance<br>
    Powered by <b>cardiffnlp/twitter-roberta-base-sentiment</b> + <b>facebook/bart-large-mnli</b><br>
    Built with Streamlit · Plotly · HuggingFace Transformers
</div>
""", unsafe_allow_html=True)

# ── Real-Time Live Feed ───────────────────────────────────────────────────────

st.markdown('<div class="section-title">🔴 Real-Time Live Feed — Latest Citizen Posts</div>',
            unsafe_allow_html=True)

_REALTIME_CSV = os.path.join("data", "realtime_feed.csv")

col_rt1, col_rt2, col_rt3 = st.columns([2, 2, 3])
with col_rt1:
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False, key="auto_refresh")
with col_rt2:
    if st.button("Refresh Now", key="manual_refresh"):
        st.cache_data.clear()
        st.rerun()
with col_rt3:
    if os.path.exists(_REALTIME_CSV):
        mtime = os.path.getmtime(_REALTIME_CSV)
        st.caption(f"Feed updated: {datetime.fromtimestamp(mtime).strftime('%d %b %Y, %I:%M %p')}")

if os.path.exists(_REALTIME_CSV):
    try:
        rt_df = pd.read_csv(_REALTIME_CSV)
        if len(rt_df) > 0:
            rt_show = rt_df.tail(20).iloc[::-1].copy()
            def _sent_icon(s):
                return {"POSITIVE": "✅ POSITIVE", "NEGATIVE": "❌ NEGATIVE",
                        "NEUTRAL":  "⚪ NEUTRAL"}.get(s, s)
            rt_show["Sentiment"] = rt_show["sentiment"].apply(_sent_icon)
            display_cols = [c for c in ["timestamp", "source", "topic", "text", "Sentiment", "urgency"]
                            if c in rt_show.columns]
            rt_show = rt_show[display_cols].rename(columns={
                "timestamp": "Time", "source": "Source", "topic": "Topic",
                "text": "Live Comment", "urgency": "Urgency"
            })
            st.dataframe(rt_show, use_container_width=True, height=340)
            c1, c2, c3 = st.columns(3)
            c1.metric("Live Posts Collected", len(rt_df))
            rt_neg = round((rt_df["sentiment"] == "NEGATIVE").sum() / max(len(rt_df),1) * 100, 1)
            c2.metric("Live Negative Rate", f"{rt_neg}%")
            rt_crit = int((rt_df.get("urgency","") == "CRITICAL").sum())
            c3.metric("Critical Alerts (live)", rt_crit, delta_color="inverse")
        else:
            st.info("Feed empty. Start the collector to populate.")
    except Exception as e:
        st.warning(f"Could not load feed: {e}")
else:
    st.info("Real-time feed not active. Open a second terminal and run: python realtime_collector.py --simulate")
    with st.expander("How to start real-time monitoring"):
        st.code(
            "# Terminal 2 (simulation — no API key needed)\n"
            "python realtime_collector.py --simulate\n\n"
            "# Terminal 2 (real Reddit API — fill credentials in realtime_collector.py first)\n"
            "python realtime_collector.py\n\n"
            "# Run once for testing\n"
            "python realtime_collector.py --simulate --once",
            language="bash"
        )

# auto-refresh logic
if auto_refresh:
    import time as _t
    _t.sleep(60)
    st.rerun()

st.divider()
