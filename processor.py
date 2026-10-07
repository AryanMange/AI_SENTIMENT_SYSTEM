"""
processor.py
------------
NLP Engine for the Governance Sentiment Monitor.

Four tasks:
  1. Sentiment Analysis   — cardiffnlp/twitter-roberta-base-sentiment
  2. Zero-Shot Dept Tag   — facebook/bart-large-mnli
  3. Topic Modeling       — BERTopic (BERT + UMAP + HDBSCAN)
  4. SHAP Explainability  — word-level attribution for every prediction
  5. Trend Forecasting    — Facebook Prophet (30-day forecast saved to data/forecast.csv)

Run:
    python processor.py
Input:
    data/comments.csv
Output:
    data/analyzed.csv
    data/forecast.csv
    data/shap_top_words.csv
    data/bertopic_topics.csv
"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ── Config ───────────────────────────────────────────────────────────────────

SENTIMENT_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment"
ZEROSHOT_MODEL   = "facebook/bart-large-mnli"
INPUT_CSV        = os.path.join("data", "comments.csv")
OUTPUT_CSV       = os.path.join("data", "analyzed.csv")
FORECAST_CSV     = os.path.join("data", "forecast.csv")
SHAP_CSV         = os.path.join("data", "shap_top_words.csv")
TOPIC_CSV        = os.path.join("data", "bertopic_topics.csv")
BATCH_SIZE       = 16
MAX_TOKEN_LENGTH = 128

DEPARTMENT_LABELS = [
    "Infrastructure and Transport",
    "Public Health Services",
    "Education and Schools",
    "Taxation and Finance",
    "Law and Order",
    "Environment and Sanitation",
    "Social Welfare",
]

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"[processor] Device: {'GPU' if DEVICE == 0 else 'CPU'}")


# ── 1. Sentiment Analysis ─────────────────────────────────────────────────────

def load_sentiment_pipeline():
    print(f"[processor] Loading sentiment model: {SENTIMENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    return pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer,
        device=DEVICE, truncation=True,
        max_length=MAX_TOKEN_LENGTH, batch_size=BATCH_SIZE,
    )


def run_sentiment(pipe, texts: list[str]) -> list[dict]:
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for pred in pipe(batch):
            results.append({
                "sentiment":       LABEL_MAP.get(pred["label"], pred["label"]),
                "sentiment_score": round(pred["score"], 4),
            })
        print(f"\r[processor] Sentiment: {min(100, int((i+BATCH_SIZE)/len(texts)*100))}%",
              end="", flush=True)
    print()
    return results


# ── 2. Zero-Shot Department Classification ────────────────────────────────────

def load_zeroshot_pipeline():
    print(f"[processor] Loading zero-shot model: {ZEROSHOT_MODEL}")
    return pipeline("zero-shot-classification", model=ZEROSHOT_MODEL, device=DEVICE)


def run_zeroshot(pipe, texts: list[str]) -> list[dict]:
    results = []
    for i, text in enumerate(texts):
        try:
            pred = pipe(text, candidate_labels=DEPARTMENT_LABELS, multi_label=False)
            results.append({
                "department":      pred["labels"][0],
                "dept_confidence": round(pred["scores"][0], 4),
            })
        except Exception:
            results.append({"department": "Unclassified", "dept_confidence": 0.0})
        if (i + 1) % 10 == 0 or (i + 1) == len(texts):
            print(f"\r[processor] Zero-shot: {int((i+1)/len(texts)*100)}%",
                  end="", flush=True)
    print()
    return results


# ── 3. BERTopic — Topic Modeling ─────────────────────────────────────────────

def run_bertopic(texts: list[str]) -> tuple[list[int], list[str], pd.DataFrame]:
    """
    Returns:
      topic_ids   — per-document topic number (-1 = outlier)
      topic_names — per-document human-readable topic label
      topics_df   — summary DataFrame saved to TOPIC_CSV
    """
    print("[processor] Running BERTopic topic modeling...")
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
        topic_model = BERTopic(
            vectorizer_model=vectorizer,
            min_topic_size=5,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(texts)

        # Build human-readable label for each topic
        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(TOPIC_CSV, index=False)
        print(f"[processor] BERTopic found {topic_info.shape[0] - 1} topics (excl. outlier topic -1)")

        # Map topic id → short label (top 3 words)
        label_map: dict[int, str] = {}
        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                label_map[-1] = "Miscellaneous"
            else:
                # topic_model.get_topic returns list of (word, score)
                top_words = [w for w, _ in topic_model.get_topic(tid)[:3]]
                label_map[tid] = " / ".join(top_words)

        topic_names = [label_map.get(t, "Unknown") for t in topics]
        return topics, topic_names, topic_info

    except ImportError:
        print("[processor] BERTopic not installed — skipping. pip install bertopic")
        n = len(texts)
        return [-1] * n, ["N/A"] * n, pd.DataFrame()


# ── 4. SHAP Explainability ────────────────────────────────────────────────────

def run_shap(texts: list[str], sentiment_labels: list[str]) -> pd.DataFrame:
    """
    Runs SHAP on a sample of comments (max 50 — SHAP is slow on CPU).
    Saves top contributing words per prediction to SHAP_CSV.
    Returns DataFrame with columns: text, sentiment, top_positive_words, top_negative_words
    """
    print("[processor] Running SHAP explainability (sample of up to 50 comments)...")
    try:
        import shap
        from transformers import pipeline as hf_pipeline

        # Use a small sample to keep runtime manageable
        sample_size = min(50, len(texts))
        indices     = list(range(sample_size))
        sample_texts  = [texts[i]  for i in indices]
        sample_labels = [sentiment_labels[i] for i in indices]

        # SHAP needs a plain predict function
        sent_pipe = hf_pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            top_k=None,          # return all class scores
        )

        explainer = shap.Explainer(sent_pipe)
        shap_values = explainer(sample_texts)

        rows = []
        for i, text in enumerate(sample_texts):
            # shap_values[i] has shape (n_tokens, n_classes)
            # classes: LABEL_0=NEG, LABEL_1=NEU, LABEL_2=POS
            try:
                tokens = shap_values[i].data          # token strings
                values = shap_values[i].values        # (n_tokens, 3)

                # For the predicted class, get top contributing tokens
                pred_label = sample_labels[i]
                class_idx  = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}.get(pred_label, 0)

                token_scores = list(zip(tokens, values[:, class_idx]))
                # Sort by absolute value
                token_scores.sort(key=lambda x: abs(x[1]), reverse=True)

                top_pos = [t for t, v in token_scores if v > 0][:3]
                top_neg = [t for t, v in token_scores if v < 0][:3]

                rows.append({
                    "text":               text[:120],
                    "sentiment":          pred_label,
                    "top_positive_words": ", ".join(top_pos),
                    "top_negative_words": ", ".join(top_neg),
                })
            except Exception:
                rows.append({
                    "text": text[:120], "sentiment": sample_labels[i],
                    "top_positive_words": "", "top_negative_words": "",
                })

        shap_df = pd.DataFrame(rows)
        shap_df.to_csv(SHAP_CSV, index=False)
        print(f"[processor] SHAP done — saved {len(shap_df)} explanations → {SHAP_CSV}")
        return shap_df

    except ImportError:
        print("[processor] SHAP not installed — skipping. pip install shap")
        return pd.DataFrame()
    except Exception as e:
        print(f"[processor] SHAP error: {e} — skipping.")
        return pd.DataFrame()


# ── 5. Prophet Trend Forecasting ─────────────────────────────────────────────

def run_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily sentiment scores into a time series and forecasts
    30 days into the future using Prophet.
    Saves combined history + forecast to FORECAST_CSV.
    Returns the forecast DataFrame.
    """
    print("[processor] Running Prophet trend forecast (30 days)...")
    try:
        from prophet import Prophet

        df_time = df.copy()
        df_time["ds"] = pd.to_datetime(df_time["timestamp"])

        # Convert sentiment to numeric score: POS=1, NEU=0, NEG=-1
        score_map = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
        df_time["score"] = df_time["sentiment"].map(score_map)

        # Daily average sentiment score
        daily = (
            df_time.groupby(df_time["ds"].dt.date)["score"]
            .mean()
            .reset_index()
            .rename(columns={"ds": "ds", "score": "y"})
        )
        daily["ds"] = pd.to_datetime(daily["ds"])

        if len(daily) < 5:
            print("[processor] Not enough daily data points for Prophet — skipping.")
            return pd.DataFrame()

        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        m.fit(daily)

        future   = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        # Keep useful columns
        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out["is_forecast"] = out["ds"] > daily["ds"].max()
        out.to_csv(FORECAST_CSV, index=False)
        print(f"[processor] Prophet forecast saved → {FORECAST_CSV}")
        return out

    except ImportError:
        print("[processor] Prophet not installed — skipping. pip install prophet")
        return pd.DataFrame()
    except Exception as e:
        print(f"[processor] Prophet error: {e} — skipping.")
        return pd.DataFrame()


# ── Urgency Scoring ───────────────────────────────────────────────────────────

def compute_urgency(row: pd.Series) -> str:
    if row["sentiment"] == "NEGATIVE":
        return "CRITICAL" if row["sentiment_score"] >= 0.80 else "HIGH"
    elif row["sentiment"] == "NEUTRAL":
        return "MEDIUM"
    return "LOW"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"[processor] ERROR: {INPUT_CSV} not found. Run data_gen.py first.")
        return

    os.makedirs("data", exist_ok=True)
    df    = pd.read_csv(INPUT_CSV)
    texts = df["text"].astype(str).tolist()
    print(f"[processor] Loaded {len(df)} rows from {INPUT_CSV}\n")
    t0 = time.time()

    # ── Step 1: Sentiment ────────────────────────────────────────────────────
    sent_pipe        = load_sentiment_pipeline()
    sentiment_results = run_sentiment(sent_pipe, texts)
    del sent_pipe

    sent_df = pd.DataFrame(sentiment_results)
    df      = pd.concat([df.reset_index(drop=True), sent_df], axis=1)
    df["urgency"] = df.apply(compute_urgency, axis=1)

    total   = len(df)
    pos_cnt = (df["sentiment"] == "POSITIVE").sum()
    df["approval_rating"] = round(pos_cnt / total * 100, 1)

    # ── Step 2: Zero-Shot Dept Classification ────────────────────────────────
    zs_pipe          = load_zeroshot_pipeline()
    zeroshot_results = run_zeroshot(zs_pipe, texts)
    del zs_pipe

    zs_df = pd.DataFrame(zeroshot_results)
    df    = pd.concat([df, zs_df], axis=1)

    # ── Step 3: BERTopic ─────────────────────────────────────────────────────
    topic_ids, topic_names, _ = run_bertopic(texts)
    df["bert_topic_id"]   = topic_ids
    df["bert_topic_name"] = topic_names

    # ── Step 4: SHAP ─────────────────────────────────────────────────────────
    sentiment_labels = df["sentiment"].tolist()
    run_shap(texts, sentiment_labels)

    # ── Step 5: Prophet ──────────────────────────────────────────────────────
    run_prophet(df)

    # ── Save final analyzed CSV ──────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = round(time.time() - t0, 1)
    print(f"\n[processor] ✅ All done in {elapsed}s")
    print("\n── Final Summary ────────────────────────────────────")
    print(df["sentiment"].value_counts().to_string())
    print(f"\nApproval Rating  : {df['approval_rating'].iloc[0]}%")
    print(f"Critical Alerts  : {(df['urgency'] == 'CRITICAL').sum()}")
    if "bert_topic_name" in df.columns:
        print(f"BERTopic topics  : {df['bert_topic_name'].nunique()}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
