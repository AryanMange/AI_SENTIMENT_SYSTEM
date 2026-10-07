# 🏛️ GovSentinel — AI-Based Public Sentiment Monitoring System for Governance

A complete, end-to-end ML project that collects public opinions from social media,
analyzes sentiment using transformer models, classifies them into government departments,
and presents everything on an interactive real-time dashboard.

---

## 📁 Project Structure

```
sentiment_gov/
│
├── data/                        # Auto-created by scripts
│   ├── comments.csv             # 500 synthetic comments (from data_gen.py)
│   └── analyzed.csv            # Sentiment + dept labels (from processor.py)
│
├── models/                      # HuggingFace models cached here (auto)
│
├── src/                         # (future: modular source code)
│
├── data_gen.py                  # Step 1 — Generate synthetic dataset
├── processor.py                 # Step 2 — Run NLP analysis
├── app.py                       # Step 3 — Launch Streamlit dashboard
├── requirements.txt             # All dependencies
└── README.md                    # This file
```

---

## 🚀 Setup — Run in 3 Steps

### Prerequisites
- Python 3.10 or higher
- 4 GB RAM minimum (8 GB recommended)
- Internet connection (to download models first time only)

---

### Step 1 — Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all libraries
pip install -r requirements.txt
```

---

### Step 2 — Generate the Dataset

```bash
python data_gen.py
```

This creates `data/comments.csv` with 500 synthetic public comments about:
- Health services
- Education
- Transport
- Taxes

Includes positive, negative, neutral, and sarcastic comments.

---

### Step 3 — Run the NLP Analysis

```bash
python processor.py
```

This runs two AI models:
1. **cardiffnlp/twitter-roberta-base-sentiment** — Classifies each comment as POSITIVE / NEUTRAL / NEGATIVE
2. **facebook/bart-large-mnli** — Zero-shot classification into government departments

⚠️ First run downloads ~1.5 GB of model weights. Subsequent runs are fast (models cached).
⏱️ On CPU: ~15–30 minutes for 500 comments. On GPU: ~2–3 minutes.

Output saved to `data/analyzed.csv`

---

### Step 4 — Launch the Dashboard

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

---

## 🧪 Quick Demo (Skip processor.py)

If you want to see the dashboard immediately without waiting for the NLP models,
the dashboard automatically falls back to `data/comments.csv` using the
`true_sentiment` labels. Just run:

```bash
python data_gen.py
streamlit run app.py
```

The dashboard will work with the synthetic labels until you run `processor.py`.

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| 🗳️ Approval Rating | % of positive comments — overall public satisfaction score |
| 📊 Sentiment by Topic | Grouped bar chart comparing Health, Education, Transport, Taxes |
| 🥧 Sentiment Donut | Overall positive/negative/neutral distribution |
| 🏢 Department Classification | AI-detected government department for each comment |
| 📈 Trend Over Time | Monthly sentiment trend line |
| 🚨 Critical Alerts | Searchable table of high-urgency negative comments |
| 🗺️ City Heatmap | Which cities have the most negative sentiment |

---

## 🤖 Models Used

| Model | Task | Size |
|-------|------|------|
| `cardiffnlp/twitter-roberta-base-sentiment` | Sentiment classification | ~499 MB |
| `facebook/bart-large-mnli` | Zero-shot department tagging | ~1.6 GB |

---

## 🔧 Configuration

Edit top of `processor.py` to change:
- `BATCH_SIZE` — increase for faster GPU processing
- `DEPARTMENT_LABELS` — add/change government department categories
- `MAX_TOKEN_LENGTH` — increase for longer text inputs

---

## 🔄 Real-Time Monitoring (Phase 2)

To connect to live data sources, replace `data_gen.py` with API collectors:

```bash
pip install praw tweepy
```

Then use `praw.Reddit(...)` for Reddit or `tweepy.Client(...)` for Twitter,
pointing to the same `data/comments.csv` output format.

---

## 📈 Extending the Project

| Upgrade | How |
|---------|-----|
| Better accuracy | Replace DistilBERT with `roberta-large` |
| More languages | Use `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| Aspect-based | Add `PyABSA` for per-aspect sentiment |
| Explanations | Add `shap` library + SHAP text explainer |
| Forecasting | Add `prophet` for 30-day sentiment forecast |

---

## ⚠️ Troubleshooting

**Model download slow?**
Models are cached after first download. Run once with internet, then works offline.

**Out of memory error?**
Reduce `BATCH_SIZE` in `processor.py` from 32 to 8 or 4.

**Streamlit not found?**
Run `pip install streamlit` and ensure your virtual environment is active.

**torch not installing?**
Try: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
