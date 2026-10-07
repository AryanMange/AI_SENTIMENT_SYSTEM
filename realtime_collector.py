"""
realtime_collector.py
---------------------
Real-Time Data Collector for GovSentinel.

Collects live posts from Reddit about governance topics every 30 minutes,
analyzes them with the sentiment model, and appends to data/realtime_feed.csv.
The Streamlit dashboard reads this file and auto-refreshes.

HOW REAL-TIME WORKS (Level 1 — Scheduled Collection):
  1. This script runs in a background terminal window
  2. Every 30 minutes it fetches new Reddit posts about governance topics
  3. Runs each post through the sentiment model
  4. Appends results to data/realtime_feed.csv
  5. app.py reads realtime_feed.csv and auto-refreshes every 60 seconds

Setup:
  pip install praw schedule

Reddit API (free):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "Create App" → choose "script"
  3. Fill in name, redirect URI = http://localhost:8080
  4. Copy your client_id and client_secret below

Run (keep this running in a separate terminal):
  python realtime_collector.py

To run WITHOUT Reddit API (simulation mode):
  python realtime_collector.py --simulate
"""

import os
import sys
import time
import csv
import random
import argparse
import schedule
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_FILE   = os.path.join("data", "realtime_feed.csv")
FETCH_LIMIT   = 25          # posts per subreddit per run
INTERVAL_MIN  = 30          # how often to collect (minutes)
MAX_ROWS      = 2000        # cap file size — rotate oldest rows

# Reddit API credentials (fill these in after creating app at reddit.com/prefs/apps)
REDDIT_CLIENT_ID     = "YOUR_CLIENT_ID_HERE"
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
REDDIT_USER_AGENT    = "GovSentinel/1.0 by YourUsername"

# Subreddits to monitor for governance sentiment
SUBREDDITS = [
    "india",
    "IndiaPolitics",
    "worldnews",
    "politics",
    "PublicFreakout",
    "Government",
]

# Search keywords for governance topics
SEARCH_QUERIES = [
    "government hospital",
    "public school education",
    "road infrastructure",
    "tax policy",
    "government corruption",
    "public health policy",
    "government services",
]

# Simulation mode: pre-written realistic posts (used when no API key)
SIMULATION_POSTS = [
    {"text": "The new government hospital has no doctors again today. Third time this week.",          "topic": "Health",    "source": "Reddit/india"},
    {"text": "Road repair scheme announced but nothing on ground. Pure election stunt.",              "topic": "Transport", "source": "Reddit/india"},
    {"text": "GST refund pending for 8 months. No response from department at all.",                  "topic": "Taxes",     "source": "Reddit/IndiaPolitics"},
    {"text": "Government school digital classroom initiative is actually working in our village!",    "topic": "Education", "source": "Reddit/india"},
    {"text": "Ambulance didn't arrive for 2 hours. Patient died. Who is accountable?",               "topic": "Health",    "source": "Reddit/worldnews"},
    {"text": "Tax portal crashed on deadline day again. Fine imposed. This is injustice.",           "topic": "Taxes",     "source": "Reddit/politics"},
    {"text": "New metro line reduced my commute by 40 minutes. Great infrastructure work.",          "topic": "Transport", "source": "Reddit/india"},
    {"text": "Scholarship money not credited for 6 months. Students suffering.",                     "topic": "Education", "source": "Reddit/india"},
    {"text": "Oh great another ribbon cutting ceremony for a hospital with no medicines inside.",    "topic": "Health",    "source": "Reddit/IndiaPolitics"},
    {"text": "Health minister gave a speech about healthcare. The PHC in his own area has no staff.","topic": "Health",    "source": "Reddit/india"},
    {"text": "Government school building collapsed due to poor maintenance. Thankfully no students.", "topic": "Education", "source": "Reddit/worldnews"},
    {"text": "The highway expansion project is finally complete ahead of schedule. Impressed.",       "topic": "Transport", "source": "Reddit/india"},
    {"text": "Property tax doubled without notice. No improvement in local services either.",        "topic": "Taxes",     "source": "Reddit/india"},
    {"text": "Mid-day meal quality is terrible. Kids are falling sick in our district.",             "topic": "Education", "source": "Reddit/india"},
    {"text": "Pothole complaint app actually works. Road fixed within a week. Surprised.",           "topic": "Transport", "source": "Reddit/india"},
]

FIELDNAMES = ["id", "timestamp", "text", "topic", "source", "sentiment",
              "sentiment_score", "urgency", "is_realtime"]


# ── Sentiment model (lightweight inline version) ──────────────────────────────

_sent_pipe = None

def get_sentiment_pipeline():
    global _sent_pipe
    if _sent_pipe is None:
        print("[collector] Loading sentiment model...")
        from transformers import pipeline
        _sent_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            truncation=True,
            max_length=128,
        )
        print("[collector] Model loaded.")
    return _sent_pipe

LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}

def analyze_sentiment(text: str) -> tuple[str, float]:
    """Returns (label, score)."""
    try:
        pipe   = get_sentiment_pipeline()
        result = pipe(text[:512])[0]
        label  = LABEL_MAP.get(result["label"], result["label"])
        return label, round(result["score"], 4)
    except Exception as e:
        print(f"[collector] Sentiment error: {e}")
        return "NEUTRAL", 0.5

def urgency_from(label: str, score: float) -> str:
    if label == "NEGATIVE":
        return "CRITICAL" if score >= 0.80 else "HIGH"
    elif label == "NEUTRAL":
        return "MEDIUM"
    return "LOW"


# ── File helpers ──────────────────────────────────────────────────────────────

def ensure_file():
    """Create output CSV with headers if it doesn't exist."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        print(f"[collector] Created {OUTPUT_FILE}")

def append_rows(rows: list[dict]):
    """Append new rows and rotate if over MAX_ROWS."""
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    # Rotate: keep only the newest MAX_ROWS entries
    import pandas as pd
    df = pd.read_csv(OUTPUT_FILE)
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)
        df.to_csv(OUTPUT_FILE, index=False)

def load_seen_ids() -> set:
    """Load already-processed post IDs to avoid duplicates."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(OUTPUT_FILE)
        return set(df["id"].astype(str).tolist())
    except Exception:
        return set()


# ── Reddit Collection ─────────────────────────────────────────────────────────

def collect_from_reddit() -> list[dict]:
    """Fetch new posts from Reddit via PRAW."""
    try:
        import praw
    except ImportError:
        print("[collector] praw not installed. Run: pip install praw")
        return []

    if "YOUR_CLIENT_ID" in REDDIT_CLIENT_ID:
        print("[collector] Reddit credentials not set. Using simulation mode.")
        return []

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        seen = load_seen_ids()
        new_posts = []

        for sub_name in SUBREDDITS:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.new(limit=FETCH_LIMIT):
                if str(post.id) in seen:
                    continue
                text = (post.title + " " + (post.selftext or ""))[:512].strip()
                if len(text) < 20:
                    continue

                # Simple topic detection from keywords
                topic = detect_topic(text)
                label, score = analyze_sentiment(text)

                new_posts.append({
                    "id":             post.id,
                    "timestamp":      datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M"),
                    "text":           text[:300],
                    "topic":          topic,
                    "source":         f"Reddit/r/{sub_name}",
                    "sentiment":      label,
                    "sentiment_score": score,
                    "urgency":        urgency_from(label, score),
                    "is_realtime":    True,
                })

        print(f"[collector] Fetched {len(new_posts)} new Reddit posts")
        return new_posts

    except Exception as e:
        print(f"[collector] Reddit error: {e}")
        return []


def detect_topic(text: str) -> str:
    """Keyword-based topic detection for real-time posts."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["hospital", "doctor", "health", "medicine", "ambulance", "vaccine", "icu"]):
        return "Health"
    if any(w in text_lower for w in ["school", "teacher", "education", "student", "college", "scholarship"]):
        return "Education"
    if any(w in text_lower for w in ["road", "transport", "bus", "metro", "highway", "traffic", "pothole"]):
        return "Transport"
    if any(w in text_lower for w in ["tax", "gst", "income tax", "refund", "revenue", "budget"]):
        return "Taxes"
    return "General Governance"


# ── Simulation Mode ───────────────────────────────────────────────────────────

def collect_simulated() -> list[dict]:
    """Generate simulated real-time posts for demo without API keys."""
    seen = load_seen_ids()
    new_posts = []

    # Pick 3-5 random posts each run
    sample = random.sample(SIMULATION_POSTS, min(5, len(SIMULATION_POSTS)))
    for i, post_template in enumerate(sample):
        post_id = f"sim_{int(time.time())}_{i}"
        if post_id in seen:
            continue

        text  = post_template["text"]
        label, score = analyze_sentiment(text)

        new_posts.append({
            "id":              post_id,
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text":            text,
            "topic":           post_template["topic"],
            "source":          post_template["source"] + " (simulated)",
            "sentiment":       label,
            "sentiment_score": score,
            "urgency":         urgency_from(label, score),
            "is_realtime":     True,
        })

    print(f"[collector] Simulated {len(new_posts)} new posts")
    return new_posts


# ── Main collection job ───────────────────────────────────────────────────────

def collection_job(simulate: bool = False):
    """Called by scheduler every INTERVAL_MIN minutes."""
    print(f"\n[collector] ── Running collection job at {datetime.now().strftime('%H:%M:%S')} ──")
    posts = collect_simulated() if simulate else collect_from_reddit()
    if posts:
        append_rows(posts)
        print(f"[collector] Saved {len(posts)} new posts → {OUTPUT_FILE}")
    else:
        print("[collector] No new posts this run.")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GovSentinel Real-Time Collector")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated posts instead of Reddit API")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (instead of looping)")
    parser.add_argument("--interval", type=int, default=INTERVAL_MIN,
                        help=f"Collection interval in minutes (default: {INTERVAL_MIN})")
    args = parser.parse_args()

    ensure_file()
    print(f"[collector] GovSentinel Real-Time Collector starting...")
    print(f"[collector] Mode     : {'SIMULATION' if args.simulate else 'REDDIT API'}")
    print(f"[collector] Interval : every {args.interval} minutes")
    print(f"[collector] Output   : {OUTPUT_FILE}")
    print(f"[collector] Press Ctrl+C to stop\n")

    # Run once immediately
    collection_job(simulate=args.simulate)

    if args.once:
        print("[collector] --once flag set. Exiting.")
        return

    # Schedule repeated runs
    schedule.every(args.interval).minutes.do(collection_job, simulate=args.simulate)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
