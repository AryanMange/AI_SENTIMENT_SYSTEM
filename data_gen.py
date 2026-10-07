"""
data_gen.py
-----------
Generates a synthetic dataset of 500 public comments about government services.
Covers 4 topics: Health, Education, Transport, Taxes
Includes varied sentiments (positive, negative, neutral) and sarcasm.

Run:
    python data_gen.py
Output:
    data/comments.csv
"""

import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# ── Template Comments ────────────────────────────────────────────────────────

COMMENTS = {
    "Health": {
        "positive": [
            "The new government hospital in our district is excellent. Doctors are attentive and medicine is free.",
            "Really impressed by the health camps organized this month. Free checkups for everyone!",
            "The ambulance response time has improved drastically. Government is doing a great job here.",
            "Got my surgery done at the government hospital. Zero cost, great care. Thank you.",
            "The new vaccination drive has been very well managed. Smooth and efficient.",
            "Health helpline 104 is actually working now. Called at 2am and got a doctor.",
            "The district hospital renovation is complete and it looks world class now.",
            "Free dialysis at government hospitals is a life-saving initiative. Grateful.",
        ],
        "negative": [
            "Waited 6 hours at the government hospital and still didn't see a doctor. Pathetic system.",
            "No medicines in the government dispensary again. Third time this month. Absolute failure.",
            "The ambulance never came. My father had to be taken in a private vehicle. Shameful.",
            "Government hospital toilets are in a disgusting state. No hygiene whatsoever.",
            "Doctors at the PHC are always absent. What is the point of these health centers?",
            "They asked us to buy medicines from outside even in a government hospital. Corruption!",
            "The ICU at district hospital has been non-functional for 3 weeks. People are dying.",
            "Blood bank is always empty. Every patient's family has to arrange blood on their own.",
            "No female doctors in the government hospital. Women refuse to go there.",
            "Mental health services are completely absent in government hospitals. Ignored completely.",
        ],
        "sarcasm": [
            "Oh great, the health minister inaugurated another building. Too bad there are no doctors inside!",
            "Wonderful, the government hospital got a new paint job. Shame the medicines are still missing.",
            "Brilliant initiative making patients wait 8 hours. Really building that patient endurance.",
            "Sure, the hospital has 5 star rated toilets on paper. In reality, bring your own water.",
            "Amazing how the government finds crores for health schemes but zero for actual medicine stock.",
        ],
        "neutral": [
            "The government has announced a new health policy. Details are yet to be released.",
            "Ministry of Health held a review meeting today on hospital infrastructure.",
            "New MBBS seats have been added to government medical colleges this year.",
            "Health department released its annual report. Mixed results across districts.",
        ],
    },
    "Education": {
        "positive": [
            "The mid-day meal program has significantly improved school attendance in our village.",
            "Government school teachers in our block are actually teaching well now. Proud parents.",
            "Free laptop distribution to Class 10 students was a wonderful initiative.",
            "The new digital classrooms in government schools are fantastic. Kids love them.",
            "Scholarship amount was credited on time this year. Very smooth process.",
            "District government school topped the board exams. Public education can work!",
            "The sports facilities added to government schools are excellent. Children are thriving.",
            "Navodaya Vidyalayas continue to be one of the best things the government ever built.",
        ],
        "negative": [
            "Government school has no blackboard, no chalk, no teacher. Just empty classrooms.",
            "Scholarship portal keeps crashing. Students have been waiting 6 months for their money.",
            "Teachers in government school spend half the day on election duty, not teaching.",
            "Mid-day meal was raw and inedible today. Kids left hungry. Who is monitoring this?",
            "Government college has 3 professors for 800 students. How is anyone supposed to learn?",
            "No textbooks delivered even 3 months into the academic year. Criminal negligence.",
            "Government school building is crumbling. Rain comes through the roof during class.",
            "The exam results are out but certificates are delayed for 8 months. Unacceptable.",
            "Teacher recruitment has been frozen for 5 years. Schools running on skeleton staff.",
        ],
        "sarcasm": [
            "Fantastic that they painted 'Smart Classroom' on the door. Shame there is no electricity inside.",
            "Our government school got a computer lab! The computers have been in boxes for 2 years.",
            "Education minister gave a great speech about literacy. The school in his own constituency has no teacher.",
            "They say education is a priority. Funny way to show it with a 40% budget cut.",
        ],
        "neutral": [
            "The Right to Education Act review is currently underway at the national level.",
            "State board exams will be conducted in March as announced by the education department.",
            "A new curriculum framework has been released for public feedback.",
            "Education department is conducting a school mapping survey across all districts.",
        ],
    },
    "Transport": {
        "positive": [
            "The new metro line has completely changed how I commute. Fast, clean, and affordable.",
            "Road from our village to the main highway is finally fixed. Life changing for us.",
            "State bus service has added new routes to remote areas. Very helpful for daily workers.",
            "The pothole complaint app actually worked. Road was fixed within a week!",
            "Electric buses in the city are excellent. Quiet, smooth, and no pollution.",
            "The flyover construction was completed on time for once. Great project management.",
            "New railway station facilities are top class. Clean platform, proper seating.",
            "Highway expansion has reduced travel time by 40 minutes. Huge relief.",
        ],
        "negative": [
            "Roads in our area have not been repaired in 5 years. Full of craters, not potholes.",
            "State buses are always overloaded and 2 hours late. Absolute chaos every morning.",
            "Traffic signals on the main road have been broken for 3 months. Accidents happening daily.",
            "The flyover project started 4 years ago. Still incomplete. Money swallowed somewhere.",
            "No buses after 8pm in our area. Women cannot travel safely because of this.",
            "Railway station has no proper waiting area, no water, and no working toilets.",
            "The road we pay toll on is in worse condition than a dirt track. Daylight robbery.",
            "Drainage work dug up the road 6 months ago and was never completed. Waterlogging everywhere.",
            "Bus depot workers are on strike again. No information to commuters. Total mismanagement.",
        ],
        "sarcasm": [
            "Love how they announce road repair every monsoon and then do nothing. Classic tradition.",
            "They built a footpath and it collapsed in the first rain. Engineering marvel.",
            "The new flyover has a toll booth but no road connecting to it. Visionary planning.",
            "Highway is 'under construction' for 7 years now. Must be the slowest construction in history.",
            "Brilliant idea painting road lines that disappear in a week. Saves paint budget for next year.",
        ],
        "neutral": [
            "State transport department has issued a tender for 200 new buses.",
            "The highway authority released traffic diversion instructions for the bypass construction.",
            "Metro rail phase 3 Environmental Impact Assessment is currently open for public comment.",
            "Transport minister held a review of toll collection data across national highways.",
        ],
    },
    "Taxes": {
        "positive": [
            "The new GST portal is much easier to use compared to the old system. Good improvement.",
            "Tax refund was processed in just 10 days this year. Very impressed with the ITR system.",
            "The simplified tax forms for small businesses are a huge relief. Finally user-friendly.",
            "Income tax helpline was actually helpful. Issue resolved in one call.",
            "Faceless assessment is genuinely reducing harassment from tax officers. Good reform.",
            "GST exemption for small traders under 40 lakhs is a very sensible policy.",
            "Tax portal did not crash during filing season this year. Small miracle, but appreciated.",
        ],
        "negative": [
            "GST return filing portal crashed on the last day again. Fine imposed even though it was government failure.",
            "Received income tax notice for taxes already paid. No response from department for 4 months.",
            "Property tax assessment is completely arbitrary. Same house, different colonies, double the rate.",
            "Tax refund stuck for 2 years. Called 10 times. Written 5 letters. Nothing moves.",
            "They audit small traders aggressively but let big companies off with nothing. Two systems of justice.",
            "The TDS deducted from my salary does not show in Form 26AS. Complete system failure.",
            "GST on medicines is morally wrong. Taxing sick people. Shameful policy.",
            "Small business is shutting down because of compliance burden. Every month a new form.",
            "Income tax officers demand cash to process legitimate refunds. Where is the accountability?",
        ],
        "sarcasm": [
            "Excellent idea adding a new tax on top of the existing tax. Tax on tax, very innovative.",
            "Filing GST is so simple now. Only takes 3 days, 2 accountants and a prayer to the server gods.",
            "The tax department says they are going digital. The only digital thing is the fine notice.",
            "Great, another amnesty scheme for big defaulters. Small honest taxpayers can keep suffering.",
            "They simplified taxes so much that my CA had to hire two more assistants to understand them.",
        ],
        "neutral": [
            "Finance ministry released the tax collection data for Q3. Numbers are being reviewed.",
            "GST council will meet next week to discuss rate revision on essential commodities.",
            "Income tax department released new guidelines for NRI tax filing.",
            "Direct tax code revision is under discussion. Public consultation process underway.",
        ],
    },
}

PLATFORMS = ["Twitter", "Reddit", "Facebook", "Survey Form", "Government Portal"]
CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Patna", "Bhopal",
    "Chandigarh", "Kochi", "Surat", "Nagpur", "Indore", "Visakhapatnam",
]
USERNAMES = [
    "citizen_raj", "angry_voter_2024", "policy_watch", "govtcritic99",
    "proudindian_88", "aam_aadmi_voice", "frustrated_taxpayer", "hopeful_citizen",
    "realitycheckIN", "public_pulse", "voice_of_voter", "news_follower_in",
    "daily_commuter", "student_speaks", "health_worker_india", "teacher_truth",
    "tax_payer_india", "rural_voice_in", "urban_observer", "ground_reality",
]


def get_label(sentiment_type: str) -> str:
    if sentiment_type == "positive":
        return "POSITIVE"
    elif sentiment_type in ("negative", "sarcasm"):
        return "NEGATIVE"
    return "NEUTRAL"


def generate_dataset(n: int = 500) -> list[dict]:
    topics = list(COMMENTS.keys())
    sentiment_types = ["positive", "negative", "neutral", "sarcasm"]

    # Weighted distribution: more negative/neutral to simulate real social media
    weights = [0.20, 0.45, 0.15, 0.20]

    rows = []
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        topic = random.choice(topics)
        s_type = random.choices(sentiment_types, weights=weights, k=1)[0]

        # fallback to negative if sarcasm pool empty for topic
        pool = COMMENTS[topic].get(s_type, COMMENTS[topic]["negative"])
        text = random.choice(pool)

        # small natural variation: sometimes add location mention
        if random.random() < 0.25:
            city = random.choice(CITIES)
            text = f"{text} (Reporting from {city})"

        days_offset = random.randint(0, 364)
        timestamp = base_date + timedelta(days=days_offset,
                                          hours=random.randint(0, 23),
                                          minutes=random.randint(0, 59))

        rows.append({
            "id": i + 1,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "username": random.choice(USERNAMES) + str(random.randint(1, 999)),
            "platform": random.choice(PLATFORMS),
            "topic": topic,
            "text": text,
            "true_sentiment": get_label(s_type),
            "is_sarcasm": s_type == "sarcasm",
            "city": random.choice(CITIES),
        })

    random.shuffle(rows)
    return rows


def save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[data_gen] Saved {len(rows)} rows → {path}")


if __name__ == "__main__":
    output_path = os.path.join("data", "comments.csv")
    dataset = generate_dataset(500)
    save_csv(dataset, output_path)

    # quick stats
    from collections import Counter
    sentiments = Counter(r["true_sentiment"] for r in dataset)
    topics = Counter(r["topic"] for r in dataset)
    print("\n── Dataset Summary ──────────────────────")
    print(f"  Total rows   : {len(dataset)}")
    print(f"  Sentiments   : {dict(sentiments)}")
    print(f"  Topics       : {dict(topics)}")
    print(f"  Sarcasm rows : {sum(1 for r in dataset if r['is_sarcasm'])}")
    print("─────────────────────────────────────────")
