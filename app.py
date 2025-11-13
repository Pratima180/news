# app.py
# Hybrid Fake News Detector (Render-Safe)
# Google Fact Check + HF API Zero-Shot + Source Credibility
# No Torch, No Local Models → Works on Render Free Tier

from flask import Flask, render_template, request
import requests
import os
from urllib.parse import urlparse
import json
import re

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# -----------------------
# API KEYS
# -----------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

GOOGLE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"  
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# -----------------------
# Load Source Credibility JSON
# -----------------------
if os.path.exists("source_cred.json"):
    with open("source_cred.json", "r", encoding="utf-8") as f:
        SOURCE_CRED = json.load(f)
else:
    SOURCE_CRED = {}

# -----------------------
# Extract Domain
# -----------------------
def domain_from_text(text):
    url_match = re.search(r"(https?://[^\s'\"<>]+)", text)
    if url_match:
        try:
            parsed = urlparse(url_match.group(1))
            return parsed.netloc.lower()
        except:
            pass

    tokens = re.findall(r"([A-Za-z0-9.-]+\.[A-Za-z]{2,6})", text)
    if tokens:
        return tokens[0].lower()

    return None

# -----------------------
# Credibility Score
# -----------------------
def credibility_score(domain):
    if not domain:
        return 0.5
    base = ".".join(domain.split(".")[-2:])
    return float(SOURCE_CRED.get(base, 0.5))

# -----------------------
# HuggingFace Zero-Shot (Via API)
# -----------------------
def ai_zero_shot_score(text):
    try:
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": ["fake", "real"],
                "hypothesis_template": "This example is {}."
            }
        }
        r = requests.post(HF_MODEL_URL, headers=HF_HEADERS, json=payload)
        out = r.json()

        labels = [l.lower() for l in out["labels"]]
        scores = out["scores"]

        return float(scores[labels.index("fake")])
    except:
        return 0.5

# -----------------------
# Combine
# -----------------------
def combine_scores(google_decision, ai_score, cred_score):
    if google_decision:
        return google_decision, (0.99 if google_decision == "FAKE" else 0.01)

    combined = (0.6 * ai_score) + (0.4 * (1 - cred_score))

    if combined >= 0.6:
        return "FAKE", combined
    elif combined <= 0.4:
        return "REAL", combined
    else:
        return "UNCERTAIN", combined

# -----------------------
# Google Fact Check
# -----------------------
def google_fact_check(text):
    if not GOOGLE_API_KEY:
        return None

    params = {"query": text, "key": GOOGLE_API_KEY}
    try:
        res = requests.get(GOOGLE_URL, params=params, timeout=10).json()
        claims = res.get("claims", [])
        if not claims:
            return None

        review = claims[0].get("claimReview", [{}])[0]
        return {
            "rating": review.get("textualRating", ""),
            "publisher": review.get("publisher", {}).get("name", ""),
            "review_url": review.get("url", "")
        }
    except:
        return None

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/check", methods=["POST"])
def check():
    news = request.form.get("news", "").strip()
    if not news:
        return render_template("prediction.html", prediction_text="⚠ Please enter some text.")

    # 1. Google Fact Check
    fc = google_fact_check(news)
    if fc:
        rating = fc["rating"].lower()
        if any(x in rating for x in ["false", "fake", "misleading", "altered"]):
            label = "FAKE"
        else:
            label = "REAL"

        result = f"""
        <b>{label}</b><br>
        Rating: {fc['rating']}<br>
        Source: {fc['publisher']}<br>
        <a href='{fc['review_url']}' target='_blank'>Fact-check link</a>
        """
        return render_template("prediction.html", prediction_text=result)

    # 2. AI + Credibility
    ai = ai_zero_shot_score(news)
    domain = domain_from_text(news)
    cred = credibility_score(domain)

    label, score = combine_scores(None, ai, cred)

    result = f"""
    <b>{label}</b><br>
    AI-Fake Score: {round(ai, 3)}<br>
    Domain: {domain or 'Unknown'} (cred: {round(cred, 2)})<br>
    Hybrid Score: {round(score, 3)}
    """

    return render_template("prediction.html", prediction_text=result)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
