# app.py
# Hybrid Fake News Detector (Improved Version)
# Google Fact Check + HuggingFace AI + Source Credibility + Dynamic Confidence
# Fully Render-Safe (NO torch, NO transformers)

from flask import Flask, render_template, request
import requests
import os
import json
import re
from urllib.parse import urlparse

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# -----------------------
# API Keys (Render env vars)
# -----------------------
GOOGLE_API_KEY = os.environ.get("AIzaSyCQh0eeyvi4TpCQYKZ4R-NN3cPzRb5s8JE", "")
HF_API_KEY = os.environ.get("AIzaSyCQh0eeyvi4TpCQYKZ4R-NN3cPzRb5s8JE", "")

GOOGLE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


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
        return 0.6

    base = ".".join(domain.split(".")[-2:])

    return float(SOURCE_CRED.get(base, 0.6))


# -----------------------
# AI Zero-Shot via HuggingFace
# -----------------------
def ai_zero_shot_score(text):

    claim_text = f"Claim: {text}. This statement refers to a news event."

    payload = {
        "inputs": claim_text,
        "parameters": {
            "candidate_labels": ["fake", "real"],
            "hypothesis_template": "This news is {}."
        }
    }

    try:

        r = requests.post(
            HF_MODEL_URL,
            headers=HF_HEADERS,
            json=payload,
            timeout=20
        )

        out = r.json()

        labels = [l.lower() for l in out["labels"]]
        scores = out["scores"]

        fake_score = float(scores[labels.index("fake")])

        return fake_score

    except:

        return None


# -----------------------
# Google Fact Check
# -----------------------
def google_fact_check(text):

    if not GOOGLE_API_KEY:
        return None

    params = {
        "query": text,
        "key": GOOGLE_API_KEY
    }

    try:

        res = requests.get(
            GOOGLE_URL,
            params=params,
            timeout=10
        ).json()

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
# Hybrid Decision Logic
# -----------------------
def improved_final_label(ai, cred, google=None):

    # Trust Google fact check if available
    if google:

        rating = google.lower()

        if any(x in rating for x in ["false", "fake", "misleading", "altered"]):
            return "FAKE"

        return "REAL"

    # Handle AI failure
    if ai is None:
        ai = 0.5

    # Hybrid score
    hybrid = (ai * 0.7) + ((1 - cred) * 0.3)

    # Strong prediction thresholds
    if hybrid >= 0.6:
        return "FAKE"

    elif hybrid <= 0.4:
        return "REAL"

    # Tie-breaker
    if ai > 0.5:
        return "FAKE (Likely)"

    else:
        return "REAL (Likely)"


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

        return render_template(
            "prediction.html",
            prediction_text="⚠ Please enter some news text."
        )

    # -----------------------
    # Google Fact Check
    # -----------------------
    fc = google_fact_check(news)

    if fc:

        label = improved_final_label(None, None, fc["rating"])

        result = f"""
        <b>{label}</b><br>
        Rating: {fc['rating']}<br>
        Source: {fc['publisher']}<br>
        <a href='{fc['review_url']}' target='_blank'>Fact-check link</a>
        """

        return render_template(
            "prediction.html",
            prediction_text=result
        )

    # -----------------------
    # AI + Credibility
    # -----------------------
    ai = ai_zero_shot_score(news)

    domain = domain_from_text(news)

    cred = credibility_score(domain)

    label = improved_final_label(ai, cred)

    if ai is None:
        ai = 0.5

    confidence = round(abs(ai - 0.5) * 2, 2)

    result = f"""
    <b>{label}</b><br>
    AI Fake Probability: {round(ai,3)}<br>
    Confidence Score: {confidence}<br>
    Domain: {domain or 'Unknown'} (credibility: {round(cred,2)})<br>
    """

    return render_template(
        "prediction.html",
        prediction_text=result
    )


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":

    app.run(debug=True)
