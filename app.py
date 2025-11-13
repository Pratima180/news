# app.py
# Hybrid Fake News Detector
# Layers: Google Fact Check -> Zero-shot AI -> Source Credibility
# Developer: (your name)

from flask import Flask, render_template, request
import requests
import os
from urllib.parse import urlparse
import json
import re

# Optional: torch import (transformers pipeline will import/require it)
try:
    import torch
except Exception:
    torch = None

from transformers import pipeline

app = Flask(__name__, template_folder="./templates", static_folder="./static")

# -----------------------
# Config: Google API (use env var)
# -----------------------
API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")  # set this in your environment
API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# -----------------------
# Load Source Credibility (simple JSON)
# -----------------------
CRED_PATH = "source_cred.json"
if os.path.exists(CRED_PATH):
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        try:
            SOURCE_CRED = json.load(f)
        except Exception:
            SOURCE_CRED = {}
else:
    SOURCE_CRED = {}

# -----------------------
# Zero-shot classifier (NLI) initialization
# -----------------------
# Warning: large model may require a lot of RAM. You can swap model to a lighter HF model if needed.
classifier = None
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Zero-shot classifier loaded.")
except Exception as e:
    print("Warning: transformers pipeline could not be initialized:", e)
    classifier = None

# -----------------------
# Utility: extract domain and credibility score (0-1)
# -----------------------
def domain_from_text(text):
    """
    Try to extract a domain from the text. Handles explicit URLs and plain domain-like tokens.
    Returns domain (lowercase) or None.
    """
    if not text:
        return None

    # try to find a URL first
    url_match = re.search(r"(https?://[^\s'\"<>]+)", text)
    if url_match:
        try:
            parsed = urlparse(url_match.group(1))
            if parsed.netloc:
                return parsed.netloc.lower()
        except Exception:
            pass

    # fallback: look for tokens that look like domains (example.com)
    tokens = re.findall(r"([A-Za-z0-9.-]+\.[A-Za-z]{2,6})", text)
    if tokens:
        # return the first plausible token (lowercased, stripped)
        return tokens[0].strip(".,'\"()[]<>").lower()
    return None

def credibility_score_for_domain(domain):
    """
    Return credibility score in 0..1 where 1 = highly credible, 0 = not credible.
    Default neutral = 0.5
    SOURCE_CRED should contain base domains (example.com) -> float
    """
    if not domain:
        return 0.5
    try:
        parts = domain.split(".")
        base = ".".join(parts[-2:]) if len(parts) >= 2 else domain
        val = SOURCE_CRED.get(base)
        if val is None:
            val = SOURCE_CRED.get(domain, 0.5)
        return float(val)
    except Exception:
        return 0.5

# -----------------------
# AI scoring (zero-shot)
# -----------------------
def ai_zero_shot_score(text):
    """
    Returns score_fake between 0-1 representing probability that text is FAKE.
    Uses zero-shot classifier labels ["fake","real"] and returns fake score.
    If classifier not available, returns 0.5 (uncertain).
    """
    if classifier is None:
        return 0.5
    try:
        labels = ["fake", "real"]
        out = classifier(text, candidate_labels=labels, hypothesis_template="This example is {}.")
        # 'out' has 'labels' and 'scores' lists; map them robustly
        score_map = {lab.lower(): sc for lab, sc in zip(out.get("labels", []), out.get("scores", []))}
        # if 'fake' not present, try approximate detection
        return float(score_map.get("fake", 0.5))
    except Exception as e:
        print("AI classifier error:", e)
        return 0.5

# -----------------------
# Combine scores (fusion)
# -----------------------
def combine_scores(google_decision, ai_score, cred_score):
    """
    google_decision: None or "REAL"/"FAKE" (strong signal)
    ai_score: 0..1 probability of fake (higher => more likely fake)
    cred_score: 0..1 credibility (higher => more credible)

    Returns (label, combined_score) where combined_score in [0,1] is fake-likelihood.
    Label thresholds:
      combined >= 0.60 -> FAKE
      combined <= 0.40 -> REAL
      otherwise -> UNCERTAIN
    If google_decision is present we trust it strongly (but still return numeric).
    """
    # If Google fact-check gave explicit decision, trust it (but keep a numeric score)
    if google_decision is not None:
        if google_decision.upper() == "FAKE":
            return "FAKE", 0.99
        else:
            return "REAL", 0.01

    # If no google_decision: fuse AI and credibility
    # cred_score: higher => more credible => less likely fake
    # We'll use a weighted average: ai contributes more than domain credibility
    try:
        ai = float(ai_score)
    except Exception:
        ai = 0.5
    try:
        cred = float(cred_score)
    except Exception:
        cred = 0.5

    # Weighted fusion (tunable):
    # - AI (zero-shot) weight = 0.6
    # - Domain credibility (inverted) weight = 0.4
    combined = (0.6 * ai) + (0.4 * (1.0 - cred))  # higher => more fake

    # clamp
    combined = max(0.0, min(1.0, combined))

    if combined >= 0.60:
        label = "FAKE"
    elif combined <= 0.40:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return label, round(combined, 3)

# -----------------------
# Google Fact Check helper
# -----------------------
def query_google_factcheck(text):
    """
    Query Google Fact Check API with the given text.
    Returns a dict with keys {rating, publisher, review_url} or None if nothing found.
    """
    if not API_KEY:
        print("Google FactCheck API key not set (set GOOGLE_FACTCHECK_API_KEY env var).")
        return None

    params = {"query": text, "key": API_KEY}
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        claims = data.get("claims") or []
        if len(claims) > 0:
            # pick the claim with a review if possible
            for claim in claims:
                reviews = claim.get("claimReview") or []
                if reviews:
                    review = reviews[0]
                    rating = review.get("textualRating") or ""
                    pub = review.get("publisher", {}).get("name") or ""
                    return {"rating": rating, "publisher": pub, "review_url": review.get("url", "")}
            # fallback: no claimReview but claims exist
            claim = claims[0]
            return {"rating": claim.get("text", ""), "publisher": "", "review_url": ""}
        return None
    except Exception as e:
        print("Google API error:", e)
        return None

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/check", methods=["POST"])
def check_news():
    # safe extraction of 'news' from form or JSON
    news_text = None
    if request.method == "POST":
        if request.form and request.form.get("news"):
            news_text = request.form.get("news")
        else:
            j = request.get_json(silent=True)
            if isinstance(j, dict):
                news_text = j.get("news")

    if not news_text or not str(news_text).strip():
        return render_template("prediction.html", prediction_text="⚠️ Please enter some text.")

    text = str(news_text).strip()

    # 1) Google Fact Check
    g = query_google_factcheck(text)
    if g:
        rating = (g.get("rating") or "").lower()
        # look for common negative tokens to flag as FAKE
        if any(tok in rating for tok in ["false", "fake", "misleading", "altered", "pants-on-fire", "not true"]):
            final_label = "FAKE"
        else:
            final_label = "REAL"
        result_html = (
            f"<b>{final_label}</b><br>"
            f"Rating: {g.get('rating') or 'N/A'}<br>"
            f"Source: {g.get('publisher') or 'N/A'}<br>"
            f"<a href='{g.get('review_url') or '#'}' target='_blank'>Fact-check link</a>"
        )
        return render_template("prediction.html", prediction_text=result_html)

    # 2) No Google result -> AI + Source credibility
    ai_score = ai_zero_shot_score(text)
    domain = domain_from_text(text)
    cred = credibility_score_for_domain(domain)

    label, combined = combine_scores(None, ai_score, cred)

    details = (
        f"<b>{label}</b><br>"
        f"AI-fake-score: {round(ai_score, 3)}<br>"
        f"Source domain: {domain or 'Unknown'} (credibility: {round(cred,3)})<br>"
        f"Hybrid-score (fake-likelihood): {combined}"
    )

    return render_template("prediction.html", prediction_text=details)

# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug should be False in production
    app.run(host="0.0.0.0", port=port, debug=True)
