import pickle
import os
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Setup Paths ---
# IMPORTANT: Ensure these files are in the same directory as app.py
MODEL_DIR = './'
MODEL_FILE = os.path.join(MODEL_DIR, 'best_svc_model.pkl')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# --- Global Variables ---
best_svc_model = None
tfidf_vectorizer = None
porter = PorterStemmer()

# --- Load Stopwords Safely ---
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception:
    # Fallback basic stopwords list
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
        'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
        'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
        'can', 'will', 'just', 'don', 'should', 'now'
    }
    print("Warning: NLTK Stopwords could not be loaded. Using basic list.")

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """Clean and stem input text."""
    if not isinstance(text, str):
        text = ""
    
    # Remove non-alphabetic characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [porter.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- Load Model & Vectorizer ---
def load_model_assets():
    """Load pre-trained model and vectorizer from disk."""
    global best_svc_model, tfidf_vectorizer
    try:
        with open(MODEL_FILE, 'rb') as f:
            best_svc_model = pickle.load(f)
        print(f"Model loaded from: {MODEL_FILE}")
        
        with open(VECTORIZER_FILE, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from: {VECTORIZER_FILE}")
    except FileNotFoundError:
        print("ERROR: Model or vectorizer file not found in app directory.")
    except Exception as e:
        print(f"Error during loading: {e}")

# --- API Routes ---
@app.route('/')
def home():
    """Home route to confirm server is running."""
    return "Fake News Detector API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fake or real news."""
    if not best_svc_model or not tfidf_vectorizer:
        return jsonify({'error': 'Model assets not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        article_text = data.get('article', '')
        
        if not article_text or len(article_text) < 50:
            return jsonify({'error': 'Article text is too short or missing.'}), 400

        cleaned_text = preprocess_text(article_text)
        text_vectorized = tfidf_vectorizer.transform([cleaned_text])
        prediction = best_svc_model.predict(text_vectorized)

        # Return result (0: Fake, 1: Real)
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({'error': 'Prediction failed due to internal error.'}), 500

# --- Load model on startup ---
load_model_assets()

# --- Run App (for local testing) ---
if __name__ == '__main__':
    app.run(debug=True)
