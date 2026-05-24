"""
Phishing Email Detection - Flask API Server
=============================================
Serves the trained Naive Bayes model via a REST API.
Receives email text and returns phishing prediction.
"""

import os
import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─── Load Model ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

model = None
vectorizer = None


def load_model():
    """Load the trained model and vectorizer from disk."""
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("[+] Model and vectorizer loaded successfully!")
        return True
    else:
        print("[!] Model files not found. Run train_model.py first.")
        print(f"    Expected: {MODEL_PATH}")
        print(f"    Expected: {VECTORIZER_PATH}")
        return False

# Load model at startup for production WSGI servers (like Gunicorn on Render)
load_model()


def preprocess_text(text):
    """Clean and preprocess email text (must match training preprocessing)."""
    if not isinstance(text, str):
        return ""

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'urllink', text)
    text = re.sub(r'\S+@\S+', 'emailaddr', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]

    return ' '.join(words)


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "service": "Phishing Email Detector API"
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict whether an email is phishing or safe.
    
    Request body:
        { "email_text": "..." }
    
    Response:
        {
            "prediction": "phishing" | "safe",
            "confidence": 0.0 - 1.0,
            "phishing_probability": 0.0 - 1.0,
            "safe_probability": 0.0 - 1.0
        }
    """
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first."
        }), 503

    data = request.get_json()
    if not data or "email_text" not in data:
        return jsonify({
            "error": "Missing 'email_text' in request body."
        }), 400

    email_text = data["email_text"]
    if not email_text or len(email_text.strip()) == 0:
        return jsonify({
            "error": "Email text is empty."
        }), 400

    # Preprocess
    processed = preprocess_text(email_text)

    # Vectorize
    features = vectorizer.transform([processed])

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    safe_prob = float(probabilities[0])
    phishing_prob = float(probabilities[1])
    confidence = float(max(probabilities))

    result = {
        "prediction": "phishing" if prediction == 1 else "safe",
        "confidence": round(confidence, 4),
        "phishing_probability": round(phishing_prob, 4),
        "safe_probability": round(safe_prob, 4),
    }

    label = "[PHISHING]" if prediction == 1 else "[SAFE]"
    print(f"{label} Confidence: {confidence:.2%} | Text preview: {email_text[:80]}...")

    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """
    Batch prediction endpoint for multiple emails.
    
    Request body:
        { "emails": ["email1 text", "email2 text", ...] }
    """
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first."
        }), 503

    data = request.get_json()
    if not data or "emails" not in data:
        return jsonify({
            "error": "Missing 'emails' array in request body."
        }), 400

    emails = data["emails"]
    results = []

    for email_text in emails:
        processed = preprocess_text(email_text)
        features = vectorizer.transform([processed])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        results.append({
            "prediction": "phishing" if prediction == 1 else "safe",
            "confidence": round(float(max(probabilities)), 4),
            "phishing_probability": round(float(probabilities[1]), 4),
            "safe_probability": round(float(probabilities[0]), 4),
        })

    return jsonify({"results": results})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  PHISHING EMAIL DETECTOR API")
    print("=" * 50)
    
    if model is not None:
        print("[*] Starting Flask server on http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        print("\n[!] Please train the model first:")
        print("    python train_model.py")
