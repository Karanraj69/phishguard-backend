PhishGuard — AI-Powered Gmail Phishing Detector

A Chrome browser extension that detects phishing emails in Gmail using a **Naive Bayes ML model** served via a **Python Flask backend**. 

---

## 🚀 How It Works

1. **Auto-Scan:** When you open an email in Gmail, the extension silently extracts the email text.
2. **Cloud Analysis:** It sends the text to the live PhishGuard Flask API hosted on Render.
3. **ML Prediction:** The backend uses a trained Multinomial Naive Bayes model with TF-IDF vectorization to determine if the email is safe or a phishing attempt.
4. **Warning Banner:** A banner drops down in Gmail (Red for Phishing, Green for Safe) showing the AI's confidence score.

---

## 📁 Project Structure

```
phishguard-backend/
├── app.py                # Flask REST API server (Production-ready via Gunicorn)
├── train_model.py        # ML training script (Naive Bayes + TF-IDF)
├── requirements.txt      # Python dependencies
└── model/                # Pre-trained models loaded by app.py
    ├── phishing_model.pkl
    └── tfidf_vectorizer.pkl
```

*(Note: The Chrome Extension code is maintained in a separate directory/repository.)*

---

## ☁️ Cloud Deployment (Render)

This backend is designed to be hosted for **free** on [Render](https://render.com) as a Web Service.

**Render Setup:**
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt.`
- **Start Command:** `gunicorn app: app` (This is critical! It ensures the ML models load into memory at startup)

---

## 💻 Local Setup & Training

If you want to train your own model or run the API locally:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This script downloads a dataset of ~18,000 phishing/safe emails from Kaggle, cleans the text (removes HTML, links, stop words), and trains the Naive Bayes classifier, saving the output to the `model/` folder.

### 3. Run Locally
```bash
python app.py
```
The server will start at `http://localhost:5000`. 

### 4. API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server status & model check |
| `/api/predict` | POST | Analyze single email |
| `/api/batch` | POST | Analyze multiple emails |

---

## ⚙️ Tech Stack

- **ML Model:** Scikit-learn Multinomial Naive Bayes
- **Text Processing:** NLTK (PorterStemmer, Stopwords), TF-IDF (10,000 features, bigrams)
- **Backend API:** Python Flask + Flask-CORS + Gunicorn
- **Deployment:** Render (Free Tier)
- **Frontend Extension:** Vanilla JS/CSS (Manifest V3)
