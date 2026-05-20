"""
Phishing Email Detection - Model Training Script
==================================================
Trains a Naive Bayes classifier on a phishing email dataset from Kaggle.
Saves the trained model and vectorizer for use by the Flask API.
"""

import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


def download_dataset():
    """
    Download the phishing email dataset from Kaggle using kagglehub.
    Falls back to generating a synthetic dataset if download fails.
    """
    try:
        import kagglehub
        print("[*] Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("subhajournal/phishingemails")
        print(f"[+] Dataset downloaded to: {path}")

        # Find the CSV file in the downloaded directory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    print(f"[+] Found CSV: {csv_path}")
                    df = pd.read_csv(csv_path)
                    print(f"[+] Dataset shape: {df.shape}")
                    print(f"[+] Columns: {list(df.columns)}")
                    return df

        print("[!] No CSV found in downloaded dataset, using synthetic data.")
        return generate_synthetic_dataset()

    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        print("[*] Generating synthetic dataset for training...")
        return generate_synthetic_dataset()


def generate_synthetic_dataset():
    """Generate a synthetic phishing email dataset for demonstration."""
    np.random.seed(42)

    phishing_templates = [
        "Your account has been compromised. Click here to verify your identity immediately.",
        "URGENT: Your bank account will be suspended. Confirm your details now.",
        "You have won a prize of $1,000,000! Click the link to claim your reward.",
        "Dear customer, we detected suspicious activity. Update your password at this link.",
        "Your PayPal account is limited. Restore access by confirming your information.",
        "Security alert: unauthorized login detected. Verify your account now.",
        "Congratulations! You've been selected for a special offer. Act now!",
        "Your Netflix subscription has expired. Update payment to avoid cancellation.",
        "IRS Notice: You have an outstanding tax refund. Click to claim immediately.",
        "Your Apple ID was used to sign in. If this wasn't you, click here.",
        "IMPORTANT: Your email account will be deactivated in 24 hours. Verify now.",
        "Dear user, your package could not be delivered. Update shipping info here.",
        "Your Microsoft account has unusual sign-in activity. Review your activity.",
        "Warning: Your credit card has been charged $499.99. Dispute this transaction.",
        "You have received a secure document. Enter your email password to view it.",
        "Immediate action required: Your social security number has been compromised.",
        "FREE gift card worth $500! Limited time offer. Click to redeem now.",
        "Your Amazon order #12345 has been placed. If you didn't order, cancel here.",
        "Verify your identity to avoid account termination. Click the secure link.",
        "You've received a payment of $3,000. Log in to your account to confirm.",
        "ALERT: Someone tried to access your account from an unknown device.",
        "Your Wells Fargo account requires immediate verification. Click below.",
        "Congratulations winner! You are our lucky draw participant. Claim prize now.",
        "Your Google account has been flagged for unusual activity. Secure it now.",
        "Important update regarding your loan application. Review details immediately.",
    ]

    safe_templates = [
        "Hi team, please find the quarterly report attached for your review.",
        "Meeting scheduled for tomorrow at 2 PM in the conference room.",
        "Thank you for your purchase. Your order will arrive in 3-5 business days.",
        "Reminder: Project deadline is next Friday. Please submit your updates.",
        "Happy birthday! Wishing you a wonderful day filled with joy.",
        "The company picnic is next Saturday. RSVP by Wednesday please.",
        "Please review the attached document and provide your feedback.",
        "Just wanted to follow up on our conversation from last week.",
        "Here are the notes from today's team meeting for your reference.",
        "Your monthly newsletter: Top 10 tips for productivity this month.",
        "Welcome to our platform! Here's how to get started with your account.",
        "Quick update: The server maintenance is scheduled for this weekend.",
        "Thanks for attending the webinar. Here's the recording link.",
        "Invitation: Join us for the annual tech conference next month.",
        "Your subscription has been renewed. Thank you for being a valued member.",
        "FYI - The office will be closed on Monday for the holiday.",
        "Great job on the presentation today! The client was very impressed.",
        "Please complete the employee satisfaction survey by end of week.",
        "Agenda for tomorrow's standup meeting attached. See you there!",
        "Sharing the updated project timeline. Let me know if you have questions.",
        "Congratulations on your work anniversary! Five years of excellence.",
        "The new software update is available. Please install at your convenience.",
        "Reminder: Submit your timesheet by end of day Friday.",
        "Let's schedule a one-on-one meeting this week to discuss your progress.",
        "Here's the link to the shared drive with all project resources.",
    ]

    data = []
    for template in phishing_templates:
        for i in range(40):
            variation = template
            if i % 3 == 0:
                variation = variation.upper()
            if i % 5 == 0:
                variation = "URGENT! " + variation
            if i % 7 == 0:
                variation = variation + " Do not ignore this message!"
            data.append({"Email Text": variation, "Email Type": "Phishing Email"})

    for template in safe_templates:
        for i in range(40):
            variation = template
            if i % 4 == 0:
                variation = "Hi, " + variation
            if i % 6 == 0:
                variation = variation + " Best regards."
            data.append({"Email Text": variation, "Email Type": "Safe Email"})

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[+] Synthetic dataset created: {df.shape[0]} samples")
    return df


def preprocess_text(text):
    """Clean and preprocess email text for model training."""
    if not isinstance(text, str):
        return ""

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', 'urllink', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'emailaddr', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and stem
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]

    return ' '.join(words)


def train_model():
    """Main training pipeline."""
    print("=" * 60)
    print("  PHISHING EMAIL DETECTOR - MODEL TRAINING")
    print("=" * 60)

    # ─── Step 1: Load Dataset ────────────────────────────────────────────
    print("\n[Step 1] Loading dataset...")
    df = download_dataset()

    # Detect column names (different datasets use different names)
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['text', 'body', 'message', 'content']):
            text_col = col
        elif 'email' in col_lower and 'type' not in col_lower and 'label' not in col_lower:
            text_col = col

        if any(kw in col_lower for kw in ['type', 'label', 'class', 'category', 'target']):
            label_col = col

    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]

    print(f"[+] Using text column: '{text_col}'")
    print(f"[+] Using label column: '{label_col}'")

    # Drop NaN values
    df = df.dropna(subset=[text_col, label_col])

    # ─── Step 2: Encode Labels ──────────────────────────────────────────
    print("\n[Step 2] Encoding labels...")
    print(f"[+] Unique labels: {df[label_col].unique()}")

    label_map = {}
    for label in df[label_col].unique():
        label_lower = str(label).lower()
        if any(kw in label_lower for kw in ['phish', 'spam', 'malicious', 'fraud', '1']):
            label_map[label] = 1
        else:
            label_map[label] = 0

    df['label'] = df[label_col].map(label_map)
    print(f"[+] Label mapping: {label_map}")
    print(f"[+] Class distribution:\n{df['label'].value_counts()}")

    # ─── Step 3: Preprocess Text ────────────────────────────────────────
    print("\n[Step 3] Preprocessing text...")
    df['processed_text'] = df[text_col].apply(preprocess_text)

    # Remove empty processed texts
    df = df[df['processed_text'].str.len() > 0]
    print(f"[+] Samples after preprocessing: {len(df)}")

    # ─── Step 4: Vectorize ──────────────────────────────────────────────
    print("\n[Step 4] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label'].values
    print(f"[+] Feature matrix shape: {X.shape}")

    # ─── Step 5: Train/Test Split ───────────────────────────────────────
    print("\n[Step 5] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Training samples: {X_train.shape[0]}")
    print(f"[+] Testing samples: {X_test.shape[0]}")

    # ─── Step 6: Train Naive Bayes ──────────────────────────────────────
    print("\n[Step 6] Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    # ─── Step 7: Evaluate ───────────────────────────────────────────────
    print("\n[Step 7] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'-' * 40}")
    print(f"  ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'-' * 40}")
    print(f"\nClassification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Safe Email', 'Phishing Email']
    ))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    # ─── Step 8: Save Model ─────────────────────────────────────────────
    print(f"\n[Step 8] Saving model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[+] Model saved to: {MODEL_PATH}")
    print(f"[+] Vectorizer saved to: {VECTORIZER_PATH}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    return model, vectorizer


if __name__ == "__main__":
    train_model()
