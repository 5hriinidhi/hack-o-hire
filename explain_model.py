import pickle
import numpy as np

# Load the trained model and vectorizer
with open('Model/fraud_detector.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Test email
test_email = """
Subject: Urgent Account Verification Required

Dear user,
Your account has been suspended. Please click here to verify your password
and account details immediately to avoid permanent suspension.
"""

print("=" * 60)
print("HOW FRAUD DETECTION WORKS")
print("=" * 60)

# Step 1: Text Vectorization
print("\n1. TEXT VECTORIZATION (TF-IDF)")
print("-" * 60)
email_vec = vectorizer.transform([test_email])
print(f"Email converted to {email_vec.shape[1]} numerical features")

# Get top features (words) in this email
feature_names = vectorizer.get_feature_names_out()
email_features = email_vec.toarray()[0]
top_indices = email_features.argsort()[-10:][::-1]

print("\nTop 10 words/features detected:")
for idx in top_indices:
    if email_features[idx] > 0:
        print(f"  - '{feature_names[idx]}': {email_features[idx]:.4f}")

# Step 2: Model Prediction
print("\n2. MODEL PREDICTION (Naive Bayes)")
print("-" * 60)
prediction = model.predict(email_vec)[0]
probabilities = model.predict_proba(email_vec)[0]

print(f"Legitimate probability: {probabilities[0]:.2%}")
print(f"Fraud probability: {probabilities[1]:.2%}")
print(f"\nPrediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")

# Step 3: Feature Importance
print("\n3. WHY IT'S CLASSIFIED AS FRAUD")
print("-" * 60)
print("The model learned from training data that emails with")
print("these characteristics are typically fraudulent:")

# Get log probabilities for fraud class
log_prob_fraud = model.feature_log_prob_[1]
log_prob_legit = model.feature_log_prob_[0]

# Calculate contribution of each word
contributions = []
for idx in top_indices:
    if email_features[idx] > 0:
        word = feature_names[idx]
        fraud_score = log_prob_fraud[idx] * email_features[idx]
        legit_score = log_prob_legit[idx] * email_features[idx]
        diff = fraud_score - legit_score
        contributions.append((word, diff))

contributions.sort(key=lambda x: x[1], reverse=True)

print("\nWords pushing toward FRAUD:")
for word, score in contributions[:5]:
    if score > 0:
        print(f"  ✓ '{word}' (score: {score:.4f})")

print("\nWords pushing toward LEGITIMATE:")
for word, score in contributions[-3:]:
    if score < 0:
        print(f"  ✗ '{word}' (score: {score:.4f})")

# Step 4: Training Summary
print("\n4. MODEL TRAINING")
print("-" * 60)
print("The model was trained on:")
print("  • Enron dataset: Real corporate emails (legitimate)")
print("  • Phishing dataset: Real phishing emails (fraud)")
print("\nIt learned patterns like:")
print("  • Fraud emails often contain: urgent, verify, suspended,")
print("    password, account, click here")
print("  • Legitimate emails discuss: business, meetings, reports,")
print("    projects, schedules")

print("\n" + "=" * 60)
