import pickle

# Load trained model
with open('Model/fraud_detector.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_fraud(email_text):
    """Predict if an email is fraudulent"""
    email_vec = vectorizer.transform([email_text])
    prediction = model.predict(email_vec)[0]
    probability = model.predict_proba(email_vec)[0]
    
    return {
        'is_fraud': bool(prediction),
        'confidence': float(probability[prediction]),
        'fraud_probability': float(probability[1])
    }

# Example usage
if __name__ == "__main__":
    test_email = """
    Subject: Urgent Account Verification Required
    
    Dear user,
    Your account has been suspended. Please click here to verify your password
    and account details immediately to avoid permanent suspension.
    """
    
    result = predict_fraud(test_email)
    print(f"Email is {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")
