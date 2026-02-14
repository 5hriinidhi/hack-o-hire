import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import os
import email
from email import policy

# Download Enron dataset (legitimate emails)
print("Downloading Enron dataset...")
enron_path = kagglehub.dataset_download("wcukierski/enron-email-dataset")
print("Enron path:", enron_path)

# Download Phishing dataset (fraud emails)
print("\nDownloading Phishing dataset...")
phishing_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
print("Phishing path:", phishing_path)

# Load Enron emails (legitimate)
print("\nLoading Enron emails...")
enron_csv = os.path.join(enron_path, 'emails.csv')
df_enron = pd.read_csv(enron_csv)
df_enron['full_text'] = df_enron['file'].fillna('') + ' ' + df_enron['message'].fillna('')
df_enron['label'] = 0  # Legitimate
print(f"Enron emails: {len(df_enron)}")

# Load Phishing emails (fraud)
print("Loading Phishing emails...")
phishing_csv = os.path.join(phishing_path, 'Phishing_Email.csv')
df_phishing = pd.read_csv(phishing_csv)
print(f"Phishing dataset columns: {df_phishing.columns.tolist()}")

# Combine text columns from phishing dataset
if 'Email Text' in df_phishing.columns:
    df_phishing['full_text'] = df_phishing['Email Text'].fillna('')
elif 'text' in df_phishing.columns:
    df_phishing['full_text'] = df_phishing['text'].fillna('')
else:
    # Use all text columns
    text_cols = [col for col in df_phishing.columns if df_phishing[col].dtype == 'object']
    df_phishing['full_text'] = df_phishing[text_cols].fillna('').agg(' '.join, axis=1)

df_phishing['label'] = 1  # Fraud
print(f"Phishing emails: {len(df_phishing)}")

# Balance the dataset
sample_size = min(len(df_enron), len(df_phishing), 5000)
df_enron_sample = df_enron.sample(n=sample_size, random_state=42)
df_phishing_sample = df_phishing.sample(n=sample_size, random_state=42)

# Combine datasets
df_combined = pd.concat([df_enron_sample, df_phishing_sample], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

emails = df_combined['full_text'].tolist()
labels = df_combined['label'].tolist()

print(f"\nTotal emails: {len(emails)}")
print(f"Legitimate: {labels.count(0)}, Fraud: {labels.count(1)}")


print("\nCreating train/test split...")
# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorize text data
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model with class balancing
print("Training model...")
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Save model
import pickle
with open('Model/fraud_detector.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('Model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel saved successfully!")
