import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import os

# Define file paths
LABELED_FILE = f'../data/labeled_reviews.csv'

# --- Step 1: Load the labeled dataset ---
try:
    labeled_df = pd.read_csv(LABELED_FILE)
    print("Successfully loaded the labeled dataset.")
except FileNotFoundError:
    print(f"Error: The file {LABELED_FILE} was not found.")
    exit()

# Drop any rows where the 'is_problematic' label is missing
labeled_df.dropna(subset=['is_problematic'], inplace=True)

# --- Step 2: Define features (X) and labels (y) ---
X = labeled_df['text'].astype(str)
y = labeled_df['is_problematic'].astype(int)
print(f"Dataset has {len(labeled_df)} labeled examples.")

# --- Step 3: Split the data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data successfully split into training and testing sets.")

# --- Step 4: Text Vectorization ---
# Convert the text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Text vectorized into numerical format.")

# --- Step 5: Train the Logistic Regression model for classification ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
print("Logistic Regression model training complete.")

# --- Step 6: Make predictions and evaluate the model ---
y_pred = model.predict(X_test_vec)

print("\n--- Model Evaluation Report ---")
print(classification_report(y_test, y_pred))

# Get the F1 score specifically
f1 = f1_score(y_test, y_pred, average='binary')
print(f"F1 Score: {f1:.4f}")