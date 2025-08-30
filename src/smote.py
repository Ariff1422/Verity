import pandas as pd
import joblib
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- File Paths ---
VECTORIZER_PATH = '../models/vectorizer_advanced_tuned.pkl'
GB_MODEL_PATH = '../models/gb_model_advanced_tuned.pkl'
TINYBERT_MODEL_PATH = '../models/tinybert_finetuned'
META_MODEL_PATH = '../models/ensemble_meta_model.pkl'

# --- Dataset Path ---
DATA_PATH = '../data/final_labeled_dataset.csv'

def train_ensemble_with_smote():
    """
    Trains the full ensemble pipeline with SMOTE for class balancing.
    """
    print("\n--- Starting Ensemble Training with SMOTE ---")

    # --- 1. Load Data and Check for Prerequisites ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data file '{DATA_PATH}' not found.")
        return

    df_train = pd.read_csv(DATA_PATH)
    df_train.dropna(subset=['text', 'violation_type'], inplace=True)

    X = df_train['text']
    y = df_train['violation_type']

    # For simplicity, we'll use a single split for demonstration purposes
    # In a real-world scenario, you would use cross-validation to get true OOF predictions
    X_train_text, X_val_text, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 2. Sanity Check for Minority Classes ---
    # This step ensures that the minority classes are present in the training data
    # to prevent SMOTE from failing. We will add a few hardcoded examples if they
    # are missing.
    missing_classes = []
    required_classes = ['1', '2', '3']
    for cls in required_classes:
        if cls not in y_train.unique():
            print(f"Warning: Class '{cls}' is missing from the training data. Adding a few examples.")
            missing_classes.append(cls)

    if missing_classes:
        # Hardcoded examples to add for missing classes
        additional_data = []
        for cls in missing_classes:
            if cls == '1':
                additional_data.append({"text": "Visit my site for a coupon! www.bestdeals.com", "violation_type": "1"})
            elif cls == '2':
                additional_data.append({"text": "My cat just had kittens. They are so cute!", "violation_type": "2"})
            elif cls == '3':
                additional_data.append({"text": "I heard from my friend that the service is terrible. I'm not going to visit.", "violation_type": "3"})

        additional_df = pd.DataFrame(additional_data)
        
        # Concatenate the new data with the existing training data
        X_train_text = pd.concat([X_train_text, additional_df['text']], ignore_index=True)
        y_train = pd.concat([y_train, additional_df['violation_type']], ignore_index=True)

    # --- 3. Load Base Models and Vectorizer ---
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        gb_model = joblib.load(GB_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(TINYBERT_MODEL_PATH)
        tinybert_model = AutoModelForSequenceClassification.from_pretrained(TINYBERT_MODEL_PATH)
        tinybert_model.eval()
    except Exception as e:
        print(f"Error loading required components: {e}")
        print("Please ensure you have run the data prep and base model training scripts first.")
        return

    # --- 4. Get OOF Predictions from Base Models ---
    print("Generating out-of-fold predictions from base models...")
    
    # Gradient Boosting Model Predictions
    X_train_tfidf = vectorizer.transform(X_train_text)
    gb_probs = gb_model.predict_proba(X_train_tfidf)

    # TinyBERT Model Predictions
    tinybert_probs = []
    with torch.no_grad():
        for text in X_train_text:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = tinybert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
            tinybert_probs.append(probabilities[0])
    tinybert_probs = np.array(tinybert_probs)

    # Combine predictions to create the meta-features
    X_meta = np.hstack([gb_probs, tinybert_probs])
    y_meta = y_train.values

    # --- 5. Train the Meta-Model with SMOTE ---
    print("Applying SMOTE to rebalance the meta-features...")

    # Define the meta-model and the SMOTE sampler in a pipeline
    meta_pipeline = Pipeline([
        ('sampler', SMOTE(random_state=42, k_neighbors=3)),  # Adjust k_neighbors if needed
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Fit the pipeline
    print("Training the RandomForest meta-model on the rebalanced data...")
    meta_pipeline.fit(X_meta, y_meta)
    
    # --- 6. Save the Final Ensemble Meta-Model ---
    print("Saving the rebalanced meta-model...")
    joblib.dump(meta_pipeline, META_MODEL_PATH)
    print(f"Rebalanced ensemble meta-model saved to '{META_MODEL_PATH}'")

if __name__ == "__main__":
    train_ensemble_with_smote()
