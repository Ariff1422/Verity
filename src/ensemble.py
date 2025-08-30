import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, SequentialSampler

# --- File Paths ---
INPUT_CSV = '../data/augmented_labeled_reviews.csv'
TRANSFORMER_MODEL_LOAD_PATH = '../models/tinybert_finetuned'
GB_MODEL_LOAD_PATH = '../models/gb_model_advanced_tuned.pkl'
VECTORIZER_LOAD_PATH = '../models/vectorizer_advanced_tuned.pkl'
ENSEMBLE_MODEL_SAVE_PATH = '../models/ensemble_meta_model.pkl'

# --- Custom Dataset Class ---
class ReviewDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle the tokenized review data.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_predictions(model_path, data_loader, tokenizer=None, vectorizer=None, gb_model=None):
    """
    A helper function to get predictions from either the transformer or GB model.
    """
    if tokenizer:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        predictions = []
        for batch in data_loader:
            with torch.no_grad():
                input_ids, attention_mask, _ = batch['input_ids'], batch['attention_mask'], batch['labels']
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions.extend(torch.softmax(logits, dim=1).tolist())
        return np.array(predictions)
    
    elif gb_model and vectorizer:
        X_val_tfidf = vectorizer.transform(data_loader)
        return gb_model.predict_proba(X_val_tfidf)
    else:
        return None

def main():
    print("\n--- Tuning the Ensemble Model ---")

    # Load the augmented dataset
    df = pd.read_csv(INPUT_CSV)
    X = df['text']
    y = df['violation_type']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load the trained models
    gb_model = joblib.load(GB_MODEL_LOAD_PATH)
    vectorizer = joblib.load(VECTORIZER_LOAD_PATH)
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    
    # Prepare data for the transformer
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True)
    val_dataset = ReviewDataset(val_encodings, list(y_val))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

    # Get predictions from each model
    transformer_preds = get_predictions(TRANSFORMER_MODEL_LOAD_PATH, val_dataloader, tokenizer=tokenizer)
    gb_preds = get_predictions(GB_MODEL_LOAD_PATH, X_val, vectorizer=vectorizer, gb_model=gb_model)

    # Ensure predictions are numpy arrays
    transformer_preds = np.array(transformer_preds)
    gb_preds = np.array(gb_preds)
    
    # Stack the predictions for meta-model training
    stacked_features = np.hstack((transformer_preds, gb_preds))

    # --- Weighted Averaging Approach ---
    print("\n--- Evaluating Weighted Averaging ---")
    # You can experiment with different weights here (e.g., 0.6, 0.4)
    # The weights should add up to 1.0.
    weights = [0.5, 0.5]
    weighted_preds_proba = (transformer_preds * weights[0] + gb_preds * weights[1])
    weighted_preds = np.argmax(weighted_preds_proba, axis=1) + 1  # Add 1 to convert from 0-indexed to 1-indexed classes
    
    weighted_accuracy = accuracy_score(y_val, weighted_preds)
    print(f"Weighted Average Ensemble Accuracy: {weighted_accuracy:.4f}")
    print("Full Weighted Average Ensemble Classification Report:")
    print(classification_report(y_val, weighted_preds, zero_division=0))

    # --- Different Meta-Model Approach (Random Forest) ---
    print("\n--- Training and Evaluating RandomForest Meta-Model ---")
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(stacked_features, y_val)
    
    # Evaluate the new meta-model
    ensemble_preds = meta_model.predict(stacked_features)
    ensemble_accuracy = accuracy_score(y_val, ensemble_preds)
    
    print(f"RandomForest Meta-Model Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print("\nFull RandomForest Meta-Model Ensemble Classification Report:")
    print(classification_report(y_val, ensemble_preds, zero_division=0))
    
    # Save the best-performing meta-model
    joblib.dump(meta_model, ENSEMBLE_MODEL_SAVE_PATH)
    print(f"\nBest-performing meta-model saved to {ENSEMBLE_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
