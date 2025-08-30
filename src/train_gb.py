import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# --- File Paths ---
INPUT_CSV = '../data/augmented_labeled_reviews.csv'
GB_MODEL_SAVE_PATH = '../models/gb_model_advanced_tuned.pkl'
VECTORIZER_SAVE_PATH = '../models/vectorizer_advanced_tuned.pkl'

def advanced_tune_gradient_boosting_model():
    """
    Performs advanced hyperparameter tuning for a Gradient Boosting Classifier
    using RandomizedSearchCV and improved feature engineering.
    """
    print("\n--- Advanced Hyperparameter Tuning Gradient Boosting Model ---")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: The input file '{INPUT_CSV}' was not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    X = df['text']
    y = df['violation_type']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert text to TF-IDF features with n-grams
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Define a broader parameter distribution to sample from
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    
    # Instantiate the Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42)
    
    # Use RandomizedSearchCV to find the best parameters
    random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_dist,
                                       n_iter=50, cv=3, n_jobs=-1, verbose=2, scoring='accuracy', random_state=42)
    
    random_search.fit(X_train_tfidf, y_train)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print("\nBest parameters found: ", best_params)
    print("Best cross-validation accuracy: {:.4f}".format(best_score))

    # Retrain the model on the full training set with the best parameters
    best_gb_model = random_search.best_estimator_
    best_gb_model.fit(X_train_tfidf, y_train)
    
    # Evaluate the tuned model on the validation set
    X_val_tfidf = vectorizer.transform(X_val)
    y_pred = best_gb_model.predict(X_val_tfidf)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Tuned Gradient Boosting Model Validation Accuracy: {accuracy:.4f}")
    
    # Save the best model and the vectorizer
    os.makedirs(os.path.dirname(GB_MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_gb_model, GB_MODEL_SAVE_PATH)
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    print(f"Tuned Gradient Boosting model and vectorizer saved to {GB_MODEL_SAVE_PATH} and {VECTORIZER_SAVE_PATH}")

if __name__ == "__main__":
    advanced_tune_gradient_boosting_model()
