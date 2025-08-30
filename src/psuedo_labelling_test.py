import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

# --- File Paths and Constants ---
GB_MODEL_PATH = '../models/gb_model_advanced_tuned.pkl'
VECTORIZER_PATH = '../models/vectorizer_advanced_tuned.pkl'
UNLABELED_DATA_PATH = '../data/cleaned_reviews.csv'
LABELED_DATA_PATH = '../data/augmented_labeled_reviews.csv'
OUTPUT_DATA_PATH = '../data/final_labeled_dataset.csv'
FINAL_SAMPLED_DATA_PATH = '../data/test_labeled_dataset_sampled.csv'
SAMPLE_SIZE = 20000

def pseudo_label_data():
    """
    Loads a pre-trained Gradient Boosting model and a TfidfVectorizer,
    uses them to pseudo-label an unlabeled dataset, saves the
    combined data to a new CSV file, and also creates a random sample.
    """
    print("\n--- Starting Pseudo-Labeling Process with Gradient Boosting Model ---")

    # Check if necessary files exist
    if not os.path.exists(GB_MODEL_PATH):
        print(f"Error: The model file '{GB_MODEL_PATH}' was not found. Please ensure you have run the advanced tuning script first.")
        return
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Error: The vectorizer file '{VECTORIZER_PATH}' was not found. Please run the advanced tuning script.")
        return
    if not os.path.exists(UNLABELED_DATA_PATH):
        print(f"Error: The unlabeled data file '{UNLABELED_DATA_PATH}' was not found.")
        return
    if not os.path.exists(LABELED_DATA_PATH):
        print(f"Error: The already labeled data file '{LABELED_DATA_PATH}' was not found.")
        return

    # Load the trained model and vectorizer
    print("Loading the Gradient Boosting model and TfidfVectorizer...")
    gb_model = joblib.load(GB_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Models loaded successfully.")

    # Load the unlabeled data
    print(f"Loading unlabeled data from {UNLABELED_DATA_PATH}...")
    unlabeled_df = pd.read_csv(UNLABELED_DATA_PATH)
    print(f"Loaded {len(unlabeled_df)} unlabeled reviews.")

    # Transform the unlabeled text data using the loaded vectorizer
    print("Transforming unlabeled text data...")
    X_unlabeled_tfidf = vectorizer.transform(unlabeled_df['text'])
    
    # Predict labels for the unlabeled data
    print("Generating pseudo-labels...")
    pseudo_labels = gb_model.predict(X_unlabeled_tfidf)

    # Add the pseudo-labels to the unlabeled DataFrame
    unlabeled_df['violation_type'] = pseudo_labels
    
    # Load the original labeled data
    print(f"Loading existing labeled data from {LABELED_DATA_PATH}...")
    labeled_df = pd.read_csv(LABELED_DATA_PATH)
    print(f"Loaded {len(labeled_df)} already labeled reviews.")

    # Combine the datasets
    print("Combining datasets...")
    final_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    
    # Save the new, expanded dataset
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_DATA_PATH, index=False)
    
    print(f"\nFinal expanded dataset saved to {OUTPUT_DATA_PATH}")
    print(f"Final dataset size: {len(final_df)} reviews.")

    # --- Create and save a random sample if the total size is greater than SAMPLE_SIZE ---
    if len(final_df) > SAMPLE_SIZE:
        print(f"\nCreating a random sample of {SAMPLE_SIZE} reviews...")
        sampled_df = final_df.sample(n=SAMPLE_SIZE, random_state=65)
        os.makedirs(os.path.dirname(FINAL_SAMPLED_DATA_PATH), exist_ok=True)
        sampled_df.to_csv(FINAL_SAMPLED_DATA_PATH, index=False)
        print(f"Randomly sampled dataset saved to {FINAL_SAMPLED_DATA_PATH}")
        print(f"Sampled dataset size: {len(sampled_df)} reviews.")
    else:
        print(f"\nDataset size is less than {SAMPLE_SIZE}. Skipping sampling.")

    print("\nPseudo-labeling process complete.")
    
if __name__ == "__main__":
    pseudo_label_data()
