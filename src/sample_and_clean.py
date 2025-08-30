import pandas as pd
import os

# --- File Paths ---
CALIFORNIA_MERGED_PATH = '../data/california_reviews_merged.csv'
CLEANED_REVIEWS_PATH = '../data/cleaned_reviews.csv'

def sample_and_clean_data():
    """
    Loads a large CSV file, samples a subset of reviews, performs basic cleaning,
    and saves the result to a new CSV file.
    """
    print("\n--- Starting Data Sampling and Cleaning Process ---")

    # Check if the source file exists
    if not os.path.exists(CALIFORNIA_MERGED_PATH):
        print(f"Error: The source file '{CALIFORNIA_MERGED_PATH}' was not found.")
        print("Please ensure the 'california_merged.csv' file is in the 'data' directory.")
        return

    # Load the entire dataset
    print(f"Loading data from '{CALIFORNIA_MERGED_PATH}'...")
    try:
        # We specify low_memory=False to handle potential mixed dtypes in a large CSV
        df = pd.read_csv(CALIFORNIA_MERGED_PATH, low_memory=False)
        print(f"Successfully loaded {len(df)} reviews.")
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        return

    # Ensure the 'text' column exists for sampling and cleaning
    if 'text' not in df.columns:
        print("Error: The column 'text' was not found in the CSV file.")
        return

    # Define the sampling size range
    min_size = 100000
    max_size = 200000
    current_size = len(df)
    
    # Check if the dataset is large enough to sample from
    if current_size < min_size:
        print(f"Warning: The dataset size ({current_size}) is smaller than the minimum sample size ({min_size}).")
        print("Using the full dataset for cleaning.")
        sampled_df = df.copy()
    else:
        # Sample a random subset of the data
        sample_size = min(max_size, current_size)
        print(f"Sampling {sample_size} reviews...")
        sampled_df = df.sample(n=sample_size, random_state=42, replace=False)

    # --- Basic Data Cleaning ---
    # Drop rows where the review text is missing
    initial_rows = len(sampled_df)
    sampled_df.dropna(subset=['text'], inplace=True)
    rows_dropped = initial_rows - len(sampled_df)
    print(f"Dropped {rows_dropped} rows with missing 'text'.")

    # Save the cleaned, sampled data
    os.makedirs(os.path.dirname(CLEANED_REVIEWS_PATH), exist_ok=True)
    sampled_df.to_csv(CLEANED_REVIEWS_PATH, index=False)
    print(f"\nCleaned and sampled data saved to {CLEANED_REVIEWS_PATH}")
    print(f"Final dataset size: {len(sampled_df)} reviews.")
    print("Data sampling and cleaning process complete.")

if __name__ == "__main__":
    sample_and_clean_data()
