import pandas as pd
import os
import json

# --- File Paths ---
# Path to the labeled dataset created in a previous step
LABELED_DATASET_PATH = os.path.join('..', 'data', 'llm_labeled_subset_50k_self_trained.csv')

# Path to the raw review data to calculate visit counts
RAW_REVIEWS_PATH = os.path.join('..', 'data', 'raw', 'review-California_10.json')

# Path to save the final dataset with the new column
OUTPUT_DATA_PATH = os.path.join('..', 'data', 'llm_labeled_subset_with_visits_updated.csv')

def add_visit_count_column():
    """
    Adds a 'visit_count' column to an existing labeled dataset.
    """
    print("--- Step 1: Loading the Labeled Dataset ---")
    try:
        labeled_df = pd.read_csv(LABELED_DATASET_PATH)
        print(f"Dataset '{os.path.basename(LABELED_DATASET_PATH)}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Labeled dataset not found at {LABELED_DATASET_PATH}.")
        return

    # --- Step 2: Calculating Visit Counts from Raw Data ---
    print("\n--- Step 2: Calculating Visit Counts ---")
    review_data = []
    try:
        with open(RAW_REVIEWS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line)
                    review_data.append({'user_id': review['user_id'], 'gmap_id': review['gmap_id']})
                except json.JSONDecodeError:
                    continue
        
        # Create a DataFrame from the raw review data
        visit_counts_df = pd.DataFrame(review_data)
        
        # Group by user and place to count the number of visits
        visit_counts_df['visit_count'] = visit_counts_df.groupby(['user_id', 'gmap_id'])['user_id'].transform('count')
        
        # Keep only unique user-place pairs to prepare for merging
        visit_counts_df = visit_counts_df.drop_duplicates().reset_index(drop=True)
        
        print("Visit counts calculated successfully.")
        
    except FileNotFoundError:
        print(f"Warning: Raw review data not found at {RAW_REVIEWS_PATH}. Cannot calculate visit counts.")
        visit_counts_df = pd.DataFrame(columns=['user_id', 'gmap_id', 'visit_count'])
    
    # --- Step 3: Merging the Visit Count Data ---
    print("\n--- Step 3: Merging dataframes ---")
    
    # Merge the visit counts into the labeled dataset on 'user_id' and 'gmap_id'
    labeled_df = pd.merge(labeled_df, visit_counts_df, on=['user_id', 'gmap_id'], how='left')
    
    # Fill any NaN values that may result from the merge with 1 (for single-visit users)
    labeled_df['visit_count'] = labeled_df['visit_count'].fillna(1).astype(int)

    print("Visit count column added to the labeled dataset.")
    print(f"Final dataframe shape: {labeled_df.shape}")

    # --- Step 4: Saving the Updated Dataset ---
    print("\n--- Step 4: Saving the Updated Dataset ---")
    labeled_df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\nFinal updated dataset saved to: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    add_visit_count_column()
