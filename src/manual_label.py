import pandas as pd
import os
import random
import sys

def interactive_labeling(file_path: str, output_path: str, sample_size: int = 200):
    """
    An interactive tool to manually label a sample of reviews in the terminal.
    
    Args:
        file_path (str): Path to the full reviews CSV file.
        output_path (str): Path to save the labeled data.
        sample_size (int): The number of reviews to sample for labeling.
    """
    if not os.path.exists(file_path):
        print(f"Error: The input file '{file_path}' was not found.")
        print("Please ensure the file is in the correct directory.")
        return
        
    labeled_count = 0  # Ensure labeled_count is always defined

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} total reviews.")

        # Take a random sample from the full dataset
        sampled_df = df.sample(n=min(sample_size, len(df)), random_state=24).copy()
        sampled_df.reset_index(drop=True, inplace=True)
        
        # Add a new column for our labels
        sampled_df['violation_type'] = ""

        labeled_count = 0

        categories = {
            '1': 'Advertisement',
            '2': 'Irrelevant Content',
            '3': 'Rant Without Visit',
            '4': 'No Violation'
        }

        print("\n--- Starting Interactive Labeling ---")
        print("Please classify each review by entering a number from the list below.")
        print("Type 'undo' to correct your last entry.")
        print("Press Ctrl+C at any time to quit and save your progress.\n")
        
        # This loop now iterates through the index to allow for "undo" functionality
        i = 0
        while i < len(sampled_df):
            row = sampled_df.iloc[i]
            review_text = row['text']
            business_category = row['category']
            
            print("---------------------------------")
            print(f"Review #{i + 1}/{len(sampled_df)}")
            print(f"Business Category: {business_category}")
            print("\nReview Text:")
            print(f"'{review_text}'")
            print("\nCategories:")
            for num, cat_name in categories.items():
                print(f"  {num}: {cat_name}")
            print("---------------------------------")
            
            user_input = input("Enter classification number: ").strip()
            
            if user_input.lower() == 'undo':
                if i > 0:
                    i -= 1 # Go back one step
                    print("\nLast entry undone. Please re-label the previous review.")
                else:
                    print("Nothing to undo.")
                continue # Restart the loop for the current (now previous) review
            
            if user_input in categories:
                # Update the DataFrame directly with the user's input
                sampled_df.loc[i, 'violation_type'] = user_input
                labeled_count += 1
                print(f"Labeled as: {categories[user_input]}\n")
                i += 1 # Move to the next review
            else:
                print("Invalid input. Please enter a number from the list or 'undo'.")
                
            # Save progress every 10 reviews
            if labeled_count > 0 and labeled_count % 10 == 0 and user_input in categories:
                print("Saving progress...")
                sampled_df.to_csv(output_path, index=False)
                print("Progress saved.")

    except KeyboardInterrupt:
        print("\nLabeling interrupted by user. Saving all progress...")
    finally:
        if labeled_count > 0 and sampled_df is not None:
            sampled_df.to_csv(output_path, index=False)
            print(f"\nSaved {labeled_count} labeled reviews to '{output_path}'.")
            print("\nDistribution of labels:")
            print(sampled_df['violation_type'].value_counts())
        else:
            print("No new labels to save.")
            print("No new labels to save.")


if __name__ == "__main__":
    interactive_labeling(
        file_path='../data/california_reviews_merged.csv', 
        output_path='../data/manual_labeled_reviews_200.csv'
    )
