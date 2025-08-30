import pandas as pd
import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- LLM Model and Tokenizer Setup ---
# This script uses the Qwen3-8B model from Hugging Face.
# The model will be downloaded and run locally.
MODEL_NAME = "Qwen/Qwen3-8B"

# The LLM model and tokenizer are initialized globally to avoid reloading for each review.
print(f"Loading model '{MODEL_NAME}' from Hugging Face...")

try:
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have a compatible GPU and sufficient VRAM.")
    model = None
    tokenizer = None

# Dictionary to map numerical classifications to text labels
LABEL_MAPPING = {
    "1": "Advertisement",
    "2": "Irrelevant Content",
    "3": "Rant Without Visit",
    "4": "No Violation"
}

# The prompt is now split to use the Qwen chat template.
# The system prompt sets the role, and the user prompt provides the task and examples.
SYSTEM_PROMPT = "You are a policy violation detector for location reviews. You ONLY respond with the category number that best describes the provided review. Do not include any extra text or punctuation."

PROMPT_TEMPLATE = """
Classify the following review into one of the following categories:
1. Advertisement
2. Irrelevant Content
3. Rant Without Visit
4. No Violation

---
Review: "Great place! Visit my website www.bestdeals.com for a coupon!"
Classification: 1

Review: "I never visited this place, but my friend said the service was awful."
Classification: 3

Review: "The coffee was amazing and the atmosphere was cozy. I highly recommend it."
Classification: 4

Review: "The atmosphere was so great that I had to bring my pet lizard with me. He loved the place, even though they don't allow pets."
Classification: 2
---

Review: "{review_text}"
Classification:
"""

def get_llm_classification(review_text: str) -> str:
    """
    Calls the locally loaded LLM to classify a single review.
    """
    if model is None or tokenizer is None:
        return "Model not loaded"

    try:
        # We construct the message with a system prompt and a user query
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(review_text=review_text)}
        ]
        
        # Apply the chat template and explicitly disable thinking mode for a direct response
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Prepare the model input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Use the model's generate method with recommended parameters for non-thinking mode
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        # Decode and clean the output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        classification_number = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Map the numerical classification to the corresponding label
        if classification_number in LABEL_MAPPING:
            return LABEL_MAPPING[classification_number]
        else:
            print(f"Warning: LLM returned an invalid classification: '{classification_number}'. Returning 'No Violation'.")
            return "No Violation"
            
    except Exception as e:
        print(f"Error classifying review: {e}")
        return "Classification Failed"

def main():
    """
    Main function to load data, perform pseudo-labeling, and save the result.
    """
    # Define file paths
    input_file = '../data/california_reviews_merged.csv'
    output_file = '../data/labeled_reviews.csv'

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please ensure you have the `california_reviews_merged.csv` file inside your `data` folder.")
        return

    # Load your cleaned data. Adjust the encoding and delimiter as needed.
    try:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file, encoding='utf-8')
        # We now check for the 'text' column as per your input
        if 'text' not in df.columns:
            print("Error: The 'text' column was not found in the CSV file.")
            print("Please ensure the column name for the review text is 'text'.")
            return
        print(f"Successfully loaded {len(df)} reviews.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Take a random sample to control processing time
    sample_size = min(len(df), 100) # Use a smaller sample for initial testing
    print(f"Sampling {sample_size} reviews for pseudo-labeling...")
    sampled_df = df.sample(n=sample_size, random_state=42).copy()

    # Create a new column for the pseudo-labels
    sampled_df['policy_violation_label'] = 'No Violation'

    # Perform pseudo-labeling for each review, using the 'text' column
    for i, (index, row) in enumerate(sampled_df.iterrows()):
        review_text = row['text']
        label = get_llm_classification(review_text)
        sampled_df.at[index, 'policy_violation_label'] = label
        print(f"Processed {i+1}/{len(sampled_df)}: Review classified as '{label}'")

    # Save the new labeled dataset
    sampled_df.to_csv(output_file, index=False)
    print(f"\nPseudo-labeled dataset saved to {output_file}")
    print(sampled_df['policy_violation_label'].value_counts())
    print("\nScript finished.")

if __name__ == "__main__":
    main()
