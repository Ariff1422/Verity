import os
from transformers import AutoTokenizer

# --- File Path ---
MODEL_SAVE_PATH = '../models/tinybert_finetuned'

def save_only_tokenizer():
    """
    Saves the pre-trained TinyBERT tokenizer to the specified model directory.
    This is a standalone script to fix the missing tokenizer files
    after model training.
    """
    print("--- Starting Tokenizer Saving Process ---")
    
    # Check if the directory for the model exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: The model directory '{MODEL_SAVE_PATH}' was not found.")
        print("Please ensure your fine-tuned model has been saved there first.")
        return

    # Load the pre-trained tokenizer
    print("Loading TinyBERT tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Save the tokenizer to the same directory as the model
    print(f"Saving the tokenizer to '{MODEL_SAVE_PATH}'...")
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print("Tokenizer saving complete. The model and tokenizer are now ready for use!")

if __name__ == "__main__":
    save_only_tokenizer()
