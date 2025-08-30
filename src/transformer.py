import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# --- File Paths ---
FINAL_LABELED_DATA_PATH = '../data/final_labeled_dataset.csv'
MODEL_SAVE_PATH = '../models/tinybert_finetuned'

# --- Custom Dataset Class ---
class ReviewDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle the tokenized review data.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Get item method for PyTorch DataLoader
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_transformer_model():
    """
    Loads the expanded dataset, fine-tunes a TinyBERT model,
    and saves the trained model.
    """
    print("\n--- Starting Transformer Model Training Process ---")
    
    # Check if the final labeled data exists
    if not os.path.exists(FINAL_LABELED_DATA_PATH):
        print(f"Error: The file '{FINAL_LABELED_DATA_PATH}' was not found.")
        print("Please ensure you have run the data sampling and pseudo-labeling scripts first.")
        return

    # Load the expanded dataset
    print(f"Loading expanded dataset from {FINAL_LABELED_DATA_PATH}...")
    df = pd.read_csv(FINAL_LABELED_DATA_PATH)
    print(f"Loaded {len(df)} total reviews.")

    # Drop any rows with missing text or labels
    df.dropna(subset=['text', 'violation_type'], inplace=True)
    print(f"Dataset size after cleaning: {len(df)} reviews.")

    # --- NEW: Sample a smaller portion for faster training ---
    sample_size = 20000  # Reduced to 5,000 for extremely fast training
    if len(df) > sample_size:
        print(f"Sampling a stratified subset of {sample_size} reviews for faster training...")
        # Use train_test_split to get a stratified sample from the full dataframe
        df_sampled, _ = train_test_split(df, train_size=sample_size, random_state=42, stratify=df['violation_type'])
    else:
        df_sampled = df
    print(f"Training on a sample of {len(df_sampled)} reviews.")

    # Prepare data for the model
    # Convert labels to be zero-indexed as transformers expect this
    texts = df_sampled['text'].tolist()
    labels = df_sampled['violation_type'].astype(int).tolist()
    labels = [label - 1 for label in labels] # Make labels 0-indexed

    # Split the data into training and validation sets
    print("Splitting data into training and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=52, stratify=labels
    )
    
    # Load the pre-trained tokenizer for TinyBERT
    print("Loading TinyBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')

    # Tokenize the datasets
    print("Tokenizing training and validation data...")
    # Explicitly set max_length to avoid the RuntimeError
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    # Create PyTorch datasets
    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)

    # Load the pre-trained TinyBERT model for sequence classification
    print("Loading TinyBERT model for sequence classification...")
    num_labels = len(np.unique(labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        'prajjwal1/bert-tiny', num_labels=num_labels
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # directory to save model and logs
        num_train_epochs=5,              # number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        gradient_accumulation_steps=4,   # Accumulate gradients over 4 batches
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,
        eval_strategy="steps",     # evaluate every 'eval_steps'
        eval_steps=1000,
        save_strategy="steps",           # save model every 'save_steps'
        save_steps=1000,
        load_best_model_at_end=True,     # load the best model when training ends
        metric_for_best_model='accuracy',# use accuracy to determine the best model
        report_to="none"                 # Do not report to external services like wandb
    )
    
    # Define a function to compute metrics
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_score(p.label_ids, preds)
        return {"accuracy": acc}

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    print("\nStarting model fine-tuning...")
    trainer.train()

    # Evaluate the final model on the validation set
    print("\nEvaluating the final model...")
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save the fine-tuned model
    print(f"Saving the fine-tuned model to '{MODEL_SAVE_PATH}'...")
    trainer.save_model(MODEL_SAVE_PATH)
    print("Model training and saving complete.")

if __name__ == "__main__":
    train_transformer_model()
