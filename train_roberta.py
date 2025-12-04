"""
RoBERTa Model Training Script for Review Rating Prediction (1-5 stars)
Trains on GPU only, with proper data preprocessing
Predicts rating based on review text
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Force GPU-only mode - NO CPU FALLBACK
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available! This script requires GPU for training. CPU is not supported.")

# Set device to GPU only (no CPU fallback)
device = torch.device('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)

# Verify GPU is available and print info
print("=" * 60)
print("GPU Configuration (GPU-ONLY MODE)")
print("=" * 60)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Device: {device}")
print("=" * 60)

# Disable CPU fallback
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def clean_text(text):
    """
    Clean and preprocess text data
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    # text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
    
    # Remove multiple consecutive punctuation marks
    text = re.sub(r'[!?.]{2,}', '.', text)
    
    return text.strip()


def load_and_preprocess_data(file_path, sample_size=None):
    """
    Load and preprocess the review dataset
    Only keeps Text and Score columns, removes all irrelevant fields
    """
    print("Loading dataset...")
    print("=" * 60)
    
    # Read in chunks - optimized for speed
    chunk_size = 100000  # Larger chunks for faster reading
    chunks = []
    total_read = 0
    
    try:
        print("Reading CSV file in chunks (optimized for speed)...")
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, usecols=['Text', 'Score']):
            # Only keep Text and Score columns (already filtered with usecols)
            chunks.append(chunk)
            total_read += len(chunk)
            if sample_size and total_read >= sample_size:
                break
            if len(chunks) % 5 == 0:
                print(f"  Read {total_read:,} rows...")
    except Exception as e:
        print(f"Error reading file in chunks: {e}")
        print("Trying to read entire file...")
        # Try reading without chunks if file is smaller
        df = pd.read_csv(file_path, low_memory=False, usecols=['Text', 'Score'])
        chunks = [df]
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"\nLoaded {len(df):,} total reviews")
    
    # Remove rows with missing text or score
    initial_count = len(df)
    df = df.dropna(subset=['Text', 'Score'])
    print(f"Removed {initial_count - len(df):,} rows with missing values")
    
    # Convert Score to integer and filter valid scores (1-5)
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score'])
    df['Score'] = df['Score'].astype(int)
    
    # Filter valid scores (1-5)
    valid_scores = df['Score'].between(1, 5)
    df = df[valid_scores].copy()
    print(f"Filtered to {len(df):,} reviews with valid scores (1-5)")
    
    # Clean text data - optimized for speed
    print("\nCleaning text data (optimized)...")
    df['Text'] = df['Text'].astype(str).str.strip()
    # Fast text cleaning using vectorized operations
    df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)
    df['Text'] = df['Text'].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    df['Text'] = df['Text'].str.replace(r'\S+@\S+', '', regex=True)
    
    # Remove empty or very short reviews
    df = df[df['Text'].str.len() >= 15].copy()
    print(f"After removing short reviews: {len(df):,} reviews")
    
    # Create labels: Convert Score (1-5) to label (0-4) for model
    df['label'] = df['Score'] - 1  # 1->0, 2->1, 3->2, 4->3, 5->4
    
    # Show label distribution
    print("\n" + "=" * 60)
    print("Label Distribution (Rating -> Label):")
    print("=" * 60)
    label_dist = df['label'].value_counts().sort_index()
    for label, count in label_dist.items():
        rating = label + 1
        percentage = (count / len(df)) * 100
        print(f"  Rating {rating} (Label {label}): {count:,} ({percentage:.2f}%)")
    
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size:,} reviews for training...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Final dataset size: {len(df):,} reviews")
    
    print("=" * 60)
    
    # Return only Text and label columns
    return df[['Text', 'label']].copy()


def tokenize_data(examples, tokenizer, max_length=512):
    """
    Tokenize the text data
    Note: Tensors will be moved to GPU during training by Trainer
    """
    return tokenizer(
        examples['Text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None  # Return as lists, Trainer will handle GPU transfer
    )


def compute_metrics(eval_pred):
    """
    Compute accuracy, F1 score, and per-class metrics for rating prediction
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
    }
    
    # Add per-class metrics (for ratings 1-5)
    for i in range(5):
        rating = i + 1
        metrics[f'f1_rating_{rating}'] = f1_per_class[i]
        metrics[f'precision_rating_{rating}'] = precision[i]
        metrics[f'recall_rating_{rating}'] = recall[i]
    
    return metrics


def main():
    # Configuration - ULTRA FAST MODE
    MODEL_NAME = 'roberta-base'
    MAX_LENGTH = 128  # Very short for maximum speed (still works well for reviews)
    BATCH_SIZE = 128  # Very large batch size (adjust if OOM)
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 5e-5  # Higher learning rate for faster convergence
    NUM_EPOCHS = 1  # Single epoch for maximum speed
    OUTPUT_DIR = './roberta_rating_model'
    DATA_FILE = 'Reviews.csv'
    
    # Use smaller sample for ultra-fast training
    SAMPLE_SIZE = 50000  # Small sample for maximum speed
    
    print("=" * 60)
    print("RoBERTa Review Rating Prediction Model (1-5 stars)")
    print("=" * 60)
    
    # Load and preprocess data
    df = load_and_preprocess_data(DATA_FILE, sample_size=SAMPLE_SIZE)
    
    # Split data
    print("\nSplitting data into train/validation sets...")
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['Text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['Text', 'label']])
    
    # Tokenize datasets - optimized for speed
    print("\nTokenizing datasets (optimized for speed)...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_data(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=1000,  # Larger batch for tokenization
        remove_columns=['Text'],
        desc="Tokenizing training data"
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_data(x, tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=1000,  # Larger batch for tokenization
        remove_columns=['Text'],
        desc="Tokenizing validation data"
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    num_labels = 5  # 5 classes for ratings 1-5
    print(f"Number of classes: {num_labels} (ratings 1-5)")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Move model to GPU
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Training arguments - GPU ONLY (ULTRA FAST MODE)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 4,  # Very large eval batch
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=1000,  # Minimal logging
        eval_strategy='no',  # NO EVALUATION DURING TRAINING (fastest)
        save_strategy='epoch',  # Save only at end
        load_best_model_at_end=False,  # Don't load best (no eval anyway)
        fp16=True,  # Mixed precision for 2x speed
        bf16=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,  # Windows optimization
        report_to='none',
        save_total_limit=1,
        # Force GPU usage
        no_cuda=False,
        # Maximum speed optimizations
        remove_unused_columns=True,
        optim='adamw_torch',
        max_steps=-1,
        # Disable all unnecessary features
        prediction_loss_only=True,
        include_inputs_for_metrics=False,
        # Skip evaluation entirely
        eval_delay=0,
        # Faster data loading
        dataloader_drop_last=True,  # Drop last incomplete batch
    )
    
    # Verify device before training
    print(f"\nVerifying GPU usage before training...")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Expected device: {device}")
    if next(model.parameters()).device.type != 'cuda':
        raise RuntimeError(f"Model is not on GPU! Current device: {next(model.parameters()).device}")
    print("✓ Model is on GPU - Training will use GPU only")
    
    # Calculate estimated training time
    num_train_samples = len(train_df)
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = (num_train_samples // effective_batch_size) + 1
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"\nTraining Configuration Summary:")
    print(f"  Training samples: {num_train_samples:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Steps per epoch: ~{steps_per_epoch:,}")
    print(f"  Total training steps: ~{total_steps:,}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"\nNote: Training time depends on GPU speed. With RTX 3050, expect:")
    print(f"  ~{total_steps * 2 // 60} - {total_steps * 4 // 60} minutes total")
    print("=" * 60)
    
    # Initialize trainer - NO EVALUATION for maximum speed
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # No validation during training for speed
        compute_metrics=None,  # No metrics computation during training
        callbacks=[]  # No callbacks for speed
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Quick final evaluation (optional, can skip for even more speed)
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nNote: Evaluation was skipped during training for maximum speed.")
    print("You can evaluate the model separately using predict_rating.py")
    print("=" * 60)


if __name__ == '__main__':
    main()

