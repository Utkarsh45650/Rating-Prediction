"""
Evaluate the trained RoBERTa model and calculate accuracy, precision, recall, F1 score
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import Dataset
import re
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def clean_text(text):
    """Clean text data (same as training)"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()


def load_model(model_path='./roberta_rating_model'):
    """Load the trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model = model.to(device)
        print("✓ Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def prepare_data(file_path, sample_size=10000):
    """Load and prepare test data"""
    print(f"\nLoading test data from {file_path}...")
    
    # Read data
    chunk_size = 50000
    chunks = []
    total_read = 0
    
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, usecols=['Text', 'Score']):
            chunks.append(chunk)
            total_read += len(chunk)
            if sample_size and total_read >= sample_size:
                break
    except Exception as e:
        print(f"Error reading file: {e}")
        df = pd.read_csv(file_path, low_memory=False, usecols=['Text', 'Score'])
        chunks = [df]
    
    df = pd.concat(chunks, ignore_index=True)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Loaded {len(df):,} reviews")
    
    # Preprocess
    df = df.dropna(subset=['Text', 'Score'])
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score'])
    df['Score'] = df['Score'].astype(int)
    df = df[df['Score'].between(1, 5)].copy()
    
    # Clean text
    df['Text'] = df['Text'].astype(str).str.strip()
    df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)
    df['Text'] = df['Text'].str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
    df['Text'] = df['Text'].str.replace(r'\S+@\S+', '', regex=True)
    df = df[df['Text'].str.len() >= 15].copy()
    
    # Create labels (1-5 -> 0-4)
    df['label'] = df['Score'] - 1
    
    print(f"After preprocessing: {len(df):,} reviews")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().sort_index())
    
    return df[['Text', 'label']].copy()


def predict_batch(texts, model, tokenizer, batch_size=32, max_length=128):
    """Predict ratings for a batch of texts"""
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def evaluate_model(model_path='./roberta_rating_model', data_file='Reviews.csv', test_size=10000):
    """Main evaluation function"""
    print("=" * 70)
    print("RoBERTa Model Evaluation")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load test data
    df = prepare_data(data_file, sample_size=test_size)
    
    # Get texts and labels
    texts = df['Text'].tolist()
    true_labels = df['label'].values
    
    print(f"\nEvaluating on {len(texts):,} samples...")
    print("Running predictions...")
    
    # Predict
    predicted_labels, probabilities = predict_batch(
        texts, model, tokenizer, 
        batch_size=64,  # Larger batch for faster evaluation
        max_length=128  # Match training max_length
    )
    
    # Convert predictions back to ratings (0-4 -> 1-5)
    predicted_ratings = predicted_labels + 1
    true_ratings = true_labels + 1
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Overall metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (W):   {precision_weighted:.4f}")
    print(f"  Precision (M):   {precision_macro:.4f}")
    print(f"  Recall (W):      {recall_weighted:.4f}")
    print(f"  Recall (M):      {recall_macro:.4f}")
    print(f"  F1 Score (W):   {f1_weighted:.4f}")
    print(f"  F1 Score (M):    {f1_macro:.4f}")
    
    # Per-class metrics
    precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
    
    print(f"\n📈 PER-RATING METRICS:")
    print(f"{'Rating':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i in range(5):
        rating = i + 1
        support = np.sum(true_labels == i)
        print(f"{rating:<8} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support:<10}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"\n📋 CONFUSION MATRIX:")
    print("     ", " ".join([f"Pred {i+1}" for i in range(5)]))
    for i in range(5):
        print(f"Act {i+1}  ", " ".join([f"{cm[i][j]:6d}" for j in range(5)]))
    
    # Classification Report
    print(f"\n📄 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(
        true_labels, 
        predicted_labels, 
        target_names=[f'Rating {i+1}' for i in range(5)],
        zero_division=0
    ))
    
    # Calculate accuracy per rating
    print(f"\n🎯 ACCURACY BY RATING:")
    for i in range(5):
        rating = i + 1
        mask = true_labels == i
        if np.sum(mask) > 0:
            rating_accuracy = accuracy_score(true_labels[mask], predicted_labels[mask])
            print(f"  Rating {rating}: {rating_accuracy:.4f} ({rating_accuracy*100:.2f}%) - {np.sum(mask)} samples")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'precision_macro': precision_macro,
        'recall_weighted': recall_weighted,
        'recall_macro': recall_macro,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = './roberta_rating_model'
    DATA_FILE = 'Reviews.csv'
    TEST_SIZE = 20000  # Number of samples to test on (None for all data)
    
    try:
        results = evaluate_model(
            model_path=MODEL_PATH,
            data_file=DATA_FILE,
            test_size=TEST_SIZE
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the model has been trained first!")
        print("Run: python train_roberta.py")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()



