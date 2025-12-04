"""
Evaluate RNN Rating Prediction Model
Load trained RNN model and evaluate on test data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class TextDataset(Dataset):
    """Custom Dataset for text classification"""
    def __init__(self, texts, labels, word_to_idx, max_length=150):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                   for word in text.split()[:self.max_length]]
        
        # Pad sequence
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class ImprovedRNNClassifier(nn.Module):
    """Enhanced RNN-based rating classifier with GRU and multi-layer head"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=3, 
                 num_classes=5, dropout=0.4, bidirectional=True):
        super(ImprovedRNNClassifier, self).__init__()
        
        # Enhanced embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Use GRU instead of vanilla RNN for better gradient flow and accuracy
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Multi-layer classification head for better feature learning
        self.batch_norm1 = nn.BatchNorm1d(rnn_output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_output_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # GRU processing
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state from both directions
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Multi-layer classification
        x = self.batch_norm1(hidden)
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def preprocess_text(text):
    """Enhanced text preprocessing for sentiment/rating analysis"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    
    # Preserve important sentiment indicators
    text = text.replace("n't", " not")
    text = text.replace("'m", " am")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def evaluate_model(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['1★', '2★', '3★', '4★', '5★'],
                yticklabels=['1★', '2★', '3★', '4★', '5★'])
    plt.title('Confusion Matrix - RNN Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history_path='./rnn_rating_model/training_history.csv'):
    """Plot training history"""
    try:
        history = pd.read_csv(history_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(history['epoch'], history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['epoch'], history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(history['epoch'], history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./rnn_rating_model/training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to ./rnn_rating_model/training_history.png")
        plt.close()
    except Exception as e:
        print(f"Could not plot training history: {e}")


def main():
    # Configuration
    MODEL_PATH = './rnn_rating_model/model.pt'
    DATA_FILE = 'Reviews.csv'
    BATCH_SIZE = 256
    SAMPLE_SIZE = 200000
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    config = checkpoint['config']
    word_to_idx = checkpoint['word_to_idx']
    
    print(f"Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create improved model
    model = ImprovedRNNClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=5,
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training accuracy from checkpoint: {checkpoint['accuracy']:.4f}")
    
    # Load data
    print("\nLoading test data...")
    df = pd.read_csv(DATA_FILE)
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    # Preprocess
    df['clean_text'] = df['Text'].apply(preprocess_text)
    df = df[df['clean_text'].str.len() > 0]
    df['Rating'] = df['Score'] - 1
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'].values, 
        df['Rating'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['Rating'].values
    )
    
    print(f"Test samples: {len(X_test)}")
    
    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test, y_test, word_to_idx, config['max_length'])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS - RNN MODEL")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    # Per-class accuracy
    print("\n" + "-"*60)
    print("Per-Class Accuracy:")
    print("-"*60)
    cm = confusion_matrix(y_true, y_pred)
    for i in range(5):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{i+1} Star: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    # Classification report
    print("\n" + "-"*60)
    print("Detailed Classification Report:")
    print("-"*60)
    print(classification_report(y_true, y_pred, 
                                target_names=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
                                digits=4))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, './rnn_rating_model/confusion_matrix.png')
    
    # Plot training history
    plot_training_history()
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        text = X_test[idx][:100] + "..." if len(X_test[idx]) > 100 else X_test[idx]
        true_rating = y_true[idx] + 1
        pred_rating = y_pred[idx] + 1
        confidence = y_probs[idx][y_pred[idx]]
        
        print(f"\nText: {text}")
        print(f"True Rating: {true_rating}★ | Predicted: {pred_rating}★ | Confidence: {confidence:.4f}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
