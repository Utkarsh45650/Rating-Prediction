"""
RNN Rating Prediction Model - Optimized for GPU Training
Train a vanilla RNN model for Amazon review rating prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import os
from tqdm import tqdm
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    """Enhanced RNN-based rating classifier with LSTM, attention, and multi-layer head"""
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300, num_layers=3, 
                 num_classes=5, dropout=0.35, bidirectional=True):
        super(ImprovedRNNClassifier, self).__init__()
        
        # Enhanced embedding layer with larger dimension
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.15)
        
        # Use LSTM instead of GRU for better long-term dependencies
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Simple attention mechanism for better focus on important words
        self.attention = nn.Linear(rnn_output_dim, 1)
        
        # Deeper classification head with residual connection
        self.batch_norm1 = nn.BatchNorm1d(rnn_output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_output_dim, hidden_dim * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = self.embedding_dropout(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attention_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Also use last hidden state
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Combine attention output and last hidden state
        combined = attention_output + last_hidden
        
        # Multi-layer classification with residual connections
        x = self.batch_norm1(combined)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        
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


def build_vocab(texts, min_freq=2, max_vocab=40000):
    """Build vocabulary from texts with larger vocab for better coverage"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and limit vocabulary size
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in sorted_words:
        if freq >= min_freq and len(word_to_idx) < max_vocab:
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, scheduler=None):
    """Train for one epoch with mixed precision support and gradient clipping"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training on GPU
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update scheduler after each batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    # Configuration - HIGHLY OPTIMIZED FOR REVIEW RATING PREDICTION (1-5 STARS)
    MAX_LENGTH = 250  # Even longer context for nuanced reviews
    BATCH_SIZE = 96  # Smaller batches for better generalization
    LEARNING_RATE = 0.0008  # Fine-tuned learning rate
    NUM_EPOCHS = 20  # More epochs for convergence with early stopping
    EMBEDDING_DIM = 300  # Larger embeddings for richer representations
    HIDDEN_DIM = 300  # Increased hidden dimension
    NUM_LAYERS = 3  # 3-layer LSTM for hierarchical features
    DROPOUT = 0.35  # Optimized dropout rate
    BIDIRECTIONAL = True  # Capture context from both directions
    SAMPLE_SIZE = 250000  # More training data
    DATA_FILE = 'Reviews.csv'
    OUTPUT_DIR = './rnn_rating_model'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for GPU - REQUIRE GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available! This script requires CUDA-enabled GPU.")
    
    device = torch.device('cuda')
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Total reviews in dataset: {len(df)}")
    
    # Sample data
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Using {len(df)} reviews for training")
    
    # Prepare data
    print("\nPreprocessing texts...")
    df['clean_text'] = df['Text'].apply(preprocess_text)
    df = df[df['clean_text'].str.len() > 0]  # Remove empty texts
    df['Rating'] = df['Score'] - 1  # Convert ratings 1-5 to 0-4
    
    print(f"\nRating distribution:")
    print(df['Rating'].value_counts().sort_index())
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    word_to_idx = build_vocab(df['clean_text'].values)
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'].values, 
        df['Rating'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['Rating'].values
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create datasets and dataloaders
    print("\nCreating data loaders...")
    train_dataset = TextDataset(X_train, y_train, word_to_idx, MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, word_to_idx, MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True  # GPU memory optimization
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True  # GPU memory optimization
    )
    
    # Create improved model
    print(f"\nCreating Enhanced GRU Rating Prediction Model...")
    model = ImprovedRNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=5,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer with class weighting for imbalanced data
    # Calculate class weights based on distribution
    class_counts = np.bincount(df['Rating'].values)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights for imbalanced data:")
    for i, weight in enumerate(class_weights):
        print(f"  Rating {i+1}: {weight:.4f} (count: {class_counts[i]})")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02, betas=(0.9, 0.999))
    
    # Learning rate scheduler with warmup and cosine decay
    warmup_epochs = 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE * 2,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/NUM_EPOCHS,
        anneal_strategy='cos'
    )
    
    # Mixed precision training scaler (GPU required)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    best_accuracy = 0
    patience = 5
    patience_counter = 0
    training_history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train (scheduler updates per-batch for OneCycleLR)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, scheduler)
        
        # Evaluate
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
        
        # Save best model with early stopping
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            print(f"New best accuracy! Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'word_to_idx': word_to_idx,
                'config': {
                    'vocab_size': vocab_size,
                    'embedding_dim': EMBEDDING_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'bidirectional': BIDIRECTIONAL,
                    'max_length': MAX_LENGTH
                }
            }, f'{OUTPUT_DIR}/model.pt')
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(f'{OUTPUT_DIR}/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f'{OUTPUT_DIR}/training_history.csv', index=False)
    print(f"\nTraining history saved to {OUTPUT_DIR}/training_history.csv")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Model saved to: {OUTPUT_DIR}/model.pt")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
