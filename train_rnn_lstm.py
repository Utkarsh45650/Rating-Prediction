"""
RNN-LSTM Rating Prediction Model - Optimized for GPU Training
Fast and accurate LSTM model for Amazon review rating prediction
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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class TextDataset(Dataset):
    """Dataset for text classification"""
    def __init__(self, texts, labels, word_to_idx, max_length=200):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                   for word in text.split()[:self.max_length]]
        
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class RNN_LSTM_Model(nn.Module):
    """Enhanced RNN-LSTM for maximum accuracy"""
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300, num_layers=4, 
                 num_classes=5, dropout=0.3, bidirectional=True):
        super(RNN_LSTM_Model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Deep Bidirectional LSTM with layer normalization
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Enhanced attention with layer norm
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Deeper classification head for better feature transformation
        self.batch_norm1 = nn.BatchNorm1d(lstm_output_dim * 2)  # attention + hidden
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim * 2, hidden_dim * 3)
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim * 3)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.fc2 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(dropout * 0.3)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attention_context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Get last hidden state from both directions
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Combine attention and hidden state
        combined = torch.cat([attention_context, last_hidden], dim=1)
        
        # Deep classification with skip connections
        x = self.batch_norm1(combined)
        x = self.dropout1(x)
        x = F.gelu(self.fc1(x))  # GELU for better performance
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.gelu(self.fc2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.gelu(self.fc3(x))
        
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.fc4(x)
        
        return x


def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    
    text = text.replace("n't", " not")
    text = text.replace("'m", " am")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def build_vocab(texts, min_freq=2, max_vocab=40000):
    """Build vocabulary"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in sorted_words:
        if freq >= min_freq and len(word_to_idx) < max_vocab:
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, scheduler):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
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
    # Configuration - OPTIMIZED FOR MAXIMUM ACCURACY
    MAX_LENGTH = 250  # Longer sequences for better context
    BATCH_SIZE = 96  # Smaller batches for better generalization
    LEARNING_RATE = 0.0008  # Lower LR for more stable convergence
    NUM_EPOCHS = 20  # More epochs with early stopping
    EMBEDDING_DIM = 300  # Larger embeddings for richer representations
    HIDDEN_DIM = 300  # Larger hidden states
    NUM_LAYERS = 4  # Deeper LSTM for hierarchical features
    DROPOUT = 0.3  # Optimized dropout
    BIDIRECTIONAL = True
    SAMPLE_SIZE = 300000  # More training data
    DATA_FILE = 'Reviews.csv'
    OUTPUT_DIR = './rnn_lstm_rating_model'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Require GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available! This script requires CUDA.")
    
    device = torch.device('cuda')
    print(f"\nUsing device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Total reviews: {len(df)}")
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Using {len(df)} reviews")
    
    # Preprocess
    print("\nPreprocessing...")
    df['clean_text'] = df['Text'].apply(preprocess_text)
    df = df[df['clean_text'].str.len() > 0]
    df['Rating'] = df['Score'] - 1
    
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
    
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = TextDataset(X_train, y_train, word_to_idx, MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, word_to_idx, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    # Create model
    print(f"\nCreating RNN-LSTM Model...")
    model = RNN_LSTM_Model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=5,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with class weighting
    class_counts = np.bincount(df['Rating'].values)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights:")
    for i, w in enumerate(class_weights):
        print(f"  {i+1} stars: {w:.4f} (n={class_counts[i]})")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE*2, epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Training
    print("\n" + "="*50)
    print("Training RNN-LSTM Model")
    print("="*50)
    
    best_accuracy = 0
    patience = 5
    patience_counter = 0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, scheduler)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Time: {elapsed:.1f}s | LR: {lr:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': lr
        })
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            print(f"✓ New best accuracy! Saving...")
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
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    checkpoint = torch.load(f'{OUTPUT_DIR}/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['1-star', '2-star', '3-star', '4-star', '5-star'], digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save history
    pd.DataFrame(history).to_csv(f'{OUTPUT_DIR}/training_history.csv', index=False)
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Model: {OUTPUT_DIR}/model.pt")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
