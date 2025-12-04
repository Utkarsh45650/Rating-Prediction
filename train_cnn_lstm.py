"""
CNN/LSTM Model Training for Review Rating Prediction
Fast training with high accuracy using traditional deep learning
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Force GPU-only mode
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available! This script requires GPU for training.")

device = torch.device('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)

print("=" * 60)
print("GPU Configuration (GPU-ONLY MODE)")
print("=" * 60)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class ReviewDataset(Dataset):
    """Dataset class for reviews"""
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence of indices
        tokens = text.lower().split()
        sequence = [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens[:self.max_length]]
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence += [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class CNNTextClassifier(nn.Module):
    """CNN-based text classifier - Fast and accurate"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], num_classes=5, dropout=0.5):
        super(CNNTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_seq_length)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LSTMTextClassifier(nn.Module):
    """LSTM-based text classifier - Good for sequential patterns"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.5, bidirectional=True):
        super(LSTMTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        x = self.dropout(hidden)
        x = self.fc(x)
        return x


class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM model - Best of both worlds"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], 
                 lstm_hidden=128, num_classes=5, dropout=0.5):
        super(HybridCNNLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # LSTM layer
        self.lstm = nn.LSTM(
            len(filter_sizes) * num_filters,
            lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # CNN feature extraction
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_seq_length)
            conv_outputs.append(conv_out)
        
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters, seq_length)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, features)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Bidirectional
        
        x = self.dropout(hidden)
        x = self.fc(x)
        return x


def build_vocab(texts, min_freq=2):
    """Build vocabulary from texts"""
    word_counts = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def clean_text(text):
    """Clean text data"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()


def load_and_preprocess_data(file_path, sample_size=None, balance_classes=True):
    """Load and preprocess data"""
    print("Loading dataset...")
    print("=" * 60)
    
    chunk_size = 100000
    chunks = []
    total_read = 0
    
    try:
        print("Reading CSV file in chunks...")
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, usecols=['Text', 'Score']):
            chunks.append(chunk)
            total_read += len(chunk)
            if sample_size and total_read >= sample_size * 1.5:
                break
            if len(chunks) % 5 == 0:
                print(f"  Read {total_read:,} rows...")
    except Exception as e:
        print(f"Error reading file: {e}")
        df = pd.read_csv(file_path, low_memory=False, usecols=['Text', 'Score'])
        chunks = [df]
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"\nLoaded {len(df):,} total reviews")
    
    # Preprocess
    df = df.dropna(subset=['Text', 'Score'])
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df = df.dropna(subset=['Score'])
    df['Score'] = df['Score'].astype(int)
    df = df[df['Score'].between(1, 5)].copy()
    
    # Clean text
    df['Text'] = df['Text'].apply(clean_text)
    df = df[df['Text'].str.len() >= 15].copy()
    
    # Create labels
    df['label'] = df['Score'] - 1
    
    # Class balancing
    if balance_classes:
        print("\nBalancing classes...")
        label_counts = df['label'].value_counts()
        max_count = label_counts.max()
        target_count = int(max_count * 0.8)
        
        balanced_dfs = []
        for label in range(5):
            label_df = df[df['label'] == label].copy()
            current_count = len(label_df)
            if current_count < target_count:
                n_samples = target_count - current_count
                oversampled = label_df.sample(n=n_samples, replace=True, random_state=42)
                label_df = pd.concat([label_df, oversampled], ignore_index=True)
            balanced_dfs.append(label_df)
        
        df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal dataset size: {len(df):,} reviews")
    print("=" * 60)
    
    return df[['Text', 'label']].copy()


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in tqdm(dataloader, desc="Training"):
        texts = texts.to(device, non_blocking=True)  # Faster transfer
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # Update LR per step for OneCycleLR
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating"):
            texts = texts.to(device)
            labels = labels.to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def main():
    # Configuration - OPTIMIZED FOR MAXIMUM SPEED AND ACCURACY
    MODEL_TYPE = 'cnn'  # Options: 'cnn' (fastest), 'lstm' (good accuracy), 'hybrid' (best accuracy)
    MAX_LENGTH = 250  # Shorter for speed (was 200)
    BATCH_SIZE = 96  # Very large batch for maximum speed (was 128)
    LEARNING_RATE = 0.0008  # Higher LR for faster convergence (was 0.001)
    NUM_EPOCHS = 20  # Optimized epochs (was 10)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    DROPOUT = 0.3  # Lower dropout for faster learning (was 0.3)
    SAMPLE_SIZE = 300000  # More data for better accuracy (was 150000)
    DATA_FILE = 'Reviews.csv'
    OUTPUT_DIR = './cnn_lstm_rating_model'
    
    print("=" * 60)
    print(f"CNN/LSTM Model Training - {MODEL_TYPE.upper()}")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Model type: {MODEL_TYPE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Max length: {MAX_LENGTH}")
    print(f"  - Sample size: {SAMPLE_SIZE:,}")
    print("=" * 60)
    
    # Load data
    df = load_and_preprocess_data(DATA_FILE, sample_size=SAMPLE_SIZE, balance_classes=True)
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    all_texts = train_df['Text'].tolist()
    vocab = build_vocab(all_texts, min_freq=2)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Create datasets
    train_dataset = ReviewDataset(
        train_df['Text'].tolist(),
        train_df['label'].tolist(),
        vocab,
        max_length=MAX_LENGTH
    )
    val_dataset = ReviewDataset(
        val_df['Text'].tolist(),
        val_df['label'].tolist(),
        vocab,
        max_length=MAX_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating {MODEL_TYPE.upper()} model...")
    if MODEL_TYPE == 'cnn':
        model = CNNTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            num_classes=5,
            dropout=DROPOUT
        )
    elif MODEL_TYPE == 'lstm':
        model = LSTMTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            num_classes=5,
            dropout=DROPOUT,
            bidirectional=True
        )
    else:  # hybrid
        model = HybridCNNLSTM(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            lstm_hidden=HIDDEN_DIM,
            num_classes=5,
            dropout=DROPOUT
        )
    
    model = model.to(device)
    print(f"Model moved to {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate class weights
    labels = train_df['label'].values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Loss and optimizer - OPTIMIZED FOR SPEED
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE * 2,  # Peak LR
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        
        # Evaluate (less frequently for speed)
        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:  # Evaluate every 2 epochs
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_state = model.state_dict().copy()
                print(f"✓ New best model! Accuracy: {best_accuracy*100:.2f}%")
        
        if (epoch + 1) % 2 == 0:
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    val_loss, val_acc, val_prec, val_rec, val_f1, all_preds, all_labels = evaluate(
        model, val_loader, criterion, device
    )
    
    print(f"\nFinal Results:")
    print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall: {val_rec:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Rating Performance:")
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    for i in range(5):
        rating = i + 1
        print(f"  Rating {rating}: Precision={precision_per_class[i]:.4f}, "
              f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print("     ", " ".join([f"Pred {i+1}" for i in range(5)]))
    for i in range(5):
        print(f"Act {i+1}  ", " ".join([f"{cm[i][j]:6d}" for j in range(5)]))
    
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'max_length': MAX_LENGTH,
        'model_type': MODEL_TYPE,
        'accuracy': val_acc
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()

