"""
Flask API for Customer Feedback Rating Prediction
Serves predictions from trained ML models: CNN-LSTM, RNN, RNN-LSTM, RoBERTa
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
import os

# Try to import transformers for RoBERTa (optional)
try:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model paths (relative to project root)
MODEL_PATHS = {
    'cnn-lstm': '../cnn_lstm_rating_model/model.pt',
    'rnn': '../rnn_rating_model/model.pt',
    'rnn-lstm': '../rnn_lstm_rating_model/model.pt',
    'roberta': '../roberta_rating_model'
}

# Global model cache
models_cache = {}


# ============ Model Architectures ============

class ImprovedRNNClassifier(nn.Module):
    """RNN Model Architecture"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=3,
                 num_classes=5, dropout=0.35, bidirectional=True):
        super(ImprovedRNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.15)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(lstm_output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim * 2)
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        x = self.batch_norm1(context)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x


class RNN_LSTM_Model(nn.Module):
    """RNN-LSTM Model Architecture"""
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300, num_layers=4,
                 num_classes=5, dropout=0.3, bidirectional=True):
        super(RNN_LSTM_Model, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.1)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.bidirectional = bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(lstm_output_dim * 2)
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
        
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attention_context = torch.sum(attention_weights * lstm_out, dim=1)
        
        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]
        
        combined = torch.cat([attention_context, last_hidden], dim=1)
        
        x = self.batch_norm1(combined)
        x = self.dropout1(x)
        x = F.gelu(self.fc1(x))
        
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


class CNNTextClassifier(nn.Module):
    """CNN-based text classifier (actual CNN-LSTM model architecture)"""
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


# ============ Text Preprocessing ============

def preprocess_text(text):
    """Enhanced text preprocessing"""
    text = text.lower()
    
    # Preserve negations
    text = text.replace("n't", " not")
    text = text.replace("'m", " am")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def text_to_indices(text, word_to_idx, max_length=200):
    """Convert text to indices"""
    words = text.split()[:max_length]
    indices = [word_to_idx.get(word, word_to_idx.get('<UNK>', 1)) for word in words]
    
    # Pad
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    
    return indices


# ============ Model Loading ============

def load_model(model_name):
    """Load model from cache or disk"""
    if model_name in models_cache:
        return models_cache[model_name]
    
    model_path = MODEL_PATHS.get(model_name)
    
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")
    
    try:
        if model_name == 'roberta':
            if not TRANSFORMERS_AVAILABLE:
                raise Exception("RoBERTa model requires transformers library. Please install it.")
            # Load RoBERTa
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            model.to(device)
            model.eval()
            models_cache[model_name] = {'model': model, 'tokenizer': tokenizer}
            
        else:
            # Load PyTorch models
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if model_name == 'cnn-lstm':
                # CNN-LSTM uses different key names
                word_to_idx = checkpoint.get('vocab', {})
                vocab_size = len(word_to_idx)
                max_length = checkpoint.get('max_length', 200)
                
                model = CNNTextClassifier(
                    vocab_size=vocab_size,
                    embedding_dim=128,
                    num_filters=100,
                    filter_sizes=[3, 4, 5],
                    num_classes=5,
                    dropout=0.5
                )
            else:
                # RNN and RNN-LSTM use 'config' and 'word_to_idx'
                config = checkpoint.get('config', {})
                word_to_idx = checkpoint.get('word_to_idx', {})
                max_length = config.get('max_length', 200)
                
                if model_name == 'rnn':
                    model = ImprovedRNNClassifier(
                        vocab_size=config['vocab_size'],
                        embedding_dim=config.get('embedding_dim', 256),
                        hidden_dim=config.get('hidden_dim', 256),
                        num_layers=config.get('num_layers', 3),
                        dropout=config.get('dropout', 0.35),
                        bidirectional=config.get('bidirectional', True)
                    )
                elif model_name == 'rnn-lstm':
                    model = RNN_LSTM_Model(
                        vocab_size=config['vocab_size'],
                        embedding_dim=config.get('embedding_dim', 300),
                        hidden_dim=config.get('hidden_dim', 300),
                        num_layers=config.get('num_layers', 4),
                        dropout=config.get('dropout', 0.3),
                        bidirectional=config.get('bidirectional', True)
                    )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            models_cache[model_name] = {
                'model': model,
                'word_to_idx': word_to_idx,
                'max_length': max_length
            }
        
        print(f"Loaded {model_name} model successfully")
        return models_cache[model_name]
        
    except Exception as e:
        raise Exception(f"Error loading {model_name} model: {str(e)}")


# ============ Prediction ============

def predict_rating(text, model_name):
    """Predict rating for given text"""
    # Validate input
    if not text or len(text.strip()) < 5:
        raise ValueError("Text is too short. Please provide meaningful feedback.")
    
    # Load model
    model_data = load_model(model_name)
    
    if model_name == 'roberta':
        # RoBERTa prediction
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            rating = predicted.item() + 1
            
    else:
        # PyTorch models prediction
        model = model_data['model']
        word_to_idx = model_data['word_to_idx']
        max_length = model_data['max_length']
        
        # Preprocess
        clean_text = preprocess_text(text)
        indices = text_to_indices(clean_text, word_to_idx, max_length)
        
        # Convert to tensor
        input_tensor = torch.LongTensor([indices]).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            rating = predicted.item() + 1
    
    return {
        'rating': int(rating),
        'confidence': float(confidence.item())
    }


# ============ API Endpoints ============

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        model = data.get('model', 'rnn-lstm').lower()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if model not in MODEL_PATHS:
            return jsonify({'error': f'Invalid model. Choose from: {list(MODEL_PATHS.keys())}'}), 400
        
        # Predict
        result = predict_rating(text, model)
        
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    available_models = []
    
    for model_name, path in MODEL_PATHS.items():
        exists = os.path.exists(path)
        # Check if RoBERTa is actually usable
        if model_name == 'roberta' and not TRANSFORMERS_AVAILABLE:
            exists = False
        available_models.append({
            'name': model_name,
            'available': exists,
            'path': path
        })
    
    return jsonify({'models': available_models}), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': list(models_cache.keys())
    }), 200


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Customer Feedback Rating Predictor API',
        'endpoints': {
            'POST /predict': 'Predict rating from text',
            'GET /models': 'List available models',
            'GET /health': 'Health check'
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Customer Feedback Rating Predictor API")
    print("="*50)
    print(f"Device: {device}")
    print(f"Available models: {list(MODEL_PATHS.keys())}")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
