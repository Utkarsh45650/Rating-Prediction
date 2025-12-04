# Customer Feedback Rating Predictor

A full-stack Machine Learning application that predicts customer satisfaction ratings (1-5 stars) from text feedback using multiple trained ML models.

## 🚀 Features

- **4 ML Models**: CNN-LSTM, RNN, RNN-LSTM, and RoBERTa
- **Real-time Predictions**: Instant rating predictions with confidence scores
- **Interactive UI**: Modern React frontend with pure CSS styling
- **Prediction History**: Tracks last 5 predictions in browser
- **Model Comparison**: Switch between different models to compare predictions
- **Responsive Design**: Works on mobile and desktop devices

## 🛠️ Technology Stack

### Frontend
- React 18
- React Router for navigation
- Axios for API calls
- Pure CSS (no frameworks)
- Vite for build tooling

### Backend
- Flask (Python)
- PyTorch for ML models
- Transformers (Hugging Face) for RoBERTa
- Flask-CORS for API access

### ML Models
- **CNN-LSTM**: Convolutional + LSTM architecture
- **RNN**: Recurrent network with attention mechanism
- **RNN-LSTM**: 4-layer bidirectional LSTM with enhanced attention
- **RoBERTa**: Transformer-based pre-trained model

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (optional, but recommended)

## ⚙️ Installation

### 1. Backend Setup

```powershell
# Navigate to project directory
cd C:\Users\utkar\Desktop\Study\Projects\ML_Project

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend_requirements.txt

# Start Flask server
python app.py
```

The API will run on `http://localhost:5000`

### 2. Frontend Setup

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will run on `http://localhost:3000`

## 📁 Project Structure

```
ML_Project/
├── app.py                          # Flask API server
├── backend_requirements.txt        # Backend dependencies
├── Reviews.csv                     # Training dataset
├── train_rnn.py                    # RNN training script
├── train_rnn_lstm.py              # RNN-LSTM training script
├── train_cnn_lstm.py              # CNN-LSTM training script
├── train_roberta.py               # RoBERTa training script
├── evaluate_*.py                   # Model evaluation scripts
├── cnn_lstm_rating_model/         # CNN-LSTM model files
│   └── model.pt
├── rnn_rating_model/              # RNN model files
│   └── model.pt
├── rnn_lstm_rating_model/         # RNN-LSTM model files
│   └── model.pt
├── roberta_rating_model/          # RoBERTa model files
│   └── config.json, model.safetensors, etc.
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.jsx                 # Main app component
        ├── main.jsx               # Entry point
        ├── index.css              # Global styles
        ├── components/            # Reusable components
        │   ├── Navbar.jsx
        │   ├── ModelSelector.jsx
        │   ├── FeedbackInput.jsx
        │   ├── ResultCard.jsx
        │   ├── StarRating.jsx
        │   ├── HistoryCard.jsx
        │   └── Loader.jsx
        ├── pages/                 # Page components
        │   ├── Home.jsx
        │   ├── Predict.jsx
        │   └── About.jsx
        └── services/              # API integration
            └── api.js
```

## 🔌 API Endpoints

### POST /predict
Predict rating from feedback text

**Request:**
```json
{
  "text": "This product is amazing! Highly recommend it.",
  "model": "rnn-lstm"
}
```

**Response:**
```json
{
  "rating": 5,
  "confidence": 0.9234
}
```

### GET /models
List available models

**Response:**
```json
{
  "models": [
    {"name": "cnn-lstm", "available": true, "path": "./cnn_lstm_rating_model/model.pt"},
    {"name": "rnn", "available": true, "path": "./rnn_rating_model/model.pt"},
    {"name": "rnn-lstm", "available": true, "path": "./rnn_lstm_rating_model/model.pt"},
    {"name": "roberta", "available": true, "path": "./roberta_rating_model"}
  ]
}
```

### GET /health
Health check

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": ["rnn-lstm"]
}
```

## 🎯 Usage

1. **Start the Backend**:
   ```powershell
   python app.py
   ```

2. **Start the Frontend**:
   ```powershell
   cd frontend
   npm run dev
   ```

3. **Open Browser**: Navigate to `http://localhost:3000`

4. **Make Predictions**:
   - Go to the "Predict" page
   - Select an ML model from the dropdown
   - Enter customer feedback text
   - Click "Predict Rating"
   - View the predicted rating and confidence score

5. **View History**: Recent predictions are automatically saved and displayed below the prediction form

## 🧪 Model Training

Each model has its own training script:

```powershell
# Train RNN model
python train_rnn.py

# Train RNN-LSTM model
python train_rnn_lstm.py

# Train CNN-LSTM model
python train_cnn_lstm.py

# Train RoBERTa model
python train_roberta.py
```

## 📊 Model Performance

Models are trained on 300,000+ Amazon product reviews:

- **RNN**: ~79.65% accuracy
- **RNN-LSTM**: Optimized for maximum accuracy (4-layer deep architecture)
- **CNN-LSTM**: Fast inference with good accuracy
- **RoBERTa**: State-of-the-art transformer performance

## 🎨 UI Features

- **Modern gradient backgrounds**
- **Smooth animations and transitions**
- **Star rating visualization**
- **Confidence score progress bar**
- **Prediction history cards**
- **Responsive mobile design**
- **Clean, minimalist interface**

## ⚠️ Input Validation

The frontend validates that:
- Text is not empty
- Text has at least 10 characters (excluding URLs)
- Text contains meaningful feedback (not just links or names)

## 🔒 CORS Configuration

The Flask API is configured with CORS to allow requests from the React frontend running on a different port.

## 🚀 Production Build

To build the frontend for production:

```powershell
cd frontend
npm run build
```

This creates an optimized build in the `frontend/dist` directory.

## 💡 Tips for Best Results

1. Provide clear, descriptive feedback (10+ words)
2. Include specific details about products/services
3. Express opinions naturally as in real reviews
4. Avoid only URLs, names, or very short phrases
5. Try different models to compare predictions

## 🐛 Troubleshooting

**Backend won't start:**
- Ensure all model files exist in their directories
- Check Python dependencies are installed
- Verify virtual environment is activated

**Frontend won't connect to API:**
- Ensure Flask server is running on port 5000
- Check browser console for CORS errors
- Verify proxy settings in `vite.config.js`

**Model not found:**
- Ensure models are trained and saved
- Check model paths in `app.py` match your directory structure

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Built as a Machine Learning demonstration project showcasing multiple deep learning architectures for sentiment analysis and rating prediction.
