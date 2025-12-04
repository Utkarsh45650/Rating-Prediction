# 🚀 Quick Setup Guide

## ✅ Prerequisites Checklist

- [x] Python 3.8+ installed
- [x] Node.js 16+ installed
- [x] Virtual environment created at `./venv`
- [x] Trained models available

## 📦 Installation Complete

### Backend Dependencies
- Flask 3.1.2
- Flask-CORS 6.0.1
- PyTorch (already installed)
- Transformers (already installed)

### Frontend Dependencies
- React 18.2.0
- React Router DOM 6.20.0
- Axios 1.6.2
- Vite 5.0.8

## 🎯 Running the Application

### Option 1: Quick Start (Automated)

```powershell
# Run the start script
.\start.ps1
```

This will:
1. Start the backend server (http://localhost:5000)
2. Start the frontend dev server (http://localhost:3000)
3. Open your browser automatically

### Option 2: Manual Start

**Terminal 1 - Backend:**
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Flask server
python app.py
```

**Terminal 2 - Frontend:**
```powershell
# Navigate to frontend
cd frontend

# Start dev server
npm run dev
```

## 🧪 Testing the Application

### 1. Test Backend API

```powershell
# Test health endpoint
curl http://localhost:5000/health

# Test models endpoint
curl http://localhost:5000/models

# Test prediction endpoint
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"text": "This product is amazing!", "model": "rnn-lstm"}'
```

### 2. Test Frontend

1. Open browser to `http://localhost:3000`
2. Click "Start Predicting" button
3. Select a model from dropdown
4. Enter feedback text (e.g., "Great product, highly recommend!")
5. Click "Predict Rating"
6. View the predicted rating and confidence score

## 📱 Available Models

| Model | Status | Accuracy | Description |
|-------|--------|----------|-------------|
| **RNN-LSTM** | ✅ Ready | Highest | 4-layer deep bidirectional architecture |
| **RNN** | ✅ Ready | ~79.65% | LSTM with attention mechanism |
| **CNN-LSTM** | ✅ Ready | Good | Convolutional + LSTM hybrid |
| **RoBERTa** | ✅ Ready | SOTA | Transformer-based model |

## 🎨 Features to Test

### Homepage (`/`)
- Modern gradient design
- Feature cards
- "Start Predicting" button navigation

### Predict Page (`/predict`)
- Model selector dropdown
- Text input with character counter
- Real-time prediction
- Confidence score visualization
- Star rating display
- Prediction history (last 5)
- Input validation

### About Page (`/about`)
- Project information
- Model descriptions
- Use cases
- Technology stack

## 🔧 Troubleshooting

### Backend Issues

**Problem: ModuleNotFoundError for flask or flask_cors**
```powershell
.\venv\Scripts\Activate.ps1
pip install flask flask-cors
```

**Problem: Model not found**
```
Error: Model 'rnn-lstm' not found
```
Solution: Ensure model is trained. Run:
```powershell
python train_rnn_lstm.py
```

**Problem: CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
Solution: The API will automatically fall back to CPU if needed.

### Frontend Issues

**Problem: Cannot connect to backend**
- Ensure Flask server is running on port 5000
- Check browser console for CORS errors
- Verify `vite.config.js` proxy settings

**Problem: npm dependencies not installed**
```powershell
cd frontend
npm install
```

## 📊 Example Predictions

### Test Cases

**Very Positive (5 stars):**
```
"This product exceeded my expectations! Amazing quality and fast shipping. Highly recommend to everyone!"
```

**Positive (4 stars):**
```
"Good product overall. Works as described. Minor issues but nothing major."
```

**Neutral (3 stars):**
```
"It's okay. Does the job but nothing special. Average quality."
```

**Negative (2 stars):**
```
"Disappointed with the quality. Not as described. Had some issues."
```

**Very Negative (1 star):**
```
"Terrible product. Complete waste of money. Do not buy!"
```

## 🎯 Key Features

### Input Validation
- Minimum 10 characters (excluding URLs)
- Blocks submissions with only links or names
- Character counter
- Real-time validation feedback

### Prediction Display
- Animated star rating (1-5)
- Confidence score with color-coded progress bar
  - Green: >80% confidence
  - Orange: 60-80% confidence
  - Red: <60% confidence
- Feedback preview (first 150 characters)
- Model badge showing which model was used

### History Management
- Stores last 5 predictions
- Persists in browser localStorage
- Click to reload previous predictions
- Shows model, rating, confidence, and timestamp

## 🚀 Production Deployment

### Build Frontend
```powershell
cd frontend
npm run build
```

Output will be in `frontend/dist/`

### Serve with Production Server

Instead of Flask development server, use:
```powershell
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

For Windows:
```powershell
pip install waitress
waitress-serve --port=5000 app:app
```

## 📈 Performance Tips

1. **Model Loading**: Models are cached after first use
2. **GPU Acceleration**: Automatically uses CUDA if available
3. **Mixed Precision**: RNN-LSTM uses mixed precision for speed
4. **Text Preprocessing**: Optimized regex patterns
5. **Frontend Caching**: History stored in localStorage

## 🎓 Learning Resources

- **PyTorch**: https://pytorch.org/tutorials/
- **React**: https://react.dev/learn
- **Flask**: https://flask.palletsprojects.com/
- **Transformers**: https://huggingface.co/docs/transformers

## 📝 Notes

- Backend runs on port **5000**
- Frontend dev server on port **3000**
- CORS is enabled for cross-origin requests
- All models return ratings 1-5 with confidence 0-1

## ✨ Next Steps

1. Start both servers
2. Test each model with different inputs
3. Compare prediction accuracy across models
4. Review prediction history
5. Explore the About page for detailed model information

---

**Happy Predicting! 🎉**
