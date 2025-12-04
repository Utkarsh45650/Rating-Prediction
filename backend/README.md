# Backend API

Flask API server for Customer Feedback Rating Predictor.

## 📁 Structure

```
backend/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── start.ps1          # Quick start script
└── README.md          # This file
```

## 🚀 Quick Start

### Option 1: Using Start Script
```powershell
cd backend
.\start.ps1
```

### Option 2: Manual Start
```powershell
cd backend
..\venv\Scripts\Activate.ps1
python app.py
```

Server will run on: **http://localhost:5000**

## 📡 API Endpoints

### 1. POST /predict
Predict rating from feedback text.

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

**Parameters:**
- `text` (string, required): Customer feedback text (min 5 characters)
- `model` (string, required): Model name - `cnn-lstm`, `rnn`, `rnn-lstm`, or `roberta`

### 2. GET /models
List all available models.

**Response:**
```json
{
  "models": [
    {
      "name": "cnn-lstm",
      "available": true,
      "path": "../cnn_lstm_rating_model/model.pt"
    },
    {
      "name": "rnn",
      "available": true,
      "path": "../rnn_rating_model/model.pt"
    },
    {
      "name": "rnn-lstm",
      "available": true,
      "path": "../rnn_lstm_rating_model/model.pt"
    },
    {
      "name": "roberta",
      "available": false,
      "path": "../roberta_rating_model"
    }
  ]
}
```

### 3. GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": ["rnn-lstm"]
}
```

### 4. GET /
API information.

**Response:**
```json
{
  "message": "Customer Feedback Rating Predictor API",
  "endpoints": {
    "POST /predict": "Predict rating from text",
    "GET /models": "List available models",
    "GET /health": "Health check"
  }
}
```

## 🔧 Configuration

### Model Paths
Models are located in the parent directory:
- `../cnn_lstm_rating_model/model.pt`
- `../rnn_rating_model/model.pt`
- `../rnn_lstm_rating_model/model.pt`
- `../roberta_rating_model/`

### CORS
CORS is enabled for all origins to allow frontend communication.

### Device
Automatically uses CUDA if available, otherwise falls back to CPU.

## 🧪 Testing

### Using curl
```powershell
# Health check
curl http://localhost:5000/health

# List models
curl http://localhost:5000/models

# Make prediction
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Great product!\", \"model\": \"rnn-lstm\"}'
```

### Using Python
```python
import requests

# Make prediction
response = requests.post('http://localhost:5000/predict', json={
    'text': 'This product exceeded my expectations!',
    'model': 'rnn-lstm'
})

print(response.json())
# Output: {'rating': 5, 'confidence': 0.92}
```

## 🎯 Model Architectures

### CNN-LSTM Model
- Convolutional layers for feature extraction
- LSTM for sequence processing
- ~10M parameters

### RNN Model
- 3-layer bidirectional LSTM
- Attention mechanism
- ~15M parameters
- Accuracy: ~79.65%

### RNN-LSTM Model
- 4-layer deep bidirectional LSTM
- Enhanced attention + hidden states
- ~22M parameters
- Optimized for maximum accuracy

### RoBERTa Model (Optional)
- Transformer-based pretrained model
- ~125M parameters
- Requires transformers library

## 📊 Response Format

### Success Response
```json
{
  "rating": 4,
  "confidence": 0.8567
}
```

### Error Response
```json
{
  "error": "Text is too short. Please provide meaningful feedback."
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad request (validation error)
- `404` - Model not found
- `500` - Server error

## 🔒 Input Validation

- Text must be at least 5 characters
- Model name must be one of: `cnn-lstm`, `rnn`, `rnn-lstm`, `roberta`
- JSON payload required for POST requests

## 🚨 Error Handling

The API handles:
- Invalid input validation
- Model loading errors
- Prediction failures
- Missing model files
- CUDA/CPU compatibility

## 🔄 Model Caching

Models are cached in memory after first load for better performance:
- First prediction: ~2-5 seconds (model loading)
- Subsequent predictions: ~0.1-0.5 seconds

## 📈 Performance

### GPU (CUDA)
- Prediction time: 0.1-0.3s per request
- Concurrent requests: Up to 10

### CPU
- Prediction time: 0.5-2s per request
- Concurrent requests: Up to 5

## 🛠️ Dependencies

- Flask 3.1.2 - Web framework
- Flask-CORS 6.0.1 - CORS support
- PyTorch 2.0+ - Deep learning
- Transformers 4.35+ - RoBERTa support (optional)

## 📝 Logging

Server logs include:
- Device type (CUDA/CPU)
- Available models
- Model loading events
- Prediction requests
- Error messages

## 🔗 Frontend Integration

The frontend connects via Vite proxy configuration:
```javascript
// vite.config.js
server: {
  proxy: {
    '/predict': {
      target: 'http://localhost:5000',
      changeOrigin: true
    }
  }
}
```

## 🐛 Troubleshooting

### Model Not Found
```
Error: Model 'rnn-lstm' not found at ../rnn_lstm_rating_model/model.pt
```
**Solution:** Ensure model is trained and exists in parent directory.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Server will automatically restart and try CPU.

### Transformers Import Error
```
Warning: Transformers not available
```
**Solution:** RoBERTa will be disabled. Use other models instead.

### Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solution:** 
```powershell
Stop-Process -Name python -Force
```

## 🔐 Security Notes

- **Development Server**: Current setup uses Flask development server
- **Production**: Use Gunicorn or Waitress for production
- **CORS**: Currently allows all origins (adjust for production)
- **Rate Limiting**: Not implemented (consider for production)

## 🚀 Production Deployment

### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows)
```powershell
pip install waitress
waitress-serve --port=5000 app:app
```

## 📞 Support

For issues or questions:
1. Check model files exist
2. Verify virtual environment is activated
3. Check Flask server logs
4. Test endpoints with curl

---

**Built with Flask + PyTorch**
