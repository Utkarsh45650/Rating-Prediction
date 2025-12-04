# System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│                     http://localhost:3000                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐     ┌───────────┐     ┌──────────┐              │
│  │   Home   │────▶│  Predict  │◀────│  About   │              │
│  │  Page    │     │   Page    │     │  Page    │              │
│  └──────────┘     └─────┬─────┘     └──────────┘              │
│                          │                                       │
│                          ▼                                       │
│           ┌─────────────────────────┐                          │
│           │   React Components      │                          │
│           ├─────────────────────────┤                          │
│           │ • ModelSelector         │                          │
│           │ • FeedbackInput         │                          │
│           │ • ResultCard            │                          │
│           │ • StarRating            │                          │
│           │ • HistoryCard           │                          │
│           │ • Loader                │                          │
│           └──────────┬──────────────┘                          │
│                      │                                           │
│                      ▼                                           │
│           ┌─────────────────────────┐                          │
│           │    API Service          │                          │
│           │   (Axios Client)        │                          │
│           └──────────┬──────────────┘                          │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       │ HTTP POST /predict
                       │ { text, model }
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (Flask API)                        │
│                     http://localhost:5000                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────┐        │
│  │              API Endpoints                         │        │
│  ├────────────────────────────────────────────────────┤        │
│  │  POST /predict   - Get rating prediction          │        │
│  │  GET  /models    - List available models          │        │
│  │  GET  /health    - Health check                   │        │
│  │  GET  /          - API info                       │        │
│  └─────────────────────┬──────────────────────────────┘        │
│                        │                                         │
│                        ▼                                         │
│  ┌────────────────────────────────────────────────────┐        │
│  │         Text Preprocessing                         │        │
│  ├────────────────────────────────────────────────────┤        │
│  │ • Lowercase conversion                            │        │
│  │ • Negation preservation (n't → not)               │        │
│  │ • URL removal                                      │        │
│  │ • Special character cleaning                      │        │
│  │ • Tokenization                                     │        │
│  └─────────────────────┬──────────────────────────────┘        │
│                        │                                         │
│                        ▼                                         │
│  ┌────────────────────────────────────────────────────┐        │
│  │           Model Selector                           │        │
│  └─────────────────────┬──────────────────────────────┘        │
│                        │                                         │
│          ┌─────────────┼─────────────┐                          │
│          ▼             ▼             ▼             ▼            │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│   │ CNN-LSTM │  │   RNN    │  │ RNN-LSTM │  │ RoBERTa  │     │
│   │  Model   │  │  Model   │  │  Model   │  │  Model   │     │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│        │             │             │             │              │
│        └─────────────┴─────────────┴─────────────┘              │
│                      │                                           │
│                      ▼                                           │
│  ┌────────────────────────────────────────────────────┐        │
│  │         Prediction Output                          │        │
│  ├────────────────────────────────────────────────────┤        │
│  │ • Rating (1-5)                                    │        │
│  │ • Confidence (0-1)                                │        │
│  └────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                       │
                       │ JSON Response
                       │ { rating, confidence }
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CNN-LSTM Model:                                                │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐                   │
│  │ Embed│──▶│Conv1D│──▶│ LSTM │──▶│  FC  │──▶ [1-5]          │
│  └──────┘   └──────┘   └──────┘   └──────┘                   │
│                                                                  │
│  RNN Model:                                                     │
│  ┌──────┐   ┌──────┐   ┌─────────┐   ┌──────┐                │
│  │ Embed│──▶│ LSTM │──▶│Attention│──▶│  FC  │──▶ [1-5]       │
│  └──────┘   └──────┘   └─────────┘   └──────┘                │
│                                                                  │
│  RNN-LSTM Model (Deep):                                         │
│  ┌──────┐   ┌────────────┐   ┌─────────┐   ┌──────┐          │
│  │ Embed│──▶│4-Layer LSTM│──▶│Attention│──▶│ Deep │──▶ [1-5] │
│  └──────┘   │ Bidirect.  │   │+ Hidden │   │  FC  │          │
│             └────────────┘   └─────────┘   └──────┘          │
│                                                                  │
│  RoBERTa Model (Transformer):                                   │
│  ┌──────┐   ┌─────────────┐   ┌──────┐                        │
│  │Tokeniz│──▶│12-Layer     │──▶│ Class│──▶ [1-5]             │
│  │      │   │Transformer  │   │  FC  │                        │
│  └──────┘   └─────────────┘   └──────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘


DATA FLOW:
──────────

1. User enters feedback text in React form
2. Frontend validates input (length, content)
3. Axios sends POST request to Flask API
4. Flask receives request and extracts text + model
5. Text is preprocessed (cleaned, tokenized)
6. Appropriate model is loaded from cache or disk
7. Model processes text and generates prediction
8. Softmax applied to get confidence scores
9. Flask returns JSON: { rating, confidence }
10. React displays result with star rating
11. Prediction saved to browser localStorage
12. History updated with last 5 predictions


TECHNOLOGY STACK:
─────────────────

Frontend:
├─ React 18.2.0
├─ React Router DOM 6.20.0
├─ Axios 1.6.2
├─ Vite 5.0.8
└─ Pure CSS (no frameworks)

Backend:
├─ Flask 3.1.2
├─ Flask-CORS 6.0.1
├─ PyTorch 2.5+
├─ Transformers 4.35+
└─ Python 3.10

Models:
├─ CNN-LSTM: ~10M parameters
├─ RNN: ~15M parameters
├─ RNN-LSTM: ~22M parameters (4-layer deep)
└─ RoBERTa: ~125M parameters (pretrained)


FEATURES:
─────────

✓ Real-time predictions
✓ Multiple model comparison
✓ Confidence scoring
✓ Star rating visualization
✓ Prediction history (last 5)
✓ Input validation
✓ Responsive design
✓ GPU acceleration (CUDA)
✓ Mixed precision training
✓ Model caching
✓ Error handling
✓ CORS enabled
