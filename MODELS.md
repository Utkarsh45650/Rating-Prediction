# Trained Model Files

## 📦 Models Not Included in Repository

Due to GitHub's file size limitations, the trained model files are **not included** in this repository. 

### Required Models:

1. **CNN-LSTM Model** (~66 MB)
   - Location: `cnn_lstm_rating_model/model.pt`
   
2. **RNN Model** (~210 MB)
   - Location: `rnn_rating_model/model.pt`
   
3. **RNN-LSTM Model** (~250 MB)
   - Location: `rnn_lstm_rating_model/model.pt`
   
4. **RoBERTa Model** (~951 MB)
   - Location: `roberta_rating_model/model.safetensors`
   - Checkpoint: `roberta_rating_model/checkpoint-312/`

### 🏋️ Training the Models

To use this project, you need to train the models first:

```powershell
# Train CNN-LSTM model
python train_cnn_lstm.py

# Train RNN model
python train_rnn.py

# Train RNN-LSTM model
python train_rnn_lstm.py

# Train RoBERTa model
python train_roberta.py
```

### ⚡ Quick Start

1. **Ensure you have the dataset**: `Reviews.csv` (Amazon product reviews)
2. **Train models** using the scripts above
3. **Start the application**:
   ```powershell
   .\start_app.ps1
   ```

### 📝 Model Details

| Model | Size | Training Time (GPU) | Accuracy |
|-------|------|---------------------|----------|
| CNN-LSTM | ~66 MB | ~30 min | Good |
| RNN | ~210 MB | ~1-2 hours | 79.65% |
| RNN-LSTM | ~250 MB | ~2-3 hours | Highest |
| RoBERTa | ~951 MB | ~4-5 hours | SOTA |

### 🔗 Alternative: Download Pre-trained Models

If available, you can download pre-trained models from:
- [Google Drive / Dropbox link] *(add your own link)*
- [Hugging Face Hub] *(if uploaded)*

Place the downloaded models in their respective directories:
- `cnn_lstm_rating_model/`
- `rnn_rating_model/`
- `rnn_lstm_rating_model/`
- `roberta_rating_model/`

---

**Note**: Model files are excluded via `.gitignore` to keep the repository size manageable.
