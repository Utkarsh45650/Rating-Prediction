function ModelSelector({ selectedModel, onModelChange }) {
  const models = [
    { value: 'cnn-lstm', label: 'CNN-LSTM Model', description: 'Convolutional + LSTM layers' },
    { value: 'rnn', label: 'RNN Model', description: 'Recurrent Neural Network with attention' },
    { value: 'rnn-lstm', label: 'RNN-LSTM Model', description: 'Combined RNN and LSTM architecture' },
    { value: 'roberta', label: 'RoBERTa Model', description: 'Transformer-based model' }
  ]

  return (
    <div className="form-group">
      <label htmlFor="model-select" className="form-label">
        Select ML Model
      </label>
      <select
        id="model-select"
        className="form-select"
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
      >
        {models.map((model) => (
          <option key={model.value} value={model.value}>
            {model.label} - {model.description}
          </option>
        ))}
      </select>
    </div>
  )
}

export default ModelSelector
