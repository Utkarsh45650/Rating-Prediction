import { useState, useEffect } from 'react'
import ModelSelector from '../components/ModelSelector'
import FeedbackInput from '../components/FeedbackInput'
import ResultCard from '../components/ResultCard'
import HistoryCard from '../components/HistoryCard'
import Loader from '../components/Loader'
import { predictRating } from '../services/api'
import './Predict.css'

function Predict() {
  const [selectedModel, setSelectedModel] = useState('rnn-lstm')
  const [feedbackText, setFeedbackText] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('predictionHistory')
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory))
    }
  }, [])

  // Save history to localStorage
  const saveToHistory = (text, model, result) => {
    const newEntry = {
      text,
      model,
      rating: result.rating,
      confidence: result.confidence,
      timestamp: new Date().toLocaleString()
    }

    const updatedHistory = [newEntry, ...history].slice(0, 5) // Keep last 5
    setHistory(updatedHistory)
    localStorage.setItem('predictionHistory', JSON.stringify(updatedHistory))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Validation
    if (!feedbackText.trim()) {
      setError('Please enter some feedback text')
      return
    }

    // Check if text contains only URLs or names (simple validation)
    const urlPattern = /(https?:\/\/[^\s]+)/g
    const cleanText = feedbackText.replace(urlPattern, '').trim()
    
    if (cleanText.length < 10) {
      setError('Please enter meaningful feedback (at least 10 characters excluding links)')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const result = await predictRating(feedbackText, selectedModel)
      setPrediction(result)
      saveToHistory(feedbackText, selectedModel, result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setFeedbackText('')
    setPrediction(null)
    setError(null)
  }

  const handleSelectHistory = (item) => {
    setFeedbackText(item.text)
    setSelectedModel(item.model)
    setPrediction({
      rating: item.rating,
      confidence: item.confidence
    })
  }

  return (
    <div className="predict-page">
      <div className="container">
        <div className="predict-content">
          <div className="predict-main">
            <div className="card">
              <div className="card-header">
                <h1 className="card-title">Predict Rating</h1>
                <p className="card-subtitle">
                  Enter customer feedback to predict the rating
                </p>
              </div>

              <form onSubmit={handleSubmit}>
                <ModelSelector
                  selectedModel={selectedModel}
                  onModelChange={setSelectedModel}
                />

                <FeedbackInput
                  value={feedbackText}
                  onChange={setFeedbackText}
                />

                {error && (
                  <div className="alert alert-error">
                    {error}
                  </div>
                )}

                <div className="button-group">
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={loading || !feedbackText.trim()}
                  >
                    {loading ? 'Analyzing...' : 'Predict Rating'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleClear}
                    disabled={loading}
                  >
                    Clear
                  </button>
                </div>
              </form>

              {loading && <Loader />}

              {prediction && !loading && (
                <ResultCard
                  prediction={prediction}
                  model={selectedModel}
                  text={feedbackText}
                />
              )}
            </div>

            <HistoryCard
              history={history}
              onSelectHistory={handleSelectHistory}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Predict
