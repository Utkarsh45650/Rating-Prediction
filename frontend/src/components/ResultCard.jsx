import StarRating from './StarRating'
import './ResultCard.css'

function ResultCard({ prediction, model, text }) {
  if (!prediction) return null

  const { rating, confidence } = prediction

  const getRatingLabel = (rating) => {
    const labels = {
      1: 'Very Negative',
      2: 'Negative',
      3: 'Neutral',
      4: 'Positive',
      5: 'Very Positive'
    }
    return labels[rating] || 'Unknown'
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#4CAF50'
    if (confidence >= 0.6) return '#FF9800'
    return '#f44336'
  }

  return (
    <div className="result-card">
      <div className="result-header">
        <h3>Prediction Result</h3>
        <span className="model-badge">{model.toUpperCase()}</span>
      </div>

      <div className="result-body">
        <div className="rating-display">
          <StarRating rating={rating} />
          <div className="rating-label">
            {rating} out of 5 - {getRatingLabel(rating)}
          </div>
        </div>

        {confidence !== undefined && (
          <div className="confidence-display">
            <div className="confidence-label">Confidence Score</div>
            <div className="confidence-bar-container">
              <div
                className="confidence-bar"
                style={{
                  width: `${confidence * 100}%`,
                  backgroundColor: getConfidenceColor(confidence)
                }}
              />
            </div>
            <div className="confidence-value">
              {(confidence * 100).toFixed(2)}%
            </div>
          </div>
        )}

        {text && (
          <div className="feedback-preview">
            <strong>Analyzed Feedback:</strong>
            <p>{text.length > 150 ? text.substring(0, 150) + '...' : text}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ResultCard
