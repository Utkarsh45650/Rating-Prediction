import StarRating from './StarRating'
import './HistoryCard.css'

function HistoryCard({ history, onSelectHistory }) {
  if (!history || history.length === 0) return null

  return (
    <div className="history-card">
      <h3 className="history-title">Recent Predictions</h3>
      <div className="history-list">
        {history.map((item, index) => (
          <div
            key={index}
            className="history-item"
            onClick={() => onSelectHistory(item)}
          >
            <div className="history-item-header">
              <StarRating rating={item.rating} />
              <span className="history-model">{item.model}</span>
            </div>
            <p className="history-text">
              {item.text.substring(0, 80)}
              {item.text.length > 80 ? '...' : ''}
            </p>
            <div className="history-meta">
              {item.confidence && (
                <span className="history-confidence">
                  {(item.confidence * 100).toFixed(0)}% confidence
                </span>
              )}
              <span className="history-time">{item.timestamp}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default HistoryCard
