import { Link } from 'react-router-dom'
import './Home.css'

function Home() {
  return (
    <div className="home-page">
      <div className="container">
        <div className="home-content">
          <div className="hero-section">
            <div className="hero-icon">⭐</div>
            <h1 className="hero-title">
              Customer Feedback Rating Predictor
            </h1>
            <p className="hero-subtitle">
              Leverage advanced Machine Learning models to predict customer satisfaction ratings from text feedback
            </p>
            <div className="hero-features">
              <div className="feature-item">
                <span className="feature-icon">🤖</span>
                <span>4 ML Models</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">⚡</span>
                <span>Real-time Predictions</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">📊</span>
                <span>High Accuracy</span>
              </div>
            </div>
            <Link to="/predict" className="btn btn-primary btn-large">
              Start Predicting
            </Link>
          </div>

          <div className="info-cards">
            <div className="info-card">
              <h3>🎯 How It Works</h3>
              <ol className="info-list">
                <li>Choose your preferred ML model</li>
                <li>Enter customer feedback or review text</li>
                <li>Get instant rating prediction (1-5 stars)</li>
                <li>View confidence scores and analysis</li>
              </ol>
            </div>

            <div className="info-card">
              <h3>🚀 Available Models</h3>
              <ul className="model-list">
                <li><strong>CNN-LSTM:</strong> Combines convolutional and recurrent layers</li>
                <li><strong>RNN:</strong> Recurrent network with attention mechanism</li>
                <li><strong>RNN-LSTM:</strong> Enhanced hybrid architecture</li>
                <li><strong>RoBERTa:</strong> State-of-the-art transformer model</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home
