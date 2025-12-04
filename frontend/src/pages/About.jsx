import './About.css'

function About() {
  return (
    <div className="about-page">
      <div className="container">
        <div className="about-content">
          <div className="card">
            <h1 className="page-title">About This Project</h1>
            
            <section className="about-section">
              <h2>📖 Overview</h2>
              <p>
                This Customer Feedback Rating Predictor uses state-of-the-art Machine Learning 
                models to analyze customer reviews and predict their satisfaction rating on a 
                scale of 1 to 5 stars. The system has been trained on thousands of real customer 
                reviews to understand sentiment patterns and provide accurate predictions.
              </p>
            </section>

            <section className="about-section">
              <h2>🤖 Machine Learning Models</h2>
              
              <div className="model-detail">
                <h3>CNN-LSTM Model</h3>
                <p>
                  <strong>Architecture:</strong> Combines Convolutional Neural Networks (CNN) 
                  with Long Short-Term Memory (LSTM) layers.
                </p>
                <p>
                  <strong>Strengths:</strong> Excellent at capturing local patterns in text 
                  through convolution while maintaining sequence information via LSTM.
                </p>
              </div>

              <div className="model-detail">
                <h3>RNN Model</h3>
                <p>
                  <strong>Architecture:</strong> Recurrent Neural Network with attention mechanism 
                  for enhanced context understanding.
                </p>
                <p>
                  <strong>Strengths:</strong> Great for sequential data processing with improved 
                  focus on important parts of the text through attention.
                </p>
              </div>

              <div className="model-detail">
                <h3>RNN-LSTM Model</h3>
                <p>
                  <strong>Architecture:</strong> Hybrid model combining RNN and LSTM layers 
                  with 4-layer bidirectional architecture and attention mechanism.
                </p>
                <p>
                  <strong>Strengths:</strong> Optimized for maximum accuracy with deep architecture, 
                  capturing both short and long-term dependencies in customer feedback.
                </p>
              </div>

              <div className="model-detail">
                <h3>RoBERTa Model</h3>
                <p>
                  <strong>Architecture:</strong> Robustly Optimized BERT Pretraining Approach - 
                  a transformer-based model.
                </p>
                <p>
                  <strong>Strengths:</strong> State-of-the-art natural language understanding with 
                  pre-trained knowledge from millions of documents.
                </p>
              </div>
            </section>

            <section className="about-section">
              <h2>📊 Training Dataset</h2>
              <p>
                The models were trained on the Amazon Product Reviews dataset containing over 
                500,000 real customer reviews. The dataset includes:
              </p>
              <ul>
                <li>Product reviews from various categories</li>
                <li>Star ratings from 1 to 5</li>
                <li>Authentic customer feedback and opinions</li>
                <li>Balanced representation across different sentiment levels</li>
              </ul>
            </section>

            <section className="about-section">
              <h2>⚡ How It Works</h2>
              <ol className="steps-list">
                <li>
                  <strong>Text Preprocessing:</strong> Your feedback is cleaned and tokenized 
                  to prepare it for the model.
                </li>
                <li>
                  <strong>Model Processing:</strong> The selected ML model analyzes the text, 
                  identifying sentiment patterns and key phrases.
                </li>
                <li>
                  <strong>Prediction:</strong> Based on learned patterns, the model predicts 
                  a rating from 1-5 stars.
                </li>
                <li>
                  <strong>Confidence Score:</strong> The system also provides a confidence level 
                  indicating how certain it is about the prediction.
                </li>
              </ol>
            </section>

            <section className="about-section">
              <h2>🎯 Use Cases</h2>
              <div className="use-cases">
                <div className="use-case-item">
                  <h4>📦 E-commerce Platforms</h4>
                  <p>Automatically categorize customer reviews and identify satisfaction levels</p>
                </div>
                <div className="use-case-item">
                  <h4>🏢 Customer Support</h4>
                  <p>Prioritize responses based on predicted customer satisfaction</p>
                </div>
                <div className="use-case-item">
                  <h4>📈 Business Analytics</h4>
                  <p>Analyze feedback trends and measure customer sentiment at scale</p>
                </div>
                <div className="use-case-item">
                  <h4>🔍 Quality Assurance</h4>
                  <p>Monitor product quality through sentiment analysis of reviews</p>
                </div>
              </div>
            </section>

            <section className="about-section">
              <h2>💡 Tips for Best Results</h2>
              <ul className="tips-list">
                <li>Provide clear, descriptive feedback (at least 10-20 words)</li>
                <li>Include specific details about the product or service</li>
                <li>Express your opinion naturally as you would in a real review</li>
                <li>Avoid using only URLs, names, or very short phrases</li>
                <li>Try different models to compare predictions</li>
              </ul>
            </section>

            <section className="about-section tech-stack">
              <h2>🛠️ Technology Stack</h2>
              <div className="tech-grid">
                <div className="tech-item">
                  <strong>Frontend:</strong> React, Pure CSS
                </div>
                <div className="tech-item">
                  <strong>Backend:</strong> Flask (Python)
                </div>
                <div className="tech-item">
                  <strong>ML Framework:</strong> PyTorch
                </div>
                <div className="tech-item">
                  <strong>Models:</strong> CNN, LSTM, RNN, Transformers
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}

export default About
