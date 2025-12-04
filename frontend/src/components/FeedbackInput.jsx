function FeedbackInput({ value, onChange, placeholder }) {
  return (
    <div className="form-group">
      <label htmlFor="feedback-text" className="form-label">
        Customer Feedback
      </label>
      <textarea
        id="feedback-text"
        className="form-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder || "Enter customer feedback or product review here..."}
        rows="6"
      />
      <small style={{ color: '#666', fontSize: '0.875rem' }}>
        {value.length} characters
      </small>
    </div>
  )
}

export default FeedbackInput
