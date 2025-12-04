import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001'

export const predictRating = async (text, model) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
      text,
      model
    })
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    throw new Error(
      error.response?.data?.error || 
      'Failed to get prediction. Please try again.'
    )
  }
}

export const getModelInfo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/models`)
    return response.data
  } catch (error) {
    console.error('API Error:', error)
    return null
  }
}
