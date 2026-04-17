/**
 * ApiService.js — Centralized API Client
 * Connects to backend/api.py (FastAPI)
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Also create a root-level client for non-/api endpoints
const rootApi = axios.create({
  baseURL: '/',
  timeout: 30000,
})

// ── Response interceptor for error handling ──
const errorHandler = (error) => {
  if (error.response) {
    const { status, data } = error.response
    const message = data?.detail || `Server error (${status})`
    if (status === 404) return Promise.reject(new Error(message))
    if (status === 400) return Promise.reject(new Error(message))
    return Promise.reject(new Error(`API Error: ${message}`))
  }
  if (error.code === 'ECONNABORTED') {
    return Promise.reject(new Error('Request timed out. Is the backend running on port 8000?'))
  }
  return Promise.reject(new Error('Network error. Please check your connection and ensure the backend is running.'))
}

api.interceptors.response.use(r => r, errorHandler)
rootApi.interceptors.response.use(r => r, errorHandler)

// ═════════════════════════════════════════
// Prediction endpoints
// ═════════════════════════════════════════

/**
 * Fetch predictions for a given date (all models).
 * @param {string} date - Format: YYYY-MM-DD
 */
export async function fetchPrediction(date) {
  const response = await api.post('/predict', { date })
  return response.data
}

/**
 * Fetch available date range from the test dataset.
 */
export async function fetchAvailableDates() {
  const response = await rootApi.get('/available-dates')
  return response.data
}

/**
 * Health check for the backend.
 */
export async function checkHealth() {
  const response = await rootApi.get('/health')
  return response.data
}

// ═════════════════════════════════════════
// Results & Validation endpoints
// ═════════════════════════════════════════

/**
 * Fetch validation-phase metrics.
 */
export async function fetchValidationResults() {
  const response = await api.get('/validation-results')
  return response.data
}

/**
 * Fetch test-phase metrics.
 */
export async function fetchTestResults() {
  const response = await api.get('/test-results')
  return response.data
}

/**
 * Fetch list of available plots.
 */
export async function fetchAvailablePlots() {
  const response = await api.get('/available-plots')
  return response.data
}

/**
 * Get URL for a results plot.
 */
export function getPlotUrl(phase, filename) {
  return `/api/results-plots/${phase}/${filename}`
}

/**
 * Fetch feature importances.
 */
export async function fetchFeatureImportances() {
  const response = await api.get('/feature-importances')
  return response.data
}

// ═════════════════════════════════════════
// Energy Planner endpoints
// ═════════════════════════════════════════

/**
 * Fetch default activity presets.
 */
export async function fetchDefaultActivities() {
  const response = await api.get('/default-activities')
  return response.data
}

/**
 * Fetch live forecast for a date and specific model.
 */
export async function fetchLiveForecast(date, model = 'Ensemble') {
  const response = await api.get(`/live-forecast?date=${date}&model=${encodeURIComponent(model)}`)
  return response.data
}

/**
 * Run activity optimization.
 */
export async function optimizeSchedule(date, model, activities) {
  const response = await api.post('/optimize', { date, model, activities })
  return response.data
}

export default api
