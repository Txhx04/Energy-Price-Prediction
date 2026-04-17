import { useState, useEffect, useCallback } from 'react'
import { fetchPrediction, fetchAvailableDates } from '../services/ApiService'

/**
 * Custom hook for managing prediction state.
 * Adapts backend/api.py response format to frontend component expectations.
 */
export function usePrediction() {
  const [predictionData, setPredictionData] = useState(null)
  const [availableDates, setAvailableDates] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initialLoading, setInitialLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedDate, setSelectedDate] = useState(null)

  // ── Fetch available dates on mount ──
  useEffect(() => {
    let cancelled = false
    async function loadDates() {
      try {
        const data = await fetchAvailableDates()
        if (!cancelled) {
          setAvailableDates(data)
          setInitialLoading(false)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message)
          setInitialLoading(false)
        }
      }
    }
    loadDates()
    return () => { cancelled = true }
  }, [])

  // ── Fetch prediction for a selected date ──
  const getPrediction = useCallback(async (date) => {
    if (!date) return

    setLoading(true)
    setError(null)
    setSelectedDate(date)

    try {
      const dateStr = typeof date === 'string'
        ? date
        : date.toISOString().split('T')[0]

      const raw = await fetchPrediction(dateStr)

      // Adapt backend response to frontend component format
      // Backend returns: { models: [{name, consumption: {predictions, mae, rmse, r2, mape}, price: {...}}, ...] }
      // Frontend expects: { models: [{name, type, metrics: {consumption: {mae, rmse, r_squared, mape}, price: {...}}, hourly_predictions: {consumption: [...], price: [...]}}] }
      const adaptedModels = raw.models.map(m => ({
        name: m.name,
        type: m.name.includes('Ensemble') ? 'ensemble' : 'individual',
        metrics: {
          consumption: {
            mae: m.consumption.mae,
            rmse: m.consumption.rmse,
            r_squared: m.consumption.r2,
            mape: m.consumption.mape,
          },
          price: {
            mae: m.price.mae,
            rmse: m.price.rmse,
            r_squared: m.price.r2,
            mape: m.price.mape,
          },
        },
        hourly_predictions: {
          consumption: m.consumption.predictions,
          price: m.price.predictions,
        },
      }))

      const adapted = {
        models: adaptedModels,
        actuals: {
          consumption: raw.actual_consumption,
          price: raw.actual_price,
        },
        hour_labels: raw.hours.map(h => {
          // Convert "H+1" to "01:00" etc.
          const num = parseInt(h.replace('H+', ''))
          return `${String(num - 1).padStart(2, '0')}:00`
        }),
        hourly_timestamps: raw.hours,
        prediction_date: raw.date,
        feature_importances: null,  // Will be fetched separately if needed
      }

      setPredictionData(adapted)
    } catch (err) {
      setError(err.message)
      setPredictionData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  // ── Clear error ──
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    predictionData,
    availableDates,
    loading,
    initialLoading,
    error,
    selectedDate,
    getPrediction,
    clearError,
  }
}
