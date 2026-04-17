import { useState } from 'react'
import { usePrediction } from './hooks/usePrediction'
import { fetchFeatureImportances } from './services/ApiService'
import Header from './components/Header'
import DateSelector from './components/DateSelector'
import LoadingSpinner from './components/LoadingSpinner'
import ModelMetricCard from './components/ModelMetricCard'
import PredictiveChart from './components/PredictiveChart'
import ModelComparisonTable from './components/ModelComparisonTable'
import FeatureAnalysis from './components/FeatureAnalysis'
import DatasetInfoPanel from './components/DatasetInfoPanel'
import ResultsPage from './components/ResultsPage'
import EnergyPlanner from './components/EnergyPlanner'
import { cn } from './lib/utils'
import { BarChart3, FlaskConical, Zap } from 'lucide-react'

const TABS = [
  { id: 'predictions', label: 'Predictions', icon: BarChart3, emoji: '📊' },
  { id: 'results', label: 'Results & Validation', icon: FlaskConical, emoji: '📈' },
  { id: 'planner', label: 'Energy Planner', icon: Zap, emoji: '⚡' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('predictions')
  const {
    predictionData, availableDates, loading, initialLoading,
    error, selectedDate, getPrediction, clearError,
  } = usePrediction()
  const [featureImportances, setFeatureImportances] = useState(null)

  // Fetch feature importances after prediction
  const handlePrediction = async (date) => {
    await getPrediction(date)
    // Also load feature importances
    try {
      const fi = await fetchFeatureImportances()
      setFeatureImportances(fi)
    } catch {
      // Non-critical, ignore
    }
  }

  // Determine which model has the best RMSE for consumption
  const getBestModel = (models) => {
    if (!models) return null
    let best = null
    let bestRmse = Infinity
    models.forEach(m => {
      if (m.metrics.consumption.rmse < bestRmse) {
        bestRmse = m.metrics.consumption.rmse
        best = m.name
      }
    })
    return best
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Header />

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 bg-muted/50 rounded-xl p-1 mb-8" id="main-tabs">
          {TABS.map(tab => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all",
                  activeTab === tab.id
                    ? "bg-white text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground hover:bg-white/50"
                )}
                id={`tab-${tab.id}`}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{tab.label}</span>
                <span className="sm:hidden">{tab.emoji}</span>
              </button>
            )
          })}
        </div>

        {/* Tab Content */}
        {activeTab === 'predictions' && (
          <div className="space-y-8">
            {/* Date Selector */}
            {initialLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
              </div>
            ) : (
              <>
                <DatasetInfoPanel datasetInfo={availableDates?.dataset_info} />
                <DateSelector
                  availableDates={availableDates}
                  onDateSelect={handlePrediction}
                  loading={loading}
                  error={error}
                  onClearError={clearError}
                />
              </>
            )}

            {/* Loading State */}
            {loading && <LoadingSpinner />}

            {/* Results */}
            {predictionData && !loading && (
              <>
                {/* Model Metric Cards */}
                <div>
                  <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    📊 Model Performance — {predictionData.prediction_date}
                  </h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {predictionData.models.map((model, idx) => (
                      <ModelMetricCard
                        key={model.name}
                        model={model}
                        type="consumption"
                        isBest={model.name === getBestModel(predictionData.models)}
                        rank={idx}
                      />
                    ))}
                  </div>
                </div>

                {/* Charts */}
                <PredictiveChart predictionData={predictionData} />

                {/* Comparison Table */}
                <ModelComparisonTable
                  models={predictionData.models}
                  actuals={predictionData.actuals}
                />

                {/* Feature Analysis */}
                {featureImportances && (
                  <FeatureAnalysis featureImportances={featureImportances} />
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'results' && <ResultsPage />}
        {activeTab === 'planner' && <EnergyPlanner />}

        {/* Footer */}
        <footer className="mt-12 py-6 border-t text-center">
          <p className="text-xs text-muted-foreground">
            Spain Energy Prediction System — ML Project © 2026
            <span className="mx-2">•</span>
            Ensemble: 60% XGBoost + 30% GRU + 10% RF (Consumption) | 50% XGB + 35% GRU + 15% RF (Price)
          </p>
        </footer>
      </div>
    </div>
  )
}
