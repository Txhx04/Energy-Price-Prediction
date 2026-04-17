import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs'
import { Badge } from './ui/badge'
import { Trophy, TrendingUp, CheckCircle, AlertTriangle, BarChart3, Beaker, FlaskConical } from 'lucide-react'
import {
  fetchValidationResults, fetchTestResults, fetchAvailablePlots, getPlotUrl
} from '../services/ApiService'
import { cn } from '../lib/utils'

const MODEL_COLORS_MAP = {
  'XGBoost': '#3b82f6',
  'XGB': '#3b82f6',
  'RF': '#22c55e',
  'GRU': '#a855f7',
  'LSTM': '#8b5cf6',
  'LR': '#94a3b8',
  'DT': '#f97316',
  'SVM': '#ef4444',
  'Ensemble': '#10b981',
}

function getBarColor(modelName) {
  for (const [key, color] of Object.entries(MODEL_COLORS_MAP)) {
    if (modelName.includes(key)) return color
  }
  return '#6b7280'
}

/**
 * ResultsPage — Validation & Test Results Dashboard
 * Shows pre-computed metrics, model comparisons, and analysis plots.
 */
export default function ResultsPage() {
  const [validationData, setValidationData] = useState(null)
  const [testData, setTestData] = useState(null)
  const [plots, setPlots] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const [valRes, testRes, plotRes] = await Promise.all([
          fetchValidationResults(),
          fetchTestResults(),
          fetchAvailablePlots(),
        ])
        setValidationData(valRes)
        setTestData(testRes)
        setPlots(plotRes)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center space-y-4">
          <div className="w-8 h-8 border-2 border-primary/30 border-t-primary rounded-full animate-spin mx-auto" />
          <p className="text-sm text-muted-foreground">Loading results...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <Card className="border-destructive/50">
        <CardContent className="p-6">
          <p className="text-destructive">Error loading results: {error}</p>
        </CardContent>
      </Card>
    )
  }

  // Split metrics by target type
  const splitByTarget = (metrics) => {
    const consumption = metrics.filter(m => m.target === 'Consumption')
    const price = metrics.filter(m => m.target === 'Price')
    return { consumption, price }
  }

  const valSplit = validationData ? splitByTarget(validationData.metrics) : null
  const testSplit = testData ? splitByTarget(testData.metrics) : null

  // Prepare chart data for RMSE comparison
  const buildChartData = (metrics, metric = 'rmse') => {
    return metrics
      .sort((a, b) => a[metric] - b[metric])
      .map(m => ({
        name: m.model.replace('_Consumption', '').replace('_Cons', '').replace('_Price', ''),
        value: m[metric],
        r2: m.r2,
        fullName: m.model,
      }))
  }

  const MetricsTable = ({ metrics, type }) => {
    const sorted = [...metrics].sort((a, b) => a.rmse - b.rmse)
    const bestRmse = sorted[0]?.rmse
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/30">
              <th className="text-left p-3 font-medium">#</th>
              <th className="text-left p-3 font-medium">Model</th>
              <th className="text-right p-3 font-medium">RMSE</th>
              <th className="text-right p-3 font-medium">MAE</th>
              <th className="text-right p-3 font-medium">R²</th>
              <th className="text-right p-3 font-medium">MAPE</th>
              <th className="text-right p-3 font-medium">Samples</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, idx) => {
              const isBest = m.rmse === bestRmse
              return (
                <tr
                  key={m.model}
                  className={cn(
                    "border-b transition-colors hover:bg-muted/20",
                    isBest && "bg-emerald-50/50 border-l-2 border-l-emerald-500"
                  )}
                >
                  <td className="p-3">
                    {isBest ? <Trophy className="w-4 h-4 text-amber-500" /> : idx + 1}
                  </td>
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: getBarColor(m.model) }} />
                      <span className="font-medium">{m.model}</span>
                      {isBest && <Badge variant="success" className="text-[10px] py-0">Best</Badge>}
                    </div>
                  </td>
                  <td className={cn("p-3 text-right tabular-nums", isBest && "font-bold text-emerald-700")}>
                    {type === 'price' ? `€${m.rmse.toFixed(2)}` : m.rmse.toFixed(2)}
                  </td>
                  <td className="p-3 text-right tabular-nums">
                    {type === 'price' ? `€${m.mae.toFixed(2)}` : m.mae.toFixed(2)}
                  </td>
                  <td className={cn("p-3 text-right tabular-nums", m.r2 === Math.max(...metrics.map(x => x.r2)) && "font-bold text-emerald-700")}>
                    {m.r2.toFixed(4)}
                  </td>
                  <td className="p-3 text-right tabular-nums text-muted-foreground">
                    {type === 'price' ? '—' : `${m.mape.toFixed(2)}%`}
                  </td>
                  <td className="p-3 text-right tabular-nums text-muted-foreground">
                    {m.samples.toLocaleString()}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }

  const RmseChart = ({ data, type }) => (
    <div className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" opacity={0.4} />
          <XAxis type="number" tick={{ fontSize: 11 }} tickLine={false} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={120} tickLine={false} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="bg-white/95 backdrop-blur-sm border rounded-lg shadow-xl p-3">
                  <p className="text-sm font-semibold">{d.fullName}</p>
                  <p className="text-xs">RMSE: <span className="font-mono font-bold">{type === 'price' ? `€${d.value.toFixed(2)}` : d.value.toFixed(2)}</span></p>
                  <p className="text-xs">R²: <span className="font-mono font-bold">{d.r2.toFixed(4)}</span></p>
                </div>
              )
            }}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={28}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={getBarColor(entry.name)} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Production Readiness Banner */}
      <Card className="bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200">
        <CardContent className="p-6">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-8 h-8 text-emerald-600" />
            <div>
              <h3 className="text-lg font-bold text-emerald-800">Both Ensembles Production Ready ✅</h3>
              <p className="text-sm text-emerald-700 mt-1">
                Consumption Ensemble: RMSE 1726.51 MW, R² 0.8798 | Price Ensemble: RMSE €24.34, R² 0.7156
              </p>
              <p className="text-xs text-emerald-600 mt-1">
                Ensemble Config: Consumption (60% XGB + 30% GRU + 10% RF) | Price (50% XGB + 35% GRU + 15% RF)
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs: Validation vs Test */}
      <Tabs defaultValue="test">
        <TabsList className="mb-4">
          <TabsTrigger value="validation" className="gap-2">
            <Beaker className="w-4 h-4" /> Validation (2025)
          </TabsTrigger>
          <TabsTrigger value="test" className="gap-2">
            <FlaskConical className="w-4 h-4" /> Test (2026)
          </TabsTrigger>
          <TabsTrigger value="plots" className="gap-2">
            <BarChart3 className="w-4 h-4" /> Analysis Plots
          </TabsTrigger>
        </TabsList>

        {/* Validation Tab */}
        <TabsContent value="validation">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Beaker className="w-5 h-5 text-purple-500" />
                  Validation Results — {validationData?.period}
                </CardTitle>
                <CardDescription>
                  Year 2025 holdout set — used to tune hyperparameters and select ensemble weights
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="consumption">
                  <TabsList className="mb-4">
                    <TabsTrigger value="consumption">⚡ Consumption</TabsTrigger>
                    <TabsTrigger value="price">💰 Price</TabsTrigger>
                  </TabsList>
                  <TabsContent value="consumption">
                    {valSplit && (
                      <div className="space-y-6">
                        <RmseChart data={buildChartData(valSplit.consumption)} type="consumption" />
                        <MetricsTable metrics={valSplit.consumption} type="consumption" />
                      </div>
                    )}
                  </TabsContent>
                  <TabsContent value="price">
                    {valSplit && (
                      <div className="space-y-6">
                        <RmseChart data={buildChartData(valSplit.price)} type="price" />
                        <MetricsTable metrics={valSplit.price} type="price" />
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* MAPE Warning for Price */}
            <Card className="border-amber-200 bg-amber-50/30">
              <CardContent className="p-4 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-800">Price MAPE Anomaly</p>
                  <p className="text-xs text-amber-700 mt-1">
                    High MAPE (3000-9000%) for price models is a mathematical artifact of zero/near-zero €/MWh hours
                    in the wholesale market. Use MAE/RMSE for price model evaluation instead.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Test Tab */}
        <TabsContent value="test">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FlaskConical className="w-5 h-5 text-blue-500" />
                  Test Results — {testData?.period}
                </CardTitle>
                <CardDescription>
                  Unseen 2026 data — final production readiness evaluation (XGBoost + GRU + RF + Ensemble)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="consumption">
                  <TabsList className="mb-4">
                    <TabsTrigger value="consumption">⚡ Consumption</TabsTrigger>
                    <TabsTrigger value="price">💰 Price</TabsTrigger>
                  </TabsList>
                  <TabsContent value="consumption">
                    {testSplit && (
                      <div className="space-y-6">
                        <RmseChart data={buildChartData(testSplit.consumption)} type="consumption" />
                        <MetricsTable metrics={testSplit.consumption} type="consumption" />
                      </div>
                    )}
                  </TabsContent>
                  <TabsContent value="price">
                    {testSplit && (
                      <div className="space-y-6">
                        <RmseChart data={buildChartData(testSplit.price)} type="price" />
                        <MetricsTable metrics={testSplit.price} type="price" />
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* Ensemble Effectiveness Summary */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200">
                <CardContent className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="w-5 h-5 text-blue-600" />
                    <h4 className="font-semibold text-blue-800">Consumption Ensemble</h4>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-blue-600">RMSE</p>
                      <p className="text-xl font-bold text-blue-800">1726.51 MW</p>
                    </div>
                    <div>
                      <p className="text-xs text-blue-600">R²</p>
                      <p className="text-xl font-bold text-blue-800">0.8798</p>
                    </div>
                    <div>
                      <p className="text-xs text-blue-600">MAPE</p>
                      <p className="text-lg font-bold text-blue-800">3.24%</p>
                    </div>
                    <div>
                      <p className="text-xs text-blue-600">Status</p>
                      <Badge variant="success">Production Ready</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-amber-50 to-orange-50 border-amber-200">
                <CardContent className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="w-5 h-5 text-amber-600" />
                    <h4 className="font-semibold text-amber-800">Price Ensemble</h4>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-amber-600">RMSE</p>
                      <p className="text-xl font-bold text-amber-800">€24.34</p>
                    </div>
                    <div>
                      <p className="text-xs text-amber-600">R²</p>
                      <p className="text-xl font-bold text-amber-800">0.7156</p>
                    </div>
                    <div>
                      <p className="text-xs text-amber-600">MAE</p>
                      <p className="text-lg font-bold text-amber-800">€17.55</p>
                    </div>
                    <div>
                      <p className="text-xs text-amber-600">Status</p>
                      <Badge variant="success">Production Ready</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Plots Tab */}
        <TabsContent value="plots">
          <div className="space-y-6">
            {plots && Object.entries(plots).map(([phase, files]) => (
              files.length > 0 && (
                <Card key={phase}>
                  <CardHeader>
                    <CardTitle className="capitalize">
                      {phase === 'validation' ? '🔬 Validation Plots' : '🧪 Test Plots'}
                    </CardTitle>
                    <CardDescription>
                      Pre-generated analysis visualizations from the {phase} phase
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {files.map(filename => (
                        <div key={filename} className="border rounded-lg overflow-hidden bg-white">
                          <img
                            src={getPlotUrl(phase, filename)}
                            alt={filename.replace('.png', '').replace(/_/g, ' ')}
                            className="w-full h-auto"
                            loading="lazy"
                          />
                          <div className="p-2 bg-muted/30">
                            <p className="text-xs text-muted-foreground font-mono">
                              {filename.replace('.png', '').replace(/_/g, ' ')}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
