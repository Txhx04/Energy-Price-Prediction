import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import {
  Zap, Plus, Trash2, Clock, DollarSign, TrendingDown,
  Calendar, Sparkles, Battery, ArrowRight
} from 'lucide-react'
import {
  fetchDefaultActivities, optimizeSchedule, fetchLiveForecast
} from '../services/ApiService'
import { cn, formatCurrency } from '../lib/utils'

const ACTIVITY_ICONS = {
  '🍽️': '🍽️', '👕': '👕', '🌀': '🌀', '🔌': '🔌',
  '🍳': '🍳', '🏊': '🏊', '🚿': '🚿', '❄️': '❄️', '⚡': '⚡',
}

const MODELS = [
  { value: 'Ensemble', label: 'Ensemble (XGB+GRU+RF)', color: '#10b981' },
  { value: 'XGBoost', label: 'XGBoost', color: '#3b82f6' },
  { value: 'GRU', label: 'GRU (Deep Learning)', color: '#a855f7' },
  { value: 'LSTM', label: 'LSTM (Deep Learning)', color: '#8b5cf6' },
  { value: 'Random Forest', label: 'Random Forest', color: '#22c55e' },
  { value: 'Linear Regression', label: 'Linear Regression', color: '#94a3b8' },
]

/**
 * EnergyPlanner — Activity Scheduling & Savings Estimation
 * Select a model → predict 24h prices → schedule activities → see savings
 */
export default function EnergyPlanner() {
  const [selectedModel, setSelectedModel] = useState('Ensemble')
  const [selectedDate, setSelectedDate] = useState(
    new Date(Date.now() + 86400000).toISOString().split('T')[0]
  )
  const [activities, setActivities] = useState([])
  const [optimizationResult, setOptimizationResult] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(false)
  const [forecastLoading, setForecastLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load default activities on mount
  useEffect(() => {
    async function loadDefaults() {
      try {
        const data = await fetchDefaultActivities()
        setActivities(data.activities.map((a, i) => ({ ...a, id: i, enabled: true })))
      } catch (err) {
        // Use hardcoded defaults if API fails
        setActivities([
          { id: 0, name: 'Dishwasher', power_kw: 1.8, duration_hours: 2, priority: 3, earliest_start: 6, latest_finish: 23, icon: '🍽️', enabled: true },
          { id: 1, name: 'Washing Machine', power_kw: 2.0, duration_hours: 2, priority: 3, earliest_start: 6, latest_finish: 22, icon: '👕', enabled: true },
          { id: 2, name: 'EV Charging', power_kw: 7.4, duration_hours: 4, priority: 2, earliest_start: 20, latest_finish: 8, icon: '🔌', enabled: true },
          { id: 3, name: 'Water Heater', power_kw: 3.0, duration_hours: 2, priority: 4, earliest_start: 5, latest_finish: 9, icon: '🚿', enabled: true },
        ])
      }
    }
    loadDefaults()
  }, [])

  // Fetch live forecast when date or model changes
  const loadForecast = useCallback(async () => {
    setForecastLoading(true)
    try {
      const data = await fetchLiveForecast(selectedDate, selectedModel)
      setForecast(data)
    } catch (err) {
      setError(`Forecast error: ${err.message}`)
    } finally {
      setForecastLoading(false)
    }
  }, [selectedDate, selectedModel])

  useEffect(() => {
    loadForecast()
  }, [loadForecast])

  // Run optimization
  const handleOptimize = async () => {
    const enabledActivities = activities.filter(a => a.enabled)
    if (enabledActivities.length === 0) {
      setError('Please enable at least one activity')
      return
    }

    setLoading(true)
    setError(null)
    try {
      const result = await optimizeSchedule(
        selectedDate,
        selectedModel,
        enabledActivities.map(a => ({
          name: a.name,
          power_kw: a.power_kw,
          duration_hours: a.duration_hours,
          priority: a.priority,
          earliest_start: a.earliest_start,
          latest_finish: a.latest_finish,
          icon: a.icon,
        }))
      )
      setOptimizationResult(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Activity management
  const toggleActivity = (id) => {
    setActivities(prev => prev.map(a => a.id === id ? { ...a, enabled: !a.enabled } : a))
  }

  const removeActivity = (id) => {
    setActivities(prev => prev.filter(a => a.id !== id))
  }

  const addActivity = () => {
    const maxId = Math.max(0, ...activities.map(a => a.id))
    setActivities(prev => [...prev, {
      id: maxId + 1,
      name: 'New Activity',
      power_kw: 1.0,
      duration_hours: 1,
      priority: 3,
      earliest_start: 0,
      latest_finish: 23,
      icon: '⚡',
      enabled: true,
    }])
  }

  const updateActivity = (id, field, value) => {
    setActivities(prev => prev.map(a =>
      a.id === id ? { ...a, [field]: value } : a
    ))
  }

  // Build price chart data
  const priceChartData = forecast ? forecast.predicted_price.map((price, i) => ({
    hour: `${String(i).padStart(2, '0')}:00`,
    price,
    isCheapest: i === forecast.cheapest_hour,
    isMostExpensive: i === forecast.most_expensive_hour,
  })) : []

  return (
    <div className="space-y-6">
      {/* Config Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Model Selector */}
        <Card>
          <CardContent className="p-4">
            <label className="text-sm font-medium mb-2 block">Price Prediction Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full h-10 px-3 rounded-lg border bg-background text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary"
              id="model-selector"
            >
              {MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground mt-1.5">
              Choose which model generates the 24h price forecast
            </p>
          </CardContent>
        </Card>

        {/* Date Selector */}
        <Card>
          <CardContent className="p-4">
            <label className="text-sm font-medium mb-2 block">Forecast Date</label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="w-full h-10 px-3 rounded-lg border bg-background text-sm focus:ring-2 focus:ring-primary/20 focus:border-primary"
              id="planner-date"
            />
            <p className="text-xs text-muted-foreground mt-1.5">
              Source: {forecast?.data_source || 'Loading...'}
            </p>
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <Card className={cn(
          "transition-all",
          forecast && "bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200"
        )}>
          <CardContent className="p-4">
            {forecastLoading ? (
              <div className="flex items-center gap-2 py-4">
                <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                <span className="text-sm text-muted-foreground">Fetching forecast...</span>
              </div>
            ) : forecast ? (
              <div className="space-y-2">
                <p className="text-sm font-medium text-blue-800">Price Forecast Summary</p>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <p className="text-[10px] text-blue-600">Avg Price</p>
                    <p className="text-sm font-bold text-blue-800">€{forecast.avg_price}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-emerald-600">Cheapest</p>
                    <p className="text-sm font-bold text-emerald-700">{String(forecast.cheapest_hour).padStart(2, '0')}:00</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-red-600">Peak</p>
                    <p className="text-sm font-bold text-red-700">{String(forecast.most_expensive_hour).padStart(2, '0')}:00</p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground py-4">Select a date to see forecast</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 24h Price Forecast Chart */}
      {forecast && (
        <Card className="opacity-0 animate-fade-in-up">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              💰 24-Hour Price Forecast
            </CardTitle>
            <CardDescription>
              Predicted by {selectedModel} model — {selectedDate}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={priceChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                  <XAxis dataKey="hour" tick={{ fontSize: 10 }} tickLine={false} interval={1} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} label={{ value: '€/MWh', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const d = payload[0].payload
                      return (
                        <div className="bg-white/95 backdrop-blur-sm border rounded-lg shadow-xl p-3">
                          <p className="text-sm font-semibold">{d.hour}</p>
                          <p className="text-xs">Price: <span className="font-mono font-bold">€{d.price.toFixed(2)}/MWh</span></p>
                          {d.isCheapest && <Badge variant="success" className="mt-1 text-[10px]">Cheapest Hour</Badge>}
                          {d.isMostExpensive && <Badge variant="destructive" className="mt-1 text-[10px]">Most Expensive</Badge>}
                        </div>
                      )
                    }}
                  />
                  <ReferenceLine y={forecast.avg_price} stroke="#6b7280" strokeDasharray="4 4" label={{ value: 'Avg', position: 'right', fontSize: 10 }} />
                  <Bar dataKey="price" radius={[4, 4, 0, 0]} maxBarSize={20}>
                    {priceChartData.map((entry, idx) => (
                      <Cell
                        key={idx}
                        fill={entry.isCheapest ? '#10b981' : entry.isMostExpensive ? '#ef4444' : '#3b82f6'}
                        fillOpacity={entry.isCheapest || entry.isMostExpensive ? 1 : 0.7}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Activity List */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Battery className="w-5 h-5 text-primary" />
                Household Activities
              </CardTitle>
              <CardDescription className="mt-1">
                Select activities to schedule — the optimizer will find the cheapest time slots
              </CardDescription>
            </div>
            <Button onClick={addActivity} variant="outline" size="sm" className="gap-1">
              <Plus className="w-4 h-4" /> Add Activity
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {activities.map(activity => (
              <div
                key={activity.id}
                className={cn(
                  "flex items-center gap-3 p-3 rounded-lg border transition-all",
                  activity.enabled ? "bg-white border-border" : "bg-muted/30 border-transparent opacity-50"
                )}
              >
                <button
                  onClick={() => toggleActivity(activity.id)}
                  className={cn(
                    "w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors",
                    activity.enabled ? "bg-primary border-primary text-white" : "border-muted-foreground/30"
                  )}
                >
                  {activity.enabled && <span className="text-xs">✓</span>}
                </button>

                <span className="text-lg flex-shrink-0">{activity.icon}</span>

                <input
                  value={activity.name}
                  onChange={(e) => updateActivity(activity.id, 'name', e.target.value)}
                  className="flex-1 min-w-0 text-sm font-medium bg-transparent border-none focus:outline-none"
                />

                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Zap className="w-3 h-3" />
                    <input
                      type="number"
                      value={activity.power_kw}
                      onChange={(e) => updateActivity(activity.id, 'power_kw', parseFloat(e.target.value) || 0)}
                      className="w-14 text-xs p-1 rounded border text-center"
                      step="0.1"
                    />
                    <span>kW</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <input
                      type="number"
                      value={activity.duration_hours}
                      onChange={(e) => updateActivity(activity.id, 'duration_hours', parseInt(e.target.value) || 1)}
                      className="w-10 text-xs p-1 rounded border text-center"
                      min="1" max="24"
                    />
                    <span>hrs</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span>⏰</span>
                    <input
                      type="number"
                      value={activity.earliest_start}
                      onChange={(e) => updateActivity(activity.id, 'earliest_start', parseInt(e.target.value) || 0)}
                      className="w-10 text-xs p-1 rounded border text-center"
                      min="0" max="23"
                    />
                    <span>–</span>
                    <input
                      type="number"
                      value={activity.latest_finish}
                      onChange={(e) => updateActivity(activity.id, 'latest_finish', parseInt(e.target.value) || 23)}
                      className="w-10 text-xs p-1 rounded border text-center"
                      min="0" max="23"
                    />
                  </div>
                </div>

                <button
                  onClick={() => removeActivity(activity.id)}
                  className="text-muted-foreground hover:text-destructive transition-colors p-1"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          {/* Optimize Button */}
          <div className="mt-6 flex justify-center">
            <Button
              onClick={handleOptimize}
              disabled={loading || !forecast}
              className="h-12 px-8 text-base gap-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
              id="optimize-button"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Optimizing...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Optimize Schedule
                </>
              )}
            </Button>
          </div>

          {error && (
            <p className="text-sm text-destructive text-center mt-3">{error}</p>
          )}
        </CardContent>
      </Card>

      {/* Optimization Results */}
      {optimizationResult && (
        <div className="space-y-6 opacity-0 animate-fade-in-up">
          {/* Savings Hero Card */}
          <Card className="bg-gradient-to-r from-emerald-500 to-teal-500 text-white border-0">
            <CardContent className="p-8">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                <div>
                  <p className="text-emerald-100 text-sm font-medium mb-1">Estimated Savings</p>
                  <p className="text-5xl font-extrabold tracking-tight">
                    €{optimizationResult.total_savings.toFixed(4)}
                  </p>
                  <p className="text-emerald-200 text-sm mt-1">
                    {optimizationResult.savings_percentage.toFixed(1)}% less than running at earliest time
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-6 text-center">
                  <div>
                    <p className="text-emerald-100 text-xs">Optimized Cost</p>
                    <p className="text-2xl font-bold">€{optimizationResult.total_cost_optimized.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-emerald-100 text-xs">Baseline Cost</p>
                    <p className="text-2xl font-bold">€{optimizationResult.total_cost_baseline.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-emerald-100 text-xs">Solver</p>
                    <p className="text-lg font-bold">{optimizationResult.solver_status}</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Schedule Timeline */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="w-5 h-5 text-primary" />
                Optimal Schedule
              </CardTitle>
              <CardDescription>
                Activities placed at their cheapest available time slots
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Gantt-like timeline */}
              <div className="space-y-3">
                {/* Hour ruler */}
                <div className="flex">
                  <div className="w-32 flex-shrink-0" />
                  <div className="flex-1 flex">
                    {Array.from({ length: 24 }, (_, i) => (
                      <div key={i} className="flex-1 text-center text-[9px] text-muted-foreground border-l border-border/30">
                        {String(i).padStart(2, '0')}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Activity bars */}
                {optimizationResult.schedule.map((item, idx) => (
                  <div key={idx} className="flex items-center">
                    <div className="w-32 flex-shrink-0 text-sm font-medium truncate pr-2 flex items-center gap-1.5">
                      <span>{item.icon}</span>
                      <span className="truncate">{item.name}</span>
                    </div>
                    <div className="flex-1 relative h-8 bg-muted/20 rounded">
                      <div
                        className="absolute top-0 h-full rounded bg-gradient-to-r from-primary to-primary/70 flex items-center justify-center text-white text-[10px] font-bold shadow-sm"
                        style={{
                          left: `${(item.start_hour / 24) * 100}%`,
                          width: `${(item.duration_hours / 24) * 100}%`,
                        }}
                      >
                        {item.start_hour}:00–{item.end_hour}:00
                      </div>
                    </div>
                    <div className="w-20 flex-shrink-0 text-right text-xs tabular-nums">
                      <span className="text-emerald-600 font-medium">€{item.cost_optimized.toFixed(4)}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Cost breakdown table */}
              <div className="mt-6 rounded-lg border overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-muted/30 border-b">
                      <th className="text-left p-3 font-medium">Activity</th>
                      <th className="text-center p-3 font-medium">Time Slot</th>
                      <th className="text-right p-3 font-medium">Optimized</th>
                      <th className="text-right p-3 font-medium">Baseline</th>
                      <th className="text-right p-3 font-medium">Saved</th>
                    </tr>
                  </thead>
                  <tbody>
                    {optimizationResult.schedule.map((item, idx) => (
                      <tr key={idx} className="border-b last:border-0">
                        <td className="p-3 flex items-center gap-2">
                          <span>{item.icon}</span>
                          <span className="font-medium">{item.name}</span>
                        </td>
                        <td className="p-3 text-center tabular-nums">
                          {String(item.start_hour).padStart(2, '0')}:00 → {String(item.end_hour).padStart(2, '0')}:00
                        </td>
                        <td className="p-3 text-right tabular-nums text-emerald-600 font-medium">
                          €{item.cost_optimized.toFixed(4)}
                        </td>
                        <td className="p-3 text-right tabular-nums text-muted-foreground">
                          €{item.cost_baseline.toFixed(4)}
                        </td>
                        <td className="p-3 text-right tabular-nums">
                          {item.savings > 0 ? (
                            <span className="text-emerald-600 font-medium flex items-center justify-end gap-1">
                              <TrendingDown className="w-3 h-3" />
                              €{item.savings.toFixed(4)}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Hourly Power Profile */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                ⚡ Hourly Power Profile
              </CardTitle>
              <CardDescription>
                Combined power consumption from scheduled activities
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={optimizationResult.hourly_power_profile.map((kw, i) => ({
                      hour: `${String(i).padStart(2, '0')}:00`,
                      power: kw,
                      price: optimizationResult.hourly_price[i],
                    }))}
                    margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                    <XAxis dataKey="hour" tick={{ fontSize: 10 }} tickLine={false} interval={1} />
                    <YAxis tick={{ fontSize: 11 }} tickLine={false} label={{ value: 'kW', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null
                        const d = payload[0].payload
                        return (
                          <div className="bg-white/95 backdrop-blur-sm border rounded-lg shadow-xl p-3">
                            <p className="text-sm font-semibold">{d.hour}</p>
                            <p className="text-xs">Power: <span className="font-mono font-bold">{d.power.toFixed(1)} kW</span></p>
                            <p className="text-xs">Price: <span className="font-mono font-bold">€{d.price.toFixed(2)}/MWh</span></p>
                          </div>
                        )
                      }}
                    />
                    <Bar dataKey="power" radius={[4, 4, 0, 0]} maxBarSize={20}>
                      {optimizationResult.hourly_power_profile.map((kw, idx) => (
                        <Cell key={idx} fill={kw > 0 ? '#3b82f6' : '#e5e7eb'} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
