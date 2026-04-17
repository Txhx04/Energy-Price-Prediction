import { useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine, Area
} from 'recharts'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card'
import { cn, getModelColor } from '../lib/utils'

/**
 * PredictiveChart — Unified 24-Hour Visualization
 * 
 * KEY FEATURES:
 * - Two charts (Consumption + Price) BOTH visible simultaneously
 * - syncId="energy-forecast" makes hovering one chart highlight the same hour on the other
 * - Toggle individual models on/off via legend click
 * - X-axis shows Date X+1 timestamps (next-day temporal logic)
 * - Confidence band (±3%) around ensemble prediction
 */
export default function PredictiveChart({ predictionData }) {
  const { models, actuals, hour_labels, prediction_date, hourly_timestamps } = predictionData

  // ── Model visibility state (all on by default) ──
  const [visibleModels, setVisibleModels] = useState(() => {
    const initial = { 'Actual': true }
    models.forEach(m => { initial[m.name] = true })
    return initial
  })

  const handleLegendClick = (entry) => {
    const { value } = entry
    setVisibleModels(prev => ({ ...prev, [value]: !prev[value] }))
  }

  // ── Build chart data: one object per hour with all model predictions ──
  const consumptionData = useMemo(() => {
    return hour_labels.map((label, i) => {
      const point = { hour: label, timestamp: hourly_timestamps[i] }
      point['Actual'] = actuals.consumption[i]
      models.forEach(m => {
        point[m.name] = m.hourly_predictions.consumption[i]
      })
      // Confidence band for Ensemble
      const ensemble = models.find(m => m.name.includes('Ensemble'))
      if (ensemble) {
        point['conf_upper'] = ensemble.hourly_predictions.consumption[i] * 1.03
        point['conf_lower'] = ensemble.hourly_predictions.consumption[i] * 0.97
      }
      return point
    })
  }, [models, actuals, hour_labels, hourly_timestamps])

  const priceData = useMemo(() => {
    return hour_labels.map((label, i) => {
      const point = { hour: label, timestamp: hourly_timestamps[i] }
      point['Actual'] = actuals.price[i]
      models.forEach(m => {
        point[m.name] = m.hourly_predictions.price[i]
      })
      const ensemble = models.find(m => m.name.includes('Ensemble'))
      if (ensemble) {
        point['conf_upper'] = ensemble.hourly_predictions.price[i] * 1.03
        point['conf_lower'] = ensemble.hourly_predictions.price[i] * 0.97
      }
      return point
    })
  }, [models, actuals, hour_labels, hourly_timestamps])

  // ── All line keys (Actual + models) ──
  const lineKeys = useMemo(() => {
    return ['Actual', ...models.map(m => m.name)]
  }, [models])

  // ── Custom Tooltip ──
  const CustomTooltip = ({ active, payload, label, unit }) => {
    if (!active || !payload || !payload.length) return null
    return (
      <div className="bg-white/95 backdrop-blur-sm border rounded-xl shadow-xl p-4 min-w-[200px]">
        <p className="text-sm font-semibold text-foreground mb-2 border-b pb-1.5">
          {prediction_date} {label}
        </p>
        {payload
          .filter(p => p.dataKey !== 'conf_upper' && p.dataKey !== 'conf_lower')
          .map(entry => (
          <div key={entry.dataKey} className="flex items-center justify-between gap-4 py-0.5">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="text-xs text-muted-foreground">{entry.dataKey}</span>
            </div>
            <span className="text-xs font-mono font-semibold">
              {unit === '€/MWh' ? `€${Number(entry.value).toFixed(2)}` : `${Number(entry.value).toLocaleString()} MW`}
            </span>
          </div>
        ))}
      </div>
    )
  }

  // ── Render a single chart ──
  const renderChart = (data, title, yLabel, unit, emoji) => (
    <Card className="opacity-0 animate-fade-in-up">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <span>{emoji}</span>
          {title}
        </CardTitle>
        <CardDescription>
          Forecast for {prediction_date} — Click legend items to toggle models
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[340px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              syncId="energy-forecast"
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <defs>
                <linearGradient id={`conf-${title}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={getModelColor('Ensemble (XGB+GRU+RF)')} stopOpacity={0.15} />
                  <stop offset="100%" stopColor={getModelColor('Ensemble (XGB+GRU+RF)')} stopOpacity={0.02} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.5} />
              <XAxis
                dataKey="hour"
                tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                tickLine={false}
                axisLine={{ stroke: 'hsl(var(--border))' }}
                interval={1}
              />
              <YAxis
                tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                tickLine={false}
                axisLine={{ stroke: 'hsl(var(--border))' }}
                label={{
                  value: yLabel,
                  angle: -90,
                  position: 'insideLeft',
                  style: { fontSize: 12, fill: 'hsl(var(--muted-foreground))' }
                }}
              />
              <Tooltip content={<CustomTooltip unit={unit} />} />
              <Legend
                onClick={handleLegendClick}
                wrapperStyle={{ paddingTop: '10px', cursor: 'pointer' }}
                formatter={(value) => (
                  <span className={cn(
                    "text-xs font-medium",
                    !visibleModels[value] && "line-through opacity-40"
                  )}>
                    {value}
                  </span>
                )}
              />

              {/* Confidence band for Ensemble */}
              {visibleModels['Ensemble (XGB+GRU+RF)'] && (
                <>
                  <Area
                    type="monotone"
                    dataKey="conf_upper"
                    stroke="none"
                    fill={`url(#conf-${title})`}
                    fillOpacity={1}
                    legendType="none"
                  />
                  <Area
                    type="monotone"
                    dataKey="conf_lower"
                    stroke="none"
                    fill="white"
                    fillOpacity={1}
                    legendType="none"
                  />
                </>
              )}

              {/* Peak hour reference lines */}
              <ReferenceLine x="08:00" stroke="hsl(var(--warning))" strokeDasharray="4 4" strokeOpacity={0.4} />
              <ReferenceLine x="20:00" stroke="hsl(var(--warning))" strokeDasharray="4 4" strokeOpacity={0.4} />

              {/* Model lines (dynamic iteration) */}
              {lineKeys.map(key => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={getModelColor(key)}
                  strokeWidth={key === 'Actual' ? 3 : key.includes('Ensemble') ? 2.5 : 1.5}
                  strokeDasharray={key === 'Linear Regression' ? '6 3' : key === 'Actual' ? undefined : undefined}
                  dot={false}
                  activeDot={{ r: key === 'Actual' ? 5 : 4, strokeWidth: 2, fill: 'white' }}
                  hide={!visibleModels[key]}
                  opacity={key === 'Linear Regression' ? 0.6 : 1}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="space-y-6" id="predictive-charts">
      {/* BOTH charts visible simultaneously for synchronized tooltips */}
      {renderChart(consumptionData, '24-Hour Energy Consumption Forecast', 'Consumption (MW)', 'MW', '⚡')}
      {renderChart(priceData, '24-Hour Energy Price Forecast', 'Price (€/MWh)', '€/MWh', '💰')}

      <p className="text-xs text-center text-muted-foreground">
        💡 Hover over one chart to see the synchronized crosshair on both. Click legend items to toggle models.
        Dashed yellow lines mark peak hours (08:00 & 20:00).
      </p>
    </div>
  )
}
