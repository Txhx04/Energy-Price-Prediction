import { useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from './ui/card'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs'
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from './ui/select'
import { Layers, Brain } from 'lucide-react'
import { formatFeatureName, getModelColor } from '../lib/utils'

/**
 * FeatureAnalysis — Feature Importance Visualization
 * 
 * KEY FEATURE: Dynamic model toggle dropdown
 * User can switch between XGBoost importance vs. Linear Regression coefficients
 */
export default function FeatureAnalysis({ featureImportances }) {
  // Available models for feature importance (keys from API response)
  const availableModels = useMemo(() => {
    return Object.keys(featureImportances || {})
  }, [featureImportances])

  const [selectedModel, setSelectedModel] = useState(availableModels[0] || 'XGBoost')

  const currentImportance = featureImportances?.[selectedModel]

  if (!currentImportance) return null

  const renderBarChart = (data, type) => {
    // Sort by importance descending for display
    const sorted = [...data].sort((a, b) => a.importance - b.importance) // ascending for horizontal bars (bottom = highest)

    const maxVal = Math.max(...data.map(d => d.importance))
    const colorBase = type === 'consumption' ? '#3b82f6' : '#f59e0b'

    return (
      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sorted}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" opacity={0.4} />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
              tickLine={false}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              domain={[0, 'auto']}
            />
            <YAxis
              dataKey="name"
              type="category"
              tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
              tickLine={false}
              axisLine={false}
              width={180}
              tickFormatter={formatFeatureName}
            />
            <Tooltip
              cursor={{ fill: 'hsl(var(--muted) / 0.3)' }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const d = payload[0].payload
                return (
                  <div className="bg-white/95 backdrop-blur-sm border rounded-lg shadow-xl p-3 min-w-[200px]">
                    <p className="text-sm font-semibold mb-1">{formatFeatureName(d.name)}</p>
                    <p className="text-xs text-muted-foreground">
                      Importance: <span className="font-mono font-bold text-foreground">{d.importance.toFixed(6)}</span>
                    </p>
                    <p className="text-[10px] text-muted-foreground mt-1">
                      Model: {selectedModel}
                    </p>
                  </div>
                )
              }}
            />
            <Bar dataKey="importance" radius={[0, 6, 6, 0]} maxBarSize={24}>
              {sorted.map((entry, idx) => {
                const ratio = entry.importance / (maxVal || 1)
                const opacity = 0.4 + (ratio * 0.6)
                return (
                  <Cell
                    key={entry.name}
                    fill={colorBase}
                    fillOpacity={opacity}
                  />
                )
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <Card id="feature-analysis" className="opacity-0 animate-fade-in-up stagger-5">
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-primary" />
              Feature Importance Analysis
            </CardTitle>
            <CardDescription className="mt-1">
              Which features had the highest influence on predictions
            </CardDescription>
          </div>

          {/* Model selector dropdown */}
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-muted-foreground" />
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-[220px]" id="feature-model-select">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {availableModels.map(name => (
                  <SelectItem key={name} value={name}>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: getModelColor(name) }} />
                      {name}
                      <span className="text-[10px] text-muted-foreground ml-1">
                        {name === 'XGBoost' ? '(Gain)' : '(|Coeff|)'}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="consumption">
          <TabsList className="mb-4">
            <TabsTrigger value="consumption">⚡ Consumption Features</TabsTrigger>
            <TabsTrigger value="price">💰 Price Features</TabsTrigger>
          </TabsList>
          <TabsContent value="consumption">
            {renderBarChart(currentImportance.consumption, 'consumption')}
          </TabsContent>
          <TabsContent value="price">
            {renderBarChart(currentImportance.price, 'price')}
          </TabsContent>
        </Tabs>

        <p className="text-xs text-muted-foreground mt-2">
          {selectedModel === 'XGBoost'
            ? 'Feature importance based on average gain across all trees (h+1 target estimator).'
            : 'Feature importance based on normalized absolute coefficient values.'}
        </p>
      </CardContent>
    </Card>
  )
}
