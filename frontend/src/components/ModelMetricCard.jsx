import { TrendingDown, TrendingUp, Trophy, BarChart3, Target, Gauge } from 'lucide-react'
import { Card, CardContent } from './ui/card'
import { Badge } from './ui/badge'
import { cn, getModelColor } from '../lib/utils'

/**
 * Individual model metric card.
 * Receives a single model object from the models[] array.
 * Highlights the "Best Model" if this model has the best RMSE.
 */
export default function ModelMetricCard({ model, type = 'consumption', isBest = false, rank = 0 }) {
  const metrics = model.metrics[type]
  const color = getModelColor(model.name)

  const metricItems = [
    {
      label: 'MAE',
      value: metrics.mae,
      format: (v) => type === 'price' ? `€${v.toFixed(2)}` : v.toFixed(0),
      icon: Target,
      description: 'Mean Absolute Error',
    },
    {
      label: 'RMSE',
      value: metrics.rmse,
      format: (v) => type === 'price' ? `€${v.toFixed(2)}` : v.toFixed(0),
      icon: BarChart3,
      description: 'Root Mean Squared Error',
    },
    {
      label: 'R²',
      value: metrics.r_squared,
      format: (v) => v.toFixed(4),
      icon: Gauge,
      description: 'Coefficient of Determination',
    },
    {
      label: 'MAPE',
      value: metrics.mape,
      format: (v) => `${v.toFixed(2)}%`,
      icon: TrendingDown,
      description: 'Mean Absolute % Error',
    },
  ]

  return (
    <Card
      className={cn(
        "relative overflow-hidden opacity-0 animate-fade-in-up",
        isBest && "ring-2 ring-emerald-500/40 border-emerald-200",
        `stagger-${rank + 1}`
      )}
      id={`metric-card-${model.name.replace(/\s+/g, '-').toLowerCase()}`}
    >
      {/* Color accent strip */}
      <div className="h-1.5 w-full" style={{ background: `linear-gradient(90deg, ${color}, ${color}88)` }} />

      {/* Best Model Badge */}
      {isBest && (
        <div className="absolute top-4 right-4">
          <Badge variant="success" className="gap-1 shadow-sm">
            <Trophy className="w-3 h-3" />
            Best
          </Badge>
        </div>
      )}

      <CardContent className="p-5">
        {/* Model name */}
        <div className="flex items-center gap-2 mb-4">
          <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
          <h3 className="font-semibold text-sm text-foreground truncate">{model.name}</h3>
          <span className="text-[10px] text-muted-foreground bg-muted rounded px-1.5 py-0.5 uppercase tracking-wider">
            {model.type}
          </span>
        </div>

        {/* Metric grid */}
        <div className="grid grid-cols-2 gap-3">
          {metricItems.map((item) => {
            const Icon = item.icon
            return (
              <div key={item.label} className="space-y-1">
                <div className="flex items-center gap-1">
                  <Icon className="w-3 h-3 text-muted-foreground" />
                  <span className="text-[11px] text-muted-foreground font-medium">{item.label}</span>
                </div>
                <p className="text-lg font-bold text-foreground tabular-nums leading-none">
                  {item.format(item.value)}
                </p>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
