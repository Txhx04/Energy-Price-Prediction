import { useState, useMemo } from 'react'
import { Trophy, ArrowUpDown, ChevronUp, ChevronDown } from 'lucide-react'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from './ui/table'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs'
import { Badge } from './ui/badge'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { cn, getModelColor } from '../lib/utils'

/**
 * Model comparison table that dynamically iterates models[].
 * Sorts by selectable metric, highlights the winner.
 * NEVER hardcodes model names.
 */
export default function ModelComparisonTable({ models, actuals }) {
  const [sortMetric, setSortMetric] = useState('rmse')
  const [sortAsc, setSortAsc] = useState(true)

  const handleSort = (metric) => {
    if (sortMetric === metric) {
      setSortAsc(prev => !prev)
    } else {
      setSortMetric(metric)
      // For R², higher is better → default descending
      setSortAsc(metric !== 'r_squared')
    }
  }

  const renderTable = (type) => {
    // Sort models dynamically
    const sorted = [...models].sort((a, b) => {
      const aVal = a.metrics[type][sortMetric]
      const bVal = b.metrics[type][sortMetric]
      return sortAsc ? aVal - bVal : bVal - aVal
    })

    // Winner determination:
    // For MAE, RMSE, MAPE → lowest wins
    // For R² → highest wins
    const bestModel = sorted[0]?.name

    const metrics = ['mae', 'rmse', 'r_squared', 'mape']
    const metricLabels = {
      mae: 'MAE',
      rmse: 'RMSE',
      r_squared: 'R²',
      mape: 'MAPE (%)',
    }

    const formatValue = (metric, value) => {
      if (metric === 'r_squared') return value.toFixed(4)
      if (metric === 'mape') return `${value.toFixed(2)}%`
      if (type === 'price') return `€${value.toFixed(2)}`
      return value.toFixed(2)
    }

    // Find best value per metric for highlighting
    const bestPerMetric = {}
    metrics.forEach(m => {
      const values = models.map(model => model.metrics[type][m])
      bestPerMetric[m] = m === 'r_squared'
        ? Math.max(...values)
        : Math.min(...values)
    })

    return (
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/30">
            <TableHead className="w-12">#</TableHead>
            <TableHead>Model</TableHead>
            {metrics.map(m => (
              <TableHead
                key={m}
                className="cursor-pointer hover:text-foreground select-none"
                onClick={() => handleSort(m)}
              >
                <div className="flex items-center gap-1">
                  {metricLabels[m]}
                  {sortMetric === m ? (
                    sortAsc ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ArrowUpDown className="w-3 h-3 opacity-30" />
                  )}
                </div>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((model, idx) => {
            const isWinner = idx === 0
            const color = getModelColor(model.name)
            return (
              <TableRow
                key={model.name}
                className={cn(
                  isWinner && "bg-emerald-50/50 border-l-2 border-l-emerald-500"
                )}
                id={`comparison-row-${model.name.replace(/\s+/g, '-').toLowerCase()}`}
              >
                <TableCell className="font-mono text-muted-foreground">
                  {isWinner ? (
                    <Trophy className="w-4 h-4 text-amber-500" />
                  ) : (
                    <span className="pl-0.5">{idx + 1}</span>
                  )}
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                    <span className="font-medium">{model.name}</span>
                    {isWinner && <Badge variant="success" className="text-[10px] py-0">Winner</Badge>}
                  </div>
                </TableCell>
                {metrics.map(m => {
                  const val = model.metrics[type][m]
                  const isBest = Math.abs(val - bestPerMetric[m]) < 0.0001
                  return (
                    <TableCell
                      key={m}
                      className={cn(
                        "tabular-nums",
                        isBest && "font-bold text-emerald-700"
                      )}
                    >
                      {formatValue(m, val)}
                    </TableCell>
                  )
                })}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    )
  }

  return (
    <Card id="model-comparison-table">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trophy className="w-5 h-5 text-amber-500" />
          Model Comparison
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="consumption">
          <TabsList className="mb-4">
            <TabsTrigger value="consumption">⚡ Consumption</TabsTrigger>
            <TabsTrigger value="price">💰 Price</TabsTrigger>
          </TabsList>
          <TabsContent value="consumption">
            {renderTable('consumption')}
            <p className="text-xs text-muted-foreground mt-3">
              <span className="text-emerald-600 font-semibold">Green values</span> = best per metric. Click column headers to sort.
            </p>
          </TabsContent>
          <TabsContent value="price">
            {renderTable('price')}
            <p className="text-xs text-muted-foreground mt-3">
              <span className="text-emerald-600 font-semibold">Green values</span> = best per metric. Click column headers to sort.
            </p>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
