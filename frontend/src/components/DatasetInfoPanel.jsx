import { Database, Calendar, Cpu, BarChart3 } from 'lucide-react'
import { Card, CardContent } from './ui/card'

export default function DatasetInfoPanel({ datasetInfo }) {
  if (!datasetInfo) return null

  const items = [
    {
      icon: BarChart3,
      label: 'Features',
      value: datasetInfo.total_features || 82,
      color: 'text-blue-500 bg-blue-50',
    },
    {
      icon: Calendar,
      label: 'Train Period',
      value: datasetInfo.train_period || '2015–2024',
      color: 'text-emerald-500 bg-emerald-50',
    },
    {
      icon: Calendar,
      label: 'Test Period',
      value: datasetInfo.test_period || '2026',
      color: 'text-purple-500 bg-purple-50',
    },
    {
      icon: Cpu,
      label: 'Models',
      value: '4 Active',
      color: 'text-amber-500 bg-amber-50',
    },
  ]

  return (
    <Card className="opacity-0 animate-fade-in-up">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Dataset Overview</h3>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {items.map((item) => {
            const Icon = item.icon
            return (
              <div key={item.label} className="flex items-center gap-3 p-2 rounded-lg bg-muted/30">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${item.color}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div>
                  <p className="text-[11px] text-muted-foreground">{item.label}</p>
                  <p className="text-sm font-semibold">{item.value}</p>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
