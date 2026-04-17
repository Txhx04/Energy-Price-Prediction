import { Zap, Activity, Database } from 'lucide-react'

export default function Header() {
  return (
    <header className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#0a1628] via-[#0f2847] to-[#0c3b6e] p-8 mb-8">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-4 -right-4 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl animate-pulse-subtle" />
        <div className="absolute -bottom-8 -left-8 w-96 h-96 bg-cyan-500/8 rounded-full blur-3xl animate-pulse-subtle" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-400/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          {/* Animated lightning icon */}
          <div className="flex-shrink-0 w-14 h-14 rounded-xl bg-gradient-to-br from-yellow-400 to-amber-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
            <Zap className="w-7 h-7 text-white" fill="currentColor" />
          </div>

          <div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Spain Energy Prediction System
            </h1>
            <p className="text-blue-200/70 mt-1.5 text-sm lg:text-base max-w-2xl">
              24-Hour Electricity Consumption & Price Forecasting — Powered by XGBoost, GRU & Ensemble Models
            </p>
          </div>
        </div>

        {/* Status indicators */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-lg px-3 py-2 border border-white/10">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span className="text-xs text-white/80 font-medium">Live Models</span>
          </div>
          <div className="flex items-center gap-2 bg-white/10 backdrop-blur-sm rounded-lg px-3 py-2 border border-white/10">
            <Database className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-white/80 font-medium">Spain Grid 2015–2018</span>
          </div>
        </div>
      </div>
    </header>
  )
}
