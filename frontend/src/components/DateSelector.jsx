import { useState } from 'react'
import { format, parseISO } from 'date-fns'
import { CalendarIcon, Search, AlertCircle } from 'lucide-react'
import { Button } from './ui/button'
import { Calendar } from './ui/calendar'
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover'
import { cn } from '../lib/utils'

export default function DateSelector({
  availableDates,
  onDateSelect,
  loading,
  error,
  onClearError
}) {
  const [date, setDate] = useState(null)
  const [open, setOpen] = useState(false)

  const minDate = availableDates?.min_date ? parseISO(availableDates.min_date) : null
  const maxDate = availableDates?.max_date ? parseISO(availableDates.max_date) : null

  // Build a Set of available dates for fast lookup
  const availableDateSet = new Set(availableDates?.available_dates || [])

  const handleSelect = (selectedDay) => {
    setDate(selectedDay)
    setOpen(false)
  }

  const handlePredict = () => {
    if (date) {
      const dateStr = format(date, 'yyyy-MM-dd')
      if (!availableDateSet.has(dateStr)) {
        return // Date not in dataset
      }
      onDateSelect(dateStr)
    }
  }

  // Disable dates not in the dataset
  const isDateDisabled = (day) => {
    const dateStr = format(day, 'yyyy-MM-dd')
    return !availableDateSet.has(dateStr)
  }

  const selectedDateStr = date ? format(date, 'yyyy-MM-dd') : null
  const isDateValid = selectedDateStr && availableDateSet.has(selectedDateStr)

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-end">
        {/* Calendar Date Picker (Shadcn Popover + Calendar) */}
        <div className="flex-1 space-y-2">
          <label className="text-sm font-medium text-foreground">
            Select Prediction Date
          </label>
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full sm:w-[280px] justify-start text-left font-normal h-11",
                  !date && "text-muted-foreground"
                )}
                id="date-picker-trigger"
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {date ? format(date, "MMMM d, yyyy") : "Pick a date..."}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={date}
                onSelect={handleSelect}
                disabled={isDateDisabled}
                defaultMonth={maxDate || undefined}
                fromDate={minDate || undefined}
                toDate={maxDate || undefined}
              />
              <div className="px-3 pb-3">
                <p className="text-xs text-muted-foreground text-center">
                  Data coverage: {availableDates?.min_date} to {availableDates?.max_date}
                </p>
              </div>
            </PopoverContent>
          </Popover>
        </div>

        {/* Predict Button */}
        <Button
          onClick={handlePredict}
          disabled={!isDateValid || loading}
          className="h-11 px-6"
          id="predict-button"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Predicting...
            </>
          ) : (
            <>
              <Search className="w-4 h-4" />
              Generate Forecast
            </>
          )}
        </Button>
      </div>

      {/* Next-Day Indicator */}
      {date && isDateValid && (
        <div className="text-sm text-muted-foreground bg-secondary/50 rounded-lg px-4 py-2.5 border border-primary/10">
          <span className="font-medium text-primary">📊 Forecast Target:</span>{' '}
          Predictions will cover 24 hours of{' '}
          <span className="font-semibold text-foreground">
            {format(new Date(date.getTime() + 86400000), 'MMMM d, yyyy')}
          </span>{' '}
          (next day)
        </div>
      )}

      {/* Error State: "No Data for this Period" */}
      {error && (
        <div className="flex items-start gap-3 bg-destructive/10 text-destructive rounded-lg px-4 py-3 border border-destructive/20"
             id="error-message">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium">No Data for This Period</p>
            <p className="text-xs mt-0.5 opacity-80">{error}</p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClearError} className="text-destructive hover:text-destructive">
            Dismiss
          </Button>
        </div>
      )}
    </div>
  )
}
