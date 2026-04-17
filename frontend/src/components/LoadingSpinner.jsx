import { Skeleton } from './ui/skeleton'
import { Card, CardContent, CardHeader } from './ui/card'

/**
 * Skeleton loading state — shown while API fetch is in progress.
 * Mimics the actual layout structure for a smooth visual transition.
 */
export default function LoadingSpinner() {
  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Metric cards skeleton */}
      <div>
        <Skeleton className="h-5 w-48 mb-4" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <Card key={i}>
              <div className="h-1.5 skeleton-shimmer" />
              <CardContent className="p-5 space-y-4">
                <div className="flex items-center gap-2">
                  <Skeleton className="w-3 h-3 rounded-full" />
                  <Skeleton className="h-4 w-24" />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  {[1, 2, 3, 4].map(j => (
                    <div key={j} className="space-y-1.5">
                      <Skeleton className="h-3 w-12" />
                      <Skeleton className="h-6 w-16" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Charts skeleton */}
      <div className="space-y-6">
        {[1, 2].map(i => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <Skeleton className="h-5 w-64" />
              <Skeleton className="h-3 w-48 mt-1" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-[340px] w-full rounded-lg" />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Table skeleton */}
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-8 w-48 mb-4 rounded-lg" />
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map(i => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Feature importance skeleton */}
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-56" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[400px] w-full rounded-lg" />
        </CardContent>
      </Card>
    </div>
  )
}
