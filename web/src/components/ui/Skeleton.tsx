import type { CSSProperties } from "react";

export function Skeleton({
  className = "",
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={`animate-pulse rounded bg-line/60 ${className}`}
      style={style}
      role="status"
      aria-label="Loading"
    />
  );
}

export function PanelSkeleton({ height = 240 }: { height?: number }) {
  return (
    <div className="space-y-3 p-4">
      <Skeleton className="h-3 w-1/3" />
      <Skeleton style={{ height }} className="w-full" />
    </div>
  );
}

export function StatRowSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} className="h-[92px]" />
      ))}
    </div>
  );
}
