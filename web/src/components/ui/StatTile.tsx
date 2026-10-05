import type { ReactNode } from "react";

interface StatTileProps {
  label: string;
  value: ReactNode;
  hint?: string;
  icon?: ReactNode;
  /** Optional accent stripe colour, e.g. a series or status token. */
  accent?: string;
}

/**
 * A single headline number. Per the form heuristic this is deliberately not a
 * chart: one value has no shape to show.
 */
export function StatTile({ label, value, hint, icon, accent }: StatTileProps) {
  return (
    <div
      className="relative flex min-w-0 flex-col justify-between overflow-hidden rounded-card border border-line bg-surface p-4"
      style={{ boxShadow: "var(--shadow-card)" }}
    >
      {accent && (
        <span aria-hidden className="absolute inset-x-0 top-0 h-0.5" style={{ background: accent }} />
      )}
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-medium uppercase tracking-wide text-ink-muted">
          {label}
        </span>
        {icon && <span className="shrink-0 text-ink-muted">{icon}</span>}
      </div>
      <div className="tnum mt-2 text-2xl font-semibold leading-none">{value}</div>
      {hint && <p className="mt-1.5 truncate text-xs text-ink-secondary">{hint}</p>}
    </div>
  );
}
