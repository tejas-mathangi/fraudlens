import type { ReactNode } from "react";

/** Shared chart furniture: recessive axes, a real legend, and a tooltip shell. */

export const AXIS = {
  stroke: "var(--line)",
  tick: { fill: "var(--text-muted)", fontSize: 11 },
} as const;

export const GRID = {
  stroke: "var(--line)",
  strokeDasharray: "2 4",
} as const;

export interface LegendItem {
  label: string;
  color: string;
}

/**
 * A legend is always present for two or more series, so identity never rests on
 * colour alone.
 */
export function Legend({ items, className = "" }: { items: LegendItem[]; className?: string }) {
  return (
    <ul className={`flex flex-wrap items-center gap-x-4 gap-y-1 ${className}`}>
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5 text-xs text-ink-secondary">
          <span
            aria-hidden
            className="h-2 w-2 shrink-0 rounded-sm"
            style={{ background: item.color }}
          />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

export function TooltipShell({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div
      className="rounded-md border border-line bg-raised px-2.5 py-2 text-xs"
      style={{ boxShadow: "var(--shadow-card)" }}
    >
      <p className="mb-1 font-medium text-ink">{title}</p>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

export function TooltipRow({
  color,
  label,
  value,
}: {
  color?: string;
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-center gap-2">
      {color && (
        <span aria-hidden className="h-2 w-2 shrink-0 rounded-sm" style={{ background: color }} />
      )}
      <span className="text-ink-secondary">{label}</span>
      <span className="tnum ml-auto font-medium text-ink">{value}</span>
    </div>
  );
}
