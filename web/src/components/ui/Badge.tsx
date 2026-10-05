import type { ReactNode } from "react";

import { LABEL_BADGE } from "../../lib/colors";
import { titleCase } from "../../lib/format";

interface BadgeProps {
  children: ReactNode;
  tone?: "neutral" | "good" | "warning" | "critical" | "accent";
  className?: string;
}

const TONES: Record<string, { fg: string; bg: string }> = {
  neutral: { fg: "var(--text-secondary)", bg: "color-mix(in srgb, var(--text-secondary) 12%, transparent)" },
  good: { fg: "var(--status-good)", bg: "color-mix(in srgb, var(--status-good) 14%, transparent)" },
  warning: { fg: "var(--status-warning)", bg: "color-mix(in srgb, var(--status-warning) 18%, transparent)" },
  critical: { fg: "var(--status-critical)", bg: "color-mix(in srgb, var(--status-critical) 14%, transparent)" },
  accent: { fg: "var(--accent)", bg: "color-mix(in srgb, var(--accent) 14%, transparent)" },
};

export function Badge({ children, tone = "neutral", className = "" }: BadgeProps) {
  const { fg, bg } = TONES[tone];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${className}`}
      style={{ color: fg, background: bg }}
    >
      {children}
    </span>
  );
}

/**
 * True-label pill. The text is always rendered, so the colour is reinforcement
 * rather than the sole carrier of meaning — which is what makes red/green
 * acceptable here but not in the charts.
 */
export function LabelBadge({ label }: { label: string }) {
  const { fg, bg } = LABEL_BADGE[label] ?? LABEL_BADGE.unknown;
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ color: fg, background: bg }}
    >
      {titleCase(label)}
    </span>
  );
}
