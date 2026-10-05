/**
 * Colour helpers.
 *
 * Risk is a continuous 0..1 magnitude, so it gets a single-hue ramp (blue, light
 * to dark) rather than a green-to-red rainbow. Categorical identity in charts uses
 * slots 1 and 2; the validator rejected illicit=red / licit=green outright
 * (deuteranopia ΔE 4.1 — the two are the same colour to a red-green colourblind
 * reader).
 */

export const RISK_STEPS = [
  "var(--risk-100)",
  "var(--risk-250)",
  "var(--risk-400)",
  "var(--risk-550)",
  "var(--risk-700)",
] as const;

/** Resolve a CSS custom property to a concrete colour, for canvas drawing. */
export function cssVar(name: string, fallback = "#888888"): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** Bucket a score into one of the five ramp steps. */
export function riskStep(score: number): number {
  const s = Math.max(0, Math.min(1, score));
  return Math.min(RISK_STEPS.length - 1, Math.floor(s * RISK_STEPS.length));
}

export function riskColorVar(score: number): string {
  return RISK_STEPS[riskStep(score)];
}

/** Concrete hex for canvas, which cannot read CSS variables. */
export function riskColorHex(score: number): string {
  const names = ["--risk-100", "--risk-250", "--risk-400", "--risk-550", "--risk-700"];
  return cssVar(names[riskStep(score)], "#3987e5");
}

export const SERIES = {
  /** Charts: slot 1. Also "licit" and "Random Forest". */
  one: "var(--series-1)",
  /** Charts: slot 2. Also "illicit" and "GraphSAGE". */
  two: "var(--series-2)",
  unknown: "var(--unknown)",
} as const;

/** Series colour per true label, for charts with 2–3 label series. */
export const LABEL_SERIES: Record<string, string> = {
  illicit: SERIES.two,
  licit: SERIES.one,
  unknown: SERIES.unknown,
};

/**
 * Badge colours. Safe to use red/green here because a badge always renders its
 * text label, so colour is never the only channel carrying the meaning.
 */
export const LABEL_BADGE: Record<string, { fg: string; bg: string }> = {
  illicit: { fg: "var(--status-critical)", bg: "color-mix(in srgb, var(--status-critical) 14%, transparent)" },
  licit: { fg: "var(--status-good)", bg: "color-mix(in srgb, var(--status-good) 14%, transparent)" },
  unknown: { fg: "var(--text-muted)", bg: "color-mix(in srgb, var(--text-muted) 14%, transparent)" },
};

export function riskBand(score: number, highRisk = 0.7, decision = 0.5): string {
  if (score >= highRisk) return "High risk";
  if (score >= decision) return "Flagged";
  if (score >= 0.2) return "Low risk";
  return "Clean";
}
