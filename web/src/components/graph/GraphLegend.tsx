import { RISK_STEPS } from "../../lib/colors";

/**
 * Legend for the graph canvas: the risk ramp plus what size means.
 *
 * Explicit about the fact that colour here is the model's score, not the ground
 * truth label — those are different things and conflating them is the easiest way
 * to misread the picture.
 */
export function GraphLegend({ weighted = false }: { weighted?: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-ink-secondary">
      <div className="flex items-center gap-2">
        <span>Fraud score</span>
        <span className="flex overflow-hidden rounded-sm" aria-hidden>
          {RISK_STEPS.map((step) => (
            <span key={step} className="h-2.5 w-5" style={{ background: step }} />
          ))}
        </span>
        <span className="tnum text-ink-muted">0 → 1</span>
      </div>

      <div className="flex items-center gap-1.5">
        <svg width="26" height="12" aria-hidden>
          <circle cx="4" cy="6" r="2.5" fill="var(--text-muted)" />
          <circle cx="18" cy="6" r="5" fill="var(--text-muted)" />
        </svg>
        <span>Size = degree</span>
      </div>

      {weighted && <span>Edge width = influence on the prediction</span>}

      <span className="text-ink-muted">
        Colour is the model&apos;s score, not the true label.
      </span>
    </div>
  );
}
