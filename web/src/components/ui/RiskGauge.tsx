import { riskBand, riskColorVar } from "../../lib/colors";
import { score as fmtScore } from "../../lib/format";

interface RiskGaugeProps {
  value: number;
  decision?: number;
  highRisk?: number;
  size?: number;
}

/**
 * Radial meter for one node's fraud probability.
 *
 * The arc is filled with the single-hue risk ramp step for the value, and the
 * decision and high-risk thresholds are drawn as ticks so the number has context
 * — a 0.62 means little until you can see it sits just past the 0.5 line.
 */
export function RiskGauge({ value, decision = 0.5, highRisk = 0.7, size = 132 }: RiskGaugeProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const stroke = 10;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;

  // 270° sweep starting at the lower-left, so the gap sits at the bottom.
  const sweep = 270;
  const start = 135;
  const circumference = 2 * Math.PI * r;
  const arcLength = (sweep / 360) * circumference;

  const tick = (t: number) => {
    const angle = ((start + t * sweep) * Math.PI) / 180;
    const inner = r - stroke / 2 - 2;
    const outer = r + stroke / 2 + 2;
    return {
      x1: cx + inner * Math.cos(angle),
      y1: cy + inner * Math.sin(angle),
      x2: cx + outer * Math.cos(angle),
      y2: cy + outer * Math.sin(angle),
    };
  };

  return (
    <div className="flex items-center gap-4">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`Fraud score ${fmtScore(clamped)} of 1`}
      >
        <g transform={`rotate(${start} ${cx} ${cy})`}>
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="var(--line)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${circumference}`}
          />
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={riskColorVar(clamped)}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${arcLength * clamped} ${circumference}`}
          />
        </g>
        {[decision, highRisk].map((t) => {
          const { x1, y1, x2, y2 } = tick(t);
          return (
            <line
              key={t}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="var(--text-muted)"
              strokeWidth={1.5}
            />
          );
        })}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          className="tnum"
          fill="var(--text-primary)"
          style={{ fontSize: 24, fontWeight: 600 }}
        >
          {clamped.toFixed(2)}
        </text>
        <text
          x={cx}
          y={cy + 16}
          textAnchor="middle"
          fill="var(--text-muted)"
          style={{ fontSize: 10, letterSpacing: "0.06em" }}
        >
          P(ILLICIT)
        </text>
      </svg>

      <div className="min-w-0">
        <p className="text-sm font-semibold">{riskBand(clamped, highRisk, decision)}</p>
        <dl className="mt-2 space-y-1 text-xs text-ink-secondary">
          <div className="flex gap-2">
            <dt>Decision</dt>
            <dd className="tnum">{decision.toFixed(2)}</dd>
          </div>
          <div className="flex gap-2">
            <dt>High risk</dt>
            <dd className="tnum">{highRisk.toFixed(2)}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
