import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SERIES } from "../../lib/colors";
import { num, pct } from "../../lib/format";
import type { ScoreHistogram as Histogram } from "../../lib/types";
import { AXIS, GRID, Legend, TooltipRow, TooltipShell } from "./chartBits";

interface Props {
  histogram: Histogram;
  decisionThreshold?: number;
  /** `unknown` dwarfs the labelled classes, so it is opt-in. */
  includeUnknown?: boolean;
  /** Illicit share of the dataset, used in the caption. Derived, never hardcoded. */
  illicitShare?: number;
}

/**
 * Where the model puts labelled transactions on the 0..1 score axis.
 *
 * Grouped rather than stacked: the question is "are the two classes separated?",
 * and a stack hides exactly that. A log y-axis because illicit is 2.2% of the
 * data — on a linear axis its bars are invisible next to licit.
 */
export function ScoreHistogramChart({
  histogram,
  decisionThreshold = 0.5,
  includeUnknown = false,
  illicitShare,
}: Props) {
  const data = histogram.bin_centers.map((center, i) => ({
    center,
    illicit: histogram.illicit[i] ?? 0,
    licit: histogram.licit[i] ?? 0,
    unknown: histogram.unknown?.[i] ?? 0,
  }));

  const items = [
    { label: "Illicit (confirmed)", color: SERIES.two },
    { label: "Licit (confirmed)", color: SERIES.one },
    ...(includeUnknown ? [{ label: "Unlabelled", color: SERIES.unknown }] : []),
  ];

  return (
    <div>
      <Legend items={items} className="mb-3" />
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }} barGap={2}>
          <CartesianGrid {...GRID} vertical={false} />
          <XAxis
            dataKey="center"
            {...AXIS}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "Fraud score",
              position: "insideBottom",
              offset: -2,
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <YAxis {...AXIS} scale="log" domain={[1, "auto"]} allowDataOverflow tickFormatter={num} />
          <ReferenceLine
            x={decisionThreshold}
            stroke="var(--text-muted)"
            strokeDasharray="3 3"
            label={{
              value: "decision",
              position: "top",
              fill: "var(--text-muted)",
              fontSize: 10,
            }}
          />
          <Tooltip
            cursor={{ fill: "var(--line)", opacity: 0.3 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipShell title={`Score ≈ ${Number(label).toFixed(2)}`}>
                  {payload.map((p) => (
                    <TooltipRow
                      key={p.dataKey as string}
                      color={p.color}
                      label={String(p.name)}
                      value={num(p.value as number)}
                    />
                  ))}
                </TooltipShell>
              ) : null
            }
          />
          {includeUnknown && (
            <Bar isAnimationActive={false} dataKey="unknown" name="Unlabelled" fill={SERIES.unknown} radius={[3, 3, 0, 0]} />
          )}
          <Bar isAnimationActive={false} dataKey="licit" name="Licit" fill={SERIES.one} radius={[3, 3, 0, 0]} />
          <Bar isAnimationActive={false} dataKey="illicit" name="Illicit" fill={SERIES.two} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-ink-secondary">
        Log scale: confirmed illicit transactions are
        {illicitShare !== undefined ? ` ${pct(illicitShare)} of` : " a small fraction of"} this
        dataset and would be invisible on a linear axis.
      </p>
    </div>
  );
}
