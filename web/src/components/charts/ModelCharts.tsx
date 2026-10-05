import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SERIES } from "../../lib/colors";
import { num, score } from "../../lib/format";
import type { Curve, ModelMetrics, TrainingHistory } from "../../lib/types";
import { AXIS, GRID, Legend, TooltipRow, TooltipShell } from "./chartBits";

/** Train vs validation F1 across epochs — the overfitting read. */
export function TrainingCurves({ history }: { history: TrainingHistory }) {
  const data = history.epoch.map((epoch, i) => ({
    epoch,
    train: history.train_f1[i],
    val: history.val_f1[i],
  }));

  return (
    <div>
      <Legend
        items={[
          { label: "Train F1", color: SERIES.one },
          { label: "Validation F1", color: SERIES.two },
        ]}
        className="mb-3"
      />
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 4, right: 10, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID} vertical={false} />
          <XAxis
            dataKey="epoch"
            {...AXIS}
            label={{
              value: "Epoch",
              position: "insideBottom",
              offset: -2,
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <YAxis {...AXIS} domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipShell title={`Epoch ${label}`}>
                  {payload.map((p) => (
                    <TooltipRow
                      key={p.dataKey as string}
                      color={p.color}
                      label={String(p.name)}
                      value={score(p.value as number)}
                    />
                  ))}
                </TooltipShell>
              ) : null
            }
          />
          <Line
            isAnimationActive={false}
            type="monotone"
            dataKey="train"
            name="Train F1"
            stroke={SERIES.one}
            strokeWidth={2}
            dot={false}
          />
          <Line
            isAnimationActive={false}
            type="monotone"
            dataKey="val"
            name="Validation F1"
            stroke={SERIES.two}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface PRProps {
  curves: { label: string; curve: Curve; color: string }[];
}

/** Precision-recall, the honest curve for a 1:9 imbalanced problem. */
export function PRCurveChart({ curves }: PRProps) {
  // Recharts wants one row per x; merge the curves onto a shared recall grid.
  const grid = Array.from({ length: 101 }, (_, i) => i / 100);
  const data = grid.map((recall) => {
    const row: Record<string, number> = { recall };
    curves.forEach(({ label, curve }) => {
      // `x` is recall descending from the sklearn helper; find the nearest point.
      let best = 0;
      let bestDist = Infinity;
      curve.x.forEach((x, i) => {
        const d = Math.abs(x - recall);
        if (d < bestDist) {
          bestDist = d;
          best = curve.y[i];
        }
      });
      row[label] = best;
    });
    return row;
  });

  return (
    <div>
      <Legend items={curves.map((c) => ({ label: c.label, color: c.color }))} className="mb-3" />
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 4, right: 10, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis
            dataKey="recall"
            {...AXIS}
            domain={[0, 1]}
            type="number"
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "Recall",
              position: "insideBottom",
              offset: -2,
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <YAxis
            {...AXIS}
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "Precision",
              angle: -90,
              position: "insideLeft",
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipShell title={`Recall ${Number(label).toFixed(2)}`}>
                  {payload.map((p) => (
                    <TooltipRow
                      key={p.dataKey as string}
                      color={p.color}
                      label={String(p.name)}
                      value={score(p.value as number, 3)}
                    />
                  ))}
                </TooltipShell>
              ) : null
            }
          />
          {curves.map(({ label, color }) => (
            <Line
              key={label}
              isAnimationActive={false}
              type="monotone"
              dataKey={label}
              name={label}
              stroke={color}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Confusion matrix as a 2x2 of cells.
 *
 * Not a heatmap ramp: with four cells the numbers are the point, and a colour
 * scale over four values adds nothing a reader can use. False negatives get the
 * emphasis because a missed fraud is the expensive error here.
 */
export function ConfusionMatrix({ metrics }: { metrics: ModelMetrics }) {
  const cm = metrics.confusion_matrix;
  if (!cm) return <p className="text-xs text-ink-secondary">No confusion matrix recorded.</p>;

  const [[tn, fp], [fn, tp]] = cm;
  const cells = [
    { label: "True negative", value: tn, hint: "licit, called licit", emphasis: false },
    { label: "False positive", value: fp, hint: "licit, called fraud", emphasis: false },
    { label: "False negative", value: fn, hint: "fraud, missed", emphasis: true },
    { label: "True positive", value: tp, hint: "fraud, caught", emphasis: false },
  ];

  return (
    <div>
      <div className="grid grid-cols-2 gap-2">
        {cells.map((cell) => (
          <div
            key={cell.label}
            className="rounded-md border p-3"
            style={{
              borderColor: cell.emphasis ? "var(--status-warning)" : "var(--line)",
              background: cell.emphasis
                ? "color-mix(in srgb, var(--status-warning) 8%, transparent)"
                : "transparent",
            }}
          >
            <p className="text-xs text-ink-muted">{cell.label}</p>
            <p className="tnum mt-1 text-xl font-semibold">{num(cell.value)}</p>
            <p className="mt-0.5 text-xs text-ink-secondary">{cell.hint}</p>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-ink-secondary">
        False negatives are highlighted: the class weighting exists to make a missed fraud
        costlier than a false alarm.
      </p>
    </div>
  );
}
