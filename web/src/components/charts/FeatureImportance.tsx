import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { SERIES } from "../../lib/colors";
import { score } from "../../lib/format";
import type { FeatureImportance } from "../../lib/types";
import { AXIS, GRID, Legend, TooltipRow, TooltipShell } from "./chartBits";

/**
 * Which input features the prediction is most sensitive to.
 *
 * Horizontal, because the feature names are long. Colour marks the local vs
 * aggregated split, which is the interesting read: aggregated features are
 * neighbourhood statistics, so an aggregated top feature means the verdict came
 * from the transaction's network rather than from the transaction itself.
 */
export function FeatureImportanceChart({
  features,
  limit = 15,
}: {
  features: FeatureImportance[];
  limit?: number;
}) {
  const data = features.slice(0, limit).map((f) => ({ ...f })).reverse();
  const height = Math.max(220, data.length * 22 + 40);

  return (
    <div>
      <Legend
        items={[
          { label: "Local (the transaction itself)", color: SERIES.one },
          { label: "Aggregated (its neighbourhood)", color: SERIES.two },
        ]}
        className="mb-3"
      />
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis type="number" domain={[0, 1]} {...AXIS} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis
            type="category"
            dataKey="name"
            width={78}
            {...AXIS}
            tick={{ ...AXIS.tick, fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}
          />
          <Tooltip
            cursor={{ fill: "var(--line)", opacity: 0.3 }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as FeatureImportance;
              return (
                <TooltipShell title={row.name}>
                  <TooltipRow
                    color={row.group === "local" ? SERIES.one : SERIES.two}
                    label={row.group === "local" ? "Local feature" : "Aggregated feature"}
                    value={score(row.importance, 3)}
                  />
                  <TooltipRow label="Feature index" value={row.index} />
                </TooltipShell>
              );
            }}
          />
          <Bar isAnimationActive={false} dataKey="importance" radius={[0, 3, 3, 0]} barSize={12}>
            {data.map((row) => (
              <Cell key={row.index} fill={row.group === "local" ? SERIES.one : SERIES.two} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
