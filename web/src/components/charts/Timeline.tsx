import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SERIES } from "../../lib/colors";
import { num } from "../../lib/format";
import type { TimelinePoint } from "../../lib/types";
import { AXIS, GRID, Legend, TooltipRow, TooltipShell } from "./chartBits";

/**
 * Confirmed labels over the 49 time steps.
 *
 * Two separate areas on one shared count axis — never a second y-axis. The
 * Elliptic time steps are ~2 weeks apart but unlabelled, so the axis is the step
 * index rather than a fake date.
 */
export function TimelineChart({ points }: { points: TimelinePoint[] }) {
  return (
    <div>
      <Legend
        items={[
          { label: "Illicit", color: SERIES.two },
          { label: "Licit", color: SERIES.one },
        ]}
        className="mb-3"
      />
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={points} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="fill-illicit" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={SERIES.two} stopOpacity={0.35} />
              <stop offset="100%" stopColor={SERIES.two} stopOpacity={0.04} />
            </linearGradient>
            <linearGradient id="fill-licit" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={SERIES.one} stopOpacity={0.3} />
              <stop offset="100%" stopColor={SERIES.one} stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID} vertical={false} />
          <XAxis
            dataKey="time_step"
            {...AXIS}
            label={{
              value: "Time step",
              position: "insideBottom",
              offset: -2,
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <YAxis {...AXIS} tickFormatter={num} />
          <Tooltip
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipShell title={`Time step ${label}`}>
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
          <Area
            isAnimationActive={false}
            type="monotone"
            dataKey="licit"
            name="Licit"
            stroke={SERIES.one}
            strokeWidth={2}
            fill="url(#fill-licit)"
          />
          <Area
            isAnimationActive={false}
            type="monotone"
            dataKey="illicit"
            name="Illicit"
            stroke={SERIES.two}
            strokeWidth={2}
            fill="url(#fill-illicit)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
