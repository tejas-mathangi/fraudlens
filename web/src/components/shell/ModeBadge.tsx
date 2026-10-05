import { Database, Loader2, Radio, TriangleAlert } from "lucide-react";

import type { Health } from "../../lib/types";

/**
 * Says where the numbers on screen came from.
 *
 * This matters: the console runs against a live model, against precomputed
 * artifacts, or against bundled demo files, and a viewer deserves to know which
 * without reading the README.
 */
export function ModeBadge({ health, source }: { health: Health | null; source: string | null }) {
  if (!health) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-ink-secondary">
        <Loader2 size={12} className="animate-spin" aria-hidden /> Connecting
      </span>
    );
  }

  if (health.status === "warming") {
    return (
      <span
        className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
        style={{
          color: "var(--status-warning)",
          background: "color-mix(in srgb, var(--status-warning) 16%, transparent)",
        }}
      >
        <Loader2 size={12} className="animate-spin" aria-hidden /> Loading model
      </span>
    );
  }

  if (health.error) {
    return (
      <span
        className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
        style={{
          color: "var(--status-warning)",
          background: "color-mix(in srgb, var(--status-warning) 16%, transparent)",
        }}
        title={health.error}
      >
        <TriangleAlert size={12} aria-hidden /> Degraded
      </span>
    );
  }

  const live = source === "api" && health.mode === "live";

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={
        live
          ? {
              color: "var(--status-good)",
              background: "color-mix(in srgb, var(--status-good) 14%, transparent)",
            }
          : {
              color: "var(--accent)",
              background: "color-mix(in srgb, var(--accent) 14%, transparent)",
            }
      }
      title={
        live
          ? "Connected to a backend holding the full graph and model"
          : "Reading precomputed artifacts — any node can still be browsed, but only the exported subset"
      }
    >
      {live ? <Radio size={12} aria-hidden /> : <Database size={12} aria-hidden />}
      {live ? "Live model" : "Precomputed data"}
    </span>
  );
}
