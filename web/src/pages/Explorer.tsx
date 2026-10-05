import { ExternalLink, Focus, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { GraphCanvas } from "../components/graph/GraphCanvas";
import { GraphLegend } from "../components/graph/GraphLegend";
import { LabelBadge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { PanelSkeleton } from "../components/ui/Skeleton";
import { EmptyState, ErrorState } from "../components/ui/States";
import { SyntheticNotice } from "../components/ui/SyntheticNotice";
import { useResource } from "../hooks/useResource";
import { getGraphSample } from "../lib/api";
import { num, score } from "../lib/format";
import type { GraphSample, NodeRecord } from "../lib/types";

const SIZES = [500, 1000, 1500];

export function ExplorerPage() {
  const [size, setSize] = useState(1000);
  const [minScore, setMinScore] = useState(0);
  const [selected, setSelected] = useState<NodeRecord | null>(null);

  const { data, error, loading, reload } = useResource<GraphSample>(
    `graph-sample-${size}`,
    (signal) => getGraphSample(size, signal),
  );

  // The adjacency of the sample, so selecting a node can dim everything outside
  // its 2-hop neighbourhood.
  const adjacency = useMemo(() => {
    const map = new Map<number, Set<number>>();
    data?.edges.forEach(([a, b]) => {
      if (!map.has(a)) map.set(a, new Set());
      if (!map.has(b)) map.set(b, new Set());
      map.get(a)!.add(b);
      map.get(b)!.add(a);
    });
    return map;
  }, [data]);

  const highlight = useMemo(() => {
    if (!selected) return null;
    const keep = new Set<number>([selected.idx]);
    const first = adjacency.get(selected.idx) ?? new Set();
    first.forEach((n) => {
      keep.add(n);
      (adjacency.get(n) ?? new Set()).forEach((m) => keep.add(m));
    });
    return keep;
  }, [selected, adjacency]);

  const visible = useMemo(() => {
    if (!data) return { nodes: [], edges: [] as [number, number][] };
    if (minScore <= 0) return { nodes: data.nodes, edges: data.edges };
    const nodes = data.nodes.filter((n) => n.score >= minScore);
    const keep = new Set(nodes.map((n) => n.idx));
    return {
      nodes,
      edges: data.edges.filter(([a, b]) => keep.has(a) && keep.has(b)),
    };
  }, [data, minScore]);

  if (error) return <ErrorState error={error} onRetry={reload} />;

  return (
    <div className="space-y-4">
      <SyntheticNotice visible={Boolean(data?.synthetic)} />

      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Graph explorer</h1>
          <p className="mt-0.5 text-sm text-ink-secondary">
            A connected, fraud-rich slice of the transaction network. Click a node to isolate
            its 2-hop neighbourhood.
          </p>
        </div>

        {/* Filters sit in one row above the figure. */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-ink-secondary">
            Sample
            <select
              value={size}
              onChange={(e) => {
                setSize(Number(e.target.value));
                setSelected(null);
              }}
              className="rounded-md border border-line bg-raised px-2 py-1 text-xs text-ink"
            >
              {SIZES.map((n) => (
                <option key={n} value={n}>
                  {n} nodes
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-xs text-ink-secondary">
            Min score
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.1}
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
              className="w-24 accent-[var(--accent)]"
            />
            <span className="tnum w-7 text-ink">{minScore.toFixed(1)}</span>
          </label>

          {selected && (
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="inline-flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-xs hover:bg-raised"
            >
              <RotateCcw size={12} aria-hidden /> Clear focus
            </button>
          )}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
        <Panel flush>
          {loading || !data ? (
            <PanelSkeleton height={520} />
          ) : visible.nodes.length === 0 ? (
            <EmptyState
              title="No nodes above that score"
              description="Lower the minimum score filter to bring nodes back."
            />
          ) : (
            <>
              <div className="border-b border-line px-4 py-2.5">
                <GraphLegend />
              </div>
              <GraphCanvas
                nodes={visible.nodes}
                edges={visible.edges}
                highlight={highlight}
                selectedId={selected?.idx ?? null}
                onSelect={setSelected}
                height={520}
              />
              <div className="border-t border-line px-4 py-2 text-xs text-ink-muted">
                {num(visible.nodes.length)} nodes · {num(visible.edges.length)} edges
                {data.note ? ` · ${data.note}` : ""}
              </div>
            </>
          )}
        </Panel>

        <Panel title={selected ? `Node #${selected.idx}` : "Node details"}>
          {selected ? (
            <div className="space-y-4">
              <dl className="space-y-2 text-sm">
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-xs text-ink-muted">Fraud score</dt>
                  <dd className="tnum font-semibold">{score(selected.score)}</dd>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-xs text-ink-muted">True label</dt>
                  <dd>
                    <LabelBadge label={selected.label} />
                  </dd>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-xs text-ink-muted">Model says</dt>
                  <dd className="text-sm">{selected.prediction}</dd>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-xs text-ink-muted">Transaction id</dt>
                  <dd className="font-mono text-xs">{selected.tx_id || "—"}</dd>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-xs text-ink-muted">Time step</dt>
                  <dd className="tnum text-sm">{selected.time_step}</dd>
                </div>
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-xs text-ink-muted">Degree</dt>
                  <dd className="tnum text-sm">{selected.degree}</dd>
                </div>
              </dl>

              <div className="flex flex-col gap-2">
                <Link
                  to={`/node/${selected.idx}`}
                  className="inline-flex items-center justify-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-xs font-medium hover:bg-raised"
                >
                  <Focus size={12} aria-hidden /> Inspect node
                </Link>
                <Link
                  to={`/explain?node=${selected.idx}`}
                  className="inline-flex items-center justify-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-xs font-medium hover:bg-raised"
                >
                  <ExternalLink size={12} aria-hidden /> Explain this prediction
                </Link>
              </div>
            </div>
          ) : (
            <p className="text-xs text-ink-secondary">
              Select a node in the graph to see its score, label and neighbourhood.
            </p>
          )}
        </Panel>
      </div>
    </div>
  );
}
