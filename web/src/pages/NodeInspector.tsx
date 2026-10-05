import { ArrowLeft, Lightbulb } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { GraphCanvas } from "../components/graph/GraphCanvas";
import { GraphLegend } from "../components/graph/GraphLegend";
import { LabelBadge } from "../components/ui/Badge";
import { DataTable, type Column } from "../components/ui/DataTable";
import { Panel } from "../components/ui/Panel";
import { PanelSkeleton } from "../components/ui/Skeleton";
import { RiskGauge } from "../components/ui/RiskGauge";
import { EmptyState, ErrorState } from "../components/ui/States";
import { useHealth } from "../hooks/useHealth";
import { useResource } from "../hooks/useResource";
import { getNode, getSubgraph } from "../lib/api";
import { num, score } from "../lib/format";
import type { NeighborRef, NodeRecord, Subgraph } from "../lib/types";

export function NodeInspectorPage() {
  const { idx: raw } = useParams<{ idx: string }>();
  const idx = Number(raw);
  const valid = Number.isFinite(idx) && idx >= 0;
  const { health } = useHealth();

  const node = useResource<NodeRecord>(valid ? `node-${idx}` : null, (signal) =>
    getNode(idx, signal),
  );
  const sub = useResource<Subgraph>(valid ? `subgraph-${idx}` : null, (signal) =>
    getSubgraph(idx, 2, signal),
  );

  if (!valid) {
    return <EmptyState title="Invalid node index" description={`"${raw}" is not a node index.`} />;
  }
  if (node.error) return <ErrorState error={node.error} onRetry={node.reload} />;
  if (node.loading || !node.data) return <PanelSkeleton height={400} />;

  const record = node.data;
  const neighbors = record.neighbors ?? [];

  const columns: Column<NeighborRef>[] = [
    {
      key: "idx",
      header: "Node",
      sortValue: (r) => r.idx,
      cell: (r) => (
        <Link to={`/node/${r.idx}`} className="font-mono text-xs hover:underline">
          #{r.idx}
        </Link>
      ),
    },
    {
      key: "score",
      header: "Score",
      align: "right",
      sortValue: (r) => r.score,
      cell: (r) => <span className="tnum">{score(r.score)}</span>,
    },
    {
      key: "label",
      header: "True label",
      align: "right",
      sortValue: (r) => r.label,
      cell: (r) => <LabelBadge label={r.label} />,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link
          to="/explorer"
          className="inline-flex items-center gap-1.5 text-xs text-ink-secondary hover:text-ink"
        >
          <ArrowLeft size={13} aria-hidden /> Graph explorer
        </Link>
        <h1 className="text-lg font-semibold">
          Node <span className="font-mono">#{record.idx}</span>
        </h1>
        <LabelBadge label={record.label} />
        <Link
          to={`/explain?node=${record.idx}`}
          className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-xs font-medium hover:bg-raised"
        >
          <Lightbulb size={12} aria-hidden /> Explain
        </Link>
      </div>

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <div className="space-y-4">
          <Panel title="Risk">
            <RiskGauge
              value={record.score}
              decision={health?.thresholds.decision ?? 0.5}
              highRisk={health?.thresholds.high_risk ?? 0.7}
            />
          </Panel>

          <Panel title="Attributes">
            <dl className="space-y-2 text-sm">
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-xs text-ink-muted">Transaction id</dt>
                <dd className="font-mono text-xs">{record.tx_id || "—"}</dd>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-xs text-ink-muted">Model prediction</dt>
                <dd>{record.prediction}</dd>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-xs text-ink-muted">Time step</dt>
                <dd className="tnum">{record.time_step}</dd>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-xs text-ink-muted">Degree</dt>
                <dd className="tnum">{num(record.degree)}</dd>
              </div>
            </dl>
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel
            title="2-hop neighbourhood"
            subtitle="The receptive field the model aggregated over"
            flush
          >
            {sub.loading ? (
              <PanelSkeleton height={380} />
            ) : sub.error ? (
              <ErrorState error={sub.error} onRetry={sub.reload} title="Subgraph unavailable" />
            ) : sub.data && sub.data.nodes.length > 1 ? (
              <>
                <div className="border-b border-line px-4 py-2.5">
                  <GraphLegend />
                </div>
                <GraphCanvas
                  nodes={sub.data.nodes}
                  edges={sub.data.edges}
                  focusId={record.idx}
                  height={380}
                />
              </>
            ) : (
              <EmptyState
                title="No neighbours in the available data"
                description="This transaction is isolated in the exported subset."
              />
            )}
          </Panel>

          <Panel title={`Neighbours (${num(neighbors.length)})`} flush>
            {neighbors.length ? (
              <DataTable
                rows={neighbors}
                columns={columns}
                rowKey={(r) => r.idx}
                initialSort={{ key: "score", dir: "desc" }}
                maxHeight={280}
                caption="Direct neighbours with their fraud scores"
              />
            ) : (
              <EmptyState title="No direct neighbours recorded" />
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
