import { Users } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { GraphCanvas } from "../components/graph/GraphCanvas";
import { GraphLegend } from "../components/graph/GraphLegend";
import { LabelBadge } from "../components/ui/Badge";
import { DataTable, type Column } from "../components/ui/DataTable";
import { Panel } from "../components/ui/Panel";
import { PanelSkeleton, StatRowSkeleton } from "../components/ui/Skeleton";
import { StatTile } from "../components/ui/StatTile";
import { EmptyState, ErrorState } from "../components/ui/States";
import { SyntheticNotice } from "../components/ui/SyntheticNotice";
import { useResource } from "../hooks/useResource";
import { getRing, getRings } from "../lib/api";
import { num, pct, score } from "../lib/format";
import type { NodeRecord, RingDetail, RingList, RingSummary } from "../lib/types";

export function RingsPage() {
  const list = useResource<RingList>("rings", (signal) => getRings(25, signal));
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // Default to the top-ranked ring so the page is never empty on arrival.
  useEffect(() => {
    if (selectedId === null && list.data?.rings.length) {
      setSelectedId(list.data.rings[0].community_id);
    }
  }, [list.data, selectedId]);

  const detail = useResource<RingDetail>(
    selectedId === null ? null : `ring-${selectedId}`,
    (signal) => getRing(selectedId!, signal),
  );

  if (list.error) return <ErrorState error={list.error} onRetry={list.reload} />;

  if (list.loading || !list.data) {
    return (
      <div className="space-y-4">
        <StatRowSkeleton count={4} />
        <PanelSkeleton height={320} />
      </div>
    );
  }

  const { rings, stats } = list.data;

  const columns: Column<RingSummary>[] = [
    {
      key: "community_id",
      header: "Ring",
      sortValue: (r) => r.community_id,
      cell: (r) => <span className="font-mono text-xs">#{r.community_id}</span>,
    },
    {
      key: "size",
      header: "Members",
      align: "right",
      sortValue: (r) => r.size,
      cell: (r) => <span className="tnum">{num(r.size)}</span>,
    },
    {
      key: "illicit_count",
      header: "Confirmed",
      align: "right",
      sortValue: (r) => r.illicit_count,
      cell: (r) => <span className="tnum">{num(r.illicit_count)}</span>,
    },
    {
      key: "illicit_ratio",
      header: "Illicit share",
      align: "right",
      sortValue: (r) => r.illicit_ratio,
      cell: (r) => <span className="tnum font-medium">{pct(r.illicit_ratio)}</span>,
    },
    {
      key: "avg_fraud_score",
      header: "Avg score",
      align: "right",
      sortValue: (r) => r.avg_fraud_score,
      cell: (r) => <span className="tnum">{score(r.avg_fraud_score)}</span>,
    },
  ];

  const memberColumns: Column<NodeRecord>[] = [
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
      header: "Label",
      align: "right",
      sortValue: (r) => r.label,
      cell: (r) => <LabelBadge label={r.label} />,
    },
  ];

  return (
    <div className="space-y-4">
      <SyntheticNotice visible={list.data.synthetic} />

      <div>
        <h1 className="text-lg font-semibold">Fraud rings</h1>
        <p className="mt-0.5 text-sm text-ink-secondary">
          Louvain communities on the raw transaction topology, promoted when they are at
          least 3 nodes, more than 40% confirmed illicit, and average over 0.4 fraud score.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Rings detected" value={num(stats.count ?? rings.length)} icon={<Users size={14} />} />
        <StatTile
          label="Illicit nodes covered"
          value={num(stats.illicit_in_rings ?? 0)}
          hint={stats.pct_of_illicit ? `${stats.pct_of_illicit}% of all illicit` : undefined}
        />
        <StatTile label="Largest ring" value={num(stats.largest ?? 0)} hint="members" />
        <StatTile
          label="Communities searched"
          value={num(stats.communities_total ?? 0)}
          hint="before thresholds"
        />
      </div>

      {rings.length === 0 ? (
        <Panel>
          <EmptyState
            title="No fraud rings in the available data"
            description="Run `fraudlens rings` with the dataset present to detect them."
          />
        </Panel>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,420px)_1fr]">
          <Panel title="Ranked by illicit concentration" flush>
            <DataTable
              rows={rings}
              columns={columns}
              rowKey={(r) => r.community_id}
              onRowClick={(r) => setSelectedId(r.community_id)}
              selectedKey={selectedId}
              initialSort={{ key: "illicit_ratio", dir: "desc" }}
              maxHeight={420}
              caption="Detected fraud rings"
            />
          </Panel>

          <Panel
            title={selectedId !== null ? `Ring #${selectedId}` : "Ring detail"}
            subtitle={
              detail.data
                ? `${num(detail.data.size)} members · ${pct(detail.data.illicit_ratio)} confirmed illicit · avg score ${score(detail.data.avg_fraud_score)}`
                : undefined
            }
            flush
          >
            {detail.loading ? (
              <PanelSkeleton height={340} />
            ) : detail.error ? (
              <ErrorState error={detail.error} onRetry={detail.reload} title="Ring unavailable" />
            ) : detail.data ? (
              <>
                <div className="border-b border-line px-4 py-2.5">
                  <GraphLegend />
                </div>
                <GraphCanvas
                  nodes={
                    detail.data.member_detail.length
                      ? detail.data.member_detail
                      : detail.data.members.map((m) => ({
                          idx: m,
                          tx_id: 0,
                          score: detail.data!.avg_fraud_score,
                          label: "unknown" as const,
                          prediction: "illicit" as const,
                          time_step: 0,
                          degree: 1,
                        }))
                  }
                  edges={detail.data.edges}
                  height={340}
                />
                {detail.data.member_detail.length > 0 && (
                  <div className="border-t border-line">
                    <DataTable
                      rows={detail.data.member_detail}
                      columns={memberColumns}
                      rowKey={(r) => r.idx}
                      initialSort={{ key: "score", dir: "desc" }}
                      maxHeight={220}
                      caption="Ring members"
                    />
                  </div>
                )}
              </>
            ) : (
              <EmptyState title="Select a ring" description="Pick a ring from the list." />
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}
