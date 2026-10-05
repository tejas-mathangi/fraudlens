import { AlertTriangle, Check, FileWarning, Minus, Network, ShieldCheck, Users } from "lucide-react";

import { TimelineChart } from "../components/charts/Timeline";
import { ScoreHistogramChart } from "../components/charts/ScoreHistogram";
import { DataTable, type Column } from "../components/ui/DataTable";
import { Panel } from "../components/ui/Panel";
import { PanelSkeleton, StatRowSkeleton } from "../components/ui/Skeleton";
import { StatTile } from "../components/ui/StatTile";
import { ErrorState } from "../components/ui/States";
import { SyntheticNotice } from "../components/ui/SyntheticNotice";
import { useResource } from "../hooks/useResource";
import { getOverview } from "../lib/api";
import { SERIES } from "../lib/colors";
import { num, pct, score, short } from "../lib/format";
import type { ComparisonRow, Overview as OverviewData } from "../lib/types";

const CAPABILITY_COLUMNS: { key: keyof ComparisonRow; header: string; note: string }[] = [
  { key: "ring_detection", header: "Rings", note: "Detects coordinated groups" },
  { key: "explainability", header: "Explains", note: "Attributes a prediction" },
  { key: "inductive", header: "Inductive", note: "Generalises to new nodes" },
];

function Capability({ on }: { on: boolean }) {
  return on ? (
    <span className="inline-flex items-center gap-1" style={{ color: "var(--status-good)" }}>
      <Check size={13} aria-hidden />
      <span className="sr-only">yes</span>
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-ink-muted">
      <Minus size={13} aria-hidden />
      <span className="sr-only">no</span>
    </span>
  );
}

export function OverviewPage() {
  const { data, error, loading, reload } = useResource<OverviewData>("overview", getOverview);

  if (error) return <ErrorState error={error} onRetry={reload} />;

  if (loading || !data) {
    return (
      <div className="space-y-4">
        <StatRowSkeleton count={6} />
        <div className="grid gap-4 xl:grid-cols-2">
          <PanelSkeleton />
          <PanelSkeleton />
        </div>
      </div>
    );
  }

  const { dataset, scoring, rings } = data;
  const gnn = data.models?.graphsage;

  /*
   * The comparison copy has to follow the numbers that are actually loaded.
   * On the real Elliptic data the forest wins on F1 and the "pre-engineered
   * features" explanation is the point; on other data (a synthetic sample, a
   * retrained model) asserting that would simply be false.
   */
  const rf = data.comparison.find((r) => r.key === "random_forest");
  const sage = data.comparison.find((r) => r.key === "graphsage");
  const baselineAhead =
    rf?.f1_illicit != null && sage?.f1_illicit != null && rf.f1_illicit > sage.f1_illicit;
  const comparisonSubtitle =
    rf && sage
      ? baselineAhead
        ? "The baseline wins on raw metrics — and that is the interesting part"
        : "Capabilities, not headline metrics, are the reason to use a graph model"
      : "Run both models to compare them";

  const columns: Column<ComparisonRow>[] = [
    { key: "model", header: "Model", cell: (r) => <span className="font-medium">{r.model}</span> },
    { key: "auc", header: "AUC", align: "right", cell: (r) => <span className="tnum">{score(r.auc)}</span> },
    {
      key: "f1",
      header: "F1 (illicit)",
      align: "right",
      cell: (r) => <span className="tnum">{score(r.f1_illicit)}</span>,
    },
    {
      key: "ap",
      header: "Avg precision",
      align: "right",
      cell: (r) => <span className="tnum">{score(r.avg_precision)}</span>,
    },
    ...CAPABILITY_COLUMNS.map(({ key, header }) => ({
      key: String(key),
      header,
      align: "right" as const,
      cell: (r: ComparisonRow) => <Capability on={Boolean(r[key])} />,
    })),
  ];

  return (
    <div className="space-y-4">
      <SyntheticNotice visible={data.synthetic} />

      <div>
        <h1 className="text-lg font-semibold">Overview</h1>
        <p className="mt-0.5 text-sm text-ink-secondary">
          A 3-layer GraphSAGE model scored every transaction in the Elliptic network.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <StatTile
          label="Transactions"
          value={short(dataset.nodes)}
          hint={`${short(dataset.edges)} edges`}
          icon={<Network size={14} />}
        />
        <StatTile
          label="Confirmed illicit"
          value={short(dataset.illicit)}
          hint={`${pct(dataset.illicit / dataset.nodes)} of all nodes`}
          icon={<FileWarning size={14} />}
          accent={SERIES.two}
        />
        <StatTile
          label="Confirmed licit"
          value={short(dataset.licit)}
          hint={`${short(dataset.unknown)} unlabelled`}
          icon={<ShieldCheck size={14} />}
          accent={SERIES.one}
        />
        <StatTile
          label="High risk"
          value={short(scoring.high_risk)}
          hint={`score > ${scoring.high_risk_threshold}`}
          icon={<AlertTriangle size={14} />}
        />
        <StatTile
          label="Fraud rings"
          value={num(rings.count ?? 0)}
          hint={
            rings.pct_of_illicit
              ? `${rings.pct_of_illicit}% of illicit nodes`
              : "coordinated communities"
          }
          icon={<Users size={14} />}
        />
        <StatTile
          label="Model AUC"
          value={<span className="tnum">{score(gnn?.auc)}</span>}
          hint={gnn?.f1_illicit ? `F1 ${score(gnn.f1_illicit)}` : "not yet trained"}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel
          title="Score separation"
          subtitle="Does the model actually tell the two labelled classes apart?"
        >
          <ScoreHistogramChart
            histogram={data.score_histogram}
            decisionThreshold={scoring.decision_threshold}
            illicitShare={dataset.illicit / dataset.nodes}
          />
        </Panel>

        <Panel title="Labelled activity over time" subtitle="49 time steps, roughly two weeks apart">
          {data.timeline.length ? (
            <TimelineChart points={data.timeline} />
          ) : (
            <p className="text-xs text-ink-secondary">No timeline recorded.</p>
          )}
        </Panel>
      </div>

      <Panel title="Random Forest vs GraphSAGE" subtitle={comparisonSubtitle} flush>
        {data.comparison.length ? (
          <>
            <DataTable
              rows={data.comparison}
              columns={columns}
              rowKey={(r) => r.key}
              caption="Model comparison with capability columns"
            />
            <div className="border-t border-line px-4 py-3 text-xs leading-relaxed text-ink-secondary">
              {baselineAhead
                ? "Elliptic ships 72 pre-engineered neighbourhood-aggregate features, so a tabular model already consumes a hand-crafted summary of each transaction's graph context — which is why the forest scores so well. "
                : "GraphSAGE is ahead on the headline metrics here, but the metrics are not the reason to prefer it. "}
              The GNN learns structure from raw topology, and gains three capabilities a tabular
              model cannot have at any F1:{" "}
              {CAPABILITY_COLUMNS.map((c) => c.note.toLowerCase()).join(", ")}.
            </div>
          </>
        ) : (
          <p className="p-4 text-xs text-ink-secondary">
            No metrics recorded yet. Run <code className="font-mono">fraudlens baseline</code> and{" "}
            <code className="font-mono">fraudlens train</code>.
          </p>
        )}
      </Panel>
    </div>
  );
}
