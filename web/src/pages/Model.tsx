import { ConfusionMatrix, PRCurveChart, TrainingCurves } from "../components/charts/ModelCharts";
import { FeatureImportanceChart } from "../components/charts/FeatureImportance";
import { Panel } from "../components/ui/Panel";
import { PanelSkeleton } from "../components/ui/Skeleton";
import { StatTile } from "../components/ui/StatTile";
import { EmptyState, ErrorState } from "../components/ui/States";
import { useResource } from "../hooks/useResource";
import { getOverview } from "../lib/api";
import { SERIES } from "../lib/colors";
import { num } from "../lib/format";
import type { Curve, Overview } from "../lib/types";

export function ModelPage() {
  const { data, error, loading, reload } = useResource<Overview>("overview", getOverview);

  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (loading || !data) return <PanelSkeleton height={320} />;

  const gnn = data.models?.graphsage;
  const rf = data.models?.random_forest;
  const arch = gnn?.architecture;

  const hidden = arch?.hidden_channels;
  // The third layer halves the width, so describe the stack from the real value
  // rather than hardcoding the default.
  const inChannels = arch?.in_channels;
  const localSplit =
    inChannels === 165 ? "93 local + 72 aggregated" : inChannels ? "anonymised" : undefined;

  const curves: { label: string; curve: Curve; color: string }[] = [];
  if (rf?.pr_curve) curves.push({ label: "Random Forest", curve: rf.pr_curve, color: SERIES.one });
  if (gnn?.pr_curve) curves.push({ label: "GraphSAGE", curve: gnn.pr_curve, color: SERIES.two });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Model</h1>
        <p className="mt-0.5 text-sm text-ink-secondary">
          Three SAGEConv layers with batch normalisation, trained with class-weighted
          cross-entropy on an 80/20 stratified split of the labelled nodes.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <StatTile label="Layers" value={arch?.layers ?? 3} hint="3-hop receptive field" />
        <StatTile
          label="Hidden width"
          value={arch?.hidden_channels ?? "—"}
          hint={hidden ? `${hidden} → ${hidden} → ${hidden / 2}` : undefined}
        />
        <StatTile
          label="Input features"
          value={arch?.in_channels ?? "—"}
          hint={localSplit}
        />
        <StatTile label="Dropout" value={arch?.dropout ?? 0.3} />
        <StatTile
          label="Best epoch"
          value={gnn?.best_epoch ? num(gnn.best_epoch) : "—"}
          hint="by validation F1"
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Training curves" subtitle="Train vs validation F1 on the illicit class">
          {gnn?.history?.epoch?.length ? (
            <TrainingCurves history={gnn.history} />
          ) : (
            <EmptyState
              title="No training history recorded"
              description="Run `fraudlens train` to capture per-epoch metrics."
            />
          )}
        </Panel>

        <Panel
          title="Precision-recall"
          subtitle="The honest curve for a 1:9 imbalanced problem — ROC flatters it"
        >
          {curves.length ? (
            <PRCurveChart curves={curves} />
          ) : (
            <EmptyState title="No PR curves recorded" />
          )}
        </Panel>

        <Panel title="GraphSAGE confusion matrix" subtitle="On the held-out test split">
          {gnn ? <ConfusionMatrix metrics={gnn} /> : <EmptyState title="Not trained yet" />}
        </Panel>

        <Panel
          title="Random Forest feature importances"
          subtitle="Which of Elliptic's columns the tabular baseline leans on"
        >
          {rf?.top_features?.length ? (
            <FeatureImportanceChart features={rf.top_features} limit={12} />
          ) : (
            <EmptyState
              title="No baseline importances recorded"
              description="Run `fraudlens baseline`."
            />
          )}
        </Panel>
      </div>

      <Panel title="Why GraphSAGE" subtitle="Design decisions behind the architecture">
        <dl className="grid gap-4 text-sm sm:grid-cols-2">
          <div>
            <dt className="font-medium">Inductive, not transductive</dt>
            <dd className="mt-1 text-xs leading-relaxed text-ink-secondary">
              GraphSAGE learns an aggregation function, so it scores transactions it never saw
              in training. A GCN would need retraining as the graph grows — and this graph grows
              daily.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Three layers</dt>
            <dd className="mt-1 text-xs leading-relaxed text-ink-secondary">
              Each layer adds a hop. At three, a node sees the structure that makes a fraud ring
              a ring; at one, it sees only its immediate counterparties.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Class-weighted loss</dt>
            <dd className="mt-1 text-xs leading-relaxed text-ink-secondary">
              The labelled set is ~1:9 illicit:licit. Unweighted, calling everything licit scores
              ~90% accuracy. The weighting makes a missed fraud roughly 5× costlier than a false
              alarm — and accuracy is never reported here.
            </dd>
          </div>
          <div>
            <dt className="font-medium">Batch normalisation</dt>
            <dd className="mt-1 text-xs leading-relaxed text-ink-secondary">
              Stabilises training under that imbalance; without it the weighted loss makes early
              epochs swing hard.
            </dd>
          </div>
        </dl>
      </Panel>
    </div>
  );
}
