import { Network as NetworkIcon, User } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { FeatureImportanceChart } from "../components/charts/FeatureImportance";
import { GraphCanvas } from "../components/graph/GraphCanvas";
import { GraphLegend } from "../components/graph/GraphLegend";
import { Badge, LabelBadge } from "../components/ui/Badge";
import { Panel } from "../components/ui/Panel";
import { PanelSkeleton } from "../components/ui/Skeleton";
import { EmptyState, ErrorState } from "../components/ui/States";
import { SyntheticNotice } from "../components/ui/SyntheticNotice";
import { useResource } from "../hooks/useResource";
import { getExplainCandidates, getExplanation } from "../lib/api";
import { score } from "../lib/format";
import type { Explanation, NodeRecord } from "../lib/types";

/**
 * The plain-English verdict.
 *
 * This is the sentence the old Streamlit dashboard tried to render and crashed
 * on. It is the whole payoff of the explainability work, so it gets stated
 * directly rather than left for the reader to infer from a bar chart.
 */
function Verdict({ explanation }: { explanation: Explanation }) {
  const network = explanation.driver === "network";
  const top = explanation.top_features[0];

  return (
    <div className="flex items-start gap-3 rounded-md border border-line bg-raised p-3.5">
      <span
        aria-hidden
        className="mt-0.5 shrink-0"
        style={{ color: network ? "var(--series-2)" : "var(--series-1)" }}
      >
        {network ? <NetworkIcon size={18} /> : <User size={18} />}
      </span>
      <div className="min-w-0 text-sm">
        <p className="font-medium">
          {network
            ? "Driven by the transaction's network"
            : "Driven by the transaction's own behaviour"}
        </p>
        <p className="mt-1 text-xs leading-relaxed text-ink-secondary">
          The most influential input is{" "}
          <code className="rounded bg-surface px-1 py-0.5 font-mono">{top?.name}</code>, which is{" "}
          {network
            ? "an aggregated feature — a statistic over this transaction's neighbours. The verdict rests on the company it keeps, not on the transaction in isolation."
            : "a local feature — a property of this transaction itself. The verdict would hold even with a different neighbourhood."}
        </p>
      </div>
    </div>
  );
}

export function ExplainPage() {
  const [params, setParams] = useSearchParams();
  const requested = params.get("node");
  const [nodeIdx, setNodeIdx] = useState<number | null>(
    requested !== null && Number.isFinite(Number(requested)) ? Number(requested) : null,
  );

  const candidates = useResource<{ candidates: NodeRecord[] }>(
    "explain-candidates",
    getExplainCandidates,
  );

  // Seed with the most confident fraud node so the page is useful immediately.
  useEffect(() => {
    if (nodeIdx === null && candidates.data?.candidates.length) {
      setNodeIdx(candidates.data.candidates[0].idx);
    }
  }, [candidates.data, nodeIdx]);

  const explanation = useResource<Explanation>(
    nodeIdx === null ? null : `explain-${nodeIdx}`,
    (signal) => getExplanation(nodeIdx!, signal),
  );

  const select = (idx: number) => {
    setNodeIdx(idx);
    setParams({ node: String(idx) }, { replace: true });
  };

  return (
    <div className="space-y-4">
      <SyntheticNotice visible={false} />

      <div>
        <h1 className="text-lg font-semibold">Explainability</h1>
        <p className="mt-0.5 text-sm text-ink-secondary">
          Input-gradient saliency: the predicted-class probability is backpropagated into the
          features of the node&apos;s 2-hop subgraph, so the attribution is faithful to the
          computation rather than a post-hoc approximation.
        </p>
      </div>

      <Panel title="Pick a transaction" subtitle="Highest-confidence confirmed fraud, plus one clean contrast case">
        {candidates.loading ? (
          <PanelSkeleton height={48} />
        ) : candidates.error ? (
          <ErrorState error={candidates.error} onRetry={candidates.reload} />
        ) : candidates.data?.candidates.length ? (
          <div className="flex flex-wrap gap-2">
            {candidates.data.candidates.map((c) => (
              <button
                key={c.idx}
                type="button"
                onClick={() => select(c.idx)}
                aria-pressed={c.idx === nodeIdx}
                className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                  c.idx === nodeIdx
                    ? "border-[var(--accent)] bg-raised font-medium"
                    : "border-line hover:bg-raised"
                }`}
              >
                <span className="font-mono">#{c.idx}</span>
                <span className="tnum text-ink-secondary">{score(c.score, 3)}</span>
                <LabelBadge label={c.label} />
              </button>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No explanation candidates"
            description="Run `fraudlens export` with the dataset present."
          />
        )}
      </Panel>

      {nodeIdx !== null && (
        <>
          {explanation.loading ? (
            <PanelSkeleton height={360} />
          ) : explanation.error ? (
            <Panel>
              <ErrorState
                error={explanation.error}
                onRetry={explanation.reload}
                title="No explanation for this node"
              />
            </Panel>
          ) : explanation.data ? (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-3">
                <Link
                  to={`/node/${explanation.data.node.idx}`}
                  className="font-mono text-sm font-medium hover:underline"
                >
                  #{explanation.data.node.idx}
                </Link>
                <LabelBadge label={explanation.data.node.label} />
                <Badge tone={explanation.data.predicted_class === 1 ? "critical" : "good"}>
                  predicted {explanation.data.predicted_class === 1 ? "illicit" : "licit"}
                </Badge>
                <span className="tnum text-xs text-ink-secondary">
                  confidence {score(explanation.data.probability)}
                </span>
              </div>

              <Verdict explanation={explanation.data} />

              <div className="grid gap-4 xl:grid-cols-[1fr_minmax(0,440px)]">
                <Panel
                  title="Influence over the neighbourhood"
                  subtitle="Edge width and colour scale with how much the connection moved the prediction"
                  flush
                >
                  {explanation.data.subgraph.nodes.length ? (
                    <>
                      <div className="border-b border-line px-4 py-2.5">
                        <GraphLegend weighted />
                      </div>
                      <GraphCanvas
                        nodes={explanation.data.subgraph.nodes}
                        edges={explanation.data.subgraph.edges}
                        focusId={explanation.data.node.idx}
                        height={380}
                        weighted
                      />
                    </>
                  ) : (
                    <EmptyState title="No subgraph recorded for this node" />
                  )}
                </Panel>

                <Panel title="Most influential features" subtitle="Top 15 by |∂p/∂x|">
                  <FeatureImportanceChart features={explanation.data.top_features} limit={15} />
                </Panel>
              </div>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
