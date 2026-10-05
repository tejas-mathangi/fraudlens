/** Response shapes mirroring the FastAPI contract in `api/routes.py`. */

export type Label = "illicit" | "licit" | "unknown";
export type Prediction = "illicit" | "licit";
export type Mode = "live" | "artifact";
export type Status = "warming" | "ready" | "error";

export interface Health {
  status: Status;
  mode: Mode;
  model_loaded: boolean;
  graph_loaded: boolean;
  nodes: number;
  edges: number;
  artifacts: string[];
  error: string | null;
  thresholds: { decision: number; high_risk: number };
  n_local_features: number;
}

export interface NeighborRef {
  idx: number;
  score: number;
  label: Label;
}

export interface NodeRecord {
  idx: number;
  tx_id: number;
  score: number;
  label: Label;
  prediction: Prediction;
  time_step: number;
  degree: number;
  neighbors?: NeighborRef[];
}

export interface DatasetSummary {
  nodes: number;
  edges: number;
  features: number;
  time_steps: number;
  illicit: number;
  licit: number;
  unknown: number;
  labeled: number;
}

export interface ScoreHistogram {
  bin_centers: number[];
  illicit: number[];
  licit: number[];
  unknown: number[];
}

export interface TimelinePoint {
  time_step: number;
  transactions: number;
  illicit: number;
  licit: number;
  avg_score: number;
}

export interface ComparisonRow {
  key: string;
  model: string;
  auc: number | null;
  f1_illicit: number | null;
  avg_precision: number | null;
  ring_detection: boolean;
  explainability: boolean;
  inductive: boolean;
}

export interface Curve {
  x: number[];
  y: number[];
}

export interface TrainingHistory {
  epoch: number[];
  train_loss: number[];
  train_f1: number[];
  val_loss: number[];
  val_f1: number[];
  val_auc: number[];
}

export interface ModelMetrics {
  auc?: number;
  f1_illicit?: number;
  avg_precision?: number;
  threshold?: number;
  support?: { licit: number; illicit: number };
  confusion_matrix?: number[][];
  pr_curve?: Curve;
  history?: TrainingHistory;
  best_epoch?: number;
  architecture?: {
    layers: number;
    hidden_channels: number;
    dropout: number;
    in_channels: number;
  };
  top_features?: FeatureImportance[];
  n_estimators?: number;
}

export interface RingStats {
  count?: number;
  illicit_in_rings?: number;
  pct_of_illicit?: number;
  largest?: number;
  communities_total?: number;
  thresholds?: Record<string, number>;
}

export interface Overview {
  synthetic: boolean;
  dataset: DatasetSummary;
  scoring: {
    high_risk: number;
    flagged: number;
    high_risk_threshold: number;
    decision_threshold: number;
    mean_score: number;
  };
  score_histogram: ScoreHistogram;
  timeline: TimelinePoint[];
  rings: RingStats;
  comparison: ComparisonRow[];
  models: Record<string, ModelMetrics>;
}

export interface GraphSample {
  synthetic: boolean;
  nodes: NodeRecord[];
  /** `[source, target]` pairs of global node indices. */
  edges: [number, number][];
  note?: string;
}

export interface RingSummary {
  community_id: number;
  size: number;
  labeled: number;
  illicit_count: number;
  licit_count: number;
  illicit_ratio: number;
  avg_fraud_score: number;
  max_fraud_score: number;
}

export interface RingDetail extends RingSummary {
  members: number[];
  edges: [number, number][];
  member_detail: NodeRecord[];
}

export interface RingList {
  synthetic: boolean;
  stats: RingStats;
  rings: RingSummary[];
}

export interface FeatureImportance {
  index: number;
  name: string;
  group: "local" | "aggregated";
  importance: number;
}

export interface WeightedEdge {
  source: number;
  target: number;
  weight: number;
}

export interface Subgraph {
  nodes: NodeRecord[];
  edges: WeightedEdge[];
  target?: number;
}

export interface Explanation {
  node: NodeRecord;
  predicted_class: number;
  probability: number;
  driver: "local" | "network";
  top_features: FeatureImportance[];
  subgraph: Subgraph;
}

export interface FeatureMeta {
  synthetic: boolean;
  n_features: number;
  n_local: number;
  features: { index: number; name: string; group: "local" | "aggregated" }[];
}

export interface SearchResult {
  query: string;
  match: NodeRecord | null;
}
