/**
 * Data layer with a static fallback.
 *
 * The console has to work in three situations, and the pages must not care which:
 *
 *  1. a live FastAPI backend holding the real graph
 *  2. the same backend in artifact mode, serving precomputed JSON
 *  3. no backend at all — a static deploy reading `public/demo/*.json`
 *
 * So every request goes through `request()`. The first time a call fails to reach
 * the API we latch into static mode for the rest of the session and read the demo
 * files instead. Pages receive identical shapes either way; only the badge in the
 * top bar changes.
 */

import type {
  Explanation,
  FeatureMeta,
  GraphSample,
  Health,
  NodeRecord,
  Overview,
  RingDetail,
  RingList,
  SearchResult,
  Subgraph,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
const DEMO_BASE = `${import.meta.env.BASE_URL ?? "/"}demo`.replace(/\/{2,}/g, "/");

export type Source = "api" | "demo";

let source: Source | null = null;
const listeners = new Set<(s: Source) => void>();

/** Which source the session settled on, or null before the first request. */
export function currentSource(): Source | null {
  return source;
}

export function onSourceChange(fn: (s: Source) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function setSource(next: Source) {
  if (source === next) return;
  source = next;
  listeners.forEach((fn) => fn(next));
}

/**
 * One shared reachability probe.
 *
 * Pages mount together and fire their requests in parallel, so without this every
 * one of them independently tries the API and fails before the fallback latches —
 * a burst of ~25 failed requests on every load of a static deploy. The first
 * caller probes `/health`; everyone else awaits the same promise.
 */
let probe: Promise<Source> | null = null;
/** The health body the probe already fetched, handed to the first getHealth call. */
let probedHealth: Health | null = null;

function probeSource(): Promise<Source> {
  if (source) return Promise.resolve(source);
  if (!probe) {
    probe = fetch(`${API_BASE}/health`, { headers: { Accept: "application/json" } })
      .then(async (res) => {
        if (!res.ok) return "demo" as Source;
        probedHealth = (await res.json()) as Health;
        return "api" as Source;
      })
      .catch(() => "demo" as Source)
      .then((next) => {
        setSource(next);
        return next;
      });
  }
  return probe;
}

export class NotAvailable extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotAvailable";
  }
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal, headers: { Accept: "application/json" } });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body — keep the status line */
    }
    const err = new Error(detail);
    (err as Error & { status?: number }).status = res.status;
    throw err;
  }
  return (await res.json()) as T;
}

interface Route<T> {
  /** Path appended to the API base, e.g. `/overview`. */
  api: string;
  /** Demo file name plus an extractor, when the static shape needs narrowing. */
  demo: string;
  pick?: (raw: unknown) => T;
}

/**
 * Try the API, fall back to the bundled demo JSON.
 *
 * A 404 or 503 from a reachable API is a real answer, not a reason to fall back —
 * only a transport failure (or a non-OK response before we have ever reached the
 * API) flips us to static mode.
 */
async function request<T>(route: Route<T>, signal?: AbortSignal): Promise<T> {
  if ((await probeSource()) === "api") {
    try {
      const data = await fetchJson<T>(`${API_BASE}${route.api}`, signal);
      return data;
    } catch (err) {
      if ((err as Error).name === "AbortError") throw err;
      const status = (err as Error & { status?: number }).status;
      // A 404 or 503 from a reachable API is a real answer, not a reason to
      // abandon it; only a transport failure means the backend went away.
      if (status) throw err;
      setSource("demo");
    }
  }

  const raw = await fetchJson<unknown>(`${DEMO_BASE}/${route.demo}`, signal);
  const value = route.pick ? route.pick(raw) : (raw as T);
  if (value === undefined) {
    throw new NotAvailable("Not included in the demo dataset.");
  }
  return value;
}

/* ── Endpoints ──────────────────────────────────────────────────────────── */

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  try {
    if ((await probeSource()) === "demo") throw new Error("no backend");
    // Reuse the probe's response once, so a page load costs one health request
    // rather than two. Later polls (while the backend warms up) refetch.
    if (probedHealth) {
      const first = probedHealth;
      probedHealth = null;
      return first;
    }
    return await fetchJson<Health>(`${API_BASE}/health`, signal);
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    setSource("demo");
    // Synthesise a health record from the static overview so the shell can render.
    const overview = await fetchJson<Overview>(`${DEMO_BASE}/overview.json`, signal).catch(
      () => null,
    );
    return {
      status: "ready",
      mode: "artifact",
      model_loaded: false,
      graph_loaded: false,
      nodes: overview?.dataset.nodes ?? 0,
      edges: overview?.dataset.edges ?? 0,
      artifacts: [],
      error: null,
      thresholds: {
        decision: overview?.scoring.decision_threshold ?? 0.5,
        high_risk: overview?.scoring.high_risk_threshold ?? 0.7,
      },
      n_local_features: 93,
    };
  }
}

export const getOverview = (signal?: AbortSignal) =>
  request<Overview>({ api: "/overview", demo: "overview.json" }, signal);

export const getFeatures = (signal?: AbortSignal) =>
  request<FeatureMeta>({ api: "/features", demo: "features.json" }, signal);

export const getGraphSample = (n = 1500, signal?: AbortSignal) =>
  request<GraphSample>(
    {
      api: `/graph/sample?n=${n}`,
      demo: "graph_sample.json",
      pick: (raw) => {
        const sample = raw as GraphSample;
        const nodes = sample.nodes.slice(0, n);
        const keep = new Set(nodes.map((d) => d.idx));
        return {
          ...sample,
          nodes,
          edges: sample.edges.filter(([a, b]) => keep.has(a) && keep.has(b)),
        };
      },
    },
    signal,
  );

export const getRings = (limit = 25, signal?: AbortSignal) =>
  request<RingList>(
    {
      api: `/rings?limit=${limit}`,
      demo: "rings.json",
      pick: (raw) => {
        const list = raw as { synthetic: boolean; stats: RingList["stats"]; rings: RingDetail[] };
        return { ...list, rings: list.rings.slice(0, limit) };
      },
    },
    signal,
  );

export const getRing = (id: number, signal?: AbortSignal) =>
  request<RingDetail>(
    {
      api: `/rings/${id}`,
      demo: "rings.json",
      pick: (raw) =>
        (raw as { rings: RingDetail[] }).rings.find((r) => r.community_id === id) as RingDetail,
    },
    signal,
  );

export const getNode = (idx: number, signal?: AbortSignal) =>
  request<NodeRecord>(
    {
      api: `/nodes/${idx}`,
      demo: "nodes_index.json",
      pick: (raw) =>
        (raw as { nodes: NodeRecord[] }).nodes.find((n) => n.idx === idx) as NodeRecord,
    },
    signal,
  );

export const getSubgraph = (idx: number, hops = 2, signal?: AbortSignal) =>
  request<Subgraph>(
    {
      api: `/nodes/${idx}/subgraph?hops=${hops}`,
      demo: "nodes_index.json",
      pick: (raw) => {
        const node = (raw as { nodes: NodeRecord[] }).nodes.find((n) => n.idx === idx);
        if (!node) return undefined as unknown as Subgraph;
        const neighbors = node.neighbors ?? [];
        return {
          target: idx,
          nodes: [
            node,
            ...neighbors.map((n) => ({
              idx: n.idx,
              tx_id: 0,
              score: n.score,
              label: n.label,
              prediction: (n.score >= 0.5 ? "illicit" : "licit") as NodeRecord["prediction"],
              time_step: node.time_step,
              degree: 1,
            })),
          ],
          edges: neighbors.map((n) => ({ source: idx, target: n.idx, weight: n.score })),
        };
      },
    },
    signal,
  );

export const getExplainCandidates = (signal?: AbortSignal) =>
  request<{ candidates: NodeRecord[] }>(
    {
      api: "/explain/candidates",
      demo: "explanations.json",
      pick: (raw) => ({
        candidates: (raw as { explanations: Explanation[] }).explanations.map((e) => e.node),
      }),
    },
    signal,
  );

export const getExplanation = (idx: number, signal?: AbortSignal) =>
  request<Explanation>(
    {
      api: `/explain/${idx}`,
      demo: "explanations.json",
      pick: (raw) =>
        (raw as { explanations: Explanation[] }).explanations.find(
          (e) => e.node.idx === idx,
        ) as Explanation,
    },
    signal,
  );

export const search = (q: string, signal?: AbortSignal) =>
  request<SearchResult>(
    {
      api: `/search?q=${encodeURIComponent(q)}`,
      demo: "nodes_index.json",
      pick: (raw) => {
        const value = Number(q);
        const nodes = (raw as { nodes: NodeRecord[] }).nodes;
        const match =
          nodes.find((n) => n.idx === value) ?? nodes.find((n) => n.tx_id === value) ?? null;
        return { query: q, match };
      },
    },
    signal,
  );
