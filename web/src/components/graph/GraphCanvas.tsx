import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { ForceGraphMethods } from "react-force-graph-2d";

import { cssVar, riskColorHex } from "../../lib/colors";
import type { NodeRecord } from "../../lib/types";

export interface GraphNode extends NodeRecord {
  id: number;
  /** Written by the force simulation once the layout starts. */
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: number;
  target: number;
  weight?: number;
}

interface GraphCanvasProps {
  nodes: NodeRecord[];
  /** Either `[a, b]` pairs or weighted edges — both shapes are accepted. */
  edges: ([number, number] | { source: number; target: number; weight?: number })[];
  height?: number;
  /** Node ids to keep at full opacity; everything else dims. */
  highlight?: Set<number> | null;
  /** Draw this node with a ring. */
  focusId?: number | null;
  selectedId?: number | null;
  onSelect?: (node: NodeRecord | null) => void;
  /** Scale edge width and opacity by `weight` — used for explanations. */
  weighted?: boolean;
  /** Hard cap; the force simulation becomes unusable well before the browser does. */
  maxNodes?: number;
}

const MAX_NODES_DEFAULT = 3000;
const MAX_ZOOM = 2.2;

/** Link rest length: longer in small subgraphs, where there is room to breathe. */
function n_link(nodeCount: number): number {
  if (nodeCount <= 60) return 48;
  if (nodeCount <= 300) return 34;
  return 22;
}

/**
 * Canvas force-directed graph, shared by the explorer, node subgraphs, ring
 * subgraphs and explanations.
 *
 * Design notes:
 *  - Node colour encodes the fraud score on a single-hue ramp. The true label is
 *    deliberately *not* colour-encoded here: red/green fails colourblind
 *    separation, and the label is already available in the tooltip and the
 *    details panel.
 *  - Node radius encodes degree, so hubs read as hubs.
 *  - The simulation is frozen after it settles (`cooldownTicks`); leaving a
 *    force layout running costs a permanent frame budget for no information.
 *  - Under `prefers-reduced-motion` the layout is pre-settled with zero warmup
 *    so nothing visibly animates.
 */
export function GraphCanvas({
  nodes,
  edges,
  height = 520,
  highlight = null,
  focusId = null,
  selectedId = null,
  onSelect,
  weighted = false,
  maxNodes = MAX_NODES_DEFAULT,
}: GraphCanvasProps) {
  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(undefined);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [hovered, setHovered] = useState<GraphNode | null>(null);

  const reducedMotion = useMemo(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    [],
  );

  // Track the container width; the canvas needs explicit pixel dimensions.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    setWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  const data = useMemo(() => {
    const capped = nodes.slice(0, maxNodes);
    const present = new Set(capped.map((n) => n.idx));
    const graphNodes: GraphNode[] = capped.map((n) => ({ ...n, id: n.idx }));
    const links: GraphLink[] = [];
    for (const edge of edges) {
      const [a, b, w] = Array.isArray(edge)
        ? [edge[0], edge[1], undefined]
        : [edge.source, edge.target, edge.weight];
      if (a === b || !present.has(a) || !present.has(b)) continue;
      links.push({ source: a, target: b, weight: w });
    }
    return { nodes: graphNodes, links };
  }, [nodes, edges, maxNodes]);

  const degreeMax = useMemo(
    () => Math.max(1, ...data.nodes.map((n) => n.degree || 1)),
    [data.nodes],
  );

  const radius = useCallback(
    (node: GraphNode) => {
      const scale = Math.sqrt((node.degree || 1) / degreeMax);
      return 2.2 + scale * 5.5;
    },
    [degreeMax],
  );

  const dimmed = useCallback(
    (id: number) => highlight !== null && !highlight.has(id),
    [highlight],
  );

  const drawNode = useCallback(
    (node: GraphNode, ctx: CanvasRenderingContext2D) => {
      const r = radius(node);
      const faded = dimmed(node.id);
      ctx.globalAlpha = faded ? 0.12 : 1;

      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
      ctx.fillStyle = riskColorHex(node.score);
      ctx.fill();

      // A 2px surface ring keeps overlapping nodes readable as separate marks.
      ctx.lineWidth = 1;
      ctx.strokeStyle = cssVar("--surface", "#1a1a19");
      ctx.stroke();

      if (node.id === focusId || node.id === selectedId) {
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.arc(node.x ?? 0, node.y ?? 0, r + 3.5, 0, 2 * Math.PI);
        ctx.lineWidth = 2;
        ctx.strokeStyle = cssVar("--accent", "#3987e5");
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
    },
    [radius, dimmed, focusId, selectedId],
  );

  const drawLink = useCallback(
    (link: GraphLink, ctx: CanvasRenderingContext2D) => {
      // The simulation replaces the numeric endpoints with node objects in place.
      const s = link.source as unknown as GraphNode;
      const t = link.target as unknown as GraphNode;
      if (!s || !t) return;
      const faded = dimmed(s.id) && dimmed(t.id);
      const w = weighted ? (link.weight ?? 0.3) : 0.3;

      ctx.globalAlpha = faded ? 0.05 : weighted ? 0.25 + 0.6 * w : 0.3;
      ctx.beginPath();
      ctx.moveTo(s.x ?? 0, s.y ?? 0);
      ctx.lineTo(t.x ?? 0, t.y ?? 0);
      ctx.strokeStyle = weighted ? riskColorHex(w) : cssVar("--text-muted", "#888");
      ctx.lineWidth = weighted ? 0.4 + 2.2 * w : 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    },
    [dimmed, weighted],
  );

  /*
   * Tune the simulation, then re-fit once it settles.
   *
   * The defaults produce a hairball at this density: ~1,000 nodes with ~3,000
   * edges collapse into a disc where no structure is readable. Stronger charge
   * repulsion with a capped interaction distance, plus an explicit link distance,
   * pushes the communities apart so the rings are visible as rings.
   */
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    const charge = fg.d3Force("charge");
    if (charge) {
      const n = data.nodes.length;
      // Sparse subgraphs need less push than the full explorer sample.
      const strength = n > 400 ? -85 : -160;
      (charge as unknown as {
        strength: (v: number) => { distanceMax: (d: number) => unknown };
      })
        .strength(strength)
        .distanceMax(320);
    }
    const link = fg.d3Force("link");
    if (link) {
      (link as unknown as { distance: (d: number) => unknown }).distance(
        n_link(data.nodes.length),
      );
    }
  }, [data]);

  /*
   * Fit the view when the layout comes to rest.
   *
   * Fitting on a timer does not work: the simulation keeps spreading after the
   * timer fires, so the graph drifts out of frame. `onEngineStop` is the only
   * moment the positions are final.
   */
  const fitToView = useCallback(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.zoomToFit(350, 50);
    // zoomToFit fills the viewport, which on a two- or three-node subgraph
    // magnifies the marks to the size of dinner plates. Cap the magnification so a
    // small neighbourhood still reads as a small neighbourhood.
    window.setTimeout(() => {
      if (fg.zoom() > MAX_ZOOM) fg.zoom(MAX_ZOOM, 200);
    }, 380);
  }, []);

  const truncated = nodes.length > maxNodes;

  const nudgeZoom = (factor: number) => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.zoom(Math.max(0.2, Math.min(MAX_ZOOM * 3, fg.zoom() * factor)), 200);
  };

  return (
    <div ref={wrapRef} className="relative w-full" style={{ height }}>
      <div className="absolute right-3 top-3 z-10 flex flex-col gap-1">
        {[
          { label: "Zoom in", icon: "+", onClick: () => nudgeZoom(1.4) },
          { label: "Zoom out", icon: "\u2212", onClick: () => nudgeZoom(1 / 1.4) },
          { label: "Fit to view", icon: "\u2922", onClick: fitToView },
        ].map((btn) => (
          <button
            key={btn.label}
            type="button"
            onClick={btn.onClick}
            aria-label={btn.label}
            title={btn.label}
            className="grid h-7 w-7 place-items-center rounded-md border border-line bg-surface text-sm text-ink-secondary hover:bg-raised"
          >
            {btn.icon}
          </button>
        ))}
      </div>
      <ForceGraph2D<GraphNode, GraphLink>
        ref={fgRef}
        graphData={data}
        width={width}
        height={height}
        backgroundColor="transparent"
        nodeRelSize={1}
        nodeCanvasObject={drawNode}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x ?? 0, node.y ?? 0, radius(node) + 8, 0, 2 * Math.PI);
          ctx.fill();
        }}
        linkCanvasObject={drawLink as never}
        cooldownTicks={reducedMotion ? 0 : 120}
        warmupTicks={reducedMotion ? 200 : 0}
        d3AlphaDecay={0.035}
        d3VelocityDecay={0.35}
        onNodeHover={(node) => {
          setHovered((node as GraphNode) ?? null);
          if (wrapRef.current) {
            wrapRef.current.style.cursor = node ? "pointer" : "default";
          }
        }}
        onNodeClick={(node) => onSelect?.(node as GraphNode)}
        onBackgroundClick={() => onSelect?.(null)}
        enableNodeDrag={false}
        onEngineStop={fitToView}
      />

      {hovered && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-md border border-line bg-raised px-2.5 py-2 text-xs"
          style={{ boxShadow: "var(--shadow-card)" }}
        >
          <p className="font-mono font-medium">#{hovered.idx}</p>
          <p className="tnum mt-0.5 text-ink-secondary">
            score {hovered.score.toFixed(4)} · {hovered.label} · degree {hovered.degree}
          </p>
        </div>
      )}

      {truncated && (
        <p className="pointer-events-none absolute bottom-2 left-3 text-xs text-ink-muted">
          Showing {maxNodes.toLocaleString()} of {nodes.length.toLocaleString()} nodes
        </p>
      )}
    </div>
  );
}
