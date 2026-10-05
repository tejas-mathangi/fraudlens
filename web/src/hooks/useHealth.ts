import { useEffect, useState } from "react";

import { currentSource, getHealth, onSourceChange, type Source } from "../lib/api";
import type { Health } from "../lib/types";

export interface HealthState {
  health: Health | null;
  source: Source | null;
}

/**
 * Poll `/api/health` while the backend is still warming.
 *
 * A live backend takes ~40 s to parse the dataset and score the graph, so the
 * console renders a warming state and re-checks every three seconds until the
 * backend reports ready. Once ready (or once we have fallen back to static demo
 * data) polling stops.
 */
export function useHealth(): HealthState {
  const [health, setHealth] = useState<Health | null>(null);
  const [source, setSource] = useState<Source | null>(currentSource());

  useEffect(() => onSourceChange(setSource), []);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;

    const tick = async () => {
      const controller = new AbortController();
      try {
        const next = await getHealth(controller.signal);
        if (!active) return;
        setHealth(next);
        if (next.status === "warming") {
          timer = window.setTimeout(tick, 3000);
        }
      } catch {
        if (active) timer = window.setTimeout(tick, 5000);
      }
    };

    tick();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  return { health, source };
}
