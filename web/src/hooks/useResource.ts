import { useCallback, useEffect, useRef, useState } from "react";

export interface Resource<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  reload: () => void;
}

/**
 * Fetch-on-mount with abort, an in-memory cache and a manual reload.
 *
 * Deliberately not TanStack Query: the console has a handful of read-only
 * endpoints and no mutations, so a ~40-line hook covers it without the
 * dependency.
 */
const cache = new Map<string, unknown>();

export function useResource<T>(
  key: string | null,
  fetcher: (signal: AbortSignal) => Promise<T>,
): Resource<T> {
  const [data, setData] = useState<T | null>(() => (key ? ((cache.get(key) as T) ?? null) : null));
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);

  // Keep the latest fetcher without making it a dependency: callers pass inline
  // closures, which would otherwise re-trigger on every render.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!key) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }

    const cached = cache.get(key) as T | undefined;
    if (cached !== undefined && nonce === 0) {
      setData(cached);
      setError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setError(null);

    fetcherRef
      .current(controller.signal)
      .then((value) => {
        if (!active) return;
        cache.set(key, value);
        setData(value);
      })
      .catch((err: Error) => {
        if (!active || err.name === "AbortError") return;
        setError(err);
        setData(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [key, nonce]);

  const reload = useCallback(() => {
    if (key) cache.delete(key);
    setNonce((n) => n + 1);
  }, [key]);

  return { data, error, loading, reload };
}

export function clearResourceCache(): void {
  cache.clear();
}
