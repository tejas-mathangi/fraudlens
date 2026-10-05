import { Loader2, Search } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { search } from "../../lib/api";

/**
 * Resolve a node index or a real Elliptic txId.
 *
 * The txId path is the point: the original dashboard only accepted the internal
 * 0..203,768 index, so if you had an actual transaction id there was no way to
 * look it up.
 */
export function SearchBox() {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  // Cmd/Ctrl+K focuses search, the convention for this kind of console.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = value.trim();
    if (!query) return;

    setBusy(true);
    setMessage(null);
    try {
      const result = await search(query);
      if (result.match) {
        navigate(`/node/${result.match.idx}`);
        setValue("");
      } else {
        setMessage("No match in the available data");
      }
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="relative" role="search">
      <label htmlFor="node-search" className="sr-only">
        Search by node index or transaction id
      </label>
      <Search
        size={14}
        aria-hidden
        className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-muted"
      />
      <input
        id="node-search"
        ref={inputRef}
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          setMessage(null);
        }}
        inputMode="numeric"
        placeholder="Node index or txId"
        className="w-full rounded-md border border-line bg-raised py-1.5 pl-8 pr-8 font-mono text-xs text-ink placeholder:font-sans placeholder:text-ink-muted sm:w-56"
      />
      {busy && (
        <Loader2
          size={13}
          aria-hidden
          className="absolute right-2.5 top-1/2 -translate-y-1/2 animate-spin text-ink-muted"
        />
      )}
      {message && (
        <p
          role="status"
          className="absolute right-0 top-full z-20 mt-1 whitespace-nowrap rounded-md border border-line bg-raised px-2 py-1 text-xs text-ink-secondary"
        >
          {message}
        </p>
      )}
    </form>
  );
}
