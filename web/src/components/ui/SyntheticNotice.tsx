import { FlaskConical } from "lucide-react";

/**
 * Shown when the loaded artifacts were generated from a synthetic graph rather
 * than the real Elliptic dataset.
 *
 * The repository ships sample artifacts so the console is not empty on a fresh
 * clone, and it should be impossible to mistake those for real results.
 */
export function SyntheticNotice({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <div
      className="mb-4 flex items-start gap-2.5 rounded-card border px-3.5 py-2.5 text-xs"
      style={{
        borderColor: "var(--status-warning)",
        background: "color-mix(in srgb, var(--status-warning) 8%, transparent)",
      }}
    >
      <FlaskConical size={15} aria-hidden style={{ color: "var(--status-warning)" }} className="mt-px shrink-0" />
      <p className="text-ink-secondary">
        <span className="font-medium text-ink">Sample data.</span> These figures come from a
        synthetic graph generated so the console has something to render without the 697&nbsp;MB
        Elliptic dataset. Download the dataset and run{" "}
        <code className="rounded bg-raised px-1 py-0.5 font-mono">fraudlens export</code> to
        replace them with real results.
      </p>
    </div>
  );
}
