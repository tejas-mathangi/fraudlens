import type { ReactNode } from "react";

interface PanelProps {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  /** Remove body padding — for a canvas or a table that bleeds to the edges. */
  flush?: boolean;
}

export function Panel({ title, subtitle, actions, children, className = "", flush }: PanelProps) {
  return (
    <section
      className={`flex min-w-0 flex-col rounded-card border border-line bg-surface ${className}`}
      style={{ boxShadow: "var(--shadow-card)" }}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold leading-tight">{title}</h2>}
            {subtitle && (
              <p className="mt-0.5 text-xs leading-snug text-ink-secondary">{subtitle}</p>
            )}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={flush ? "min-w-0 flex-1" : "min-w-0 flex-1 p-4"}>{children}</div>
    </section>
  );
}
