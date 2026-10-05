import {
  Activity,
  Braces,
  Github,
  Lightbulb,
  Menu,
  Moon,
  Network,
  Sun,
  Users,
  X,
} from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { useHealth } from "../../hooks/useHealth";
import { useTheme } from "../../hooks/useTheme";
import { short } from "../../lib/format";
import { ModeBadge } from "./ModeBadge";
import { SearchBox } from "./SearchBox";

const NAV = [
  { to: "/", label: "Overview", icon: Activity, end: true },
  { to: "/explorer", label: "Graph explorer", icon: Network, end: false },
  { to: "/rings", label: "Fraud rings", icon: Users, end: false },
  { to: "/explain", label: "Explainability", icon: Lightbulb, end: false },
  { to: "/model", label: "Model", icon: Braces, end: false },
];

const REPO = "https://github.com/tejas-mathangi/fraudlens";

export function AppShell() {
  const { health, source } = useHealth();
  const { isDark, toggle } = useTheme();
  const [navOpen, setNavOpen] = useState(false);

  return (
    <div className="flex h-full">
      {/* Sidebar — a drawer below lg, a fixed rail above it. */}
      <div
        className={`fixed inset-0 z-40 bg-black/40 lg:hidden ${navOpen ? "" : "hidden"}`}
        onClick={() => setNavOpen(false)}
        aria-hidden
      />
      <nav
        aria-label="Main"
        className={`fixed inset-y-0 left-0 z-50 flex w-60 flex-col border-r border-line bg-surface transition-transform lg:static lg:translate-x-0 ${
          navOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-2 px-4 py-4">
          <span
            aria-hidden
            className="grid h-7 w-7 place-items-center rounded-md text-sm font-bold text-white"
            style={{ background: "var(--accent)" }}
          >
            F
          </span>
          <div className="min-w-0 leading-tight">
            <p className="truncate text-sm font-semibold">FraudLens</p>
            <p className="truncate text-xs text-ink-muted">Graph fraud detection</p>
          </div>
          <button
            type="button"
            onClick={() => setNavOpen(false)}
            className="ml-auto rounded p-1 text-ink-muted hover:bg-raised lg:hidden"
            aria-label="Close navigation"
          >
            <X size={16} aria-hidden />
          </button>
        </div>

        <ul className="flex-1 space-y-0.5 px-2">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                onClick={() => setNavOpen(false)}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors ${
                    isActive
                      ? "bg-raised font-medium text-ink"
                      : "text-ink-secondary hover:bg-raised hover:text-ink"
                  }`
                }
              >
                <Icon size={15} aria-hidden />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="space-y-2 border-t border-line px-4 py-3 text-xs text-ink-muted">
          {health && (
            <dl className="space-y-1">
              <div className="flex justify-between gap-2">
                <dt>Transactions</dt>
                <dd className="tnum text-ink-secondary">{short(health.nodes)}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt>Edges</dt>
                <dd className="tnum text-ink-secondary">{short(health.edges)}</dd>
              </div>
            </dl>
          )}
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-ink"
          >
            <Github size={13} aria-hidden /> Source
          </a>
        </div>
      </nav>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-line bg-surface/95 px-4 py-2.5 backdrop-blur">
          <button
            type="button"
            onClick={() => setNavOpen(true)}
            className="rounded p-1 text-ink-muted hover:bg-raised lg:hidden"
            aria-label="Open navigation"
          >
            <Menu size={18} aria-hidden />
          </button>

          <div className="ml-auto flex items-center gap-2 sm:gap-3">
            <SearchBox />
            <ModeBadge health={health} source={source} />
            <button
              type="button"
              onClick={toggle}
              className="rounded-md border border-line p-1.5 text-ink-secondary hover:bg-raised"
              aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
            >
              {isDark ? <Sun size={14} aria-hidden /> : <Moon size={14} aria-hidden />}
            </button>
          </div>
        </header>

        <main className="min-w-0 flex-1 overflow-y-auto p-4 lg:p-6">
          <Outlet context={{ health }} />
        </main>
      </div>
    </div>
  );
}
