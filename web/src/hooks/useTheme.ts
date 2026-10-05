import { useCallback, useEffect, useState } from "react";

type Theme = "light" | "dark" | "system";

const KEY = "fraudlens-theme";

function read(): Theme {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* private mode / blocked storage — fall through to system */
  }
  return "system";
}

/**
 * Light/dark toggle. Dark is the default look, but both modes are explicitly
 * designed: the dark palette is the same hues re-stepped for the dark surface,
 * not an automatic inversion.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(read);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, theme);
    } catch {
      /* non-fatal: the attribute above already applied the theme */
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((current) => {
      if (current === "system") {
        const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        return systemDark ? "light" : "dark";
      }
      return current === "dark" ? "light" : "dark";
    });
  }, []);

  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);

  return { theme, isDark, toggle };
}
