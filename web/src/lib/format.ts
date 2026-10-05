/** Number and label formatting shared across the console. */

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const plain = new Intl.NumberFormat("en");

export const num = (v: number | null | undefined): string =>
  v === null || v === undefined ? "—" : plain.format(v);

export const short = (v: number | null | undefined): string =>
  v === null || v === undefined ? "—" : compact.format(v);

export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

export const score = (v: number | null | undefined, digits = 4): string =>
  v === null || v === undefined ? "—" : v.toFixed(digits);

/** Elliptic txIds are 9 digits; truncate in dense tables. */
export const txId = (v: number | null | undefined): string =>
  v === null || v === undefined || v === 0 ? "—" : String(v);

export const titleCase = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);
