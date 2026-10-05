import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Relative base so the build works from a subpath (GitHub Pages) as well as root.
  base: "./",
  server: {
    port: 5173,
    proxy: {
      // In dev the console talks to the FastAPI service without CORS round-trips.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
});
