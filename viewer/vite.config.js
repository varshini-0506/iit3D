import { defineConfig } from "vite";

// base "./" keeps every URL relative, so the built site works from any folder:
// GitHub Pages project sites, Netlify, or a plain static file server.
export default defineConfig({
  base: "./",
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
  server: { host: true },
});
