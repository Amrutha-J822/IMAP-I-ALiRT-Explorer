import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND_HTTP = process.env.VITE_BACKEND_HTTP ?? "http://127.0.0.1:8000";
const BACKEND_WS = process.env.VITE_BACKEND_WS ?? "ws://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: BACKEND_HTTP,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target: BACKEND_WS,
        ws: true,
      },
    },
  },
});
