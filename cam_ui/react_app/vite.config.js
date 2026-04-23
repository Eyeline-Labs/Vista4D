import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const VISER_PORT   = parseInt(process.env.VISER_PORT)   || 7080
const FASTAPI_PORT = parseInt(process.env.FASTAPI_PORT) || 7000
const REACT_PORT   = parseInt(process.env.REACT_PORT)   || 7173

// Tunneling services (e.g. pinggy, cloudflared) assign random subdomains per session, so we
// allow the whole parent domain with a leading dot — vite treats that as a wildcard suffix match.
const TUNNEL_HOSTS = [
  '.pinggy-free.link',
  '.pinggy.link',
  '.pinggy.io',
  '.trycloudflare.com',
]

export default defineConfig({
  plugins: [react()],
  server: {
    port: REACT_PORT,
    allowedHosts: TUNNEL_HOSTS,
    proxy: {
      '/api': {
        target: `http://localhost:${FASTAPI_PORT}`,
        changeOrigin: true,
      },
    },
    fs: {
      allow: ['..'],  // Allow serving files from cam_ui/ (for fonts)
    },
  },
  preview: {
    port: REACT_PORT,
    allowedHosts: TUNNEL_HOSTS,
    proxy: {
      '/api': {
        target: `http://localhost:${FASTAPI_PORT}`,
        changeOrigin: true,
      },
    },
  },
  define: {
    __VISER_PORT__: VISER_PORT,
  },
})

