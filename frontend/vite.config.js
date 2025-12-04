import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/predict': {
        target: 'http://localhost:5001',
        changeOrigin: true
      },
      '/models': {
        target: 'http://localhost:5001',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:5001',
        changeOrigin: true
      }
    }
  }
})
