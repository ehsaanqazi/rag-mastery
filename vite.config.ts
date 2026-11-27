import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Served from https://ehsaanqazi.github.io/rag-mastery/ on GitHub Pages.
export default defineConfig({
  base: '/rag-mastery/',
  plugins: [react(), tailwindcss()],
})
