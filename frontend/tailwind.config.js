/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        medical: {
          bg: '#0f172a',      // slate-900
          surface: '#1e293b', // slate-800
          text: '#f8fafc',    // slate-50
          muted: '#94a3b8',   // slate-400
        }
      }
    },
  },
  plugins: [],
}
