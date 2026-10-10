/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        page: 'var(--bg-page)',
        surface: 'var(--bg-surface)',
        ink: {
          primary: 'var(--ink-primary)',
          secondary: 'var(--ink-secondary)',
          muted: 'var(--ink-muted)',
        },
        // Bright categorical accents (validated palette, fixed order:
        // blue, orange, aqua, yellow, magenta, green, violet, red).
        // Used only as identity accents on tiles/icons — body text always
        // uses the `ink` tokens above, never an accent color directly.
        bento: {
          blue: '#2a78d6',
          'blue-dark': '#3987e5',
          orange: '#eb6834',
          'orange-dark': '#d95926',
          aqua: '#1baf7a',
          'aqua-dark': '#199e70',
          yellow: '#eda100',
          'yellow-dark': '#c98500',
          violet: 'var(--accent-violet)',
          'violet-dark': '#9085e9',
          magenta: '#e87ba4',
          red: '#e34948',
        },
        status: {
          good: '#0ca30c',
          warning: '#fab219',
          serious: '#ec835a',
          critical: '#d03b3b',
        },
      },
      borderColor: {
        DEFAULT: 'var(--border)',
      },
      borderRadius: {
        bento: '1.5rem',
      },
    },
  },
  plugins: [],
}
