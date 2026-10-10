// Small helper around the manual dark-mode toggle. The initial class is set
// synchronously in index.html (see the inline script) to avoid a flash; this
// module just keeps React state and localStorage in sync after that.
export type Theme = 'light' | 'dark'

export function getInitialTheme(): Theme {
  if (typeof document === 'undefined') return 'light'
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem('theme', theme)
  } catch {
    // Storage can throw in some contexts (e.g. private browsing) — the
    // toggle still works for the current page load, it just won't persist.
  }
}
