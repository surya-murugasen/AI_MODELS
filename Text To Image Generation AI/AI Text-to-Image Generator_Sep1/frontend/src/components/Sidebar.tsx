import {
  BrainCircuit,
  Clapperboard,
  Images,
  History as HistoryIcon,
  Info,
  Settings as SettingsIcon,
  Sparkles,
  Wand2,
  Workflow,
  Cloud,
  Shield,
} from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'
import type { Theme } from '@/lib/theme'
import type { ViewKey } from '@/lib/types'

const NAV_ITEMS: { key: ViewKey; label: string; icon: React.ReactNode; badge?: string }[] = [
  { key: 'generate', label: 'Image Generate', icon: <Wand2 size={18} /> },
  { key: 'gallery', label: 'Gallery', icon: <Images size={18} /> },
  { key: 'history', label: 'History', icon: <HistoryIcon size={18} /> },
  { key: 'predictive', label: 'Predictive AI', icon: <BrainCircuit size={18} />, badge: 'LIVE' },
  { key: 'story-video', label: 'Story to Video', icon: <Clapperboard size={18} /> },
  { key: 'how-it-works', label: 'How Algorithms Work', icon: <Workflow size={18} /> },
  { key: 'settings', label: 'Settings', icon: <SettingsIcon size={18} /> },
  { key: 'cloud-learning', label: 'Cloud Learning', icon: <Cloud size={18} /> },
  { key: 'security-learning', label: 'Security Learning', icon: <Shield size={18} /> },
  { key: 'about', label: 'About', icon: <Info size={18} /> },
]

export function Sidebar({
  view,
  onNavigate,
  theme,
  onThemeChange,
}: {
  view: ViewKey
  onNavigate: (view: ViewKey) => void
  theme: Theme
  onThemeChange: (theme: Theme) => void
}) {
  return (
    <aside className="flex h-full w-60 flex-none flex-col border-r bg-surface px-4 py-6">
      <div className="mb-8 flex items-center gap-3 px-2">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-bento-violet to-bento-blue text-white">
          <Sparkles size={18} />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-bold text-ink-primary">OmniAI</p>
          <p className="text-xs text-ink-secondary">Studio</p>
        </div>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.key}
            type="button"
            onClick={() => onNavigate(item.key)}
            className={cn(
              'flex items-center justify-between rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-colors cursor-pointer',
              view === item.key
                ? 'bg-bento-blue/10 text-bento-blue font-semibold'
                : 'text-ink-secondary hover:bg-black/[0.03] hover:text-ink-primary dark:hover:bg-white/[0.05]',
            )}
          >
            <div className="flex items-center gap-3">
              {item.icon}
              {item.label}
            </div>
            {item.badge && (
              <span className="flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-1.5 py-0.5 text-[9px] font-bold text-emerald-500">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                {item.badge}
              </span>
            )}
          </button>
        ))}
      </nav>

      <div className="flex items-center justify-between rounded-xl border px-3 py-2.5">
        <span className="text-sm font-medium text-ink-primary">Dark mode</span>
        <Switch
          checked={theme === 'dark'}
          onCheckedChange={(checked) => onThemeChange(checked ? 'dark' : 'light')}
          aria-label="Toggle dark mode"
        />
      </div>
    </aside>
  )
}
