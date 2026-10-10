import { Settings } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import type { Theme } from '@/lib/theme'
import type { HealthInfo } from '@/lib/types'

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b py-3 text-sm last:border-b-0">
      <span className="text-ink-secondary">{label}</span>
      <span className="font-medium text-ink-primary">{value}</span>
    </div>
  )
}

export function SettingsView({
  health,
  theme,
  onThemeChange,
}: {
  health: HealthInfo | null
  theme: Theme
  onThemeChange: (theme: Theme) => void
}) {
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <header className="mb-6 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-ink-muted to-ink-secondary text-white">
          <Settings size={20} />
        </div>
        <div>
          <h1 className="text-xl font-bold text-ink-primary md:text-2xl">Settings</h1>
          <p className="text-sm text-ink-secondary">Backend configuration and appearance.</p>
        </div>
      </header>

      <div className="flex max-w-xl flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Appearance</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-between">
            <span className="text-sm text-ink-primary">Dark mode</span>
            <Switch checked={theme === 'dark'} onCheckedChange={(checked) => onThemeChange(checked ? 'dark' : 'light')} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Backend configuration</CardTitle>
            <CardDescription>Read live from <code>GET /api/health</code>.</CardDescription>
          </CardHeader>
          <CardContent className="pt-0">
            <Row label="Model" value={health?.model_id ?? '—'} />
            <Row label="Device" value={health?.device ?? '—'} />
            <Row label="Default inference steps" value={health ? String(health.default_steps) : '—'} />
            <Row label="Default guidance scale" value={health ? health.default_guidance_scale.toFixed(1) : '—'} />
            <Row
              label="Default image size"
              value={health ? `${health.default_width}×${health.default_height}` : '—'}
            />
          </CardContent>
        </Card>

        <p className="text-xs text-ink-muted">
          Pick a different model from the dropdown in the Generate view's Model &amp; settings card to switch for
          this session — only one model is kept in memory at a time, so switching unloads the current one and loads
          the new one on your next Generate (slower the first time, since it also has to download). To change the
          curated list itself, or the model that's loaded before anyone picks one, edit{' '}
          <code className="rounded bg-black/[0.04] px-1 py-0.5 dark:bg-white/[0.06]">AVAILABLE_MODELS</code> in{' '}
          <code className="rounded bg-black/[0.04] px-1 py-0.5 dark:bg-white/[0.06]">backend/model.py</code>, or set{' '}
          <code className="rounded bg-black/[0.04] px-1 py-0.5 dark:bg-white/[0.06]">
            MODEL_ID="runwayml/stable-diffusion-v1-5" uvicorn backend.main:app --reload
          </code>{' '}
          — see the README for details.
        </p>
      </div>
    </div>
  )
}
