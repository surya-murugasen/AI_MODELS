import { Card } from '@/components/ui/card'

export function StatTile({
  icon,
  accent,
  label,
  value,
}: {
  icon: React.ReactNode
  accent: string
  label: string
  value: string
}) {
  return (
    <Card className="flex flex-col gap-3 p-5">
      <div
        className="flex h-9 w-9 items-center justify-center rounded-lg"
        style={{ backgroundColor: `${accent}1a`, color: accent }}
      >
        {icon}
      </div>
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">{label}</p>
        <p className="truncate text-sm font-semibold text-ink-primary" title={value}>
          {value}
        </p>
      </div>
    </Card>
  )
}
