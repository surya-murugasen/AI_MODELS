import { useId, useRef, useState } from 'react'

const VIOLET = '#9085e9'
const BLUE = '#2a78d6'
const FORECAST_GREEN = '#1baf7a'

/** Circular "meter" for a single 0-100 ratio, e.g. predicted accuracy. */
export function ProgressRing({
  value,
  size = 200,
  strokeWidth = 16,
  primaryLabel,
  secondaryLabel,
}: {
  value: number
  size?: number
  strokeWidth?: number
  primaryLabel: string
  secondaryLabel: string
}) {
  const gradientId = useId()
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const filled = (clamp01(value / 100)) * circumference

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${primaryLabel}: ${secondaryLabel}`}>
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={VIOLET} />
          <stop offset="100%" stopColor={BLUE} />
        </linearGradient>
      </defs>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={VIOLET}
        strokeOpacity={0.12}
        strokeWidth={strokeWidth}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference - filled}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: 'stroke-dasharray 0.4s ease' }}
      />
      <text x="50%" y="47%" textAnchor="middle" dominantBaseline="middle" className="fill-ink-primary text-[2.1rem] font-bold">
        {primaryLabel}
      </text>
      <text x="50%" y="63%" textAnchor="middle" dominantBaseline="middle" className="fill-ink-muted text-[0.75rem] font-medium">
        {secondaryLabel}
      </text>
    </svg>
  )
}

export interface AccuracyTrendHistoryPoint {
  id: string
  label: string
  dailyAccuracy: number
  movingAverage: number
}

export interface AccuracyTrendForecastPoint {
  dateKey: string
  label: string
  predicted: number
}

/**
 * Image accuracy over time: real daily accuracy dots + line, a 7-day moving
 * average, and — only when `forecast` is non-empty — a projected trend line
 * continuing past a shaded "Prediction" region. Crosshair + tooltip on
 * hover. Renders a "Not enough data yet." empty state when `history` is
 * empty (nothing to plot).
 */
export function AccuracyTrendChart({
  history,
  forecast,
  height = 260,
}: {
  history: AccuracyTrendHistoryPoint[]
  forecast: AccuracyTrendForecastPoint[]
  height?: number
}) {
  const width = 640
  const padding = { top: 16, right: 12, bottom: 24, left: 34 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom
  const areaGradientId = useId()
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  if (history.length === 0) {
    return (
      <div className="flex items-center justify-center text-sm text-ink-muted" style={{ height }}>
        Not enough data yet.
      </div>
    )
  }

  const combined = [
    ...history.map((h) => ({
      label: h.label,
      daily: h.dailyAccuracy as number | null,
      moving: h.movingAverage as number | null,
      predicted: null as number | null,
      isForecast: false,
    })),
    ...forecast.map((f) => ({
      label: f.label,
      daily: null as number | null,
      moving: null as number | null,
      predicted: f.predicted as number | null,
      isForecast: true,
    })),
  ]
  const n = combined.length
  const xFor = (i: number) => (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW)
  const yFor = (v: number) => innerH - clamp01(v / 100) * innerH

  const dailyPath = pathFor(history.map((h, i): [number, number] => [xFor(i), yFor(h.dailyAccuracy)]))
  const movingPath = pathFor(history.map((h, i): [number, number] => [xFor(i), yFor(h.movingAverage)]))

  const lastHistoryX = xFor(history.length - 1)
  const areaPath =
    dailyPath && `M0,${innerH} ${dailyPath.replace('M', 'L')} L${lastHistoryX},${innerH} Z`

  const forecastPoints: [number, number][] =
    forecast.length > 0
      ? [
          [lastHistoryX, yFor(history[history.length - 1].movingAverage)],
          ...forecast.map((f, i) => [xFor(history.length + i), yFor(f.predicted)] as [number, number]),
        ]
      : []
  const forecastPath = pathFor(forecastPoints)
  const dividerX = forecast.length > 0 ? xFor(history.length - 0.5) : null

  const ticks = [0, 25, 50, 75, 100]
  // Evenly spaced tick indices, always including the first and last point.
  // A fixed step can seat the forced-last label right next to the previous
  // one when n doesn't divide evenly, so the tail end swaps them instead of
  // adding both.
  const desiredLabelCount = Math.min(n, 7)
  const labelIndices = new Set<number>()
  if (desiredLabelCount <= 1) {
    labelIndices.add(0)
  } else {
    const step = Math.max(1, Math.ceil((n - 1) / (desiredLabelCount - 1)))
    for (let i = 0; i < n; i += step) labelIndices.add(i)
    const lastRegular = Math.max(...labelIndices)
    if (n - 1 - lastRegular < step) labelIndices.delete(lastRegular)
    labelIndices.add(n - 1)
  }

  function handleMove(e: React.PointerEvent<SVGRectElement>) {
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const ratio = clamp01((e.clientX - rect.left) / rect.width)
    const x = ratio * width - padding.left
    const idx = n === 1 ? 0 : Math.round((x / innerW) * (n - 1))
    setHoverIndex(clamp(idx, 0, n - 1))
  }

  const hovered = hoverIndex !== null ? combined[hoverIndex] : null

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-ink-secondary">
        <LegendSwatch color={VIOLET} label="Daily Average Accuracy (%)" kind="dot" />
        <LegendSwatch color={BLUE} label="7-Day Moving Average (%)" kind="dashed" />
        {forecast.length > 0 && (
          <LegendSwatch color={FORECAST_GREEN} label={`Prediction (Next ${forecast.length} Days)`} kind="dashed" />
        )}
      </div>

      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height={height}
          role="img"
          aria-label="Image accuracy over time"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id={areaGradientId} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={VIOLET} stopOpacity={0.25} />
              <stop offset="100%" stopColor={VIOLET} stopOpacity={0} />
            </linearGradient>
          </defs>
          <g transform={`translate(${padding.left},${padding.top})`}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={0} x2={innerW} y1={yFor(t)} y2={yFor(t)} stroke="var(--border)" strokeWidth={1} />
                <text x={-8} y={yFor(t)} textAnchor="end" dominantBaseline="middle" className="fill-ink-muted text-[10px]">
                  {t}
                </text>
              </g>
            ))}

            {dividerX !== null && (
              <>
                <rect x={dividerX} y={0} width={Math.max(0, innerW - dividerX)} height={innerH} fill={FORECAST_GREEN} fillOpacity={0.06} />
                <line x1={dividerX} x2={dividerX} y1={0} y2={innerH} stroke="var(--border)" strokeWidth={1} strokeDasharray="3 3" />
                <text x={innerW} y={11} textAnchor="end" className="fill-ink-muted text-[10px] font-medium">
                  Prediction
                </text>
              </>
            )}

            {areaPath && <path d={areaPath} fill={`url(#${areaGradientId})`} stroke="none" />}
            {movingPath && (
              <path d={movingPath} fill="none" stroke={BLUE} strokeWidth={2} strokeDasharray="5 4" strokeLinecap="round" />
            )}
            {forecastPath && (
              <path d={forecastPath} fill="none" stroke={FORECAST_GREEN} strokeWidth={2} strokeDasharray="5 4" strokeLinecap="round" />
            )}
            {dailyPath && (
              <path d={dailyPath} fill="none" stroke={VIOLET} strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" />
            )}

            {history.map((h, i) => (
              <circle
                key={`d-${h.id}`}
                cx={xFor(i)}
                cy={yFor(h.dailyAccuracy)}
                r={hoverIndex === i ? 5 : 3.5}
                fill={VIOLET}
                stroke="var(--bg-surface)"
                strokeWidth={1.5}
              />
            ))}
            {forecast.map((f, i) => (
              <circle
                key={`f-${f.dateKey}`}
                cx={xFor(history.length + i)}
                cy={yFor(f.predicted)}
                r={hoverIndex === history.length + i ? 5 : 3}
                fill={FORECAST_GREEN}
                stroke="var(--bg-surface)"
                strokeWidth={1.5}
              />
            ))}

            {combined.map((c, i) =>
              labelIndices.has(i) ? (
                <text
                  key={c.label + i}
                  x={xFor(i)}
                  y={innerH + 18}
                  textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}
                  className="fill-ink-muted text-[10px]"
                >
                  {c.label}
                </text>
              ) : null,
            )}

            {hoverIndex !== null && (
              <line x1={xFor(hoverIndex)} x2={xFor(hoverIndex)} y1={0} y2={innerH} stroke={VIOLET} strokeOpacity={0.3} strokeWidth={1} />
            )}

            <rect
              x={0}
              y={0}
              width={innerW}
              height={innerH}
              fill="transparent"
              onPointerMove={handleMove}
              onPointerLeave={() => setHoverIndex(null)}
            />
          </g>
        </svg>

        {hovered && hoverIndex !== null && (
          <div
            className="pointer-events-none absolute top-1 flex flex-col gap-1 rounded-lg border bg-surface px-3 py-2 text-xs shadow-md"
            style={{
              left: `clamp(4px, ${((xFor(hoverIndex) + padding.left) / width) * 100}%, calc(100% - 160px))`,
              transform: 'translateX(-50%)',
            }}
          >
            <p className="font-medium text-ink-primary">{hovered.label}</p>
            {!hovered.isForecast ? (
              <>
                <p className="flex items-center gap-1.5 text-ink-secondary">
                  <Dot color={VIOLET} />
                  Daily Accuracy:
                  <span className="font-semibold text-ink-primary">{hovered.daily?.toFixed(1)}%</span>
                </p>
                <p className="flex items-center gap-1.5 text-ink-secondary">
                  <Dot color={BLUE} />
                  7-Day Avg:
                  <span className="font-semibold text-ink-primary">{hovered.moving?.toFixed(1)}%</span>
                </p>
              </>
            ) : (
              <p className="flex items-center gap-1.5 text-ink-secondary">
                <Dot color={FORECAST_GREEN} />
                Predicted:
                <span className="font-semibold text-ink-primary">{hovered.predicted?.toFixed(1)}%</span>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function pathFor(points: (readonly [number, number])[]): string {
  if (points.length < 2) return ''
  return points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x},${y}`).join(' ')
}

function LegendSwatch({ color, label, kind }: { color: string; label: string; kind: 'dot' | 'dashed' }) {
  return (
    <span className="flex items-center gap-1.5">
      {kind === 'dot' ? (
        <span className="h-2 w-2 flex-none rounded-full" style={{ backgroundColor: color }} />
      ) : (
        <svg width="14" height="2" className="flex-none" aria-hidden="true">
          <line x1="0" y1="1" x2="14" y2="1" stroke={color} strokeWidth={2} strokeDasharray="4 3" />
        </svg>
      )}
      {label}
    </span>
  )
}

function Dot({ color }: { color: string }) {
  return <span className="inline-block h-1.5 w-1.5 flex-none rounded-full" style={{ backgroundColor: color }} />
}

/** Horizontal magnitude bars, one hue, value directly labeled at the tip. */
export function BarList({
  items,
  max = 100,
  accent = BLUE,
  valueSuffix = '%',
}: {
  items: { label: string; value: number; sublabel?: string }[]
  max?: number
  accent?: string
  valueSuffix?: string
}) {
  if (items.length === 0) {
    return <p className="py-6 text-center text-sm text-ink-muted">Not enough data yet.</p>
  }
  return (
    <div className="flex flex-col gap-3.5">
      {items.map((item) => (
        <div key={item.label} title={`${item.label}: ${item.value.toFixed(1)}${valueSuffix}`}>
          <div className="mb-1 flex items-baseline justify-between gap-2">
            <span className="truncate text-sm font-medium text-ink-primary">{item.label}</span>
            <span className="flex-none text-xs font-semibold text-ink-secondary">
              {item.value.toFixed(1)}
              {valueSuffix}
            </span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full" style={{ backgroundColor: `${accent}1a` }}>
            <div
              className="h-full rounded-full"
              style={{ width: `${clamp01(item.value / max) * 100}%`, backgroundColor: accent, transition: 'width 0.4s ease' }}
            />
          </div>
          {item.sublabel && <p className="mt-0.5 text-[11px] text-ink-muted">{item.sublabel}</p>}
        </div>
      ))}
    </div>
  )
}

/** Part-to-whole: one segmented bar + legend, categorical colors in fixed order. */
export function StackedShareBar({
  segments,
}: {
  segments: { label: string; share: number; color: string }[]
}) {
  if (segments.length === 0) {
    return <p className="py-6 text-center text-sm text-ink-muted">Not enough data yet.</p>
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="flex h-4 w-full overflow-hidden rounded-full" style={{ backgroundColor: 'var(--border)' }}>
        {segments.map((s, i) => (
          <div
            key={s.label}
            title={`${s.label}: ${s.share.toFixed(1)}%`}
            style={{
              width: `${s.share}%`,
              backgroundColor: s.color,
              marginRight: i === segments.length - 1 ? 0 : 2,
            }}
          />
        ))}
      </div>
      <ul className="flex flex-col gap-2">
        {segments.map((s) => (
          <li key={s.label} className="flex items-center justify-between gap-3 text-sm">
            <span className="flex min-w-0 items-center gap-2">
              <span className="h-2.5 w-2.5 flex-none rounded-full" style={{ backgroundColor: s.color }} />
              <span className="truncate text-ink-primary">{s.label}</span>
            </span>
            <span className="flex-none font-semibold text-ink-secondary">{s.share.toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export const CATEGORICAL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#898781']

function clamp01(v: number) {
  return Math.min(1, Math.max(0, v))
}
function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v))
}
