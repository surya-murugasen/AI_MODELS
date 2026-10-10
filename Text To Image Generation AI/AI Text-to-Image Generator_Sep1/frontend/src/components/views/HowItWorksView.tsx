import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  ArrowDown,
  ArrowRight,
  Blend,
  BrainCircuit,
  ChartSpline,
  Check,
  CheckCircle2,
  ChevronDown,
  Columns3,
  Cpu,
  Eraser,
  FileSpreadsheet,
  Gauge,
  GitBranch,
  GitCompare,
  GitFork,
  Layers,
  LineChart,
  Percent,
  RefreshCw,
  RotateCcw,
  ScanSearch,
  Search,
  Sparkles,
  SplitSquareVertical,
  Table2,
  Tags,
  ToggleLeft,
  Trees,
  TrendingUp,
  Upload,
  Waypoints,
  Workflow,
  Wrench,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import {
  computeFullBrakingPipeline,
  DEFAULT_BRAKING_PIPELINE,
  parseUploadedBrakingRows,
  type BrakingDistancePipelineData,
} from '@/lib/brakingDistancePipeline'
import {
  computeFullPipeline,
  DEFAULT_PIPELINE,
  parseUploadedRows,
  type PipelineData,
} from '@/lib/howItWorksPipeline'
import type { PipelineCommon } from '@/lib/howItWorksShared'
import {
  computeFullHousePipeline,
  DEFAULT_HOUSE_PIPELINE,
  parseUploadedHouseRows,
  type HousePricePipelineData,
} from '@/lib/housePricePipeline'
import { cn } from '@/lib/utils'
import { HowItWorksTrainScene } from './HowItWorksTrainScene'

const ALGORITHMS = [
  'Linear Regression',
  'Multiple Linear Regression',
  'Polynomial Regression',
  'Logistic Regression',
  'Decision Tree',
  'Random Forest',
  'K-Nearest Neighbors (KNN)',
  'Support Vector Machine (SVM)',
  'Naive Bayes',
  'K-Means Clustering',
  'Neural Network',
  'Gradient Boosting',
]

// Each algorithm gets a distinct icon that reflects what it actually is
// (a branching tree for Decision Tree, a cluster of trees for Random
// Forest, a brain/circuit for Neural Network, and so on) rather than a
// single generic icon shown for every selection.
const ALGORITHM_ICONS: Record<string, typeof Cpu> = {
  'Linear Regression': LineChart,
  'Multiple Linear Regression': Layers,
  'Polynomial Regression': ChartSpline,
  'Logistic Regression': ToggleLeft,
  'Decision Tree': GitBranch,
  'Random Forest': Trees,
  'K-Nearest Neighbors (KNN)': Waypoints,
  'Support Vector Machine (SVM)': SplitSquareVertical,
  'Naive Bayes': Percent,
  'K-Means Clustering': Blend,
  'Neural Network': BrainCircuit,
  'Gradient Boosting': TrendingUp,
}

function AlgorithmDropdown({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const filtered = ALGORITHMS.filter((a) => a.toLowerCase().includes(query.toLowerCase()))

  useEffect(() => {
    if (!open) return
    const t = setTimeout(() => inputRef.current?.focus(), 10)
    return () => clearTimeout(t)
  }, [open])

  return (
    <DropdownMenu
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (!o) setQuery('')
      }}
    >
      <DropdownMenuTrigger asChild>
        <Button variant="secondary" size="sm" className="min-w-[230px] justify-between gap-2">
          <span className="flex items-center gap-2 truncate">
            <AlgorithmIcon name={value} size={14} className="flex-none text-bento-blue" />
            <span className="truncate">{value}</span>
          </span>
          <ChevronDown size={14} className="flex-none text-ink-muted" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-[280px] p-0">
        <div className="border-b p-2" onKeyDown={(e) => e.stopPropagation()}>
          <div className="flex items-center gap-2 rounded-lg border bg-page px-2 py-1.5">
            <Search size={13} className="flex-none text-ink-muted" />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search algorithms…"
              className="w-full bg-transparent text-xs text-ink-primary outline-none placeholder:text-ink-muted"
            />
          </div>
        </div>
        <div className="max-h-64 overflow-y-auto p-1">
          {filtered.length === 0 && <p className="px-2.5 py-3 text-center text-xs text-ink-muted">No matching algorithms.</p>}
          {filtered.map((algo) => (
            <DropdownMenuItem
              key={algo}
              onClick={() => {
                onChange(algo)
                setOpen(false)
                setQuery('')
              }}
              className="justify-between"
            >
              <span className="flex items-center gap-2">
                <AlgorithmIcon name={algo} size={14} className="flex-none text-ink-secondary" />
                {algo}
              </span>
              {algo === value && <Check size={14} className="flex-none text-bento-blue" />}
            </DropdownMenuItem>
          ))}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function AlgorithmIcon({ name, size, className }: { name: string; size: number; className?: string }) {
  const Icon = ALGORITHM_ICONS[name] ?? Cpu
  return <Icon size={size} className={className} />
}

function UnderConstructionPanel({ algorithm, onBack }: { algorithm: string; onBack: () => void }) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="flex flex-col items-center gap-4 px-6 py-16 text-center">
        <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-bento-orange to-bento-pink text-white shadow-sm">
          <AlgorithmIcon name={algorithm} size={28} />
          <span className="absolute -bottom-1.5 -right-1.5 flex h-6 w-6 items-center justify-center rounded-full border-2 border-page bg-ink-primary text-page">
            <Wrench size={12} />
          </span>
        </div>
        <div>
          <h2 className="text-lg font-bold text-ink-primary">{algorithm} walkthrough — in development</h2>
          <p className="mx-auto mt-1.5 max-w-md text-sm text-ink-secondary">
            This page currently walks through <strong className="text-ink-primary">Linear Regression</strong> only —
            it's the algorithm behind the app's real Train Arrival Prediction formula. A dedicated {algorithm}{' '}
            walkthrough is on the roadmap.
          </p>
        </div>
        <Button size="sm" variant="secondary" onClick={onBack}>
          Back to Linear Regression
        </Button>
      </CardContent>
    </Card>
  )
}

// Semantic color system (kept consistent across every card):
//   blue    = input / raw data
//   orange  = processing / needs-attention
//   green   = successful processing / final success
//   pink    = target / result
//   purple  = ML / model
//   muted   = "not part of the shipped app" tint; brightens when a live
//             teaching demo is actually running on uploaded data
const C = {
  blue: '#2a78d6',
  blueDark: '#3987e5',
  orange: '#eb6834',
  orangeDark: '#d95926',
  green: '#1baf7a',
  greenDark: '#199e70',
  pink: '#e87ba4',
  pinkDark: '#d4638e',
  purple: '#4a3aa7',
  purpleDark: '#9085e9',
  successGreen: '#0ca30c',
  mutedBlue: '#93a5c4',
  mutedBlueDark: '#7c93b8',
  mutedPurple: '#ab9fd4',
  mutedPurpleDark: '#8f80c2',
  mutedPink: '#d9a7bd',
  mutedPinkDark: '#c48aa4',
}

/** Demo-only cards (7-10, 12) use a muted tint by default, and brighten to
 * the vivid category color once a real live demo is actually running. */
function demoGradient(isLive: boolean, muted: [string, string], vivid: [string, string]): [string, string] {
  return isLive ? vivid : muted
}

interface Stage {
  n: number
  title: string
  icon: ReactNode
  gradient: [string, string]
  used: boolean // false = this stage doesn't actually happen in the shipped app
  explanation: string
  visual: ReactNode
}

// ---------------------------------------------------------------------------
// Small shared building blocks
// ---------------------------------------------------------------------------
function Chip({ children, tone }: { children: ReactNode; tone?: 'bad' | 'good' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md px-1.5 py-0.5 font-mono text-[10px]',
        tone === 'bad' && 'bg-status-critical/10 text-status-critical',
        tone === 'good' && 'bg-bento-aqua/10 text-bento-aqua',
        !tone && 'bg-black/[0.04] text-ink-primary dark:bg-white/[0.06]',
      )}
    >
      {children}
    </span>
  )
}

function Badge({ label, value, tone }: { label: string; value: ReactNode; tone: 'good' | 'warn' | 'default' }) {
  return (
    <div
      className={cn(
        'flex flex-col rounded-md px-1.5 py-1',
        tone === 'good' && 'bg-bento-aqua/10',
        tone === 'warn' && 'bg-bento-orange/10',
        tone === 'default' && 'bg-black/[0.04] dark:bg-white/[0.06]',
      )}
    >
      <span className="text-[8px] uppercase tracking-wide text-ink-muted">{label}</span>
      <span
        className={cn(
          'text-[11px] font-bold',
          tone === 'good' && 'text-bento-aqua',
          tone === 'warn' && 'text-bento-orange',
          tone === 'default' && 'text-ink-primary',
        )}
      >
        {value}
      </span>
    </div>
  )
}

function FlowArrow({ direction = 'right', color = C.blue, size = 16 }: { direction?: 'right' | 'down'; color?: string; size?: number }) {
  const Icon = direction === 'right' ? ArrowRight : ArrowDown
  return (
    <span className="hiw-flow-arrow inline-flex flex-none" style={{ color }}>
      <Icon size={size} />
    </span>
  )
}

function FlowNode({ icon, label, delay, color }: { icon: ReactNode; label: string; delay?: number; color: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="hiw-seq-pulse flex h-7 w-7 items-center justify-center rounded-lg text-white"
        style={{ backgroundColor: color, animationDelay: delay !== undefined ? `${delay}s` : undefined }}
      >
        {icon}
      </div>
      <span className="text-[8px] font-medium text-ink-muted">{label}</span>
    </div>
  )
}

function MiniTable({ headers, rows }: { headers: string[]; rows: ReactNode[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left text-[10px]">
        <thead>
          <tr className="text-ink-muted">
            {headers.map((h) => (
              <th key={h} className="pb-1 pr-2 font-semibold uppercase tracking-wide">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-t border-black/[0.04] dark:border-white/[0.06]">
              {row.map((cell, ci) => (
                <td key={ci} className="py-1 pr-2 font-mono font-medium text-ink-primary">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LiveTag({ isLive }: { isLive: boolean }) {
  return (
    <span
      className={cn(
        'w-fit rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide',
        isLive ? 'bg-bento-aqua/15 text-bento-aqua' : 'bg-black/[0.05] text-ink-muted dark:bg-white/[0.08]',
      )}
    >
      {isLive ? 'Live demo — from your file' : 'Not used in this app'}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Step 1 — Upload: file card + progress bar entering the pipeline
// ---------------------------------------------------------------------------
function UploadVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 rounded-lg border border-dashed border-bento-blue/40 bg-bento-blue/5 px-2.5 py-2">
        <FileSpreadsheet size={16} className="flex-none text-bento-blue" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-[11px] font-medium text-ink-primary">{p.fileName}</p>
          <p className="text-[9px] text-ink-muted">{p.fileSizeKB.toFixed(1)} KB</p>
        </div>
        <CheckCircle2 size={14} className="hiw-check-pop-fast flex-none text-status-good" />
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/[0.06] dark:bg-white/[0.08]">
        <div className="hiw-progress-fill h-full rounded-full bg-gradient-to-r from-bento-blue to-bento-aqua" />
      </div>
      <p className="text-center text-[9px] text-ink-muted">entering the pipeline…</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 2 — Read: Raw File → Parser → Table, with a crossfading preview
// ---------------------------------------------------------------------------
function ReadExcelVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-center gap-1">
        <FlowNode icon={<FileSpreadsheet size={13} />} label="Raw File" delay={0} color={C.blue} />
        <FlowArrow size={11} />
        <FlowNode icon={<Cpu size={13} />} label="Parser" delay={0.6} color={C.blueDark} />
        <FlowArrow size={11} />
        <FlowNode icon={<Table2 size={13} />} label="Table" delay={1.2} color={C.green} />
      </div>
      <div className="relative h-[52px]">
        <div className="hiw-crossfade-a absolute inset-0 flex items-center justify-center rounded-md bg-black/[0.04] px-2 text-center font-mono text-[9px] text-ink-muted dark:bg-white/[0.06]">
          {p.sampleRawLine}
        </div>
        <div className="hiw-crossfade-b absolute inset-0">
          <MiniTable headers={p.sampleRow.map((c) => c.label)} rows={[p.sampleRow.map((c) => c.value)]} />
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 3 — Inspect: a small data-quality dashboard
// ---------------------------------------------------------------------------
function InspectVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="grid grid-cols-2 gap-1.5">
      <Badge label="Rows found" value={p.totalRows} tone="default" />
      <Badge label="Columns" value={p.columnsDetected} tone="default" />
      <Badge label="Missing" value={`${p.missingCount} flagged`} tone={p.missingCount > 0 ? 'warn' : 'good'} />
      <Badge label="Duplicates" value={`${p.duplicateCount} found`} tone={p.duplicateCount > 0 ? 'warn' : 'good'} />
      <Badge label="Valid rows" value={`${p.validCount} / ${p.totalRows}`} tone="good" />
      <Badge label="Ready?" value="Yes" tone="good" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 4 — Clean: Before → Cleaning → After
// ---------------------------------------------------------------------------
function CleanVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="flex flex-col gap-1.5">
      {p.cleaningExamples.map(({ field, before, after }) => (
        <div key={field} className="flex items-center gap-1.5 text-[10px]">
          <span className="w-12 flex-none text-ink-muted">{field}</span>
          <Chip tone="bad">{before}</Chip>
          <span className="hiw-seq-pulse flex h-5 w-5 flex-none items-center justify-center rounded-full bg-bento-orange/15 text-bento-orange">
            <Eraser size={10} />
          </span>
          <Chip tone="good">{after}</Chip>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 5 — Prepare Features: raw → encoded, plus already-numeric fields
// ---------------------------------------------------------------------------
function FeaturesVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="flex flex-col gap-1.5 text-[10px]">
      {p.encodingExamples.map(({ field, raw, code }) => (
        <div key={field} className="flex items-center gap-1.5">
          <Chip>
            {field}: &quot;{raw}&quot;
          </Chip>
          <FlowArrow size={11} color={C.green} />
          <Chip tone="good">code: {code}</Chip>
        </div>
      ))}
      <div className="flex flex-wrap gap-1 pt-0.5">
        {p.numericFeatureNames.map((f) => (
          <span key={f} className="rounded-full bg-bento-aqua/10 px-2 py-0.5 text-[9px] font-medium text-bento-aqua">
            {f} (already numeric)
          </span>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 6 — Separate Input & Target: dataset columns fan out into X / y
// ---------------------------------------------------------------------------
function SeparateXYVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap justify-center gap-1">
        {[...p.xFields, p.yField].map((f) => (
          <span key={f} className="rounded bg-black/[0.04] px-1.5 py-0.5 font-mono text-[9px] text-ink-muted dark:bg-white/[0.06]">
            {f}
          </span>
        ))}
      </div>
      <div className="flex justify-center">
        <FlowArrow direction="down" size={12} />
      </div>
      <div className="flex items-stretch gap-2 text-[10px]">
        <div className="flex-1 rounded-lg bg-bento-blue/10 p-1.5">
          <p className="mb-1 font-mono font-bold text-bento-blue">X (input)</p>
          <p className="leading-snug text-ink-muted">{p.xFields.join(', ')}</p>
        </div>
        <div className="flex-1 rounded-lg bg-bento-magenta/10 p-1.5">
          <p className="mb-1 font-mono font-bold text-bento-magenta">y (target)</p>
          <p className="leading-snug text-ink-muted">{p.yField}</p>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 7 — Split: dataset divides into training/test, each routed onward
// ---------------------------------------------------------------------------
function SplitVisual({ p }: { p: PipelineCommon }) {
  const total = p.splitTrainCount + p.splitTestCount || 1
  const trainPct = Math.round((p.splitTrainCount / total) * 100)
  return (
    <div className="flex flex-col gap-2">
      <p className="text-center text-[9px] text-ink-muted">
        {p.isLive ? 'Your uploaded rows' : `Example: ${p.totalRows} collected rows`}
      </p>
      <div className="flex h-3 w-full overflow-hidden rounded-full">
        <div className="bg-[color:var(--hiw-muted-blue)]" style={{ width: `${trainPct}%` }} />
        <div className="bg-[color:var(--hiw-muted-purple)]" style={{ width: `${100 - trainPct}%` }} />
      </div>
      <div className="flex items-center justify-between text-[9px]">
        <span className="text-ink-muted">
          Training ({p.splitTrainCount}, {trainPct}%)
        </span>
        <span className="text-ink-muted">
          Test ({p.splitTestCount}, {100 - trainPct}%)
        </span>
      </div>
      <div className="flex flex-col gap-1 pt-1 text-[9px]">
        <div className="flex items-center gap-1.5 text-ink-muted">
          <span className="h-2 w-2 flex-none rounded-full bg-[color:var(--hiw-muted-blue)]" />
          <span>Training</span>
          <FlowArrow size={10} color="var(--hiw-muted-blue-dark)" />
          <Cpu size={11} className="text-[color:var(--hiw-muted-blue-dark)]" />
          <span>Model</span>
        </div>
        <div className="flex items-center gap-1.5 text-ink-muted">
          <span className="h-2 w-2 flex-none rounded-full bg-[color:var(--hiw-muted-purple)]" />
          <span>Test</span>
          <FlowArrow size={10} color="var(--hiw-muted-purple-dark)" />
          <GitCompare size={11} className="text-[color:var(--hiw-muted-purple-dark)]" />
          <span>held back for Evaluation</span>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 8 — Choose Model: tiny glyph per model, one selected, with a reason
// ---------------------------------------------------------------------------
function LineGlyph() {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" fill="none">
      {[2, 6, 10, 14].map((y, i) => (
        <circle key={i} cx={2 + i * 6} cy={14 - y} r="1.2" fill="currentColor" />
      ))}
      <path d="M2 12 L8 8 L14 5 L20 2" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  )
}
function TreeGlyph() {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" fill="none">
      <circle cx="11" cy="2.5" r="1.6" fill="currentColor" />
      <path d="M11 4 L5 9 M11 4 L17 9" stroke="currentColor" strokeWidth="1.1" />
      <circle cx="5" cy="10" r="1.4" fill="currentColor" />
      <circle cx="17" cy="10" r="1.4" fill="currentColor" />
      <path d="M5 11.4 L2 15 M5 11.4 L8 15 M17 11.4 L14 15 M17 11.4 L20 15" stroke="currentColor" strokeWidth="1" />
    </svg>
  )
}
function MarginGlyph() {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" fill="none">
      <line x1="1" y1="14" x2="21" y2="2" stroke="currentColor" strokeWidth="1.1" strokeDasharray="2 1.5" />
      <line x1="1" y1="8" x2="21" y2="-4" stroke="currentColor" strokeWidth="1" opacity="0.4" />
      <circle cx="6" cy="4" r="1.3" fill="currentColor" />
      <circle cx="10" cy="3" r="1.3" fill="currentColor" />
      <circle cx="14" cy="12" r="1.3" fill="currentColor" opacity="0.5" />
      <circle cx="18" cy="13" r="1.3" fill="currentColor" opacity="0.5" />
    </svg>
  )
}
function CurveGlyph() {
  return (
    <svg width="22" height="16" viewBox="0 0 22 16" fill="none">
      {[2, 4, 9, 13].map((y, i) => (
        <circle key={i} cx={2 + i * 6} cy={14 - y} r="1.2" fill="currentColor" />
      ))}
      <path d="M2 13 Q11 -2 20 13" stroke="currentColor" strokeWidth="1.2" fill="none" />
    </svg>
  )
}
function ModelPickerVisual({ p }: { p: PipelineCommon }) {
  const options: { name: string; glyph: ReactNode }[] = [
    { name: 'Linear Regression', glyph: <LineGlyph /> },
    { name: 'Polynomial Regression', glyph: <CurveGlyph /> },
    { name: 'Random Forest', glyph: <TreeGlyph /> },
    { name: 'SVM', glyph: <MarginGlyph /> },
  ]
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-center gap-1.5">
        {options.map((opt) => {
          const selected = opt.name === p.modelName || (opt.name === 'Linear Regression' && p.modelName.includes('Linear Regression'))
          return (
            <div
              key={opt.name}
              className={cn(
                'flex flex-col items-center gap-1 rounded-lg border px-2 py-1.5',
                selected
                  ? 'border-[color:var(--hiw-muted-purple)] bg-[color:var(--hiw-muted-purple)]/15'
                  : 'border-dashed text-ink-muted',
              )}
            >
              <span className={selected ? 'text-[color:var(--hiw-muted-purple-dark)]' : 'text-ink-muted'}>{opt.glyph}</span>
              <span
                className={cn(
                  'text-center text-[8px] font-medium leading-tight',
                  selected ? 'font-bold text-[color:var(--hiw-muted-purple-dark)]' : 'text-ink-muted',
                )}
              >
                {opt.name}
              </span>
              {selected && (
                <span className="rounded-full bg-[color:var(--hiw-muted-purple)] px-1.5 py-0.5 text-[7px] font-bold uppercase text-white">
                  Selected
                </span>
              )}
            </div>
          )
        })}
      </div>
      <p className="rounded-md bg-black/[0.03] p-1.5 text-center text-[9px] leading-snug text-ink-muted dark:bg-white/[0.05]">
        <span className="font-semibold text-[color:var(--hiw-muted-purple-dark)]">Why?</span> {p.whyModel}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 9 — Train Model: animated loop + shrinking-error chart + weight ticker
// ---------------------------------------------------------------------------
function TrainingLoopVisual({ p }: { p: PipelineCommon }) {
  const nodes = ['Data', 'Model', 'Predict', 'Error', 'Adjust']
  const maxErr = Math.max(...p.errorHistory, 1)
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-center gap-0.5">
        {nodes.map((node, i) => (
          <span key={node} className="flex items-center gap-0.5">
            <span
              className="hiw-seq-pulse rounded-md border border-dashed px-1.5 py-1 text-[8px] font-medium text-ink-muted"
              style={{ animationDelay: `${i * 0.4}s` }}
            >
              {node}
            </span>
            {i < nodes.length - 1 && <ArrowRight size={9} className="text-ink-muted" />}
          </span>
        ))}
        <RefreshCw size={11} className="hiw-spin-slow ml-0.5 text-[color:var(--hiw-muted-purple-dark)]" />
      </div>

      <div>
        <p className="mb-1 text-center text-[8px] text-ink-muted">
          {p.isLive ? 'Real error, measured while training on your rows' : 'Error shrinking over training iterations'}
        </p>
        <div className="flex h-8 items-end justify-center gap-1">
          {p.errorHistory.map((e, i) => (
            <div key={i} className="w-2 rounded-t bg-[color:var(--hiw-muted-purple)]" style={{ height: `${Math.max(6, (e / maxErr) * 100)}%` }} />
          ))}
        </div>
      </div>

      <div className="flex justify-center gap-2 text-[8px]">
        <span className="hiw-ticker-a rounded bg-black/[0.04] px-1.5 py-0.5 font-mono text-ink-muted dark:bg-white/[0.06]">
          {p.weightTickerLabel} ≈ {p.weightEarly}
        </span>
        <span className="hiw-ticker-b -ml-[74px] rounded bg-[color:var(--hiw-muted-purple)]/15 px-1.5 py-0.5 font-mono font-bold text-[color:var(--hiw-muted-purple-dark)]">
          {p.weightTickerLabel} ≈ {p.weightFinal}
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 10 — Learned Model: a tiny weighted network diagram
// ---------------------------------------------------------------------------
function LearnedNetworkVisual({ p }: { p: PipelineCommon }) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-center text-[8px] text-ink-muted">
        {p.isLive ? 'Learned network (fit on your rows)' : 'Illustrative learned network (hypothetical)'}
      </p>
      <svg viewBox="0 0 140 70" className="w-full">
        {p.learnedWeights.map((inp, i) => {
          const y = 8 + i * 18
          return (
            <line
              key={inp.label}
              x1={26}
              y1={y}
              x2={112}
              y2={35}
              stroke="var(--hiw-muted-purple)"
              strokeWidth={Math.max(0.6, inp.magnitude * 2.4)}
              opacity={0.35 + inp.magnitude * 0.5}
            />
          )
        })}
        {p.learnedWeights.map((inp, i) => {
          const y = 8 + i * 18
          return (
            <g key={inp.label}>
              <circle cx={20} cy={y} r="6" fill="var(--hiw-muted-blue)" />
              <text x={2} y={y + 2.5} fontSize="6" fill="var(--ink-muted)">
                {inp.label}
              </text>
            </g>
          )
        })}
        <circle cx={118} cy={35} r="8" fill="var(--hiw-muted-purple-dark)" />
        <text x={106} y={53} fontSize="6" fill="var(--ink-muted)">
          {p.yFieldShort}
        </text>
      </svg>
      <p className="text-center text-[8px] text-ink-muted">Thicker line = stronger learned influence</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 11 — Test / Predict: unseen data flows into the (real) formula
// ---------------------------------------------------------------------------
function PredictVisual({ p }: { p: PipelineData }) {
  const e = p.predictExample
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative flex items-center justify-center gap-2 text-[9px] text-ink-muted">
        <span className="rounded border px-1.5 py-0.5">New data</span>
        <span className="relative w-6">
          <FlowArrow size={14} color={C.pink} />
          <span className="hiw-particle absolute left-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-bento-magenta" />
        </span>
        <span className="rounded border px-1.5 py-0.5">
          <Cpu size={10} className="mr-1 inline" />
          Formula
        </span>
        <FlowArrow size={14} color={C.pink} />
        <span className="rounded border border-bento-magenta/40 bg-bento-magenta/10 px-1.5 py-0.5 font-semibold text-bento-magenta">
          Prediction
        </span>
      </div>
      <div className="font-mono text-[10px] text-ink-muted">
        {e.previousDelay} + {e.speedPenalty.toFixed(2)} + {e.trafficDelay} + {e.weatherDelay} = {e.delayMin}m
      </div>
      <div className="flex items-center justify-between rounded-lg bg-bento-magenta/10 px-2 py-1.5">
        <span className="text-base font-black text-ink-primary">{e.predictedArrival}</span>
        <span className="text-sm font-bold text-bento-magenta">{e.confidence}%</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Step 12 — Evaluate: comparison table + metric cards + mini scatter chart
// ---------------------------------------------------------------------------
function fmtMetric(v: number | null, suffix: string): string {
  return v === null ? 'n/a' : `${v.toFixed(2)}${suffix}`
}

function EvaluateVisual({ p }: { p: PipelineCommon }) {
  const rows = p.evaluationRows
  const maxVal = Math.max(...rows.map((r) => Math.max(r.actual, r.predicted)), 1)
  return (
    <div className="flex flex-col gap-2">
      <p className="text-center text-[9px] text-ink-muted">
        {p.isLive ? 'Held-out test rows from your file' : 'Illustrative example rows'}
      </p>
      <MiniTable
        headers={['Actual', 'Predicted', 'Error']}
        rows={rows.map((r) => [`${r.actual}${p.metricUnit}`, `${r.predicted}${p.metricUnit}`, `${r.error}${p.metricUnit}`])}
      />
      <div className="grid grid-cols-3 gap-1">
        {[
          ['MAE', fmtMetric(p.metrics.mae, p.metricUnit)],
          ['RMSE', fmtMetric(p.metrics.rmse, p.metricUnit)],
          ['R²', p.metrics.r2 === null ? 'n/a' : p.metrics.r2.toFixed(2)],
        ].map(([label, val]) => (
          <div key={label} className="rounded-md bg-[color:var(--hiw-muted-pink)]/15 px-1 py-1 text-center">
            <p className="text-[7px] uppercase tracking-wide text-[color:var(--hiw-muted-pink-dark)]">{label}</p>
            <p className="text-[11px] font-bold text-[color:var(--hiw-muted-pink-dark)]">{val}</p>
          </div>
        ))}
      </div>
      {rows.length > 0 && (
        <svg viewBox="0 0 100 50" className="w-full">
          <line x1="4" y1="46" x2="96" y2="4" stroke="var(--hiw-muted-pink)" strokeWidth="0.8" strokeDasharray="2 2" opacity="0.6" />
          {rows.map((r, i) => {
            const x = 4 + (r.actual / maxVal) * 92
            const y = 46 - (r.predicted / maxVal) * 42
            return <circle key={i} cx={x} cy={y} r="2.4" fill="var(--hiw-muted-pink-dark)" />
          })}
          <text x="2" y="8" fontSize="6" fill="var(--ink-muted)">
            Predicted
          </text>
          <text x="66" y="48" fontSize="6" fill="var(--ink-muted)">
            Actual
          </text>
        </svg>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Final Result — GSAP-animated train illustration + prominent success panel
// ---------------------------------------------------------------------------
function FinalResultVisual({ p }: { p: PipelineData }) {
  return <HowItWorksTrainScene result={p.finalResult} />
}

// ---------------------------------------------------------------------------
// Step card shell
// ---------------------------------------------------------------------------
function StepBadge({ n, gradient }: { n: number; gradient: [string, string] }) {
  return (
    <span
      className="flex h-7 w-7 flex-none items-center justify-center rounded-full text-xs font-bold text-white shadow-sm"
      style={{ backgroundImage: `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})` }}
    >
      {n}
    </span>
  )
}

function StageCard({ stage, isLive }: { stage: Stage; isLive: boolean }) {
  return (
    <Card
      className={cn(
        'h-full overflow-hidden transition-shadow hover:shadow-md',
        !stage.used && 'border-dashed bg-black/[0.015] dark:bg-white/[0.02]',
      )}
    >
      <div
        className="h-1 w-full"
        style={{ backgroundImage: `linear-gradient(90deg, ${stage.gradient[0]}, ${stage.gradient[1]})`, opacity: stage.used ? 1 : 0.55 }}
      />
      <CardContent className="flex h-full flex-col gap-3 p-4">
        <div className="flex items-center gap-2.5">
          <StepBadge n={stage.n} gradient={stage.gradient} />
          <div
            className="flex h-8 w-8 flex-none items-center justify-center rounded-lg text-white shadow-sm"
            style={{
              backgroundImage: `linear-gradient(135deg, ${stage.gradient[0]}, ${stage.gradient[1]})`,
              opacity: stage.used ? 1 : 0.85,
            }}
          >
            {stage.icon}
          </div>
          <h3 className="text-sm font-bold leading-tight text-ink-primary">{stage.title}</h3>
        </div>

        {!stage.used && <LiveTag isLive={isLive} />}

        <p className="text-xs leading-relaxed text-ink-secondary">{stage.explanation}</p>

        <div className="mt-auto overflow-hidden rounded-lg border bg-page p-2.5">{stage.visual}</div>
      </CardContent>
    </Card>
  )
}

function buildStages(p: PipelineData): Stage[] {
  return [
    {
      n: 1,
      title: 'Upload Excel',
      icon: <FileSpreadsheet size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'You upload a CSV/Excel file of train records — raw bytes the browser hasn’t looked inside yet.',
      visual: <UploadVisual p={p} />,
    },
    {
      n: 2,
      title: 'Read Excel',
      icon: <Table2 size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'The file is read into raw text, then a parser splits it on line breaks and commas into an actual table.',
      visual: <ReadExcelVisual p={p} />,
    },
    {
      n: 3,
      title: 'Inspect Data',
      icon: <ScanSearch size={16} />,
      gradient: [C.orange, C.orangeDark],
      used: true,
      explanation: 'The system checks which columns exist, what type each is, and where values are missing or duplicated.',
      visual: <InspectVisual p={p} />,
    },
    {
      n: 4,
      title: 'Clean Data',
      icon: <Eraser size={16} />,
      gradient: [C.orange, C.orangeDark],
      used: true,
      explanation: 'Blank or unrecognized values are replaced with safe defaults, so no row is skipped.',
      visual: <CleanVisual p={p} />,
    },
    {
      n: 5,
      title: 'Prepare Features',
      icon: <Tags size={16} />,
      gradient: [C.green, C.greenDark],
      used: true,
      explanation: 'Category text is converted into numeric codes; already-numeric columns are kept as-is.',
      visual: <FeaturesVisual p={p} />,
    },
    {
      n: 6,
      title: 'Separate Input & Target',
      icon: <Columns3 size={16} />,
      gradient: [C.blue, C.pink],
      used: true,
      explanation: 'Speed/delay/traffic/weather become the input (X); minutes late is the target (y) to estimate.',
      visual: <SeparateXYVisual p={p} />,
    },
    {
      n: 7,
      title: 'Split Data',
      icon: <GitFork size={16} />,
      gradient: demoGradient(p.isLive, [C.mutedBlue, C.mutedBlueDark], [C.blue, C.blueDark]),
      used: false,
      explanation: 'A trained model would hold out test rows; the shipped app scores every row immediately instead.',
      visual: <SplitVisual p={p} />,
    },
    {
      n: 8,
      title: 'Choose Model',
      icon: <Cpu size={16} />,
      gradient: demoGradient(p.isLive, [C.mutedPurple, C.mutedPurpleDark], [C.purple, C.purpleDark]),
      used: false,
      explanation: 'The shipped app runs one fixed formula for every row instead of a selectable model type.',
      visual: <ModelPickerVisual p={p} />,
    },
    {
      n: 9,
      title: 'Train Model',
      icon: <RefreshCw size={16} />,
      gradient: demoGradient(p.isLive, [C.mutedPurple, C.mutedPurpleDark], [C.purple, C.purpleDark]),
      used: false,
      explanation: p.isLive
        ? 'This demo actually ran gradient descent on your rows, adjusting weights each pass to shrink error.'
        : 'A real model repeats this loop, adjusting its weights to shrink error each time.',
      visual: <TrainingLoopVisual p={p} />,
    },
    {
      n: 10,
      title: 'Learned Model',
      icon: <BrainCircuit size={16} />,
      gradient: demoGradient(p.isLive, [C.mutedPurple, C.mutedPurpleDark], [C.purple, C.purpleDark]),
      used: false,
      explanation: p.isLive
        ? 'These are the actual weights this demo learned from your rows — shown below.'
        : 'The shipped app saves nothing between runs; the same input always gives the exact same answer.',
      visual: <LearnedNetworkVisual p={p} />,
    },
    {
      n: 11,
      title: 'Test / Predict',
      icon: <Gauge size={16} />,
      gradient: [C.pink, C.pinkDark],
      used: true,
      explanation: "This row's numbers flow into the formula, producing a predicted arrival and confidence.",
      visual: <PredictVisual p={p} />,
    },
    {
      n: 12,
      title: 'Evaluate',
      icon: <GitCompare size={16} />,
      gradient: demoGradient(p.isLive, [C.mutedPink, C.mutedPinkDark], [C.pink, C.pinkDark]),
      used: false,
      explanation: p.isLive
        ? "This demo's model was scored against your held-out test rows — see the metrics below."
        : 'There are no recorded actual arrival times in a schedule file to compare predictions against.',
      visual: <EvaluateVisual p={p} />,
    },
    {
      n: 13,
      title: 'Final Result',
      icon: <CheckCircle2 size={16} />,
      gradient: [C.successGreen, C.green],
      used: true,
      explanation: 'The completed pipeline’s output: a predicted time, delay, and confidence, ready for the screen.',
      visual: <FinalResultVisual p={p} />,
    },
  ]
}

// ---------------------------------------------------------------------------
// Multiple Linear Regression — house price example (predict + final result)
// ---------------------------------------------------------------------------
function HousePricePredictVisual({ p }: { p: HousePricePipelineData }) {
  const e = p.predictExample
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative flex items-center justify-center gap-2 text-[9px] text-ink-muted">
        <span className="rounded border px-1.5 py-0.5">New data</span>
        <span className="relative w-6">
          <FlowArrow size={14} color={C.pink} />
          <span className="hiw-particle absolute left-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-bento-magenta" />
        </span>
        <span className="rounded border px-1.5 py-0.5">
          <Cpu size={10} className="mr-1 inline" />
          Formula
        </span>
        <FlowArrow size={14} color={C.pink} />
        <span className="rounded border border-bento-magenta/40 bg-bento-magenta/10 px-1.5 py-0.5 font-semibold text-bento-magenta">
          Prediction
        </span>
      </div>
      <div className="font-mono text-[9px] leading-snug text-ink-muted">
        50,000 + {Math.round(e.sqftTerm).toLocaleString()} + {Math.round(e.bedroomTerm).toLocaleString()} −{' '}
        {Math.round(e.agePenaltyTerm).toLocaleString()} + {Math.round(e.locationTerm).toLocaleString()} = $
        {e.predictedPrice.toLocaleString()}
      </div>
      <div className="flex items-center justify-between rounded-lg bg-bento-magenta/10 px-2 py-1.5">
        <span className="text-base font-black text-ink-primary">${e.predictedPrice.toLocaleString()}</span>
        <span className="text-sm font-bold text-bento-magenta">{e.confidence}%</span>
      </div>
    </div>
  )
}

function HousePriceFinalResultVisual({ p }: { p: HousePricePipelineData }) {
  const r = p.finalResult
  const terms = [
    { label: 'Base price', value: r.basePrice },
    { label: `${r.sqft} sqft`, value: r.sqftContribution },
    { label: `${r.bedrooms} bedrooms`, value: r.bedroomContribution },
    { label: `${r.ageYears} yrs old`, value: r.ageContribution },
    { label: `${r.locationTier} location`, value: r.locationContribution },
  ]
  const maxAbs = Math.max(...terms.map((t) => Math.abs(t.value)), 1)
  return (
    <div className="flex flex-col gap-2">
      <div className="rounded-xl border border-status-good/30 bg-status-good/10 p-3">
        <div className="mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={14} className="text-status-good" />
            <span className="text-[10px] font-bold uppercase tracking-wide text-status-good">Pipeline complete</span>
          </div>
          <span className="font-mono text-[9px] text-ink-muted">
            {r.sqft} sqft · {r.bedrooms}bd · {r.locationTier}
          </span>
        </div>
        <div className="flex items-end justify-between">
          <span className="text-2xl font-black leading-none text-ink-primary">${r.predictedPrice.toLocaleString()}</span>
          <span className="text-xs font-bold text-bento-orange">{r.confidence}% confident</span>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        {terms.map((t) => (
          <div key={t.label} className="flex items-center gap-1.5 text-[9px]">
            <span className="w-20 flex-none truncate text-ink-muted">{t.label}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-black/[0.04] dark:bg-white/[0.06]">
              <div
                className={cn('h-full rounded-full', t.value >= 0 ? 'bg-status-good' : 'bg-status-critical')}
                style={{ width: `${(Math.abs(t.value) / maxAbs) * 100}%` }}
              />
            </div>
            <span className={cn('w-16 flex-none text-right font-mono', t.value >= 0 ? 'text-status-good' : 'text-status-critical')}>
              {t.value >= 0 ? '+' : '−'}${Math.abs(Math.round(t.value)).toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function buildHousePriceStages(p: HousePricePipelineData): Stage[] {
  return [
    {
      n: 1,
      title: 'Upload Excel',
      icon: <FileSpreadsheet size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'You upload a CSV/Excel file of property listings — raw bytes the browser hasn’t looked inside yet.',
      visual: <UploadVisual p={p} />,
    },
    {
      n: 2,
      title: 'Read Excel',
      icon: <Table2 size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'The file is read into raw text, then a parser splits it on line breaks and commas into an actual table.',
      visual: <ReadExcelVisual p={p} />,
    },
    {
      n: 3,
      title: 'Inspect Data',
      icon: <ScanSearch size={16} />,
      gradient: [C.orange, C.orangeDark],
      used: true,
      explanation: 'The system checks which columns exist, what type each is, and where values are missing or duplicated.',
      visual: <InspectVisual p={p} />,
    },
    {
      n: 4,
      title: 'Clean Data',
      icon: <Eraser size={16} />,
      gradient: [C.orange, C.orangeDark],
      used: true,
      explanation: 'Blank or unrecognized values are replaced with safe defaults, so no listing is skipped.',
      visual: <CleanVisual p={p} />,
    },
    {
      n: 5,
      title: 'Prepare Features',
      icon: <Tags size={16} />,
      gradient: [C.green, C.greenDark],
      used: true,
      explanation: 'Location tier (text) is converted into a numeric code; square footage, bedrooms, and age are kept as-is.',
      visual: <FeaturesVisual p={p} />,
    },
    {
      n: 6,
      title: 'Separate Input & Target',
      icon: <Columns3 size={16} />,
      gradient: [C.blue, C.pink],
      used: true,
      explanation: 'Sqft/bedrooms/age/location become the input (X); sale price is the target (y) to estimate.',
      visual: <SeparateXYVisual p={p} />,
    },
    {
      n: 7,
      title: 'Split Data',
      icon: <GitFork size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'A slice of listings is held back as a test set, never seen during training, so evaluation stays honest.',
      visual: <SplitVisual p={p} />,
    },
    {
      n: 8,
      title: 'Choose Model',
      icon: <Cpu size={16} />,
      gradient: [C.purple, C.purpleDark],
      used: true,
      explanation: 'Multiple Linear Regression is picked because several independent numeric factors add up to the price.',
      visual: <ModelPickerVisual p={p} />,
    },
    {
      n: 9,
      title: 'Train Model',
      icon: <RefreshCw size={16} />,
      gradient: [C.purple, C.purpleDark],
      used: true,
      explanation: p.isLive
        ? 'This demo actually ran gradient descent on your listings, adjusting weights each pass to shrink error.'
        : 'Gradient descent repeats this loop, adjusting weights to shrink error each time.',
      visual: <TrainingLoopVisual p={p} />,
    },
    {
      n: 10,
      title: 'Learned Model',
      icon: <BrainCircuit size={16} />,
      gradient: [C.purple, C.purpleDark],
      used: true,
      explanation: p.isLive
        ? 'These are the actual weights this demo learned from your listings — shown below.'
        : 'Relative influence of each factor on the formula’s own price output for the worked example.',
      visual: <LearnedNetworkVisual p={p} />,
    },
    {
      n: 11,
      title: 'Test / Predict',
      icon: <Gauge size={16} />,
      gradient: [C.pink, C.pinkDark],
      used: true,
      explanation: "This listing's numbers flow into the formula, producing a predicted price and confidence.",
      visual: <HousePricePredictVisual p={p} />,
    },
    {
      n: 12,
      title: 'Evaluate',
      icon: <GitCompare size={16} />,
      gradient: [C.pink, C.pinkDark],
      used: true,
      explanation: p.isLive
        ? "This demo's model was scored against your held-out test listings — see the metrics below."
        : 'Illustrative held-out listings, scored against the formula’s own predicted prices.',
      visual: <EvaluateVisual p={p} />,
    },
    {
      n: 13,
      title: 'Final Result',
      icon: <CheckCircle2 size={16} />,
      gradient: [C.successGreen, C.green],
      used: true,
      explanation: 'The completed pipeline’s output: a predicted price broken down by contributing factor, ready for the screen.',
      visual: <HousePriceFinalResultVisual p={p} />,
    },
  ]
}

// ---------------------------------------------------------------------------
// Polynomial Regression — vehicle stopping-distance example (predict + final result)
// ---------------------------------------------------------------------------
function BrakingPredictVisual({ p }: { p: BrakingDistancePipelineData }) {
  const e = p.predictExample
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative flex items-center justify-center gap-2 text-[9px] text-ink-muted">
        <span className="rounded border px-1.5 py-0.5">New data</span>
        <span className="relative w-6">
          <FlowArrow size={14} color={C.pink} />
          <span className="hiw-particle absolute left-0 top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full bg-bento-magenta" />
        </span>
        <span className="rounded border px-1.5 py-0.5">
          <Cpu size={10} className="mr-1 inline" />
          Formula
        </span>
        <FlowArrow size={14} color={C.pink} />
        <span className="rounded border border-bento-magenta/40 bg-bento-magenta/10 px-1.5 py-0.5 font-semibold text-bento-magenta">
          Prediction
        </span>
      </div>
      <div className="font-mono text-[9px] leading-snug text-ink-muted">
        {e.reactionDistance}m (reaction) + {e.brakingDistance}m (braking, ∝ speed²) = {e.totalDistance}m
      </div>
      <div className="flex items-center justify-between rounded-lg bg-bento-magenta/10 px-2 py-1.5">
        <span className="text-base font-black text-ink-primary">{e.totalDistance}m</span>
        <span className="text-sm font-bold text-bento-magenta">{e.confidence}%</span>
      </div>
    </div>
  )
}

function BrakingFinalResultVisual({ p }: { p: BrakingDistancePipelineData }) {
  const r = p.finalResult
  const total = r.totalDistance || 1
  const reactionPct = (r.reactionDistance / total) * 100
  const brakingPct = 100 - reactionPct
  return (
    <div className="flex flex-col gap-2">
      <div className="rounded-xl border border-status-good/30 bg-status-good/10 p-3">
        <div className="mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={14} className="text-status-good" />
            <span className="text-[10px] font-bold uppercase tracking-wide text-status-good">Pipeline complete</span>
          </div>
          <span className="font-mono text-[9px] text-ink-muted">
            {r.speedKmh} km/h · {r.roadCondition}
          </span>
        </div>
        <div className="flex items-end justify-between">
          <span className="text-2xl font-black leading-none text-ink-primary">{r.totalDistance}m</span>
          <span className="text-xs font-bold text-bento-orange">{r.confidence}% confident</span>
        </div>
      </div>
      <div className="flex h-4 w-full overflow-hidden rounded-full text-[8px] font-bold text-white">
        <div className="flex items-center justify-center bg-bento-blue" style={{ width: `${reactionPct}%` }}>
          {reactionPct > 18 && `${r.reactionDistance}m`}
        </div>
        <div className="flex items-center justify-center bg-bento-orange" style={{ width: `${brakingPct}%` }}>
          {brakingPct > 18 && `${r.brakingDistance}m`}
        </div>
      </div>
      <div className="flex justify-between text-[8px] text-ink-muted">
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-bento-blue" />
          Reaction distance
        </span>
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-bento-orange" />
          Braking distance (∝ speed²)
        </span>
      </div>
    </div>
  )
}

function buildBrakingStages(p: BrakingDistancePipelineData): Stage[] {
  return [
    {
      n: 1,
      title: 'Upload Excel',
      icon: <FileSpreadsheet size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'You upload a CSV/Excel file of vehicle speed tests — raw bytes the browser hasn’t looked inside yet.',
      visual: <UploadVisual p={p} />,
    },
    {
      n: 2,
      title: 'Read Excel',
      icon: <Table2 size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'The file is read into raw text, then a parser splits it on line breaks and commas into an actual table.',
      visual: <ReadExcelVisual p={p} />,
    },
    {
      n: 3,
      title: 'Inspect Data',
      icon: <ScanSearch size={16} />,
      gradient: [C.orange, C.orangeDark],
      used: true,
      explanation: 'The system checks which columns exist, what type each is, and where values are missing or duplicated.',
      visual: <InspectVisual p={p} />,
    },
    {
      n: 4,
      title: 'Clean Data',
      icon: <Eraser size={16} />,
      gradient: [C.orange, C.orangeDark],
      used: true,
      explanation: 'Blank or unrecognized values are replaced with safe defaults, so no test run is skipped.',
      visual: <CleanVisual p={p} />,
    },
    {
      n: 5,
      title: 'Prepare Features',
      icon: <Tags size={16} />,
      gradient: [C.green, C.greenDark],
      used: true,
      explanation: 'Road condition (text) becomes a numeric code, and a speed² term is engineered — the standard trick for fitting a curve with linear regression.',
      visual: <FeaturesVisual p={p} />,
    },
    {
      n: 6,
      title: 'Separate Input & Target',
      icon: <Columns3 size={16} />,
      gradient: [C.blue, C.pink],
      used: true,
      explanation: 'Speed/speed²/road condition become the input (X); stopping distance is the target (y) to estimate.',
      visual: <SeparateXYVisual p={p} />,
    },
    {
      n: 7,
      title: 'Split Data',
      icon: <GitFork size={16} />,
      gradient: [C.blue, C.blueDark],
      used: true,
      explanation: 'A slice of test runs is held back as a test set, never seen during training, so evaluation stays honest.',
      visual: <SplitVisual p={p} />,
    },
    {
      n: 8,
      title: 'Choose Model',
      icon: <Cpu size={16} />,
      gradient: [C.purple, C.purpleDark],
      used: true,
      explanation: 'Polynomial Regression is picked because distance grows with the square of speed, not a straight line.',
      visual: <ModelPickerVisual p={p} />,
    },
    {
      n: 9,
      title: 'Train Model',
      icon: <RefreshCw size={16} />,
      gradient: [C.purple, C.purpleDark],
      used: true,
      explanation: p.isLive
        ? 'This demo actually ran gradient descent on your test runs, adjusting weights each pass to shrink error.'
        : 'Gradient descent repeats this loop, adjusting weights to shrink error each time.',
      visual: <TrainingLoopVisual p={p} />,
    },
    {
      n: 10,
      title: 'Learned Model',
      icon: <BrainCircuit size={16} />,
      gradient: [C.purple, C.purpleDark],
      used: true,
      explanation: p.isLive
        ? 'These are the actual weights this demo learned from your test runs — shown below.'
        : 'Relative influence of speed, speed², and road condition on the formula’s own distance output.',
      visual: <LearnedNetworkVisual p={p} />,
    },
    {
      n: 11,
      title: 'Test / Predict',
      icon: <Gauge size={16} />,
      gradient: [C.pink, C.pinkDark],
      used: true,
      explanation: "This run's numbers flow into the physics formula, producing a predicted stopping distance and confidence.",
      visual: <BrakingPredictVisual p={p} />,
    },
    {
      n: 12,
      title: 'Evaluate',
      icon: <GitCompare size={16} />,
      gradient: [C.pink, C.pinkDark],
      used: true,
      explanation: p.isLive
        ? "This demo's model was scored against your held-out test runs — see the metrics below."
        : 'Illustrative held-out test runs, scored against the formula’s own predicted distances.',
      visual: <EvaluateVisual p={p} />,
    },
    {
      n: 13,
      title: 'Final Result',
      icon: <CheckCircle2 size={16} />,
      gradient: [C.successGreen, C.green],
      used: true,
      explanation: 'The completed pipeline’s output: predicted stopping distance split into reaction vs. braking, ready for the screen.',
      visual: <BrakingFinalResultVisual p={p} />,
    },
  ]
}

const COLS = 3 // matches lg:grid-cols-3 below — used to decide where row-wrap arrows go

const HIW_STYLES = `
  @keyframes hiwArrowNudge { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
  @keyframes hiwSeqPulse { 0%, 100% { opacity: 0.4; transform: scale(1); } 50% { opacity: 1; transform: scale(1.08); } }
  @keyframes hiwCrossfadeA { 0%, 40% { opacity: 1; } 50%, 90% { opacity: 0; } 100% { opacity: 1; } }
  @keyframes hiwCrossfadeB { 0%, 40% { opacity: 0; } 50%, 90% { opacity: 1; } 100% { opacity: 0; } }
  @keyframes hiwProgressFill { 0% { width: 0%; } 70%, 100% { width: 100%; } }
  @keyframes hiwCheckPopFast { 0%, 60% { opacity: 0; transform: scale(0.5); } 80%, 100% { opacity: 1; transform: scale(1); } }
  @keyframes hiwSpinSlow { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  @keyframes hiwParticleMove { 0% { transform: translateX(0); opacity: 1; } 90%, 100% { transform: translateX(24px); opacity: 0; } }
  @keyframes hiwTickerA { 0%, 45% { opacity: 1; } 50%, 100% { opacity: 0; } }
  @keyframes hiwTickerB { 0%, 45% { opacity: 0; } 50%, 100% { opacity: 1; } }
  .hiw-flow-arrow { animation: hiwArrowNudge 1.6s ease-in-out infinite; }
  .hiw-seq-pulse { animation: hiwSeqPulse 2s ease-in-out infinite; }
  .hiw-crossfade-a { animation: hiwCrossfadeA 6s ease-in-out infinite; }
  .hiw-crossfade-b { animation: hiwCrossfadeB 6s ease-in-out infinite; }
  .hiw-progress-fill { animation: hiwProgressFill 3s ease-out infinite; }
  .hiw-check-pop-fast { animation: hiwCheckPopFast 3s ease-out infinite; }
  .hiw-spin-slow { animation: hiwSpinSlow 3s linear infinite; }
  .hiw-particle { animation: hiwParticleMove 1.8s ease-in infinite; }
  .hiw-ticker-a { animation: hiwTickerA 3.2s ease-in-out infinite; }
  .hiw-ticker-b { animation: hiwTickerB 3.2s ease-in-out infinite; }
  @media (prefers-reduced-motion: reduce) {
    .hiw-flow-arrow, .hiw-seq-pulse, .hiw-crossfade-a, .hiw-crossfade-b, .hiw-progress-fill,
    .hiw-check-pop-fast, .hiw-spin-slow, .hiw-particle, .hiw-ticker-a, .hiw-ticker-b {
      animation: none !important;
    }
  }
`

const LIVE_ALGORITHMS = new Set(['Linear Regression', 'Multiple Linear Regression', 'Polynomial Regression'])

export function HowItWorksView() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [pipeline, setPipeline] = useState<PipelineData>(DEFAULT_PIPELINE)
  const [housePipeline, setHousePipeline] = useState<HousePricePipelineData>(DEFAULT_HOUSE_PIPELINE)
  const [brakingPipeline, setBrakingPipeline] = useState<BrakingDistancePipelineData>(DEFAULT_BRAKING_PIPELINE)
  const [notice, setNotice] = useState<{ kind: 'error' | 'info'; text: string } | null>(null)
  const [algorithm, setAlgorithm] = useState('Linear Regression')

  const isMultipleLinear = algorithm === 'Multiple Linear Regression'
  const isPolynomial = algorithm === 'Polynomial Regression'
  const isLiveAlgorithm = LIVE_ALGORITHMS.has(algorithm)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const content = event.target?.result as string
      if (!content) return

      try {
        if (isMultipleLinear) {
          const { rows, columnsDetected } = parseUploadedHouseRows(content)
          if (rows.length === 0) {
            setNotice({ kind: 'error', text: 'No valid rows found — check the file has a header row plus at least one data row.' })
            return
          }
          setHousePipeline(computeFullHousePipeline(rows, file.name, file.size / 1024, columnsDetected))
        } else if (isPolynomial) {
          const { rows, columnsDetected } = parseUploadedBrakingRows(content)
          if (rows.length === 0) {
            setNotice({ kind: 'error', text: 'No valid rows found — check the file has a header row plus at least one data row.' })
            return
          }
          setBrakingPipeline(computeFullBrakingPipeline(rows, file.name, file.size / 1024, columnsDetected))
        } else {
          const { rows, columnsDetected } = parseUploadedRows(content)
          if (rows.length === 0) {
            setNotice({ kind: 'error', text: 'No valid rows found — check the file has a header row plus at least one data row.' })
            return
          }
          setPipeline(computeFullPipeline(rows, file.name, file.size / 1024, columnsDetected))
        }
        setNotice({ kind: 'info', text: `Loaded ${file.name} — every card below now reflects your data.` })
      } catch {
        setNotice({ kind: 'error', text: 'Could not read that file as CSV. Try a plain .csv export of your spreadsheet.' })
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  function handleReset() {
    if (isMultipleLinear) setHousePipeline(DEFAULT_HOUSE_PIPELINE)
    else if (isPolynomial) setBrakingPipeline(DEFAULT_BRAKING_PIPELINE)
    else setPipeline(DEFAULT_PIPELINE)
    setNotice(null)
  }

  const stages = isMultipleLinear ? buildHousePriceStages(housePipeline) : isPolynomial ? buildBrakingStages(brakingPipeline) : buildStages(pipeline)
  const currentIsLive = isMultipleLinear ? housePipeline.isLive : isPolynomial ? brakingPipeline.isLive : pipeline.isLive

  return (
    <div
      className="flex-1 overflow-y-auto p-6 md:p-8"
      style={
        {
          '--hiw-muted-blue': C.mutedBlue,
          '--hiw-muted-blue-dark': C.mutedBlueDark,
          '--hiw-muted-purple': C.mutedPurple,
          '--hiw-muted-purple-dark': C.mutedPurpleDark,
          '--hiw-muted-pink': C.mutedPink,
          '--hiw-muted-pink-dark': C.mutedPinkDark,
        } as React.CSSProperties
      }
    >
      <style>{HIW_STYLES}</style>

      <header className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-blue to-bento-aqua text-white shadow-sm">
            <Workflow size={20} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-ink-primary md:text-2xl">How Algorithms Work</h1>
            <p className="text-sm text-ink-secondary">See what happens from your Excel upload to the final result.</p>
          </div>
        </div>
      </header>

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <AlgorithmDropdown value={algorithm} onChange={setAlgorithm} />
        {isLiveAlgorithm && (
          <div className="flex items-center gap-2">
            <input ref={fileInputRef} type="file" accept=".csv,.xlsx,.xls,text/csv" className="hidden" onChange={handleFileChange} />
            <Button size="sm" onClick={() => fileInputRef.current?.click()}>
              <Upload size={14} />
              Upload Your Excel / CSV
            </Button>
            {currentIsLive && (
              <Button size="sm" variant="secondary" onClick={handleReset}>
                <RotateCcw size={14} />
                Reset to example
              </Button>
            )}
          </div>
        )}
      </div>

      {!isLiveAlgorithm ? (
        <UnderConstructionPanel algorithm={algorithm} onBack={() => setAlgorithm('Linear Regression')} />
      ) : (
        <>
          {notice && (
            <div
              className={cn(
                'mb-4 rounded-lg px-3 py-2 text-xs font-medium',
                notice.kind === 'error' ? 'bg-status-critical/10 text-status-critical' : 'bg-bento-aqua/10 text-bento-aqua',
              )}
            >
              {notice.text}
            </div>
          )}

          <Card className="mb-6 overflow-hidden border-bento-blue/30">
        <div className="h-1 w-full bg-gradient-to-r from-bento-blue via-bento-aqua to-bento-violet" />
        <CardContent className="flex items-start gap-3 bg-bento-blue/5 p-4 text-sm text-ink-secondary">
          <Sparkles size={16} className="mt-0.5 flex-none text-bento-blue" />
          <p>
            {isMultipleLinear ? (
              housePipeline.isLive ? (
                <>
                  Every card below is now computed from <strong className="text-ink-primary">your file</strong> (
                  {housePipeline.fileName}, {housePipeline.totalRows} rows). Every step is genuinely computed — there's
                  no separate "production feature" for house prices elsewhere in this app to contrast against, so this
                  page is the full, honest demonstration end to end.
                </>
              ) : (
                <>
                  Every card traces <strong className="text-ink-primary">the same one listing</strong> — a worked
                  example — with real numbers from a hand-designed multiple linear regression formula. Upload your own
                  CSV/Excel above to see every card update with your real data instead, including a genuinely trained
                  model on your rows.
                </>
              )
            ) : isPolynomial ? (
              brakingPipeline.isLive ? (
                <>
                  Every card below is now computed from <strong className="text-ink-primary">your file</strong> (
                  {brakingPipeline.fileName}, {brakingPipeline.totalRows} rows), using real physics — braking distance
                  genuinely grows with the square of speed.
                </>
              ) : (
                <>
                  Every card traces <strong className="text-ink-primary">the same one test run</strong> — a worked
                  example — using a real physics formula (stopping distance ∝ speed²), not an invented curve. Upload
                  your own CSV/Excel above to see every card update with your real data instead.
                </>
              )
            ) : pipeline.isLive ? (
              <>
                Every card below is now computed from <strong className="text-ink-primary">your file</strong> (
                {pipeline.fileName}, {pipeline.totalRows} rows). Steps 1-6, 11 and 13 mirror the app's real formula
                exactly. Steps 7-10 and 12 are a genuinely real (if small) linear regression trained live on your
                rows as a teaching demo — its target is this app's own formula output, since a schedule file has no
                real historical outcome to learn from. The production Train Arrival Prediction screen still only
                does steps 1-6/11/13.
              </>
            ) : (
              <>
                Every card traces <strong className="text-ink-primary">the same one train</strong> — a worked example
                — with real numbers from the app's own code. Upload your own CSV/Excel above to see every card update
                with your real data instead. Cards with a dashed border and a{' '}
                <span className="font-semibold">Not used in this app</span> tag show a small mock-up of what a fully
                trained ML pipeline would do at that stage — the shipped app's simpler formula-based approach doesn't
                do it.
              </>
            )}
          </p>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-x-3 gap-y-6 lg:grid-cols-3">
        {stages.map((stage, i) => {
          const isRowEnd = (i + 1) % COLS === 0
          const isLast = i === stages.length - 1
          return (
            <div key={stage.n} className="relative">
              <StageCard stage={stage} isLive={currentIsLive} />
              {!isLast && (
                <>
                  <div className="mt-3 flex justify-center lg:hidden">
                    <FlowArrow direction="down" color={C.blue} />
                  </div>
                  {!isRowEnd && (
                    <div className="pointer-events-none absolute -right-2.5 top-1/2 z-10 hidden -translate-y-1/2 lg:block">
                      <FlowArrow direction="right" color={C.blue} />
                    </div>
                  )}
                  {isRowEnd && (
                    <div className="mt-3 hidden justify-center lg:flex">
                      <FlowArrow direction="down" color={C.blue} />
                    </div>
                  )}
                </>
              )}
            </div>
          )
        })}
      </div>

          <p className="mx-auto mt-6 max-w-2xl text-center text-xs text-ink-muted">
            A hand-designed formula is a completely valid, transparent baseline — it's just a different (and simpler,
            more explainable) approach than a trained statistical model. This page shows exactly which one you're
            looking at.
          </p>
        </>
      )}
    </div>
  )
}
