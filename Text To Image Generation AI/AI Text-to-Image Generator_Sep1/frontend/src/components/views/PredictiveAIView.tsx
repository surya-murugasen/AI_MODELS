import { useMemo, useState } from 'react'
import {
  ArrowRight,
  BrainCircuit,
  ChevronDown,
  Clock,
  Download,
  Image as ImageIcon,
  Plus,
  Sparkles,
  Star,
  Target,
  Train,
  TrendingDown,
  TrendingUp,
  Trophy,
  Wand2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { StatTile } from '@/components/StatTile'
import { AccuracyTrendChart, BarList, CATEGORICAL_COLORS, ProgressRing, StackedShareBar } from '@/components/charts'
import { TrainArrivalPredictionView } from '@/components/views/TrainArrivalPredictionView'
import {
  accuracyBadgeTone,
  accuracyBreakdownForGeneration,
  accuracyByModel,
  accuracyByResolution,
  comparePredictionToActual,
  computeAccuracyTrend,
  periodOverPeriodDelta,
  predictNextAccuracy,
  recommendSettings,
  type SettingsRecommendation,
} from '@/lib/predictive'
import type { GenerationRecord, ModelsInfo } from '@/lib/types'
import { cn } from '@/lib/utils'

const TIME_RANGES = [
  { key: '7d', label: '7 Days', days: 7 },
  { key: '30d', label: '30 Days', days: 30 },
  { key: 'all', label: 'All time', days: 0 },
] as const

function modelLabel(models: ModelsInfo | null, modelId: string): string {
  return models?.available.find((m) => m.id === modelId)?.label ?? modelId
}

function formatDelta(delta: number | null, unit: string, decimals = 1) {
  if (delta === null || Math.abs(delta) < 0.05) return null
  const up = delta > 0
  return { up, text: `${up ? '+' : ''}${delta.toFixed(decimals)}${unit} vs last 7 days` }
}

export function PredictiveAIView({
  generations,
  models,
  draftPrompt,
  draftModelId,
  draftSteps,
  draftGuidanceScale,
  draftWidth,
  draftHeight,
  onApplyRecommended,
  onOpenGenerate,
}: {
  generations: GenerationRecord[]
  models: ModelsInfo | null
  draftPrompt: string
  draftModelId: string
  draftSteps: number
  draftGuidanceScale: number
  draftWidth: number
  draftHeight: number
  onApplyRecommended: (rec: SettingsRecommendation) => void
  onOpenGenerate: () => void
}) {
  const [rangeKey, setRangeKey] = useState<(typeof TIME_RANGES)[number]['key']>('7d')
  const [modelFilter, setModelFilter] = useState<string>('all')
  const [compareId, setCompareId] = useState<string | null>(null)

  const compareGeneration = generations.find((g) => g.id === compareId) ?? generations[0] ?? null
  const comparison = compareGeneration
    ? comparePredictionToActual(compareGeneration.predictedAccuracy, compareGeneration.accuracy)
    : null

  const overallAccuracy = generations.length
    ? generations.reduce((sum, g) => sum + g.accuracy, 0) / generations.length
    : 0
  const highAccuracyCount = generations.filter((g) => g.accuracy >= 80).length
  const avgSeconds = generations.length ? generations.reduce((sum, g) => sum + g.seconds, 0) / generations.length : 0
  const byModel = useMemo(() => accuracyByModel(generations), [generations])
  const topModel = byModel[0] ?? null

  const accuracyDelta = formatDelta(periodOverPeriodDelta(generations, 7), '%')
  const timeDelta = formatDelta(
    periodOverPeriodDelta(generations, 7, (g) => g.seconds),
    's',
  )

  const prediction = useMemo(
    () =>
      predictNextAccuracy(generations, {
        modelId: draftModelId,
        steps: draftSteps,
        guidanceScale: draftGuidanceScale,
        width: draftWidth,
        height: draftHeight,
        prompt: draftPrompt,
      }),
    [generations, draftModelId, draftSteps, draftGuidanceScale, draftWidth, draftHeight, draftPrompt],
  )

  const recommendation = useMemo(() => recommendSettings(generations), [generations])

  const selectedRange = TIME_RANGES.find((r) => r.key === rangeKey) ?? TIME_RANGES[0]
  const trend = useMemo(() => computeAccuracyTrend(generations, selectedRange.days), [generations, selectedRange])

  const resolutionShares = useMemo(() => accuracyByResolution(generations), [generations])
  const resolutionSegments = resolutionShares.map((r, i) => ({
    label: r.label,
    share: r.share,
    color: r.label === 'Others' ? '#898781' : CATEGORICAL_COLORS[i % CATEGORICAL_COLORS.length],
  }))

  const distinctModelIds = useMemo(() => [...new Set(generations.map((g) => g.modelId))], [generations])
  const filteredGenerations = useMemo(
    () => (modelFilter === 'all' ? generations : generations.filter((g) => g.modelId === modelFilter)),
    [generations, modelFilter],
  )

  const [activePredictiveTab, setActivePredictiveTab] = useState<'image-accuracy' | 'train-arrival'>('train-arrival')

  function handleExportCsv() {
    const header = [
      '#',
      'Prompt',
      'Model',
      'Steps',
      'Guidance',
      'Width',
      'Height',
      'Prompt Match (%)',
      'Visual Quality (%)',
      'Composition (%)',
      'Artifact-Free (%)',
      'Overall Accuracy (%)',
      'Generation Time (s)',
      'Created At',
    ]
    const rows = filteredGenerations.map((g, i) => {
      const b = accuracyBreakdownForGeneration(g)
      return [
        String(i + 1),
        `"${g.prompt.replace(/"/g, '""')}"`,
        modelLabel(models, g.modelId),
        String(g.steps),
        g.guidanceScale.toFixed(1),
        String(g.width),
        String(g.height),
        b.promptMatch.toFixed(1),
        b.visualQuality.toFixed(1),
        b.composition.toFixed(1),
        b.artifactFree.toFixed(1),
        b.overall.toFixed(1),
        g.seconds.toFixed(1),
        new Date(g.createdAt).toLocaleString(),
      ]
    })
    const csv = [header, ...rows].map((r) => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `predictive-ai-generations-${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-1 flex-col h-full overflow-hidden bg-page">
      {activePredictiveTab === 'train-arrival' ? (
        <TrainArrivalPredictionView onSwitchToOverview={() => setActivePredictiveTab('image-accuracy')} />
      ) : (
        <div className="flex-1 overflow-y-auto p-6 md:p-8">
          {/* Top Tab Switcher Bar for Image Accuracy AI */}
          <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b pb-4">
            <div className="flex items-center gap-2">
            {/* Tab Switcher Pill */}
          <div className="flex items-center rounded-xl border bg-surface p-1 shadow-sm">
            {/* Image Accuracy AI — always active in this branch */}
            <button
              type="button"
              onClick={() => setActivePredictiveTab('image-accuracy')}
              className="flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold bg-bento-blue text-white shadow transition-all cursor-pointer"
            >
              <BrainCircuit size={14} />
              Image Accuracy AI
            </button>
            {/* Train Arrival — inactive, click to switch */}
            <button
              type="button"
              onClick={() => setActivePredictiveTab('train-arrival')}
              className="flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold text-ink-secondary hover:text-ink-primary hover:bg-black/[0.04] dark:hover:bg-white/[0.06] transition-all cursor-pointer"
            >
              <Train size={14} />
              Train Arrival Prediction
              <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            </button>
          </div>
        </div>

        <Button size="sm" onClick={onOpenGenerate}>
          <Plus size={16} />
          New prompt
        </Button>
      </div>

          <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-violet to-bento-blue text-white">
                <BrainCircuit size={20} />
              </div>
              <div>
                <h1 className="text-xl font-bold text-ink-primary md:text-2xl">Image Accuracy Predictive AI</h1>
                <p className="text-sm text-ink-secondary">Analyze past generations and predict the accuracy of your next image.</p>
              </div>
            </div>
          </header>

          <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
            <StatTile
              icon={<Target size={18} />}
              accent="#2a78d6"
              label="Overall accuracy"
              value={generations.length ? `${overallAccuracy.toFixed(1)}%` : 'No data available'}
            />
        <StatTile icon={<ImageIcon size={18} />} accent="#1baf7a" label="Total generations" value={String(generations.length)} />
        <StatTile
          icon={<Trophy size={18} />}
          accent="#eda100"
          label="High accuracy (≥ 80%)"
          value={
            generations.length
              ? `${highAccuracyCount} · ${((highAccuracyCount / generations.length) * 100).toFixed(1)}%`
              : 'No data available'
          }
        />
        <StatTile
          icon={<Clock size={18} />}
          accent="var(--accent-violet)"
          label="Avg generation time"
          value={generations.length ? `${avgSeconds.toFixed(1)}s` : 'No data available'}
        />
        <StatTile
          icon={<Star size={18} />}
          accent="#e87ba4"
          label="Top performing model"
          value={topModel ? `${modelLabel(models, topModel.modelId)}` : 'No data available'}
        />
      </div>

      {(accuracyDelta || timeDelta) && (
        <div className="mb-6 flex flex-wrap gap-4 text-xs">
          {accuracyDelta && (
            <span className={cn('flex items-center gap-1 font-medium', accuracyDelta.up ? 'text-status-good' : 'text-status-critical')}>
              {accuracyDelta.up ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
              {accuracyDelta.text}
            </span>
          )}
          {timeDelta && (
            <span className={cn('flex items-center gap-1 font-medium', timeDelta.up ? 'text-status-critical' : 'text-status-good')}>
              {timeDelta.up ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
              {timeDelta.text}
            </span>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Next Image Accuracy Prediction</CardTitle>
            <CardDescription>
              {prediction
                ? 'Based on your past generations and currently selected settings.'
                : 'Unlocks once you have real generation history to predict from.'}
            </CardDescription>
          </CardHeader>
          {prediction ? (
            <>
              <CardContent className="flex flex-col gap-6 sm:flex-row sm:items-center">
                <div className="flex flex-none justify-center">
                  <ProgressRing
                    value={prediction.breakdown.overall}
                    primaryLabel={`${prediction.breakdown.overall.toFixed(1)}%`}
                    secondaryLabel="Predicted Accuracy"
                  />
                </div>
                <div className="flex-1">
                  <p className="mb-3 text-xs font-medium uppercase tracking-wide text-ink-muted">Why this prediction?</p>
                  <BarList
                    accent="#4a3aa7"
                    items={[
                      { label: 'Prompt Match', value: prediction.breakdown.promptMatch },
                      { label: 'Visual Quality', value: prediction.breakdown.visualQuality },
                      { label: 'Composition', value: prediction.breakdown.composition },
                      { label: 'Artifact-Free', value: prediction.breakdown.artifactFree },
                    ]}
                  />
                </div>
              </CardContent>
              <CardContent className="pt-0">
                <div className="flex items-start gap-2 rounded-xl bg-bento-blue/10 px-4 py-3 text-sm text-ink-secondary">
                  <Sparkles size={16} className="mt-0.5 flex-none text-bento-blue" />
                  <span>
                    {prediction.breakdown.overall >= 85
                      ? 'Your selected settings are strong — this combination has historically produced high-quality images.'
                      : 'There is room to improve — check the recommended settings for a higher predicted accuracy.'}
                  </span>
                </div>
              </CardContent>
            </>
          ) : (
            <CardContent>
              <EmptyState message="No data available — generate an image to start predicting the accuracy of your next one." />
            </CardContent>
          )}
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recommended Settings</CardTitle>
            <CardDescription>
              {recommendation
                ? `Based on analysis of ${recommendation.sampleSize} real generation${recommendation.sampleSize === 1 ? '' : 's'}.`
                : 'Unlocks once you have real generation history to recommend from.'}
            </CardDescription>
          </CardHeader>
          {recommendation ? (
            <>
              <CardContent className="flex flex-col gap-0 pt-0">
                <RecommendationRow label="Model" value={modelLabel(models, recommendation.modelId)} />
                <RecommendationRow label="Steps" value={String(recommendation.steps)} />
                <RecommendationRow label="Guidance scale" value={recommendation.guidanceScale.toFixed(1)} />
                <RecommendationRow label="Resolution" value={`${recommendation.width} × ${recommendation.height}`} />
              </CardContent>
              <CardContent className="pt-2">
                <Button
                  size="lg"
                  className="w-full bg-gradient-to-r from-bento-violet via-bento-magenta to-bento-blue hover:opacity-90"
                  onClick={() => onApplyRecommended(recommendation)}
                >
                  <Wand2 size={16} />
                  Apply recommended settings
                </Button>
              </CardContent>
            </>
          ) : (
            <CardContent>
              <EmptyState message="No data available — generate an image so real settings performance can be analyzed." />
            </CardContent>
          )}
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-3 space-y-0">
          <div>
            <CardTitle>Prediction vs Actual Result</CardTitle>
            <CardDescription>How close was the prediction made before this image was generated?</CardDescription>
          </div>
          {generations.length > 0 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="secondary" size="sm" className="max-w-[220px]">
                  <span className="truncate">{compareGeneration?.prompt ?? 'Select a generation'}</span>
                  <ChevronDown size={14} className="flex-none" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="max-h-80 w-[320px] overflow-y-auto">
                {generations.map((g) => (
                  <DropdownMenuItem key={g.id} onClick={() => setCompareId(g.id)} className="justify-between gap-2">
                    <span className="truncate">{g.prompt}</span>
                    <span className="flex-none text-xs text-ink-muted">{new Date(g.createdAt).toLocaleDateString()}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </CardHeader>
        <CardContent>
          {!compareGeneration ? (
            <EmptyState message="No data available — generate an image to compare its predicted and actual accuracy." />
          ) : (
            <div className="flex flex-col gap-6">
              <div className="flex flex-col items-center justify-center gap-4 sm:flex-row sm:gap-6">
                <ComparisonMetric label="Predicted Accuracy" value={comparison?.predicted ?? null} color="#2a78d6" />
                <ArrowRight className="hidden flex-none text-ink-muted sm:block" size={20} />
                <div className="flex flex-col items-center gap-2">
                  <img
                    src={compareGeneration.imageSrc}
                    alt={compareGeneration.prompt}
                    className="h-28 w-28 flex-none rounded-xl border object-cover"
                  />
                  <p className="max-w-[11rem] truncate text-xs text-ink-secondary" title={compareGeneration.prompt}>
                    {compareGeneration.prompt}
                  </p>
                  <p className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">Generated Image</p>
                </div>
                <ArrowRight className="hidden flex-none text-ink-muted sm:block" size={20} />
                <ComparisonMetric
                  label="Actual Accuracy"
                  value={compareGeneration.accuracy}
                  tone={accuracyBadgeTone(compareGeneration.accuracy)}
                />
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <MetricTile
                  label="Prediction Error"
                  value={comparison ? `${comparison.error.toFixed(1)} pts` : 'No data available'}
                />
                <MetricTile
                  label="Prediction Accuracy"
                  value={comparison ? `${comparison.predictionAccuracy.toFixed(1)}%` : 'No data available'}
                />
              </div>

              {!comparison && (
                <p className="text-xs text-ink-muted">
                  No prediction was made before this generation ran — this happens when there wasn't yet any earlier
                  history to predict from (typically the very first generation in your history).
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-3 space-y-0">
          <div>
            <CardTitle>Image Accuracy Over Time</CardTitle>
            <CardDescription>Daily accuracy from real generations, with a 7-day moving average and trend projection.</CardDescription>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="secondary" size="sm">
                {selectedRange.label}
                <ChevronDown size={14} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {TIME_RANGES.map((r) => (
                <DropdownMenuItem key={r.key} onClick={() => setRangeKey(r.key)}>
                  {r.label}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </CardHeader>
        <CardContent>
          <AccuracyTrendChart history={trend.history} forecast={trend.forecast} />
        </CardContent>
      </Card>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Accuracy by Model</CardTitle>
            <CardDescription>Average accuracy comparison.</CardDescription>
          </CardHeader>
          <CardContent>
            <BarList items={byModel.map((m) => ({ label: modelLabel(models, m.modelId), value: m.avgAccuracy }))} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generations by Resolution</CardTitle>
            <CardDescription>Which resolution do you use most?</CardDescription>
          </CardHeader>
          <CardContent>
            <StackedShareBar segments={resolutionSegments} />
          </CardContent>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-3 space-y-0">
          <div>
            <CardTitle>Recent Generations Analysis</CardTitle>
            <CardDescription>Detailed accuracy of your recent generations.</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="secondary" size="sm">
                  {modelFilter === 'all' ? 'All models' : modelLabel(models, modelFilter)}
                  <ChevronDown size={14} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => setModelFilter('all')}>All models</DropdownMenuItem>
                {distinctModelIds.map((id) => (
                  <DropdownMenuItem key={id} onClick={() => setModelFilter(id)}>
                    {modelLabel(models, id)}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button variant="secondary" size="sm" onClick={handleExportCsv} disabled={filteredGenerations.length === 0}>
              <Download size={14} />
              Export CSV
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-4">
          {filteredGenerations.length === 0 ? (
            <EmptyState
              message={
                generations.length === 0
                  ? 'No data available — head to Generate to create your first image.'
                  : 'No data available for this model filter.'
              }
              className="py-16"
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] border-collapse text-sm">
                <thead>
                  <tr className="border-b text-left text-xs font-medium uppercase tracking-wide text-ink-muted">
                    <th className="pb-2 pr-3">#</th>
                    <th className="pb-2 pr-3">Prompt</th>
                    <th className="pb-2 pr-3">Model</th>
                    <th className="pb-2 pr-3">Steps</th>
                    <th className="pb-2 pr-3">Guidance</th>
                    <th className="pb-2 pr-3">Resolution</th>
                    <th className="pb-2 pr-3">Accuracy</th>
                    <th className="pb-2 pr-3">Gen. time</th>
                    <th className="pb-2">Created at</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredGenerations.map((g, i) => {
                    const tone = accuracyBadgeTone(g.accuracy)
                    const breakdown = accuracyBreakdownForGeneration(g)
                    const breakdownTitle = `Prompt Match: ${breakdown.promptMatch}% · Visual Quality: ${breakdown.visualQuality}% · Composition: ${breakdown.composition}% · Artifact-Free: ${breakdown.artifactFree}% · Overall Accuracy: ${breakdown.overall}%`
                    return (
                      <tr key={g.id} className="border-b last:border-b-0">
                        <td className="py-2.5 pr-3 text-ink-muted">{i + 1}</td>
                        <td className="py-2.5 pr-3">
                          <div className="flex max-w-[220px] items-center gap-2.5">
                            <img src={g.imageSrc} alt="" className="h-9 w-9 flex-none rounded-lg object-cover" />
                            <span className="truncate text-ink-primary" title={g.prompt}>
                              {g.prompt}
                            </span>
                          </div>
                        </td>
                        <td className="py-2.5 pr-3 text-ink-secondary">{modelLabel(models, g.modelId)}</td>
                        <td className="py-2.5 pr-3 text-ink-secondary">{g.steps}</td>
                        <td className="py-2.5 pr-3 text-ink-secondary">{g.guidanceScale.toFixed(1)}</td>
                        <td className="py-2.5 pr-3 text-ink-secondary">
                          {g.width}×{g.height}
                        </td>
                        <td className="py-2.5 pr-3">
                          <span
                            title={breakdownTitle}
                            className={cn(
                              'inline-flex cursor-help rounded-full px-2 py-0.5 text-xs font-semibold',
                              tone === 'good' && 'bg-status-good/10 text-status-good',
                              tone === 'warning' && 'bg-status-warning/15 text-status-warning',
                              tone === 'critical' && 'bg-status-critical/10 text-status-critical',
                            )}
                          >
                            {g.accuracy.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-2.5 pr-3 text-ink-secondary">{g.seconds.toFixed(1)}s</td>
                        <td className="py-2.5 text-ink-secondary">{new Date(g.createdAt).toLocaleString()}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

        <p className="mt-6 text-xs text-ink-muted">
          Overall Accuracy is the average of four estimated categories — Prompt Match, Visual Quality, Composition, and
          Artifact-Free — derived from your model, steps, guidance scale, resolution, and prompt detail (hover a row's
          accuracy badge for the full breakdown). There's no rating system in this app, so treat it as a heuristic guide
          rather than a measured score.
        </p>
      </div>
      )}
    </div>
  )
}

function RecommendationRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b py-3 text-sm last:border-b-0">
      <span className="text-ink-secondary">{label}</span>
      <span className="font-semibold text-ink-primary">{value}</span>
    </div>
  )
}

function EmptyState({ message, className }: { message: string; className?: string }) {
  return (
    <div className={cn('flex flex-col items-center gap-2 py-8 text-center text-ink-muted', className)}>
      <BrainCircuit size={28} />
      <p className="text-sm">{message}</p>
    </div>
  )
}

function ComparisonMetric({
  label,
  value,
  color,
  tone,
}: {
  label: string
  value: number | null
  color?: string
  tone?: 'good' | 'warning' | 'critical'
}) {
  const toneClass =
    tone === 'good' ? 'text-status-good' : tone === 'warning' ? 'text-status-warning' : tone === 'critical' ? 'text-status-critical' : undefined
  return (
    <div className="flex w-32 flex-col items-center gap-1 text-center">
      <p className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">{label}</p>
      {value === null ? (
        <p className="text-sm font-semibold text-ink-muted">No data available</p>
      ) : (
        <p className={cn('text-3xl font-bold', toneClass)} style={!toneClass && color ? { color } : undefined}>
          {value.toFixed(1)}%
        </p>
      )}
    </div>
  )
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border bg-page px-4 py-3 text-center">
      <p className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="mt-1 text-lg font-semibold text-ink-primary">{value}</p>
    </div>
  )
}
