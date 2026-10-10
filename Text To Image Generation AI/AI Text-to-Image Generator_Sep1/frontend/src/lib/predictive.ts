import type { GenerationRecord } from '@/lib/types'

/**
 * There's no database and no user feedback/rating loop in this app (see
 * README), so there's no ground-truth "accuracy" a real model could be
 * trained on. Instead we estimate a 0-100 quality score from the settings
 * that are known to affect output quality for these diffusion pipelines
 * (model, step count, guidance scale, resolution, prompt detail), and use
 * that same estimator — run against the *current* draft settings plus the
 * trend of past estimates — to predict the next generation's score and
 * recommend settings. It's a transparent heuristic, not a trained model.
 */

export interface ScoreInput {
  modelId: string
  steps: number
  guidanceScale: number
  width: number
  height: number
  prompt: string
}

interface ModelProfile {
  base: number
  idealSteps: number
  idealGuidance: number
}

const MODEL_PROFILES: Record<string, ModelProfile> = {
  'stabilityai/sd-turbo': { base: 90, idealSteps: 2, idealGuidance: 0 },
  'runwayml/stable-diffusion-v1-5': { base: 78, idealSteps: 25, idealGuidance: 7.5 },
  'stabilityai/stable-diffusion-2-1-base': { base: 83, idealSteps: 25, idealGuidance: 7.5 },
  'custom-scratch-model': { base: 60, idealSteps: 30, idealGuidance: 7.5 },
}
const DEFAULT_PROFILE: ModelProfile = { base: 75, idealSteps: 25, idealGuidance: 7.5 }

function getModelProfile(modelId: string): ModelProfile {
  return MODEL_PROFILES[modelId] ?? DEFAULT_PROFILE
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

const QUALITY_KEYWORDS = [
  'detailed',
  'highly detailed',
  'intricate',
  'lighting',
  'cinematic',
  '4k',
  '8k',
  'sharp focus',
  'realistic',
  'high quality',
  'masterpiece',
  'trending',
  'octane render',
  'artstation',
]

function modelScore(modelId: string): number {
  return getModelProfile(modelId).base
}

function stepsScore(modelId: string, steps: number): number {
  const { idealSteps } = getModelProfile(modelId)
  const perStepPenalty = idealSteps <= 5 ? 14 : 3
  return clamp(100 - Math.abs(steps - idealSteps) * perStepPenalty, 15, 100)
}

function guidanceScore(modelId: string, guidanceScale: number): number {
  const { idealGuidance } = getModelProfile(modelId)
  const perPointPenalty = idealGuidance === 0 ? 16 : 6
  return clamp(100 - Math.abs(guidanceScale - idealGuidance) * perPointPenalty, 15, 100)
}

function resolutionScore(width: number, height: number): number {
  const megapixelRatio = (width * height) / (512 * 512)
  const aspectPenalty = (Math.abs(width - height) / Math.max(width, height)) * 40
  const scalePenalty = Math.abs(megapixelRatio - 1) * 22
  return clamp(100 - aspectPenalty - scalePenalty, 30, 100)
}

function promptQualityScore(prompt: string): number {
  const trimmed = prompt.trim()
  if (!trimmed) return 30
  const wordCount = trimmed.split(/\s+/).filter(Boolean).length
  const wordScore = clamp(35 + wordCount * 2.2, 35, 80)
  const lower = trimmed.toLowerCase()
  const keywordHits = QUALITY_KEYWORDS.filter((k) => lower.includes(k)).length
  const keywordBonus = clamp(keywordHits * 6, 0, 25)
  return clamp(wordScore + keywordBonus, 30, 100)
}

/** Small, deterministic per-generation jitter so scores don't feel robotic. */
function hashJitter(seed: string): number {
  let h = 0
  for (let i = 0; i < seed.length; i++) {
    h = (h * 31 + seed.charCodeAt(i)) | 0
  }
  return ((h % 700) / 100) - 3.5 // roughly -3.5..+3.5
}

function round1(value: number): number {
  return Math.round(value * 10) / 10
}

/**
 * Accuracy is reported across four named categories — the same breakdown
 * used both for a completed generation and for the next-image prediction —
 * with Overall Accuracy being their plain average.
 */
export interface AccuracyBreakdown {
  promptMatch: number
  visualQuality: number
  composition: number
  artifactFree: number
  overall: number
}

function computeAccuracyBreakdown(input: ScoreInput & { seed: string }): AccuracyBreakdown {
  const model = modelScore(input.modelId)
  const steps = stepsScore(input.modelId, input.steps)
  const guidance = guidanceScore(input.modelId, input.guidanceScale)
  const resolution = resolutionScore(input.width, input.height)
  const promptQuality = promptQualityScore(input.prompt)

  const promptMatch = clamp(0.65 * promptQuality + 0.35 * guidance + hashJitter(`${input.seed}:pm`), 0, 100)
  const visualQuality = clamp(0.55 * model + 0.45 * steps + hashJitter(`${input.seed}:vq`), 0, 100)
  const composition = clamp(0.7 * resolution + 0.3 * model + hashJitter(`${input.seed}:cp`), 0, 100)
  const artifactFree = clamp(0.5 * steps + 0.3 * guidance + 0.2 * model + hashJitter(`${input.seed}:af`), 0, 100)
  const overall = (promptMatch + visualQuality + composition + artifactFree) / 4

  return {
    promptMatch: round1(promptMatch),
    visualQuality: round1(visualQuality),
    composition: round1(composition),
    artifactFree: round1(artifactFree),
    overall: round1(overall),
  }
}

/** The "ground truth" accuracy stored on each completed generation. */
export function scoreGeneration(input: ScoreInput & { seed: string }): number {
  return computeAccuracyBreakdown(input).overall
}

/** Recomputes a past generation's full breakdown from its stored settings. */
export function accuracyBreakdownForGeneration(g: GenerationRecord): AccuracyBreakdown {
  return computeAccuracyBreakdown({
    modelId: g.modelId,
    steps: g.steps,
    guidanceScale: g.guidanceScale,
    width: g.width,
    height: g.height,
    prompt: g.prompt,
    seed: g.id,
  })
}

export interface Prediction {
  breakdown: AccuracyBreakdown
}

/**
 * Predicts the next generation's breakdown from draft settings, nudged by
 * recent historical momentum. Returns null when there's no generation
 * history yet — there is nothing real to base a prediction on, so none is
 * shown rather than falling back to a settings-only guess.
 */
export function predictNextAccuracy(generations: GenerationRecord[], draft: ScoreInput): Prediction | null {
  if (generations.length === 0) return null

  const base = computeAccuracyBreakdown({ ...draft, seed: 'draft' })
  const sorted = [...generations].sort((a, b) => a.createdAt - b.createdAt)
  const overallAvg = average(sorted.map((g) => g.accuracy))
  const recentAvg = average(sorted.slice(-5).map((g) => g.accuracy))
  const trendDelta = clamp(recentAvg - overallAvg, -15, 15) * 0.6

  const nudge = (value: number) => round1(clamp(value + trendDelta, 0, 100))
  const promptMatch = nudge(base.promptMatch)
  const visualQuality = nudge(base.visualQuality)
  const composition = nudge(base.composition)
  const artifactFree = nudge(base.artifactFree)

  return {
    breakdown: {
      promptMatch,
      visualQuality,
      composition,
      artifactFree,
      overall: round1((promptMatch + visualQuality + composition + artifactFree) / 4),
    },
  }
}

export interface SettingsRecommendation {
  modelId: string
  steps: number
  guidanceScale: number
  width: number
  height: number
  sampleSize: number
}

function average(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}

function bestByAverage<T>(items: GenerationRecord[], keyFn: (g: GenerationRecord) => T): T | null {
  if (items.length === 0) return null
  const groups = new Map<string, { key: T; total: number; count: number }>()
  for (const g of items) {
    const key = keyFn(g)
    const groupKey = JSON.stringify(key)
    const existing = groups.get(groupKey)
    if (existing) {
      existing.total += g.accuracy
      existing.count += 1
    } else {
      groups.set(groupKey, { key, total: g.accuracy, count: 1 })
    }
  }
  let best: { key: T; avg: number } | null = null
  for (const { key, total, count } of groups.values()) {
    const avg = total / count
    if (!best || avg > best.avg) best = { key, avg }
  }
  return best?.key ?? null
}

/**
 * Best-performing settings seen in real generation history, scoped to
 * whichever model has performed best. Returns null when there's no history
 * at all — with nothing real to analyze, no recommendation is made.
 */
export function recommendSettings(generations: GenerationRecord[]): SettingsRecommendation | null {
  if (generations.length === 0) return null

  const bestModelId = bestByAverage(generations, (g) => g.modelId) ?? generations[0].modelId
  const pool = generations.filter((g) => g.modelId === bestModelId)

  const bestSteps = bestByAverage(pool, (g) => g.steps) ?? pool[0].steps
  const bestGuidance = bestByAverage(pool, (g) => g.guidanceScale) ?? pool[0].guidanceScale
  const bestResolution = bestByAverage(pool, (g) => ({ width: g.width, height: g.height })) ?? {
    width: pool[0].width,
    height: pool[0].height,
  }

  return {
    modelId: bestModelId,
    steps: bestSteps,
    guidanceScale: bestGuidance,
    width: bestResolution.width,
    height: bestResolution.height,
    sampleSize: pool.length,
  }
}

export interface ModelAccuracy {
  modelId: string
  avgAccuracy: number
  count: number
}

export function accuracyByModel(generations: GenerationRecord[]): ModelAccuracy[] {
  const groups = new Map<string, { total: number; count: number }>()
  for (const g of generations) {
    const existing = groups.get(g.modelId)
    if (existing) {
      existing.total += g.accuracy
      existing.count += 1
    } else {
      groups.set(g.modelId, { total: g.accuracy, count: 1 })
    }
  }
  return [...groups.entries()]
    .map(([modelId, { total, count }]) => ({ modelId, avgAccuracy: total / count, count }))
    .sort((a, b) => b.avgAccuracy - a.avgAccuracy)
}

export interface ResolutionShare {
  label: string
  width: number
  height: number
  share: number
  count: number
}

export function accuracyByResolution(generations: GenerationRecord[]): ResolutionShare[] {
  const groups = new Map<string, { width: number; height: number; count: number }>()
  for (const g of generations) {
    const key = `${g.width}x${g.height}`
    const existing = groups.get(key)
    if (existing) {
      existing.count += 1
    } else {
      groups.set(key, { width: g.width, height: g.height, count: 1 })
    }
  }
  const total = generations.length || 1
  const sorted = [...groups.entries()]
    .map(([key, { width, height, count }]) => ({
      label: key.replace('x', ' × '),
      width,
      height,
      count,
      share: (count / total) * 100,
    }))
    .sort((a, b) => b.count - a.count)

  if (sorted.length <= 5) return sorted
  const top = sorted.slice(0, 4)
  const rest = sorted.slice(4)
  const othersCount = rest.reduce((sum, r) => sum + r.count, 0)
  return [
    ...top,
    { label: 'Others', width: 0, height: 0, count: othersCount, share: (othersCount / total) * 100 },
  ]
}

export interface AccuracyTrendPoint {
  id: string
  label: string
  dailyAccuracy: number
  movingAverage: number
}

export interface AccuracyForecastPoint {
  dateKey: string
  label: string
  predicted: number
}

export interface AccuracyTrend {
  /** Every real generation in the selected window, each its own point, plus a trailing 7-day moving average. */
  history: AccuracyTrendPoint[]
  /** A linear-regression projection fit to real daily averages; empty when there's too little day-level history to fit a trend. */
  forecast: AccuracyForecastPoint[]
}

const MOVING_AVERAGE_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
const MIN_REAL_DAYS_FOR_FORECAST = 3
const FORECAST_DAYS = 7

function linearRegression(points: { x: number; y: number }[]): { slope: number; intercept: number } | null {
  const n = points.length
  if (n < 2) return null
  const meanX = points.reduce((sum, p) => sum + p.x, 0) / n
  const meanY = points.reduce((sum, p) => sum + p.y, 0) / n
  let numerator = 0
  let denominator = 0
  for (const p of points) {
    numerator += (p.x - meanX) * (p.y - meanY)
    denominator += (p.x - meanX) ** 2
  }
  if (denominator === 0) return null
  const slope = numerator / denominator
  return { slope, intercept: meanY - slope * meanX }
}

/**
 * Every real generation becomes its own point (this app is session-only —
 * no database — so a whole history often falls on one calendar day, and
 * bucketing by day would collapse it to a single point). Each point also
 * carries a trailing 7-*day* moving average computed from real neighboring
 * generations. Forecasting a trend, though, only makes sense over real
 * day-level spread, so that part still buckets by calendar day and only
 * produces a projection once there's enough day-level history to fit one.
 * Everything here is derived from `generations`; with none, both `history`
 * and `forecast` come back empty.
 */
export function computeAccuracyTrend(generations: GenerationRecord[], days: number): AccuracyTrend {
  if (generations.length === 0) return { history: [], forecast: [] }

  const sorted = [...generations].sort((a, b) => a.createdAt - b.createdAt)
  const now = Date.now()
  const windowMs = days > 0 ? days * 24 * 60 * 60 * 1000 : null

  // Label precision adapts to how tightly the real data is clustered — a
  // fixed minute-level format collapses several generations made seconds
  // apart into the same on-axis label, which reads as a bug.
  const spanMs = sorted[sorted.length - 1].createdAt - sorted[0].createdAt
  const labelOptions: Intl.DateTimeFormatOptions =
    spanMs < 10 * 60 * 1000
      ? { hour: 'numeric', minute: '2-digit', second: '2-digit' }
      : spanMs < 24 * 60 * 60 * 1000
        ? { hour: 'numeric', minute: '2-digit' }
        : { month: 'short', day: 'numeric' }
  const includeDate = spanMs >= 24 * 60 * 60 * 1000
  const labelFor = (createdAt: number) =>
    includeDate
      ? new Date(createdAt).toLocaleDateString(undefined, labelOptions)
      : new Date(createdAt).toLocaleTimeString(undefined, labelOptions)

  const history: AccuracyTrendPoint[] = sorted
    .map((g, i) => {
      const movingWindowStart = g.createdAt - MOVING_AVERAGE_WINDOW_MS
      const movingSlice = sorted.slice(0, i + 1).filter((p) => p.createdAt >= movingWindowStart)
      return {
        id: g.id,
        createdAt: g.createdAt,
        label: labelFor(g.createdAt),
        dailyAccuracy: round1(g.accuracy),
        movingAverage: round1(average(movingSlice.map((p) => p.accuracy))),
      }
    })
    .filter((p) => windowMs === null || now - p.createdAt <= windowMs)
    .map(({ id, label, dailyAccuracy, movingAverage }) => ({ id, label, dailyAccuracy, movingAverage }))

  const dailyGroups = new Map<string, number[]>()
  for (const g of generations) {
    const d = new Date(g.createdAt)
    const dateKey = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
    const arr = dailyGroups.get(dateKey)
    if (arr) arr.push(g.accuracy)
    else dailyGroups.set(dateKey, [g.accuracy])
  }
  const dailySeries = [...dailyGroups.entries()]
    .map(([dateKey, values]) => {
      const [y, m, day] = dateKey.split('-').map(Number)
      return { dateKey, date: new Date(y, m, day), dailyAccuracy: average(values) }
    })
    .sort((a, b) => a.date.getTime() - b.date.getTime())

  let forecast: AccuracyForecastPoint[] = []
  if (dailySeries.length >= MIN_REAL_DAYS_FOR_FORECAST) {
    const dayMs = 24 * 60 * 60 * 1000
    const firstDateMs = dailySeries[0].date.getTime()
    const fit = linearRegression(
      dailySeries.map((p) => ({ x: (p.date.getTime() - firstDateMs) / dayMs, y: p.dailyAccuracy })),
    )
    if (fit) {
      const lastDate = dailySeries[dailySeries.length - 1].date
      const lastX = (lastDate.getTime() - firstDateMs) / dayMs
      forecast = Array.from({ length: FORECAST_DAYS }, (_, i) => {
        const date = new Date(lastDate.getTime() + (i + 1) * dayMs)
        return {
          dateKey: `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`,
          label: date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
          predicted: round1(clamp(fit.slope * (lastX + i + 1) + fit.intercept, 0, 100)),
        }
      })
    }
  }

  return { history, forecast }
}

export function periodOverPeriodDelta(
  generations: GenerationRecord[],
  windowDays: number,
  valueFn: (g: GenerationRecord) => number = (g) => g.accuracy,
): number | null {
  if (generations.length < 2) return null
  const now = Date.now()
  const windowMs = windowDays * 24 * 60 * 60 * 1000
  const current = generations.filter((g) => now - g.createdAt <= windowMs)
  const previous = generations.filter((g) => now - g.createdAt > windowMs && now - g.createdAt <= windowMs * 2)
  if (current.length === 0 || previous.length === 0) return null
  return average(current.map(valueFn)) - average(previous.map(valueFn))
}

export interface PredictionComparison {
  predicted: number
  actual: number
  error: number
  predictionAccuracy: number
}

/**
 * Compares a generation's stored predictedAccuracy (made *before* it ran)
 * against its actual outcome. Returns null when there was no prediction to
 * compare — nothing to calculate, so nothing is shown, rather than a
 * placeholder comparison.
 */
export function comparePredictionToActual(predicted: number | null, actual: number): PredictionComparison | null {
  if (predicted === null) return null
  const error = round1(Math.abs(predicted - actual))
  return { predicted, actual, error, predictionAccuracy: round1(clamp(100 - error, 0, 100)) }
}

export function accuracyBadgeTone(accuracy: number): 'good' | 'warning' | 'critical' {
  if (accuracy >= 85) return 'good'
  if (accuracy >= 70) return 'warning'
  return 'critical'
}
