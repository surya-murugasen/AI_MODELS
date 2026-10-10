/**
 * Shared types + small dependency-free ML math reused across every "How
 * Algorithms Work" domain (trains/Linear Regression, house prices/Multiple
 * Linear Regression, braking distance/Polynomial Regression). Each domain
 * has its own pipeline module and its own predict/final-result visuals, but
 * they all share this common data shape so the same 11 stage-card visual
 * components (Upload, Read, Inspect, Clean, Features, Separate X/y, Split,
 * Choose Model, Train, Learned Model, Evaluate) can render any of them
 * without duplicating code.
 */

export interface CleaningExample {
  field: string
  before: string
  after: string
}

export interface EncodingExample {
  field: string
  raw: string
  code: number
}

export interface EvaluationRow {
  actual: number
  predicted: number
  error: number
}

export interface SampleCell {
  label: string
  value: string | number
}

export interface PipelineCommon {
  isLive: boolean
  fileName: string
  fileSizeKB: number

  // Step 1-2: upload / read
  sampleRawLine: string
  sampleRow: SampleCell[]

  // Step 3: inspect
  totalRows: number
  columnsDetected: number
  missingCount: number
  duplicateCount: number
  validCount: number

  // Step 4: clean
  cleaningExamples: CleaningExample[]

  // Step 5: prepare features
  encodingExamples: EncodingExample[]
  numericFeatureNames: string[]

  // Step 6: X / y
  xFields: string[]
  yField: string
  yFieldShort: string // short display label used in the learned-network diagram

  // Step 7: split
  splitTrainCount: number
  splitTestCount: number

  // Step 8: choose model
  modelName: string
  whyModel: string

  // Step 9: train (real gradient descent when live)
  errorHistory: number[] // sampled MSE, decreasing
  weightTickerLabel: string
  weightEarly: number
  weightFinal: number

  // Step 10: learned model
  learnedWeights: { label: string; magnitude: number }[] // 0..1, relative importance

  // Step 12: evaluate
  evaluationRows: EvaluationRow[]
  metrics: { mae: number | null; rmse: number | null; r2: number | null }
  metricUnit: string // display suffix for MAE/RMSE and table values, e.g. "m", "$", ""
}

// ---------------------------------------------------------------------------
// Small linear algebra: z-score normalize + batch gradient descent + metrics
// ---------------------------------------------------------------------------
export function zscore(values: number[]): { normalized: number[]; mean: number; std: number } {
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length
  const std = Math.sqrt(variance) || 1
  return { normalized: values.map((v) => (v - mean) / std), mean, std }
}

export interface GDResult {
  weights: number[]
  bias: number
  errorHistory: number[]
  weightEarly: number
  weightFinal: number
}

/** Batch gradient descent for multi-variate linear regression, on already
 * z-score-normalized features. Small, dependency-free, and fast enough for
 * the tiny row counts a demo CSV upload will realistically have. Also what
 * "polynomial regression" reduces to once x² is added as its own feature
 * column — still ordinary multi-variate linear regression under the hood. */
export function trainLinearRegressionGD(X: number[][], y: number[], iterations = 300, lr = 0.15): GDResult {
  const n = X.length
  const nFeatures = X[0]?.length ?? 0
  let weights = Array.from({ length: nFeatures }, () => 0)
  let bias = 0
  const errorHistory: number[] = []
  let weightEarly = 0

  for (let it = 0; it < iterations; it++) {
    const predictions = X.map((row) => row.reduce((sum, x, j) => sum + x * weights[j], bias))
    const errors = predictions.map((p, i) => p - y[i])

    const gradW = Array.from({ length: nFeatures }, () => 0)
    for (let j = 0; j < nFeatures; j++) {
      gradW[j] = (2 / n) * X.reduce((sum, row, i) => sum + row[j] * errors[i], 0)
    }
    const gradB = (2 / n) * errors.reduce((a, b) => a + b, 0)

    weights = weights.map((w, j) => w - lr * gradW[j])
    bias -= lr * gradB

    if (it === Math.floor(iterations * 0.1)) weightEarly = weights[0]

    if (it % Math.max(1, Math.floor(iterations / 8)) === 0 || it === iterations - 1) {
      const mse = errors.reduce((a, b) => a + b * b, 0) / n
      errorHistory.push(mse)
    }
  }

  return { weights, bias, errorHistory, weightEarly, weightFinal: weights[0] }
}

export function computeMetrics(actual: number[], predicted: number[]): { mae: number | null; rmse: number | null; r2: number | null } {
  const n = actual.length
  if (n === 0) return { mae: null, rmse: null, r2: null }

  const errors = actual.map((a, i) => a - predicted[i])
  const mae = errors.reduce((s, e) => s + Math.abs(e), 0) / n
  const rmse = Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / n)

  if (n < 2) return { mae, rmse, r2: null }
  const meanActual = actual.reduce((a, b) => a + b, 0) / n
  const ssTot = actual.reduce((s, a) => s + (a - meanActual) ** 2, 0)
  const ssRes = errors.reduce((s, e) => s + e * e, 0)
  const r2 = ssTot === 0 ? null : 1 - ssRes / ssTot

  return { mae, rmse, r2 }
}

// ---------------------------------------------------------------------------
// Small shared CSV helper
// ---------------------------------------------------------------------------
export function getCol(headers: string[], row: string[], aliases: string[]): string | null {
  for (const alias of aliases) {
    const idx = headers.findIndex((h) => h.includes(alias))
    if (idx !== -1 && row[idx] !== undefined && row[idx].trim() !== '') {
      return row[idx].trim().replace(/^["']|["']$/g, '')
    }
  }
  return null
}

export function splitCsvLines(csvText: string): string[] {
  return csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
}
