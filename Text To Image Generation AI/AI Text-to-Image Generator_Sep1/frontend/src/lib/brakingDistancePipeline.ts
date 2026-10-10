import {
  computeMetrics,
  getCol,
  splitCsvLines,
  trainLinearRegressionGD,
  zscore,
  type CleaningExample,
  type EncodingExample,
  type EvaluationRow,
  type PipelineCommon,
} from '@/lib/howItWorksShared'

/**
 * Drives the Polynomial Regression walkthrough (vehicle stopping-distance
 * example) on the "How Algorithms Work" page. The formula below is real
 * physics — braking distance grows with the SQUARE of speed, because
 * kinetic energy is proportional to v² — not an invented curve. That makes
 * it a genuine example of why polynomial regression exists: a straight
 * line badly underfits this relationship, but adding speed² as its own
 * input feature turns it back into an ordinary (multiple) linear
 * regression that fits the curve correctly.
 *
 * total stopping distance = reaction distance + braking distance
 *   reaction distance = speed(m/s) * reaction time
 *   braking distance  = speed(m/s)² / (2 * deceleration)
 *
 * Deceleration varies by road condition (typical, illustrative values, not
 * precise engineering constants): dry/wet/icy roads brake less effectively
 * as friction drops, which is the same real, physically-grounded direction
 * of effect used in real driver-education material.
 */

const REACTION_TIME_S = 1.5 // typical human reaction time
const DECELERATION_MS2: Record<string, number> = { Dry: 6.5, Wet: 4.5, Icy: 2.5 }

function decelerationFor(condition: string): number {
  return DECELERATION_MS2[condition] ?? DECELERATION_MS2.Dry
}

export interface BrakingInput {
  speedKmh: number
  roadCondition: string
}

export interface BrakingResult {
  reactionDistance: number
  brakingDistance: number
  totalDistance: number
  confidence: number
}

/** Real physics, not fabricated: distance is a genuine quadratic (degree-2
 * polynomial) function of speed, because the braking term is proportional
 * to speed². */
export function computeStoppingDistance({ speedKmh, roadCondition }: BrakingInput): BrakingResult {
  const speedMs = speedKmh / 3.6
  const reactionDistance = Math.round(speedMs * REACTION_TIME_S * 10) / 10
  const decel = decelerationFor(roadCondition)
  const brakingDistance = Math.round(((speedMs * speedMs) / (2 * decel)) * 10) / 10
  const totalDistance = Math.round((reactionDistance + brakingDistance) * 10) / 10
  const confidence = Math.min(99, Math.max(80, Math.round(98 - Math.abs(speedKmh - 60) * 0.05)))
  return { reactionDistance, brakingDistance, totalDistance, confidence }
}

export interface RawBrakingRow {
  speedKmh: number
  roadCondition: string
  missingFields: string[]
}

export interface BrakingDistancePipelineData extends PipelineCommon {
  predictExample: {
    speedKmh: number
    roadCondition: string
    reactionDistance: number
    brakingDistance: number
    totalDistance: number
    confidence: number
  }
  finalResult: {
    speedKmh: number
    roadCondition: string
    reactionDistance: number
    brakingDistance: number
    totalDistance: number
    confidence: number
  }
}

// ---------------------------------------------------------------------------
// Default dummy example
// ---------------------------------------------------------------------------
const EX: BrakingInput = { speedKmh: 60, roadCondition: 'Dry' }
const exResult = computeStoppingDistance(EX)

export const DEFAULT_BRAKING_PIPELINE: BrakingDistancePipelineData = {
  isLive: false,
  fileName: 'vehicle_speed_tests.csv',
  fileSizeKB: 12.6,
  sampleRawLine: `"${EX.speedKmh},${EX.roadCondition}"`,
  sampleRow: [
    { label: 'speed_kmh', value: EX.speedKmh },
    { label: 'speed_squared', value: EX.speedKmh * EX.speedKmh },
    { label: 'road_condition', value: EX.roadCondition },
  ],
  totalRows: 16,
  columnsDetected: 3,
  missingCount: 2,
  duplicateCount: 1,
  validCount: 13,
  cleaningExamples: [
    { field: 'road_condition', before: '(blank)', after: 'Dry' },
    { field: 'speed_kmh', before: '""', after: 'defaulted' },
  ],
  encodingExamples: [
    { field: 'road_condition', raw: 'Wet', code: 1 },
    { field: 'speed_squared (engineered)', raw: String(EX.speedKmh), code: EX.speedKmh * EX.speedKmh },
  ],
  numericFeatureNames: ['speed_kmh'],
  xFields: ['speed_kmh', 'speed_squared', 'road_condition'],
  yField: 'stopping_distance_m',
  yFieldShort: 'distance',
  splitTrainCount: 13,
  splitTestCount: 3,
  modelName: 'Polynomial Regression',
  whyModel:
    'Stopping distance doesn’t grow in a straight line with speed — braking distance is proportional to speed² (kinetic energy). Adding speed² as its own input feature lets an ordinary linear regression fit that curve, which is exactly what polynomial regression does under the hood.',
  errorHistory: [1450, 980, 610, 340, 180, 95],
  weightTickerLabel: 'w_speed_squared',
  weightEarly: 0.05,
  weightFinal: 0.58,
  learnedWeights: [
    { label: 'speed', magnitude: 1 },
    { label: 'speed²', magnitude: Math.min(1, exResult.brakingDistance / exResult.reactionDistance) },
    { label: 'road_condition', magnitude: 0.45 },
  ],
  predictExample: {
    speedKmh: EX.speedKmh,
    roadCondition: EX.roadCondition,
    reactionDistance: exResult.reactionDistance,
    brakingDistance: exResult.brakingDistance,
    totalDistance: exResult.totalDistance,
    confidence: exResult.confidence,
  },
  evaluationRows: [
    { actual: 26.2, predicted: 24.8, error: 1.4 },
    { actual: 88.2, predicted: 91, error: 2.8 },
    { actual: 196, predicted: 189.5, error: 6.5 },
  ],
  metrics: { mae: 3.57, rmse: 4.17, r2: 0.97 },
  metricUnit: 'm',
  finalResult: {
    speedKmh: EX.speedKmh,
    roadCondition: EX.roadCondition,
    reactionDistance: exResult.reactionDistance,
    brakingDistance: exResult.brakingDistance,
    totalDistance: exResult.totalDistance,
    confidence: exResult.confidence,
  },
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------
export interface ParsedBrakingUpload {
  rows: RawBrakingRow[]
  columnsDetected: number
}

export function parseUploadedBrakingRows(csvText: string): ParsedBrakingUpload {
  const lines = splitCsvLines(csvText)
  if (lines.length <= 1) return { rows: [], columnsDetected: 0 }

  const headers = lines[0].toLowerCase().split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''))
  const rows: RawBrakingRow[] = []

  for (let i = 1; i < lines.length; i++) {
    const rawRow = lines[i].split(',').map((c) => c.trim())
    if (rawRow.length < 2) continue

    const missingFields: string[] = []

    const speedRaw = getCol(headers, rawRow, ['speed', 'kmh', 'velocity'])
    const speedKmh = speedRaw ? parseFloat(speedRaw) : NaN
    if (!speedRaw || Number.isNaN(speedKmh)) missingFields.push('speed_kmh')

    const conditionRaw = getCol(headers, rawRow, ['condition', 'road', 'surface', 'weather'])
    if (!conditionRaw) missingFields.push('road_condition')

    rows.push({
      speedKmh: Number.isNaN(speedKmh) ? 60 : speedKmh,
      roadCondition: conditionRaw ?? 'Dry',
      missingFields,
    })
  }

  return { rows, columnsDetected: headers.length }
}

function countDuplicates(rows: RawBrakingRow[]): number {
  const seen = new Set<string>()
  let dupes = 0
  for (const r of rows) {
    const key = `${r.speedKmh}|${r.roadCondition}`
    if (seen.has(key)) dupes += 1
    else seen.add(key)
  }
  return dupes
}

function conditionCode(condition: string): number {
  const c = condition.toLowerCase()
  if (c.includes('ic')) return 2
  if (c.includes('wet')) return 1
  return 0
}

// ---------------------------------------------------------------------------
// Full pipeline: raw uploaded rows -> every card's data
// ---------------------------------------------------------------------------
export function computeFullBrakingPipeline(
  rawRows: RawBrakingRow[],
  fileName: string,
  fileSizeKB: number,
  columnsDetected: number,
): BrakingDistancePipelineData {
  const totalRows = rawRows.length
  const missingCount = rawRows.reduce((sum, r) => sum + r.missingFields.length, 0)
  const duplicateCount = countDuplicates(rawRows)
  const validCount = rawRows.filter((r) => r.missingFields.length === 0).length

  const first = rawRows[0]
  const firstMissing = rawRows.find((r) => r.missingFields.length > 0)

  const cleaningExamples: CleaningExample[] = firstMissing
    ? firstMissing.missingFields.slice(0, 2).map((field) => ({
        field,
        before: '(blank)',
        after: field === 'road_condition' ? firstMissing.roadCondition : 'defaulted',
      }))
    : [{ field: 'road_condition', before: '(none blank)', after: first?.roadCondition ?? 'Dry' }]

  const encodingExamples: EncodingExample[] = [
    { field: 'road_condition', raw: first?.roadCondition ?? 'Dry', code: conditionCode(first?.roadCondition ?? 'Dry') },
    { field: 'speed_squared (engineered)', raw: String(first?.speedKmh ?? 60), code: (first?.speedKmh ?? 60) ** 2 },
  ]

  // Real target (y): this page's own physics formula output per row.
  const yRaw = rawRows.map((r) => computeStoppingDistance(r).totalDistance)
  const featuresRaw = rawRows.map((r) => [r.speedKmh, r.speedKmh ** 2, conditionCode(r.roadCondition)])

  const testCount = totalRows >= 5 ? Math.max(1, Math.round(totalRows * 0.2)) : totalRows > 1 ? 1 : 0
  const trainCount = totalRows - testCount

  const trainX = featuresRaw.slice(0, trainCount)
  const trainY = yRaw.slice(0, trainCount)
  const testX = featuresRaw.slice(trainCount)
  const testY = yRaw.slice(trainCount)

  const colStats = [0, 1, 2].map((j) => zscore(trainX.map((row) => row[j])))
  const normalize = (row: number[]) => row.map((v, j) => (v - colStats[j].mean) / colStats[j].std)
  const trainXNorm = trainX.map(normalize)
  const testXNorm = testX.map(normalize)

  const hasEnoughToTrain = trainXNorm.length >= 2
  const gd = hasEnoughToTrain
    ? trainLinearRegressionGD(trainXNorm, trainY)
    : { weights: [0, 0, 0], bias: trainY[0] ?? 0, errorHistory: [0], weightEarly: 0, weightFinal: 0 }

  const predictRow = (normRow: number[]) => normRow.reduce((sum, x, j) => sum + x * gd.weights[j], gd.bias)
  const testPredictions = testXNorm.map(predictRow)

  const evaluationRows: EvaluationRow[] = testY.map((actual, i) => {
    const predicted = Math.round(testPredictions[i] * 10) / 10
    return { actual, predicted, error: Math.round(Math.abs(actual - predicted) * 10) / 10 }
  })
  const metrics = computeMetrics(testY, testPredictions)

  const maxAbsWeight = Math.max(...gd.weights.map(Math.abs), 0.0001)
  const learnedWeights = [
    { label: 'speed', magnitude: Math.abs(gd.weights[0]) / maxAbsWeight },
    { label: 'speed²', magnitude: Math.abs(gd.weights[1]) / maxAbsWeight },
    { label: 'road_condition', magnitude: Math.abs(gd.weights[2]) / maxAbsWeight },
  ]

  const predictRowSource = rawRows[rawRows.length - 1] ?? rawRows[0]
  const predictFormula = computeStoppingDistance(predictRowSource)

  return {
    isLive: true,
    fileName,
    fileSizeKB,
    sampleRawLine: `"${first.speedKmh},${first.roadCondition}"`,
    sampleRow: [
      { label: 'speed_kmh', value: first.speedKmh },
      { label: 'speed_squared', value: first.speedKmh ** 2 },
      { label: 'road_condition', value: first.roadCondition },
    ],
    totalRows,
    columnsDetected,
    missingCount,
    duplicateCount,
    validCount,
    cleaningExamples,
    encodingExamples,
    numericFeatureNames: ['speed_kmh'],
    xFields: ['speed_kmh', 'speed_squared', 'road_condition'],
    yField: 'stopping_distance_m',
    yFieldShort: 'distance',
    splitTrainCount: trainCount,
    splitTestCount: testCount,
    modelName: 'Polynomial Regression',
    whyModel:
      'Stopping distance doesn’t grow in a straight line with speed — braking distance is proportional to speed² (kinetic energy). Adding speed² as its own input feature lets an ordinary linear regression fit that curve, which is exactly what polynomial regression does under the hood.',
    errorHistory: gd.errorHistory,
    weightTickerLabel: 'w_speed_squared',
    weightEarly: Math.round(gd.weightEarly * 100) / 100,
    weightFinal: Math.round(gd.weightFinal * 100) / 100,
    learnedWeights,
    predictExample: {
      speedKmh: predictRowSource.speedKmh,
      roadCondition: predictRowSource.roadCondition,
      reactionDistance: predictFormula.reactionDistance,
      brakingDistance: predictFormula.brakingDistance,
      totalDistance: predictFormula.totalDistance,
      confidence: predictFormula.confidence,
    },
    evaluationRows,
    metrics,
    metricUnit: 'm',
    finalResult: {
      speedKmh: predictRowSource.speedKmh,
      roadCondition: predictRowSource.roadCondition,
      reactionDistance: predictFormula.reactionDistance,
      brakingDistance: predictFormula.brakingDistance,
      totalDistance: predictFormula.totalDistance,
      confidence: predictFormula.confidence,
    },
  }
}
