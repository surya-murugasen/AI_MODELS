import { computeDelayPrediction } from '@/components/views/TrainArrivalPredictionView'
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
 * Drives the Linear Regression walkthrough (the train-delay example) on the
 * "How Algorithms Work" page. With no upload, every card shows a fixed
 * worked example (`DEFAULT_PIPELINE`). Once the user uploads a real
 * CSV/Excel file, `computeFullPipeline()` re-derives every field below from
 * their actual rows, so all 13 cards re-render with real values.
 *
 * IMPORTANT — what's real vs. what's a teaching demo:
 *   Steps 1-6, 11, 13 mirror the app's ACTUAL production formula
 *   (`computeDelayPrediction`, imported from the real feature so the two can
 *   never drift apart) — these numbers are exactly what the shipped Train
 *   Arrival Prediction screen would show for the same input.
 *   Steps 7-10 and 12 (split/choose model/train/learned/evaluate) do NOT
 *   exist in the shipped feature at all. When a file is uploaded, this page
 *   runs a genuinely real (if small) linear regression — real train/test
 *   split, real gradient descent, real MAE/RMSE/R² — as a self-contained
 *   teaching demo. Its target (y) is this app's OWN formula output, not a
 *   real historical outcome, because no such ground truth exists in an
 *   uploaded schedule file. That's still real machine learning math running
 *   on real data; it's just approximating the formula above rather than
 *   predicting a real-world result. Every "live demo" card says so.
 */

export interface RawParsedRow {
  trainNo: string
  name: string
  origin: string
  destination: string
  distanceKm: number
  scheduled: string
  speed: number
  previousDelay: number
  trafficLevel: string
  weatherCondition: string
  missingFields: string[] // which of the 4 model fields were absent/blank in the source row
}

export interface PipelineData extends PipelineCommon {
  // Step 11: predict (real formula, real or example row)
  predictExample: {
    trainNo: string
    speed: number
    previousDelay: number
    traffic: string
    weather: string
    speedPenalty: number
    trafficDelay: number
    weatherDelay: number
    scheduled: string
    predictedArrival: string
    delayMin: number
    confidence: number
  }

  // Step 13: final result
  finalResult: {
    trainNo: string
    origin: string
    destination: string
    nextStopName: string
    distanceKm: number
    scheduledDeparture: string
    predictedArrival: string
    delayMin: number
    confidence: number
    speed: number
    previousDelay: number
    trafficLevel: string
    weatherCondition: string
  }
}

// ---------------------------------------------------------------------------
// Default dummy example (used whenever no file has been uploaded)
// ---------------------------------------------------------------------------
const EX = {
  trainNo: '12627',
  name: 'Konkan Superfast Express',
  origin: 'Mumbai Central',
  destination: 'New Delhi',
  nextStop: 'Kanpur Central',
  distanceKm: 1384,
  scheduled: '07:00 PM',
  speed: 55,
  previousDelay: 8,
  traffic: 'High',
  weather: 'Rain',
}
const exResult = computeDelayPrediction({
  speed: EX.speed,
  previousDelay: EX.previousDelay,
  trafficLevel: EX.traffic,
  weatherCondition: EX.weather,
  scheduled: EX.scheduled,
})

export const DEFAULT_PIPELINE: PipelineData = {
  isLive: false,
  fileName: 'railway_data.csv',
  fileSizeKB: 24.8,
  sampleRawLine: `"${EX.trainNo},${EX.name},...,${EX.speed},${EX.traffic}"`,
  sampleRow: [
    { label: 'train_no', value: EX.trainNo },
    { label: 'speed', value: EX.speed },
    { label: 'traffic', value: EX.traffic },
  ],
  totalRows: 12,
  columnsDetected: 9,
  missingCount: 3,
  duplicateCount: 0,
  validCount: 9,
  cleaningExamples: [
    { field: 'traffic', before: '""', after: 'Moderate' },
    { field: 'weather', before: 'null', after: 'Clear' },
  ],
  encodingExamples: [
    { field: 'traffic', raw: EX.traffic, code: 2 },
    { field: 'weather', raw: EX.weather, code: 1 },
  ],
  numericFeatureNames: ['speed', 'prev_delay'],
  xFields: ['speed', 'prev_delay', 'traffic', 'weather'],
  yField: 'delay_min',
  yFieldShort: 'delay',
  splitTrainCount: 10,
  splitTestCount: 2,
  modelName: 'Linear Regression',
  whyModel: 'The target (minutes late) is a continuous number, and the effect of speed/traffic/weather is roughly additive.',
  errorHistory: [92, 70, 48, 30, 18, 10],
  weightTickerLabel: 'w_speed',
  weightEarly: 0.06,
  weightFinal: 0.35,
  learnedWeights: [
    { label: 'speed', magnitude: 0.62 },
    { label: 'delay', magnitude: 0.88 },
    { label: 'traffic', magnitude: 0.54 },
    { label: 'weather', magnitude: 0.4 },
  ],
  predictExample: {
    trainNo: EX.trainNo,
    speed: EX.speed,
    previousDelay: EX.previousDelay,
    traffic: EX.traffic,
    weather: EX.weather,
    speedPenalty: exResult.speedPenalty,
    trafficDelay: exResult.routeFactor,
    weatherDelay: exResult.weatherDelay,
    scheduled: EX.scheduled,
    predictedArrival: exResult.predicted,
    delayMin: exResult.delayMin,
    confidence: exResult.confidence,
  },
  evaluationRows: [
    { actual: 22, predicted: 20, error: 2 },
    { actual: 5, predicted: 7, error: 2 },
    { actual: 14, predicted: 13, error: 1 },
  ],
  metrics: { mae: 1.7, rmse: 2.1, r2: 0.91 },
  metricUnit: 'm',
  finalResult: {
    trainNo: EX.trainNo,
    origin: EX.origin,
    destination: EX.destination,
    nextStopName: EX.nextStop,
    distanceKm: EX.distanceKm,
    scheduledDeparture: EX.scheduled,
    predictedArrival: exResult.predicted,
    delayMin: exResult.delayMin,
    confidence: exResult.confidence,
    speed: EX.speed,
    previousDelay: EX.previousDelay,
    trafficLevel: EX.traffic,
    weatherCondition: EX.weather,
  },
}

// ---------------------------------------------------------------------------
// CSV parsing (independent of, but alias-compatible with, the production
// parser) — this one also tracks which fields were missing/defaulted per
// row and detects duplicate rows, since that's what the Inspect/Clean cards
// need to show real counts.
// ---------------------------------------------------------------------------
export interface ParsedUpload {
  rows: RawParsedRow[]
  columnsDetected: number
}

export function parseUploadedRows(csvText: string): ParsedUpload {
  const lines = splitCsvLines(csvText)
  if (lines.length <= 1) return { rows: [], columnsDetected: 0 }

  const headers = lines[0].toLowerCase().split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''))
  const rows: RawParsedRow[] = []

  for (let i = 1; i < lines.length; i++) {
    const rawRow = lines[i].split(',').map((c) => c.trim())
    if (rawRow.length < 2) continue

    const missingFields: string[] = []

    const trainNoRaw = getCol(headers, rawRow, ['no', 'number', 'train'])
    const trainNo = trainNoRaw ?? `TR-${1000 + i}`

    const nameRaw = getCol(headers, rawRow, ['name', 'title'])
    const name = nameRaw ?? `Express ${trainNo}`

    const origin = getCol(headers, rawRow, ['origin', 'from', 'source']) ?? 'Departure'
    const destination = getCol(headers, rawRow, ['destination', 'to']) ?? 'Arrival'
    const distanceRaw = getCol(headers, rawRow, ['distance', 'km'])
    const distanceKm = distanceRaw ? parseFloat(distanceRaw) : NaN

    const scheduledRaw = getCol(headers, rawRow, ['scheduled', 'time', 'sched'])
    const scheduled = scheduledRaw ?? '07:00 PM'
    if (!scheduledRaw) missingFields.push('scheduled')

    const speedRaw = getCol(headers, rawRow, ['speed', 'velocity'])
    const speed = speedRaw ? parseFloat(speedRaw) : NaN
    if (!speedRaw || Number.isNaN(speed)) missingFields.push('speed')

    const prevDelayRaw = getCol(headers, rawRow, ['prev', 'hold', 'delay'])
    const previousDelay = prevDelayRaw ? parseFloat(prevDelayRaw) : NaN
    if (!prevDelayRaw || Number.isNaN(previousDelay)) missingFields.push('previous_delay')

    const trafficRaw = getCol(headers, rawRow, ['traffic', 'congestion', 'route'])
    if (!trafficRaw) missingFields.push('traffic')

    const weatherRaw = getCol(headers, rawRow, ['weather', 'climate'])
    if (!weatherRaw) missingFields.push('weather')

    rows.push({
      trainNo,
      name,
      origin,
      destination,
      distanceKm: Number.isNaN(distanceKm) ? 120 : distanceKm,
      scheduled,
      speed: Number.isNaN(speed) ? 68 : speed,
      previousDelay: Number.isNaN(previousDelay) ? 0 : previousDelay,
      trafficLevel: trafficRaw ?? 'Moderate',
      weatherCondition: weatherRaw ?? 'Clear',
      missingFields,
    })
  }

  return { rows, columnsDetected: headers.length }
}

function countDuplicates(rows: RawParsedRow[]): number {
  const seen = new Set<string>()
  let dupes = 0
  for (const r of rows) {
    const key = `${r.trainNo}|${r.speed}|${r.previousDelay}|${r.trafficLevel}|${r.weatherCondition}`
    if (seen.has(key)) dupes += 1
    else seen.add(key)
  }
  return dupes
}

function trafficCode(level: string): number {
  const l = level.toLowerCase()
  if (l.includes('high')) return 2
  if (l.includes('mod')) return 1
  return 0
}
function weatherCode(cond: string): number {
  const l = cond.toLowerCase()
  if (l.includes('fog')) return 2
  if (l.includes('rain')) return 1
  return 0
}

// ---------------------------------------------------------------------------
// Full pipeline: raw uploaded rows -> every card's data
// ---------------------------------------------------------------------------
export function computeFullPipeline(rawRows: RawParsedRow[], fileName: string, fileSizeKB: number, columnsDetected: number): PipelineData {
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
        after: field === 'traffic' ? firstMissing.trafficLevel : field === 'weather' ? firstMissing.weatherCondition : 'defaulted',
      }))
    : [
        { field: 'traffic', before: '(none blank)', after: first?.trafficLevel ?? 'Moderate' },
        { field: 'weather', before: '(none blank)', after: first?.weatherCondition ?? 'Clear' },
      ]

  const encodingExamples: EncodingExample[] = [
    { field: 'traffic', raw: first?.trafficLevel ?? 'Moderate', code: trafficCode(first?.trafficLevel ?? 'Moderate') },
    { field: 'weather', raw: first?.weatherCondition ?? 'Clear', code: weatherCode(first?.weatherCondition ?? 'Clear') },
  ]

  // Real target (y): this app's own formula output per row — see module docstring for why.
  const yRaw = rawRows.map((r) =>
    computeDelayPrediction({
      speed: r.speed,
      previousDelay: r.previousDelay,
      trafficLevel: r.trafficLevel,
      weatherCondition: r.weatherCondition,
      scheduled: r.scheduled,
    }).delayMin,
  )

  const featuresRaw = rawRows.map((r) => [r.speed, r.previousDelay, trafficCode(r.trafficLevel), weatherCode(r.weatherCondition)])

  // Deterministic 80/20 split (first 80% train, remainder test) — no shuffling,
  // so results are reproducible for the same file.
  const testCount = totalRows >= 5 ? Math.max(1, Math.round(totalRows * 0.2)) : totalRows > 1 ? 1 : 0
  const trainCount = totalRows - testCount

  const trainX = featuresRaw.slice(0, trainCount)
  const trainY = yRaw.slice(0, trainCount)
  const testX = featuresRaw.slice(trainCount)
  const testY = yRaw.slice(trainCount)

  // Normalize using TRAIN statistics only (no test leakage into scaling).
  const colStats = [0, 1, 2, 3].map((j) => zscore(trainX.map((row) => row[j])))
  const normalize = (row: number[]) => row.map((v, j) => (v - colStats[j].mean) / colStats[j].std)
  const trainXNorm = trainX.map(normalize)
  const testXNorm = testX.map(normalize)

  const hasEnoughToTrain = trainXNorm.length >= 2
  const gd = hasEnoughToTrain
    ? trainLinearRegressionGD(trainXNorm, trainY)
    : { weights: [0, 0, 0, 0], bias: trainY[0] ?? 0, errorHistory: [0], weightEarly: 0, weightFinal: 0 }

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
    { label: 'delay', magnitude: Math.abs(gd.weights[1]) / maxAbsWeight },
    { label: 'traffic', magnitude: Math.abs(gd.weights[2]) / maxAbsWeight },
    { label: 'weather', magnitude: Math.abs(gd.weights[3]) / maxAbsWeight },
  ]

  // "Test / Predict" (step 11) always shows the REAL formula, on the last uploaded row.
  const predictRowSource = rawRows[rawRows.length - 1] ?? rawRows[0]
  const predictFormula = computeDelayPrediction({
    speed: predictRowSource.speed,
    previousDelay: predictRowSource.previousDelay,
    trafficLevel: predictRowSource.trafficLevel,
    weatherCondition: predictRowSource.weatherCondition,
    scheduled: predictRowSource.scheduled,
  })

  return {
    isLive: true,
    fileName,
    fileSizeKB,
    sampleRawLine: `"${first.trainNo},${first.name.slice(0, 10)}...,${first.speed},${first.trafficLevel}"`,
    sampleRow: [
      { label: 'train_no', value: first.trainNo },
      { label: 'speed', value: first.speed },
      { label: 'traffic', value: first.trafficLevel },
    ],
    totalRows,
    columnsDetected,
    missingCount,
    duplicateCount,
    validCount,
    cleaningExamples,
    encodingExamples,
    numericFeatureNames: ['speed', 'prev_delay'],
    xFields: ['speed', 'prev_delay', 'traffic', 'weather'],
    yField: 'delay_min',
    yFieldShort: 'delay',
    splitTrainCount: trainCount,
    splitTestCount: testCount,
    modelName: 'Linear Regression',
    whyModel: 'The target (minutes late) is a continuous number, and the effect of speed/traffic/weather is roughly additive.',
    errorHistory: gd.errorHistory,
    weightTickerLabel: 'w_speed',
    weightEarly: Math.round(gd.weightEarly * 100) / 100,
    weightFinal: Math.round(gd.weightFinal * 100) / 100,
    learnedWeights,
    predictExample: {
      trainNo: predictRowSource.trainNo,
      speed: predictRowSource.speed,
      previousDelay: predictRowSource.previousDelay,
      traffic: predictRowSource.trafficLevel,
      weather: predictRowSource.weatherCondition,
      speedPenalty: predictFormula.speedPenalty,
      trafficDelay: predictFormula.routeFactor,
      weatherDelay: predictFormula.weatherDelay,
      scheduled: predictRowSource.scheduled,
      predictedArrival: predictFormula.predicted,
      delayMin: predictFormula.delayMin,
      confidence: predictFormula.confidence,
    },
    evaluationRows,
    metrics,
    metricUnit: 'm',
    finalResult: {
      trainNo: predictRowSource.trainNo,
      origin: predictRowSource.origin,
      destination: predictRowSource.destination,
      nextStopName: 'Next stop',
      distanceKm: predictRowSource.distanceKm,
      scheduledDeparture: predictRowSource.scheduled,
      predictedArrival: predictFormula.predicted,
      delayMin: predictFormula.delayMin,
      confidence: predictFormula.confidence,
      speed: predictRowSource.speed,
      previousDelay: predictRowSource.previousDelay,
      trafficLevel: predictRowSource.trafficLevel,
      weatherCondition: predictRowSource.weatherCondition,
    },
  }
}
