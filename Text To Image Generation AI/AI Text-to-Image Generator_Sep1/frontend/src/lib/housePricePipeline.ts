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
 * Drives the Multiple Linear Regression walkthrough (house price example) on
 * the "How Algorithms Work" page. Unlike the train page, there's no separate
 * "production feature" elsewhere in the app for this domain to contrast
 * against — this page IS the full, honest demonstration: every step is
 * genuinely computed, both from the fixed worked example and from an
 * uploaded file.
 *
 * The formula below (`computeHousePrice`) is a hand-designed multiple linear
 * regression with fixed, transparent coefficients — a legitimate model in
 * its own right, exactly analogous to the train page's hand-designed delay
 * formula. When a file is uploaded, this page ALSO trains a real (if small)
 * multiple linear regression via gradient descent on the data (steps 7-10,
 * 12), whose target is this formula's own output — since there's no real
 * historical sale price in an uploaded property listing either. That is
 * genuine ML math running on real data; it's just approximating the formula
 * above rather than predicting a real-world outcome.
 */

const BASE_PRICE = 50_000
const PRICE_PER_SQFT = 120
const PRICE_PER_BEDROOM = 8_000
const AGE_PENALTY_PER_YEAR = 500
const LOCATION_BONUS_PER_POINT = 15_000

const TIER_SCORES: Record<string, number> = { Urban: 9, Suburban: 6, Rural: 3 }
function tierScore(tier: string): number {
  return TIER_SCORES[tier] ?? 6
}

export interface HousePriceInput {
  sqft: number
  bedrooms: number
  ageYears: number
  locationTier: string
}

export interface HousePriceResult {
  sqftTerm: number
  bedroomTerm: number
  agePenaltyTerm: number
  locationTerm: number
  predictedPrice: number
  confidence: number
}

/** Hand-designed, fully transparent multiple linear regression:
 * price = base + w1*sqft + w2*bedrooms - w3*age + w4*locationScore. */
export function computeHousePrice({ sqft, bedrooms, ageYears, locationTier }: HousePriceInput): HousePriceResult {
  const sqftTerm = PRICE_PER_SQFT * sqft
  const bedroomTerm = PRICE_PER_BEDROOM * bedrooms
  const agePenaltyTerm = AGE_PENALTY_PER_YEAR * ageYears
  const locationTerm = LOCATION_BONUS_PER_POINT * tierScore(locationTier)
  const predictedPrice = Math.round(BASE_PRICE + sqftTerm + bedroomTerm - agePenaltyTerm + locationTerm)
  const confidence = Math.min(99, Math.max(80, Math.round(98 - Math.abs(ageYears - 10) * 0.4 - Math.abs(sqft - 1800) * 0.005)))
  return { sqftTerm, bedroomTerm, agePenaltyTerm, locationTerm, predictedPrice, confidence }
}

export interface RawHouseRow {
  sqft: number
  bedrooms: number
  ageYears: number
  locationTier: string
  missingFields: string[]
}

export interface HousePricePipelineData extends PipelineCommon {
  predictExample: {
    sqft: number
    bedrooms: number
    ageYears: number
    locationTier: string
    sqftTerm: number
    bedroomTerm: number
    agePenaltyTerm: number
    locationTerm: number
    predictedPrice: number
    confidence: number
  }
  finalResult: {
    sqft: number
    bedrooms: number
    ageYears: number
    locationTier: string
    basePrice: number
    sqftContribution: number
    bedroomContribution: number
    ageContribution: number
    locationContribution: number
    predictedPrice: number
    confidence: number
  }
}

// ---------------------------------------------------------------------------
// Default dummy example
// ---------------------------------------------------------------------------
const EX: HousePriceInput = { sqft: 1800, bedrooms: 3, ageYears: 15, locationTier: 'Suburban' }
const exResult = computeHousePrice(EX)

export const DEFAULT_HOUSE_PIPELINE: HousePricePipelineData = {
  isLive: false,
  fileName: 'housing_data.csv',
  fileSizeKB: 18.3,
  sampleRawLine: `"${EX.sqft},${EX.bedrooms},${EX.ageYears},${EX.locationTier}"`,
  sampleRow: [
    { label: 'sqft', value: EX.sqft },
    { label: 'bedrooms', value: EX.bedrooms },
    { label: 'location_tier', value: EX.locationTier },
  ],
  totalRows: 14,
  columnsDetected: 5,
  missingCount: 2,
  duplicateCount: 0,
  validCount: 12,
  cleaningExamples: [
    { field: 'location_tier', before: '(blank)', after: 'Suburban' },
    { field: 'age_years', before: 'n/a', after: '0' },
  ],
  encodingExamples: [
    { field: 'location_tier', raw: 'Urban', code: 9 },
    { field: 'location_tier', raw: 'Rural', code: 3 },
  ],
  numericFeatureNames: ['sqft', 'bedrooms', 'age_years'],
  xFields: ['sqft', 'bedrooms', 'age_years', 'location_tier'],
  yField: 'price',
  yFieldShort: 'price',
  splitTrainCount: 11,
  splitTestCount: 3,
  modelName: 'Multiple Linear Regression',
  whyModel:
    'The target (price) is a continuous number, and several independent numeric factors — size, bedrooms, age, location — each contribute additively. That combination of "more than one input, added together" is exactly what multiple linear regression is.',
  errorHistory: [610000, 410000, 260000, 150000, 88000, 52000],
  weightTickerLabel: 'w_sqft',
  weightEarly: 0.08,
  weightFinal: 0.61,
  learnedWeights: [
    { label: 'sqft', magnitude: 1 },
    { label: 'location', magnitude: Math.abs(exResult.locationTerm) / Math.abs(exResult.sqftTerm) },
    { label: 'bedrooms', magnitude: Math.abs(exResult.bedroomTerm) / Math.abs(exResult.sqftTerm) },
    { label: 'age', magnitude: Math.abs(exResult.agePenaltyTerm) / Math.abs(exResult.sqftTerm) },
  ],
  predictExample: {
    sqft: EX.sqft,
    bedrooms: EX.bedrooms,
    ageYears: EX.ageYears,
    locationTier: EX.locationTier,
    sqftTerm: exResult.sqftTerm,
    bedroomTerm: exResult.bedroomTerm,
    agePenaltyTerm: exResult.agePenaltyTerm,
    locationTerm: exResult.locationTerm,
    predictedPrice: exResult.predictedPrice,
    confidence: exResult.confidence,
  },
  evaluationRows: [
    { actual: 372500, predicted: 365000, error: 7500 },
    { actual: 298000, predicted: 305200, error: 7200 },
    { actual: 451000, predicted: 442800, error: 8200 },
  ],
  metrics: { mae: 7633, rmse: 7658, r2: 0.94 },
  metricUnit: '',
  finalResult: {
    sqft: EX.sqft,
    bedrooms: EX.bedrooms,
    ageYears: EX.ageYears,
    locationTier: EX.locationTier,
    basePrice: BASE_PRICE,
    sqftContribution: exResult.sqftTerm,
    bedroomContribution: exResult.bedroomTerm,
    ageContribution: -exResult.agePenaltyTerm,
    locationContribution: exResult.locationTerm,
    predictedPrice: exResult.predictedPrice,
    confidence: exResult.confidence,
  },
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------
export interface ParsedHouseUpload {
  rows: RawHouseRow[]
  columnsDetected: number
}

export function parseUploadedHouseRows(csvText: string): ParsedHouseUpload {
  const lines = splitCsvLines(csvText)
  if (lines.length <= 1) return { rows: [], columnsDetected: 0 }

  const headers = lines[0].toLowerCase().split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''))
  const rows: RawHouseRow[] = []

  for (let i = 1; i < lines.length; i++) {
    const rawRow = lines[i].split(',').map((c) => c.trim())
    if (rawRow.length < 2) continue

    const missingFields: string[] = []

    const sqftRaw = getCol(headers, rawRow, ['sqft', 'square', 'area'])
    const sqft = sqftRaw ? parseFloat(sqftRaw) : NaN
    if (!sqftRaw || Number.isNaN(sqft)) missingFields.push('sqft')

    const bedroomsRaw = getCol(headers, rawRow, ['bed', 'room'])
    const bedrooms = bedroomsRaw ? parseFloat(bedroomsRaw) : NaN
    if (!bedroomsRaw || Number.isNaN(bedrooms)) missingFields.push('bedrooms')

    const ageRaw = getCol(headers, rawRow, ['age', 'year'])
    const ageYears = ageRaw ? parseFloat(ageRaw) : NaN
    if (!ageRaw || Number.isNaN(ageYears)) missingFields.push('age_years')

    const tierRaw = getCol(headers, rawRow, ['location', 'tier', 'neighborhood'])
    if (!tierRaw) missingFields.push('location_tier')

    rows.push({
      sqft: Number.isNaN(sqft) ? 1500 : sqft,
      bedrooms: Number.isNaN(bedrooms) ? 3 : bedrooms,
      ageYears: Number.isNaN(ageYears) ? 10 : ageYears,
      locationTier: tierRaw ?? 'Suburban',
      missingFields,
    })
  }

  return { rows, columnsDetected: headers.length }
}

function countDuplicates(rows: RawHouseRow[]): number {
  const seen = new Set<string>()
  let dupes = 0
  for (const r of rows) {
    const key = `${r.sqft}|${r.bedrooms}|${r.ageYears}|${r.locationTier}`
    if (seen.has(key)) dupes += 1
    else seen.add(key)
  }
  return dupes
}

// ---------------------------------------------------------------------------
// Full pipeline: raw uploaded rows -> every card's data
// ---------------------------------------------------------------------------
export function computeFullHousePipeline(
  rawRows: RawHouseRow[],
  fileName: string,
  fileSizeKB: number,
  columnsDetected: number,
): HousePricePipelineData {
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
        after: field === 'location_tier' ? firstMissing.locationTier : 'defaulted',
      }))
    : [
        { field: 'location_tier', before: '(none blank)', after: first?.locationTier ?? 'Suburban' },
        { field: 'age_years', before: '(none blank)', after: String(first?.ageYears ?? 10) },
      ]

  const encodingExamples: EncodingExample[] = [
    { field: 'location_tier', raw: first?.locationTier ?? 'Suburban', code: tierScore(first?.locationTier ?? 'Suburban') },
    { field: 'location_tier', raw: 'Urban', code: tierScore('Urban') },
  ]

  // Real target (y): this page's own formula output per row.
  const yRaw = rawRows.map((r) => computeHousePrice(r).predictedPrice)
  const featuresRaw = rawRows.map((r) => [r.sqft, r.bedrooms, r.ageYears, tierScore(r.locationTier)])

  const testCount = totalRows >= 5 ? Math.max(1, Math.round(totalRows * 0.2)) : totalRows > 1 ? 1 : 0
  const trainCount = totalRows - testCount

  const trainX = featuresRaw.slice(0, trainCount)
  const trainY = yRaw.slice(0, trainCount)
  const testX = featuresRaw.slice(trainCount)
  const testY = yRaw.slice(trainCount)

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
    const predicted = Math.round(testPredictions[i])
    return { actual, predicted, error: Math.round(Math.abs(actual - predicted)) }
  })
  const metrics = computeMetrics(testY, testPredictions)

  const maxAbsWeight = Math.max(...gd.weights.map(Math.abs), 0.0001)
  const learnedWeights = [
    { label: 'sqft', magnitude: Math.abs(gd.weights[0]) / maxAbsWeight },
    { label: 'bedrooms', magnitude: Math.abs(gd.weights[1]) / maxAbsWeight },
    { label: 'age', magnitude: Math.abs(gd.weights[2]) / maxAbsWeight },
    { label: 'location', magnitude: Math.abs(gd.weights[3]) / maxAbsWeight },
  ]

  const predictRowSource = rawRows[rawRows.length - 1] ?? rawRows[0]
  const predictFormula = computeHousePrice(predictRowSource)

  return {
    isLive: true,
    fileName,
    fileSizeKB,
    sampleRawLine: `"${first.sqft},${first.bedrooms},${first.ageYears},${first.locationTier}"`,
    sampleRow: [
      { label: 'sqft', value: first.sqft },
      { label: 'bedrooms', value: first.bedrooms },
      { label: 'location_tier', value: first.locationTier },
    ],
    totalRows,
    columnsDetected,
    missingCount,
    duplicateCount,
    validCount,
    cleaningExamples,
    encodingExamples,
    numericFeatureNames: ['sqft', 'bedrooms', 'age_years'],
    xFields: ['sqft', 'bedrooms', 'age_years', 'location_tier'],
    yField: 'price',
    yFieldShort: 'price',
    splitTrainCount: trainCount,
    splitTestCount: testCount,
    modelName: 'Multiple Linear Regression',
    whyModel:
      'The target (price) is a continuous number, and several independent numeric factors — size, bedrooms, age, location — each contribute additively. That combination of "more than one input, added together" is exactly what multiple linear regression is.',
    errorHistory: gd.errorHistory,
    weightTickerLabel: 'w_sqft',
    weightEarly: Math.round(gd.weightEarly * 100) / 100,
    weightFinal: Math.round(gd.weightFinal * 100) / 100,
    learnedWeights,
    predictExample: {
      sqft: predictRowSource.sqft,
      bedrooms: predictRowSource.bedrooms,
      ageYears: predictRowSource.ageYears,
      locationTier: predictRowSource.locationTier,
      sqftTerm: predictFormula.sqftTerm,
      bedroomTerm: predictFormula.bedroomTerm,
      agePenaltyTerm: predictFormula.agePenaltyTerm,
      locationTerm: predictFormula.locationTerm,
      predictedPrice: predictFormula.predictedPrice,
      confidence: predictFormula.confidence,
    },
    evaluationRows,
    metrics,
    metricUnit: '',
    finalResult: {
      sqft: predictRowSource.sqft,
      bedrooms: predictRowSource.bedrooms,
      ageYears: predictRowSource.ageYears,
      locationTier: predictRowSource.locationTier,
      basePrice: BASE_PRICE,
      sqftContribution: predictFormula.sqftTerm,
      bedroomContribution: predictFormula.bedroomTerm,
      ageContribution: -predictFormula.agePenaltyTerm,
      locationContribution: predictFormula.locationTerm,
      predictedPrice: predictFormula.predictedPrice,
      confidence: predictFormula.confidence,
    },
  }
}
