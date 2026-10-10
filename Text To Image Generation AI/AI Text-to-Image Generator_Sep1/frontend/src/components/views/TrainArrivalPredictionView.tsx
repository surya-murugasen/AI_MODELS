import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Bell,
  BrainCircuit,
  Calculator,
  CheckCircle2,
  Clock,
  Download,
  FileSpreadsheet,
  Gauge,
  GitFork,
  MapPin,
  Radio,
  Sparkles,
  Train,
  TrendingUp,
  Users,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { LineageCalculationModal } from '@/components/views/LineageCalculationModal'
import { cn } from '@/lib/utils'

interface StationTrain {
  trainNo: string
  name: string
  destination: string
  source: string
  platform: string
  scheduled: string
  predicted: string
  delayMin: number
  status: 'Approaching' | 'On Time' | 'Delayed'
  speed: number
  distanceKm: number
  confidence: number
  previousDelay?: number
  routeFactor?: number
  trafficLevel?: string
  weatherCondition?: string
}

const DEMO_12_TRAINS: StationTrain[] = [
  {
    trainNo: '12951',
    name: 'Mumbai Rajdhani Express',
    source: 'Mumbai Central',
    destination: 'New Delhi',
    platform: '4',
    scheduled: '06:30 PM',
    predicted: '06:42 PM',
    delayMin: 12,
    status: 'Approaching',
    speed: 68,
    distanceKm: 42,
    confidence: 94,
    previousDelay: 8,
    routeFactor: 3,
    trafficLevel: 'Moderate',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '12109',
    name: 'Panchavati Superfast',
    source: 'Mumbai Central',
    destination: 'Mumbai',
    platform: '2',
    scheduled: '06:48 PM',
    predicted: '06:47 PM',
    delayMin: 0,
    status: 'On Time',
    speed: 74,
    distanceKm: 28,
    confidence: 97,
    previousDelay: 0,
    routeFactor: 0,
    trafficLevel: 'Low',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '12267',
    name: 'Ahmedabad AC Duronto',
    source: 'Mumbai Central',
    destination: 'Ahmedabad',
    platform: '6',
    scheduled: '07:05 PM',
    predicted: '07:08 PM',
    delayMin: 3,
    status: 'Delayed',
    speed: 62,
    distanceKm: 55,
    confidence: 91,
    previousDelay: 2,
    routeFactor: 1,
    trafficLevel: 'Moderate',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '12618',
    name: 'Mangala Superfast',
    source: 'Mumbai Central',
    destination: 'Chennai',
    platform: '3',
    scheduled: '07:15 PM',
    predicted: '07:14 PM',
    delayMin: 0,
    status: 'On Time',
    speed: 80,
    distanceKm: 64,
    confidence: 96,
    previousDelay: 0,
    routeFactor: 0,
    trafficLevel: 'Low',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '12921',
    name: 'Flying Ranee Express',
    source: 'Mumbai Central',
    destination: 'Jaipur',
    platform: '5',
    scheduled: '07:30 PM',
    predicted: '07:35 PM',
    delayMin: 5,
    status: 'Delayed',
    speed: 58,
    distanceKm: 78,
    confidence: 89,
    previousDelay: 4,
    routeFactor: 1,
    trafficLevel: 'Moderate',
    weatherCondition: 'Rain',
  },
  {
    trainNo: '12009',
    name: 'Mumbai Shatabdi Express',
    source: 'Mumbai Central',
    destination: 'Ahmedabad',
    platform: '1',
    scheduled: '07:45 PM',
    predicted: '07:44 PM',
    delayMin: 0,
    status: 'On Time',
    speed: 85,
    distanceKm: 34,
    confidence: 98,
    previousDelay: 0,
    routeFactor: 0,
    trafficLevel: 'Low',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '22221',
    name: 'NZM Rajdhani Express',
    source: 'Mumbai Central',
    destination: 'Hazrat Nizamuddin',
    platform: '4',
    scheduled: '08:10 PM',
    predicted: '08:16 PM',
    delayMin: 6,
    status: 'Delayed',
    speed: 72,
    distanceKm: 50,
    confidence: 93,
    previousDelay: 5,
    routeFactor: 1,
    trafficLevel: 'Moderate',
    weatherCondition: 'Fog',
  },
  {
    trainNo: '12903',
    name: 'Golden Temple Mail',
    source: 'Mumbai Central',
    destination: 'Amritsar',
    platform: '2',
    scheduled: '08:35 PM',
    predicted: '08:51 PM',
    delayMin: 16,
    status: 'Delayed',
    speed: 64,
    distanceKm: 68,
    confidence: 86,
    previousDelay: 9,
    routeFactor: 3,
    trafficLevel: 'High',
    weatherCondition: 'Rain',
  },
  {
    trainNo: '12925',
    name: 'Paschim Superfast Express',
    source: 'Mumbai Central',
    destination: 'Amritsar',
    platform: '5',
    scheduled: '09:00 PM',
    predicted: '09:04 PM',
    delayMin: 4,
    status: 'Delayed',
    speed: 70,
    distanceKm: 82,
    confidence: 92,
    previousDelay: 3,
    routeFactor: 1,
    trafficLevel: 'Moderate',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '19015',
    name: 'Saurashtra Express',
    source: 'Mumbai Central',
    destination: 'Porbandar',
    platform: '3',
    scheduled: '09:30 PM',
    predicted: '09:48 PM',
    delayMin: 18,
    status: 'Delayed',
    speed: 55,
    distanceKm: 95,
    confidence: 84,
    previousDelay: 12,
    routeFactor: 3,
    trafficLevel: 'High',
    weatherCondition: 'Rain',
  },
  {
    trainNo: '12953',
    name: 'August Kranti Tejas Rajdhani',
    source: 'Mumbai Central',
    destination: 'Hazrat Nizamuddin',
    platform: '4',
    scheduled: '10:00 PM',
    predicted: '10:00 PM',
    delayMin: 0,
    status: 'On Time',
    speed: 88,
    distanceKm: 40,
    confidence: 99,
    previousDelay: 0,
    routeFactor: 0,
    trafficLevel: 'Low',
    weatherCondition: 'Clear',
  },
  {
    trainNo: '12216',
    name: 'Delhi Sarai Rohilla Garib Rath',
    source: 'Mumbai Central',
    destination: 'Delhi',
    platform: '6',
    scheduled: '10:30 PM',
    predicted: '10:37 PM',
    delayMin: 7,
    status: 'Delayed',
    speed: 65,
    distanceKm: 110,
    confidence: 91,
    previousDelay: 6,
    routeFactor: 1,
    trafficLevel: 'Moderate',
    weatherCondition: 'Clear',
  },
]

const INITIAL_STATION_TRAINS: StationTrain[] = DEMO_12_TRAINS.slice(0, 5)

const DELAY_TREND_POINTS = [
  { time: '00:00', delay: 4 },
  { time: '02:00', delay: 3 },
  { time: '04:00', delay: 8 },
  { time: '06:00', delay: 6 },
  { time: '08:00', delay: 14 },
  { time: '10:00', delay: 10 },
  { time: '12:00', delay: 18 },
  { time: '14:00', delay: 24 },
  { time: '16:00', delay: 26 },
  { time: '18:00', delay: 16 },
  { time: '20:00', delay: 8 },
  { time: '22:00', delay: 9 },
  { time: '24:00', delay: 4 },
]

/** The app's actual (hand-written, not trained) delay formula — extracted so
 * `HowItWorksView.tsx` can run the exact same math on user-uploaded data
 * instead of maintaining a second, potentially-drifting copy of it. */
export interface DelayFormulaInput {
  speed: number
  previousDelay: number
  trafficLevel: string
  weatherCondition: string
  scheduled: string
}
export interface DelayFormulaResult {
  speedPenalty: number
  routeFactor: number
  weatherDelay: number
  delayMin: number
  predicted: string
  confidence: number
  status: 'Delayed' | 'On Time'
}
export function computeDelayPrediction({ speed, previousDelay, trafficLevel, weatherCondition, scheduled }: DelayFormulaInput): DelayFormulaResult {
  const speedPenalty = speed < 70 ? (70 - speed) * 0.15 : 0
  const routeFactor = trafficLevel.toLowerCase().includes('high') ? 6 : trafficLevel.toLowerCase().includes('mod') ? 3 : 0
  const weatherDelay = weatherCondition.toLowerCase().includes('rain') ? 4 : weatherCondition.toLowerCase().includes('fog') ? 6 : 0
  const delayMin = Math.max(0, Math.round(previousDelay + speedPenalty + routeFactor + weatherDelay))
  const predicted = addMinutesToTimeString(scheduled, delayMin)
  const confidence = Math.min(99, Math.max(82, Math.round(98 - delayMin * 0.45)))
  const status: 'Delayed' | 'On Time' = delayMin > 2 ? 'Delayed' : 'On Time'
  return { speedPenalty, routeFactor, weatherDelay, delayMin, predicted, confidence, status }
}

export function addMinutesToTimeString(timeStr: string, minutesToAdd: number): string {
  const match = timeStr.trim().match(/^(\d{1,2}):(\d{2})(?:\s*([AP]M))?$/i)
  if (!match) return timeStr
  let hours = parseInt(match[1], 10)
  const mins = parseInt(match[2], 10)
  const ampm = match[3]?.toUpperCase()

  if (ampm === 'PM' && hours < 12) hours += 12
  if (ampm === 'AM' && hours === 12) hours = 0

  const totalMinutes = hours * 60 + mins + minutesToAdd
  const newHours24 = Math.floor(((totalMinutes % 1440) + 1440) % 1440 / 60)
  const newMins = Math.floor(((totalMinutes % 1440) + 1440) % 60)

  const finalHours = newHours24 % 12 || 12
  const finalAmPm = newHours24 >= 12 ? 'PM' : 'AM'
  return `${String(finalHours).padStart(2, '0')}:${String(newMins).padStart(2, '0')} ${finalAmPm}`
}

function parseCSVToTrains(csvText: string): StationTrain[] {
  const lines = csvText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)
  if (lines.length <= 1) return []

  const headers = lines[0].toLowerCase().split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''))
  const getCol = (row: string[], aliases: string[]): string => {
    for (const a of aliases) {
      const idx = headers.findIndex((h) => h.includes(a))
      if (idx !== -1 && row[idx] !== undefined) return row[idx].trim().replace(/^["']|["']$/g, '')
    }
    return ''
  }

  const result: StationTrain[] = []
  for (let i = 1; i < lines.length; i++) {
    const rawRow = lines[i].split(',').map((c) => c.trim())
    if (rawRow.length < 2) continue

    const trainNo = getCol(rawRow, ['no', 'number', 'train']) || `TR-${1000 + i}`
    const name = getCol(rawRow, ['name', 'title']) || `Express ${trainNo}`
    const source = getCol(rawRow, ['origin', 'source', 'from']) || 'Mumbai Central'
    const destination = getCol(rawRow, ['destination', 'to']) || 'New Delhi'
    const platform = getCol(rawRow, ['platform', 'plt']) || String((i % 6) + 1)
    const scheduled = getCol(rawRow, ['scheduled', 'time', 'sched']) || '07:00 PM'
    const speed = parseFloat(getCol(rawRow, ['speed', 'velocity'])) || 68
    const distanceKm = parseFloat(getCol(rawRow, ['distance', 'dist', 'km'])) || 42
    const previousDelay = parseFloat(getCol(rawRow, ['prev', 'hold', 'delay'])) || 0
    const trafficLevel = getCol(rawRow, ['traffic', 'congestion', 'route']) || 'Moderate'
    const weatherCondition = getCol(rawRow, ['weather', 'climate']) || 'Clear'

    // Multi-variate Linear Regression Prediction Engine:
    // Delay = prevDelay + speedPenalty + congestionFactor + weatherFactor
    const { routeFactor, delayMin, predicted, confidence, status } = computeDelayPrediction({
      speed,
      previousDelay,
      trafficLevel,
      weatherCondition,
      scheduled,
    })

    result.push({
      trainNo,
      name,
      source,
      destination,
      platform,
      scheduled,
      predicted,
      delayMin,
      status,
      speed,
      distanceKm,
      confidence,
      previousDelay,
      routeFactor,
      trafficLevel,
      weatherCondition,
    })
  }

  return result
}

export function TrainArrivalPredictionView({ onSwitchToOverview }: { onSwitchToOverview?: () => void }) {
  const [trains, setTrains] = useState<StationTrain[]>(INITIAL_STATION_TRAINS)
  const [selectedTrain, setSelectedTrain] = useState<StationTrain>(INITIAL_STATION_TRAINS[0])
  const [clockTime, setClockTime] = useState<string>('06:30:45 PM')
  const [hoveredTrend, setHoveredTrend] = useState<{ time: string; delay: number } | null>(null)
  const [speedJitter, setSpeedJitter] = useState(0)
  const [isLineageModalOpen, setIsLineageModalOpen] = useState(false)
  const [importNotice, setImportNotice] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Live ticking clock in IST format
  useEffect(() => {
    function updateClock() {
      const now = new Date()
      const hours = String(now.getHours() % 12 || 12).padStart(2, '0')
      const minutes = String(now.getMinutes()).padStart(2, '0')
      const seconds = String(now.getSeconds()).padStart(2, '0')
      const ampm = now.getHours() >= 12 ? 'PM' : 'AM'
      setClockTime(`${hours}:${minutes}:${seconds} ${ampm}`)
    }
    updateClock()
    const interval = setInterval(updateClock, 1000)
    return () => clearInterval(interval)
  }, [])

  // Minor telemetry jitter for realistic live system feel
  useEffect(() => {
    const interval = setInterval(() => {
      setSpeedJitter((Math.random() - 0.5) * 2)
    }, 2500)
    return () => clearInterval(interval)
  }, [])

  const currentSpeed = Math.max(0, Math.round((selectedTrain.speed || 68) + speedJitter))

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const content = event.target?.result as string
      if (!content) return

      try {
        const parsed = parseCSVToTrains(content)
        if (parsed.length > 0) {
          setTrains(parsed)
          setSelectedTrain(parsed[0])
          setImportNotice(`Successfully loaded ${parsed.length} trains from ${file.name}! Predictions calculated.`)
          window.setTimeout(() => setImportNotice(null), 5000)
        } else {
          setImportNotice('File loaded, but no valid train rows were detected. Check template format.')
          window.setTimeout(() => setImportNotice(null), 5000)
        }
      } catch (err) {
        console.error(err)
        setImportNotice('Failed to parse Excel/CSV sheet. Please use the template format.')
        window.setTimeout(() => setImportNotice(null), 5000)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  function handleDownloadExcel() {
    // Download the pre-generated binary XLSX file
    const a = document.createElement('a')
    a.href = '/demo_railway_train_data.xlsx'
    a.download = 'demo_railway_train_data.xlsx'
    a.click()
  }

  function handleDownloadCSV() {
    const a = document.createElement('a')
    a.href = '/demo_railway_train_data.csv'
    a.download = 'demo_railway_train_data.csv'
    a.click()
  }

  function handleLoadDemoData() {
    setTrains(DEMO_12_TRAINS)
    setSelectedTrain(DEMO_12_TRAINS[0])
    setImportNotice('Loaded full 12-train demo dataset! Predictions, delays, and telemetry recalculated.')
    window.setTimeout(() => setImportNotice(null), 5000)
  }

  return (
    <div className="flex-1 overflow-y-auto bg-page text-ink-primary p-4 md:p-6 font-sans">
      {/* Hidden File Input for Excel/CSV */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.xlsx,.xls,text/csv,application/vnd.ms-excel"
        className="hidden"
        onChange={handleFileUpload}
      />

      {/* Top Tab Switcher — matches Image Accuracy AI view style */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b pb-3">
        <div className="flex items-center gap-2">
          <div className="flex items-center rounded-xl border bg-surface p-1 shadow-sm">
            {/* Image Accuracy AI — inactive, click to switch back */}
            {onSwitchToOverview && (
              <button
                type="button"
                onClick={onSwitchToOverview}
                className="flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold text-ink-secondary hover:text-ink-primary hover:bg-black/[0.04] dark:hover:bg-white/[0.06] transition-all cursor-pointer"
              >
                <BrainCircuit size={14} />
                Image Accuracy AI
              </button>
            )}
            {/* Train Arrival Prediction — active */}
            <div className="flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold bg-purple-600 text-white shadow">
              <Train size={14} />
              <span>Train Arrival Prediction</span>
              <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            </div>
          </div>
        </div>


        {/* Live Indicator, Lineage Button, Excel Tools, Notifications & Clock */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Load Demo Data Button */}
          <Button
            size="sm"
            onClick={handleLoadDemoData}
            className="bg-blue-500/10 hover:bg-blue-500/20 dark:bg-blue-600/30 dark:hover:bg-blue-600/40 border border-blue-500/30 dark:border-blue-500/50 text-blue-700 dark:text-blue-200 text-xs font-bold h-8 cursor-pointer flex items-center gap-1.5 shadow-sm"
            title="Load 12 Demo Trains Dataset into System"
          >
            <Sparkles size={14} className="text-blue-600 dark:text-blue-400" />
            <span>Load Demo Data</span>
          </Button>

          {/* Excel Upload Button */}
          <Button
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            className="bg-emerald-500/10 hover:bg-emerald-500/20 dark:bg-emerald-600/30 dark:hover:bg-emerald-600/40 border border-emerald-500/30 dark:border-emerald-500/50 text-emerald-700 dark:text-emerald-200 text-xs font-bold h-8 cursor-pointer flex items-center gap-1.5 shadow-sm"
          >
            <FileSpreadsheet size={14} className="text-emerald-600 dark:text-emerald-400" />
            <span>Upload Excel / CSV</span>
          </Button>

          {/* Lineage & Calculation Modal Button */}
          <Button
            size="sm"
            onClick={() => setIsLineageModalOpen(true)}
            className="bg-purple-500/10 hover:bg-purple-500/20 dark:bg-purple-600/30 dark:hover:bg-purple-600/40 border border-purple-500/30 dark:border-purple-500/50 text-purple-700 dark:text-purple-200 text-xs font-bold h-8 cursor-pointer flex items-center gap-1.5 shadow-sm"
          >
            <Calculator size={14} className="text-purple-600 dark:text-purple-400" />
            <span>View AI Lineage</span>
          </Button>

          {/* Download Demo Excel .xlsx */}
          <button
            type="button"
            onClick={handleDownloadExcel}
            className="flex h-8 items-center gap-1.5 rounded-lg border bg-surface px-2.5 text-xs font-medium text-ink-secondary hover:bg-black/[0.03] hover:text-ink-primary dark:hover:bg-white/[0.05] transition-colors cursor-pointer"
            title="Download demo_railway_train_data.xlsx"
          >
            <Download size={13} className="text-emerald-600 dark:text-emerald-400" />
            <span>Download .XLSX</span>
          </button>

          {/* Download Demo CSV */}
          <button
            type="button"
            onClick={handleDownloadCSV}
            className="flex h-8 items-center gap-1.5 rounded-lg border bg-surface px-2.5 text-xs font-medium text-ink-secondary hover:bg-black/[0.03] hover:text-ink-primary dark:hover:bg-white/[0.05] transition-colors cursor-pointer"
            title="Download demo_railway_train_data.csv"
          >
            <Download size={13} className="text-cyan-600 dark:text-cyan-400" />
            <span>Download .CSV</span>
          </button>

          {/* Live Indicator */}
          <div className="flex items-center gap-1.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 text-xs font-bold text-emerald-600 dark:text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.2)]">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
            <span className="h-2 w-2 -ml-3.5 rounded-full bg-emerald-500" />
            <span>LIVE</span>
          </div>

          <div className="relative flex h-8 w-8 items-center justify-center rounded-lg border bg-surface text-ink-secondary">
            <Bell size={15} />
            <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-rose-500 text-[9px] font-bold text-white shadow-sm">
              3
            </span>
          </div>

          <div className="flex items-center gap-1.5 rounded-lg border bg-surface px-3 py-1 text-xs font-mono font-bold tracking-wider text-ink-primary shadow-inner">
            <Clock size={13} className="text-ink-muted" />
            <span>{clockTime}</span>
          </div>
        </div>
      </div>

      {/* Import Notification Banner */}
      {importNotice && (
        <div className="mb-4 flex items-center justify-between rounded-xl border border-emerald-500/40 bg-emerald-500/15 p-3 text-xs font-bold text-emerald-700 dark:text-emerald-300 shadow-lg animate-in slide-in-from-top duration-300">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={16} className="text-emerald-600 dark:text-emerald-400" />
            <span>{importNotice}</span>
          </div>
          <button
            type="button"
            onClick={() => setImportNotice(null)}
            className="text-emerald-600 dark:text-emerald-400 hover:text-ink-primary"
          >
            ✕
          </button>
        </div>
      )}

      {/* Page Title & Station Pill Bar */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-black text-ink-primary tracking-tight md:text-2xl flex items-center gap-2">
              Train Arrival Prediction
            </h1>
            <span className="rounded-full bg-purple-500/15 px-2.5 py-0.5 text-[11px] font-semibold text-purple-700 dark:text-purple-300 border border-purple-500/30">
              AI Neural Telemetry
            </span>
          </div>
          <p className="text-xs text-ink-secondary mt-0.5">
            Real-time arrival estimation powered by predictive AI &amp; signal track telematics
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 rounded-lg border bg-surface px-3 py-1.5 text-xs font-medium text-ink-primary">
            <Train size={14} className="text-blue-500" />
            <span>{selectedTrain.source}</span>
          </div>
          <div className="flex items-center gap-1.5 rounded-lg border bg-surface px-3 py-1.5 text-xs font-medium text-ink-primary">
            <span>⛅ 29°C</span>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 1. CINEMATIC RAILWAY PLATFORM HERO SCENE (REALISTIC MOVING TRAIN BG)      */}
      {/* ========================================================================= */}
      <div className="relative mb-6 overflow-hidden rounded-2xl md:rounded-3xl border border-slate-200/80 dark:border-white/15 bg-slate-100 dark:bg-slate-950 shadow-xl min-h-[390px] md:min-h-[430px] flex flex-col justify-between p-4 md:p-6 transition-all duration-300">
        {/* Light Mode Moving Indian Railway Background Image */}
        <div
          className="absolute -inset-8 bg-cover bg-center dark:hidden opacity-95 animate-train-motion"
          style={{
            backgroundImage: `url('/train-daylight-bg.jpg')`,
            backgroundPosition: 'center 42%',
          }}
        />

        {/* Dark Mode Moving Railway Station Background Image */}
        <div
          className="absolute -inset-8 bg-cover bg-center hidden dark:block opacity-90 animate-train-motion"
          style={{
            backgroundImage: `url('/train-station-bg.jpg')`,
            backgroundPosition: 'center 40%',
          }}
        />

        {/* Light Mode Soft Translucent Overlays & Gradients (Ensures Crisp Text Contrast) */}
        <div className="absolute inset-0 bg-white/20 dark:hidden backdrop-blur-[0.5px]" />
        <div className="absolute inset-0 bg-gradient-to-t from-white/70 via-transparent to-white/40 dark:hidden" />
        <div className="absolute inset-0 bg-gradient-to-r from-white/75 via-transparent to-white/75 dark:hidden" />

        {/* Dark Mode Atmospheric Vignette */}
        <div className="absolute inset-0 bg-gradient-to-t from-slate-950/95 via-slate-950/30 to-slate-950/20 hidden dark:block" />
        <div className="absolute inset-0 bg-gradient-to-r from-slate-950/80 via-transparent to-slate-950/80 hidden dark:block" />
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950/60 via-transparent to-transparent hidden dark:block" />

        {/* Dynamic Light Sheen & Flare simulation */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-transparent via-amber-200/15 dark:via-blue-500/10 to-transparent animate-track-sheen" />

        {/* Platform Signage Floating in center-top & Live Animation Badge */}
        <div className="relative z-10 flex items-center justify-between">
          {/* Platform 4 Floating Signage */}
          <div className="flex items-center gap-1.5 rounded-xl border border-blue-500/40 bg-blue-600 px-3.5 py-1 text-xs font-black tracking-widest text-white shadow-md">
            <span>PLATFORM</span>
            <span className="rounded bg-white px-1.5 text-blue-900 font-extrabold">{selectedTrain.platform}</span>
          </div>

          {/* Telemetry Live Movement Badge */}
          <div className="flex items-center gap-1.5 rounded-full bg-white/80 dark:bg-black/50 border border-slate-200 dark:border-white/20 px-2.5 py-1 text-[10px] font-bold text-slate-700 dark:text-slate-200 backdrop-blur-md shadow-sm">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
            <span className="h-2 w-2 -ml-3.5 rounded-full bg-emerald-500" />
            <span>Telemetry Active</span>
          </div>
        </div>

        {/* Hero Overlay Cards (Left: Predicted Arrival, Right: Train Details) */}
        <div className="relative z-10 grid grid-cols-1 gap-4 md:grid-cols-12 items-end mt-6">
          {/* Left Floating Frosted Glass Card — Predicted Arrival */}
          <div className="md:col-span-6 lg:col-span-5 rounded-2xl md:rounded-3xl border border-slate-200/90 dark:border-white/20 bg-white/95 dark:bg-slate-950/85 p-5 md:p-6 backdrop-blur-2xl shadow-xl shadow-slate-300/40 dark:shadow-2xl transition-all">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                PREDICTED ARRIVAL
              </span>
              <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-500/15 px-2.5 py-0.5 rounded-full border border-emerald-200 dark:border-emerald-500/30">
                <Radio size={11} className="animate-pulse text-emerald-600 dark:text-emerald-400" />
                Live Sensor Feed
              </span>
            </div>

            <div className="mt-2.5 flex items-baseline gap-3">
              <span className="text-4xl md:text-5xl font-black tracking-tight text-slate-900 dark:text-white font-mono">
                {selectedTrain.predicted}
              </span>
            </div>

            <div className="mt-1 flex items-center gap-2">
              {selectedTrain.delayMin > 0 ? (
                <span className="text-base font-bold text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                  <AlertTriangle size={16} />
                  {selectedTrain.delayMin} min delay
                </span>
              ) : (
                <span className="text-base font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <CheckCircle2 size={16} />
                  On Time
                </span>
              )}
            </div>

            <div className="mt-4 grid grid-cols-3 gap-2 border-t border-slate-100 dark:border-white/10 pt-3 text-left">
              <div>
                <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-slate-400">Scheduled</span>
                <p className="text-xs font-bold text-slate-900 dark:text-white font-mono mt-0.5">{selectedTrain.scheduled}</p>
              </div>
              <div>
                <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-slate-400">Predicted</span>
                <p className="text-xs font-bold text-slate-900 dark:text-white font-mono mt-0.5">{selectedTrain.predicted}</p>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-slate-400">Confidence</span>
                  <p className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">{selectedTrain.confidence}%</p>
                </div>
                {/* Confidence Circular Gauge */}
                <div className="relative flex h-8 w-8 items-center justify-center">
                  <svg className="h-8 w-8 -rotate-90 transform" viewBox="0 0 36 36">
                    <path
                      className="text-slate-200 dark:text-white/10"
                      strokeWidth="3.5"
                      stroke="currentColor"
                      fill="none"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path
                      className="text-emerald-500 drop-shadow-[0_0_4px_rgba(16,185,129,0.5)]"
                      strokeDasharray={`${selectedTrain.confidence}, 100`}
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      stroke="currentColor"
                      fill="none"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          <div className="hidden md:block md:col-span-1 lg:col-span-2" />

          {/* Right Floating Frosted Glass Card — Train Telemetry */}
          <div className="md:col-span-5 rounded-2xl md:rounded-3xl border border-slate-200/90 dark:border-white/20 bg-white/95 dark:bg-slate-950/85 p-4 md:p-5 backdrop-blur-2xl shadow-xl shadow-slate-300/40 dark:shadow-2xl transition-all">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-white/10 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-50 dark:bg-purple-600/30 border border-purple-200 dark:border-purple-500/40 text-purple-700 dark:text-purple-300">
                  <Train size={18} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white leading-tight">
                    Train {selectedTrain.trainNo}
                  </h3>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    {selectedTrain.source} → {selectedTrain.destination}
                  </p>
                </div>
              </div>
              <span className="rounded-full bg-purple-50 dark:bg-purple-500/20 px-2.5 py-0.5 text-[11px] font-bold text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-500/30">
                {selectedTrain.status}
              </span>
            </div>

            <div className="mt-3 flex flex-col gap-2.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400 flex items-center gap-1.5 font-medium">
                  <MapPin size={13} className="text-blue-500" /> Current Location
                </span>
                <span className="font-semibold text-slate-900 dark:text-white">{selectedTrain.distanceKm} km from station</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400 flex items-center gap-1.5 font-medium">
                  <Gauge size={13} className="text-cyan-500" /> Current Speed
                </span>
                <span className="font-bold text-cyan-600 dark:text-cyan-300 font-mono">{currentSpeed} km/h</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400 flex items-center gap-1.5 font-medium">
                  <Activity size={13} className="text-purple-500" /> Distance Remaining
                </span>
                <span className="font-semibold text-slate-900 dark:text-white font-mono">{selectedTrain.distanceKm} km</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400 flex items-center gap-1.5 font-medium">
                  <Clock size={13} className="text-amber-500" /> Estimated Arrival
                </span>
                <span className="font-semibold text-slate-900 dark:text-white font-mono">{selectedTrain.predicted}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400 flex items-center gap-1.5 font-medium">
                  <Train size={13} className="text-indigo-500" /> Platform
                </span>
                <span className="rounded-md bg-blue-50 dark:bg-blue-600/40 px-2 py-0.5 font-bold text-blue-700 dark:text-blue-200 border border-blue-200 dark:border-blue-500/30">
                  Platform {selectedTrain.platform}
                </span>
              </div>

              <div className="flex items-center justify-between pt-1 border-t border-slate-100 dark:border-white/10">
                <span className="text-slate-600 dark:text-slate-400 font-medium">Status</span>
                <span className="font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                  {selectedTrain.status}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* 2. MIDDLE ROW: JOURNEY PROGRESS & WHY PREDICTED (PREDICTION FACTORS)      */}
      {/* ========================================================================= */}
      <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Journey Progress Timeline (6 cols) */}
        <Card className="lg:col-span-6 border-slate-200/80 dark:border-border">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-bold text-ink-primary flex items-center gap-2">
                <Activity size={16} className="text-purple-600 dark:text-purple-400" />
                Journey Progress
              </CardTitle>
              <span className="text-[11px] font-mono text-ink-muted">
                Train {selectedTrain.trainNo}
              </span>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-5">
            {/* Animated Track Line with Nodes */}
            <div className="relative pt-6 pb-2 px-3">
              {/* Floating train indicator */}
              <div
                className="absolute -top-1 transition-all duration-1000 flex flex-col items-center"
                style={{ left: '22%' }}
              >
                <div className="flex items-center gap-1 rounded-full bg-emerald-50 dark:bg-emerald-500/15 border border-emerald-300/80 dark:border-emerald-500/30 px-2.5 py-0.5 text-[10px] font-bold text-emerald-700 dark:text-emerald-300 shadow-sm whitespace-nowrap">
                  <Train size={11} className="text-emerald-600 dark:text-emerald-400 animate-pulse" />
                  <span>{selectedTrain.distanceKm} km away</span>
                </div>
                <div className="h-2 w-0.5 bg-emerald-500" />
              </div>

              {/* Progress Line */}
              <div className="relative h-2.5 w-full rounded-full bg-slate-200 dark:bg-white/10 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 via-purple-500 to-blue-500 rounded-full shadow-[0_0_8px_rgba(168,85,247,0.4)]"
                  style={{ width: '68%' }}
                />
              </div>

              {/* Track Station Stop Nodes */}
              <div className="mt-3 flex items-start justify-between text-xs">
                <div className="flex flex-col items-start">
                  <div className="flex h-3.5 w-3.5 items-center justify-center rounded-full bg-emerald-500 ring-2 ring-emerald-500/20" />
                  <span className="mt-1 text-[11px] font-bold text-ink-primary">Origin</span>
                  <span className="text-[10px] text-ink-muted">{selectedTrain.source}</span>
                </div>

                <div className="flex flex-col items-center text-center">
                  <div className="flex h-4 w-4 items-center justify-center rounded-full bg-blue-600 text-[9px] font-bold text-white ring-4 ring-blue-500/20">
                    {selectedTrain.platform}
                  </div>
                  <span className="mt-1 text-[11px] font-bold text-ink-primary">
                    {selectedTrain.source}
                  </span>
                  <span className="text-[10px] text-blue-600 dark:text-blue-400 font-bold">
                    Platform {selectedTrain.platform}
                  </span>
                </div>

                <div className="flex flex-col items-end text-right">
                  <div className="flex h-3.5 w-3.5 items-center justify-center rounded-full bg-slate-400 dark:bg-slate-600 ring-2 ring-slate-400/20" />
                  <span className="mt-1 text-[11px] font-bold text-ink-primary">
                    {selectedTrain.destination}
                  </span>
                  <span className="text-[10px] text-ink-muted">Final Destination</span>
                </div>
              </div>
            </div>

            {/* 4 Stat Boxes below progress */}
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4 pt-2 border-t">
              <div className="rounded-xl border bg-slate-50/80 dark:bg-page/60 p-2.5 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-ink-muted">Distance Covered</span>
                <p className="text-xs font-bold text-ink-primary font-mono mt-0.5">1,128 km</p>
              </div>
              <div className="rounded-xl border bg-slate-50/80 dark:bg-page/60 p-2.5 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-ink-muted">Total Distance</span>
                <p className="text-xs font-bold text-ink-primary font-mono mt-0.5">1,170 km</p>
              </div>
              <div className="rounded-xl border bg-slate-50/80 dark:bg-page/60 p-2.5 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-ink-muted">Time Elapsed</span>
                <p className="text-xs font-bold text-ink-primary font-mono mt-0.5">16h 35m</p>
              </div>
              <div className="rounded-xl border bg-slate-50/80 dark:bg-page/60 p-2.5 text-center">
                <span className="text-[10px] uppercase font-semibold text-slate-500 dark:text-ink-muted">Avg. Speed</span>
                <p className="text-xs font-bold text-ink-primary font-mono mt-0.5">{currentSpeed} km/h</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Why is it predicted at 06:42 PM? (Prediction Factors) (6 cols) */}
        <Card className="lg:col-span-6 border-slate-200/80 dark:border-border">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-bold text-ink-primary flex items-center gap-2">
                <Sparkles size={16} className="text-purple-600 dark:text-purple-400" />
                Why is it predicted at {selectedTrain.predicted}?
              </CardTitle>
              <span className="text-[10px] font-bold text-purple-700 dark:text-purple-300 bg-purple-50 dark:bg-purple-500/15 px-2.5 py-0.5 rounded-full border border-purple-200 dark:border-purple-500/30">
                5 ML Feature Weights
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-5">
              {/* Feature 1: Current Speed */}
              <div className="flex flex-col items-center justify-between rounded-xl border border-cyan-200 dark:border-cyan-500/20 bg-cyan-50/70 dark:bg-cyan-500/5 p-3 text-center transition-all hover:shadow-sm">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-100 dark:bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 mb-1">
                  <Gauge size={16} />
                </div>
                <span className="text-[10px] font-semibold text-slate-600 dark:text-ink-muted">Current Speed</span>
                <span className="text-xs font-black text-cyan-700 dark:text-cyan-300 font-mono mt-0.5">{currentSpeed} km/h</span>
                <span className="mt-1 text-[9px] font-bold text-cyan-800 dark:text-cyan-400">High Impact</span>
              </div>

              {/* Feature 2: Previous Delay */}
              <div className="flex flex-col items-center justify-between rounded-xl border border-amber-200 dark:border-amber-500/20 bg-amber-50/70 dark:bg-amber-500/5 p-3 text-center transition-all hover:shadow-sm">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-100 dark:bg-amber-500/15 text-amber-700 dark:text-amber-300 mb-1">
                  <Clock size={16} />
                </div>
                <span className="text-[10px] font-semibold text-slate-600 dark:text-ink-muted">Previous Delay</span>
                <span className="text-xs font-black text-amber-700 dark:text-amber-400 font-mono mt-0.5">+{selectedTrain.previousDelay ?? 8} min</span>
                <span className="mt-1 text-[9px] font-bold text-amber-800 dark:text-amber-400">Medium Impact</span>
              </div>

              {/* Feature 3: Route Pattern */}
              <div className="flex flex-col items-center justify-between rounded-xl border border-orange-200 dark:border-orange-500/20 bg-orange-50/70 dark:bg-orange-500/5 p-3 text-center transition-all hover:shadow-sm">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-100 dark:bg-orange-500/15 text-orange-700 dark:text-orange-300 mb-1">
                  <GitFork size={16} />
                </div>
                <span className="text-[10px] font-semibold text-slate-600 dark:text-ink-muted">Route Pattern</span>
                <span className="text-xs font-black text-orange-700 dark:text-orange-300 font-mono mt-0.5">+{selectedTrain.routeFactor ?? 3} min</span>
                <span className="mt-1 text-[9px] font-bold text-orange-800 dark:text-orange-300">Medium Impact</span>
              </div>

              {/* Feature 4: Station Traffic */}
              <div className="flex flex-col items-center justify-between rounded-xl border border-emerald-200 dark:border-emerald-500/20 bg-emerald-50/70 dark:bg-emerald-500/5 p-3 text-center transition-all hover:shadow-sm">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 mb-1">
                  <Users size={16} />
                </div>
                <span className="text-[10px] font-semibold text-slate-600 dark:text-ink-muted">Station Traffic</span>
                <span className="text-xs font-black text-emerald-700 dark:text-emerald-400 mt-0.5">{selectedTrain.trafficLevel ?? 'Moderate'}</span>
                <span className="mt-1 text-[9px] font-bold text-emerald-800 dark:text-emerald-400">Low Impact</span>
              </div>

              {/* Feature 5: Historical Pattern */}
              <div className="flex flex-col items-center justify-between rounded-xl border border-purple-200 dark:border-purple-500/20 bg-purple-50/70 dark:bg-purple-500/5 p-3 text-center transition-all hover:shadow-sm">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-500/15 text-purple-700 dark:text-purple-300 mb-1">
                  <TrendingUp size={16} />
                </div>
                <span className="text-[10px] font-semibold text-slate-600 dark:text-ink-muted">Historical Pattern</span>
                <span className="text-xs font-black text-purple-700 dark:text-purple-300 mt-0.5">Normal</span>
                <span className="mt-1 text-[9px] font-bold text-purple-800 dark:text-purple-300">Low Impact</span>
              </div>
            </div>

            <div className="mt-3.5 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200/80 dark:border-white/10 bg-slate-50 dark:bg-page/80 p-3 text-xs text-ink-secondary">
              <div className="flex items-start gap-2 flex-1 min-w-[240px]">
                <Sparkles size={14} className="flex-none text-purple-600 dark:text-purple-400 mt-0.5" />
                <p className="leading-snug text-[11px] text-slate-700 dark:text-ink-secondary">
                  <span className="font-bold text-purple-800 dark:text-purple-300">AI Explainability: </span>
                  Delay is accumulated from upstream hold (+{selectedTrain.previousDelay ?? 8}m) and corridor congestion (+{selectedTrain.routeFactor ?? 3}m), offset by {currentSpeed} km/h speed.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsLineageModalOpen(true)}
                className="flex items-center gap-1.5 rounded-lg bg-purple-600 dark:bg-purple-500/20 hover:bg-purple-700 dark:hover:bg-purple-500/30 text-white dark:text-purple-200 px-3 py-1.5 text-[11px] font-bold shadow-sm transition-colors cursor-pointer"
              >
                <Calculator size={13} />
                <span>View Mathematical Lineage →</span>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ========================================================================= */}
      {/* 3. BOTTOM ROW: LIVE STATION BOARD & DELAY TREND (TODAY)                   */}
      {/* ========================================================================= */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Live Station Board (7 cols) */}
        <Card className="lg:col-span-7">
          <CardHeader className="pb-3 flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle className="text-sm font-bold text-ink-primary flex items-center gap-2">
                <Train size={16} className="text-blue-500" />
                Live Station Board
              </CardTitle>
              <p className="text-xs text-ink-secondary mt-0.5">
                Click any train to load real-time arrival telemetry
              </p>
            </div>
            <span className="text-[10px] font-mono text-ink-muted bg-page border px-2 py-1 rounded-md">
              {trains.length} Trains Tracked
            </span>
          </CardHeader>
          <CardContent className="p-0 overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-y bg-page/60 text-[11px] font-semibold text-ink-muted">
                  <th className="py-2.5 px-4">Train No.</th>
                  <th className="py-2.5 px-3">Destination</th>
                  <th className="py-2.5 px-3">Platform</th>
                  <th className="py-2.5 px-3">Scheduled</th>
                  <th className="py-2.5 px-3">Predicted</th>
                  <th className="py-2.5 px-4 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {trains.map((train) => {
                  const isSelected = selectedTrain.trainNo === train.trainNo
                  const isDelayed = train.delayMin > 0
                  return (
                    <tr
                      key={train.trainNo}
                      onClick={() => setSelectedTrain(train)}
                      className={cn(
                        'transition-colors cursor-pointer',
                        isSelected
                          ? 'bg-purple-500/10 font-semibold text-ink-primary'
                          : 'hover:bg-black/[0.02] dark:hover:bg-white/[0.03] text-ink-secondary',
                      )}
                    >
                      <td className="py-2.5 px-4 font-mono font-bold text-ink-primary flex items-center gap-1.5">
                        {isSelected && <span className="h-1.5 w-1.5 rounded-full bg-purple-500" />}
                        {train.trainNo}
                      </td>
                      <td className="py-2.5 px-3">{train.destination}</td>
                      <td className="py-2.5 px-3">
                        <span className="rounded bg-black/5 dark:bg-white/10 px-1.5 py-0.5 font-mono text-[11px] text-ink-primary">
                          {train.platform}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-ink-muted">{train.scheduled}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-ink-primary">{train.predicted}</td>
                      <td className="py-2.5 px-4 text-right">
                        <span
                          className={cn(
                            'rounded-full px-2 py-0.5 text-[10px] font-bold border',
                            isDelayed
                              ? 'bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30'
                              : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30',
                          )}
                        >
                          {isDelayed ? `${train.delayMin}m Delay` : 'On Time'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Delay Trend (Today) (5 cols) */}
        <Card className="lg:col-span-5">
          <CardHeader className="pb-2 flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle className="text-sm font-bold text-ink-primary flex items-center gap-2">
                <TrendingUp size={16} className="text-purple-600 dark:text-purple-400" />
                Delay Trend (Today)
              </CardTitle>
              <p className="text-xs text-ink-secondary mt-0.5">Average corridor arrival offset (minutes)</p>
            </div>
            {hoveredTrend && (
              <span className="text-xs font-mono font-bold text-purple-700 dark:text-purple-300 bg-purple-500/15 px-2 py-0.5 rounded border border-purple-500/30">
                {hoveredTrend.time}: {hoveredTrend.delay} min
              </span>
            )}
          </CardHeader>
          <CardContent className="pt-2">
            {/* SVG Area Chart */}
            <div className="relative h-44 w-full">
              <svg className="h-full w-full overflow-visible" viewBox="0 0 320 120" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="delayGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a855f7" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#a855f7" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Gridlines */}
                <line x1="0" y1="30" x2="320" y2="30" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" className="text-ink-muted" />
                <line x1="0" y1="60" x2="320" y2="60" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" className="text-ink-muted" />
                <line x1="0" y1="90" x2="320" y2="90" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" className="text-ink-muted" />

                {/* Y-axis Labels */}
                <text x="5" y="32" fill="currentColor" fontSize="8" className="text-ink-muted">30m</text>
                <text x="5" y="62" fill="currentColor" fontSize="8" className="text-ink-muted">20m</text>
                <text x="5" y="92" fill="currentColor" fontSize="8" className="text-ink-muted">10m</text>

                {/* Smooth Area Path */}
                <path
                  d="M 0,105 C 30,110 50,90 80,95 C 110,100 130,65 160,70 C 190,40 220,30 250,60 C 270,90 290,85 320,105 L 320,120 L 0,120 Z"
                  fill="url(#delayGradient)"
                />

                {/* Smooth Stroke Line */}
                <path
                  d="M 0,105 C 30,110 50,90 80,95 C 110,100 130,65 160,70 C 190,40 220,30 250,60 C 270,90 290,85 320,105"
                  fill="none"
                  stroke="#9333ea"
                  className="dark:stroke-purple-400"
                  strokeWidth="2.5"
                />

                {/* Interactive Points */}
                {DELAY_TREND_POINTS.map((pt, idx) => {
                  const x = (idx / (DELAY_TREND_POINTS.length - 1)) * 300 + 10
                  const y = 115 - (pt.delay / 30) * 85
                  return (
                    <circle
                      key={pt.time}
                      cx={x}
                      cy={y}
                      r="3.5"
                      className="fill-purple-600 dark:fill-purple-400 hover:r-5 transition-all cursor-pointer hover:fill-purple-800 dark:hover:fill-white"
                      onMouseEnter={() => setHoveredTrend(pt)}
                      onMouseLeave={() => setHoveredTrend(null)}
                    />
                  )
                })}
              </svg>
            </div>

            {/* Time labels below chart */}
            <div className="mt-1 flex items-center justify-between text-[10px] font-mono text-ink-muted px-1">
              <span>00:00</span>
              <span>04:00</span>
              <span>08:00</span>
              <span>12:00</span>
              <span>16:00</span>
              <span>20:00</span>
              <span>24:00</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Status Footer (Railway Control Room Operator) */}
      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-surface p-3 text-xs text-ink-secondary">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-semibold text-ink-primary">System Status: LIVE</span>
          <span className="text-ink-muted">|</span>
          <span>All telematics signals &amp; neural arrival models operating normally</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-purple-500/20 text-purple-700 dark:text-purple-300 font-bold text-[10px]">
            AI
          </div>
          <span className="font-medium text-ink-primary">AI Operator: Control Room 04</span>
        </div>
      </div>

      {/* AI Lineage & Mathematical Calculation Modal */}
      <LineageCalculationModal
        isOpen={isLineageModalOpen}
        onClose={() => setIsLineageModalOpen(false)}
        train={selectedTrain}
      />
    </div>
  )
}
