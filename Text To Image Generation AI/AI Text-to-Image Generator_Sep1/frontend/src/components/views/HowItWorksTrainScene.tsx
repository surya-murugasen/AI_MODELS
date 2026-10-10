import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { CheckCircle2, Gauge, MapPin, Milestone, Timer } from 'lucide-react'
import type { PipelineData } from '@/lib/howItWorksPipeline'

/**
 * Cinematic, GSAP-driven "final result" dashboard for the How It Works page.
 * Follows the app's own light/dark toggle (via Tailwind `dark:` classes,
 * same convention as the rest of the app) rather than a fixed palette.
 *
 * The route's origin/destination/distance/speed/delay/traffic/weather are
 * all real fields threaded through from the pipeline (the uploaded file's
 * own columns when present, or the worked example's consistent synthetic
 * values) — nothing here is fabricated GPS/live-tracking data. `nextStop`
 * is a fixed illustrative waypoint fraction along the route (there is no
 * real multi-stop itinerary in an uploaded schedule file), and its ETA/
 * distance are computed proportionally from the real total distance and
 * real scheduled/predicted times rather than copied from a mockup — so the
 * "distance covered + distance to go" pair always adds back up to the real
 * total distance. The train graphic is hand-drawn SVG, not a generated
 * image; only its animation mechanics are GSAP-driven.
 *
 * The train is positioned by its NOSE (not its left edge): the wrapper is
 * anchored via `right` in the same 0-100% track coordinate space as the
 * route line and nodes, so the nose lines up exactly with how far along
 * the journey the train has traveled, with its body trailing backward
 * toward the origin.
 */
type FinalResult = PipelineData['finalResult']

const WAYPOINT_PCTS = [30, 70]
const TRAVEL_START = -6
const TRAVEL_END = 100
const NEXT_STOP_FRAC = 0.42 // fixed illustrative position of the "next stop" along the route

export function HowItWorksTrainScene({ result }: { result: FinalResult }) {
  const trainRef = useRef<HTMLDivElement>(null)
  const streakRefs = useRef<(HTMLDivElement | null)[]>([])
  const reflectionRef = useRef<HTMLDivElement>(null)
  const glowRef = useRef<HTMLDivElement>(null)
  const hotPointRef = useRef<HTMLDivElement>(null)
  const checkRef = useRef<SVGSVGElement>(null)
  const wheelRefs = useRef<(SVGGElement | null)[]>([])
  const waypointRefs = useRef<(HTMLSpanElement | null)[]>([])
  const waypointRingRefs = useRef<(HTMLSpanElement | null)[]>([])
  const farMountainsRef = useRef<HTMLDivElement>(null)
  const nearHillsRef = useRef<HTMLDivElement>(null)
  const tiesRef = useRef<HTMLDivElement>(null)
  const headlightRef = useRef<HTMLDivElement>(null)
  const motionBlurRef = useRef<SVGFEGaussianBlurElement>(null)

  useEffect(() => {
    const train = trainRef.current
    const glow = glowRef.current
    const check = checkRef.current
    if (!train || !glow || !check) return

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const wheels = wheelRefs.current.filter(Boolean) as SVGGElement[]
    const wheelTween = reduceMotion
      ? null
      : gsap.to(wheels, { rotate: 360, duration: 0.5, repeat: -1, ease: 'none', transformOrigin: 'center' })
    const headlightTween =
      reduceMotion || !headlightRef.current
        ? null
        : gsap.to(headlightRef.current, { opacity: 0.55, scale: 0.88, duration: 1.1, repeat: -1, yoyo: true, ease: 'sine.inOut' })
    // Continuous pulse on the progress line's "hot point" — travels with the
    // train (via its parent's width) while separately pulsing in place, so
    // the glow itself reads as alive, not just a static marker riding along.
    const pulseTween =
      reduceMotion || !hotPointRef.current
        ? null
        : gsap.to(hotPointRef.current, { scale: 1.6, duration: 0.65, repeat: -1, yoyo: true, ease: 'sine.inOut' })

    // Seamless infinite-loop parallax (two copies of each layer, side by
    // side, sliding left by 50% forever) — guarantees visible continuous
    // background motion regardless of the train's own accelerate/cruise/
    // decelerate phase. Speed is then linked to the train's real velocity
    // via timeScale so the background genuinely tracks how fast it's going.
    const farTween = reduceMotion
      ? null
      : gsap.to(farMountainsRef.current, { xPercent: -50, duration: 26, repeat: -1, ease: 'none' })
    const nearTween = reduceMotion
      ? null
      : gsap.to(nearHillsRef.current, { xPercent: -50, duration: 11, repeat: -1, ease: 'none' })
    const tiesTween = reduceMotion
      ? null
      : gsap.to(tiesRef.current, { xPercent: -50, duration: 1.1, repeat: -1, ease: 'none' })
    farTween?.pause()
    nearTween?.pause()
    tiesTween?.pause()

    if (reduceMotion) {
      gsap.set(train, { right: '0%' })
      gsap.set(glow, { width: '100%' })
      gsap.set(check, { opacity: 1, scale: 1 })
      waypointRefs.current.forEach((el) => el && gsap.set(el, { opacity: 1, scale: 1.2 }))
      streakRefs.current.forEach((el) => el && gsap.set(el, { opacity: 0 }))
      motionBlurRef.current?.setAttribute('stdDeviation', '0 0')
      return () => {
        wheelTween?.kill()
        headlightTween?.kill()
        pulseTween?.kill()
      }
    }

    gsap.set(train, { right: `${100 - TRAVEL_START}%`, opacity: 0 })
    gsap.set(glow, { width: '0%' })
    gsap.set(check, { opacity: 0, scale: 0.4, rotate: -12, transformOrigin: 'center' })
    waypointRefs.current.forEach((el) => el && gsap.set(el, { opacity: 0.3, scale: 1 }))

    // Drive position off a plain numeric object rather than tweening a CSS
    // property directly — keeps every derived effect (glow, parallax speed,
    // ping triggers) reading from one known-good number each frame.
    const progress = { pct: TRAVEL_START }
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.5 })

    tl.to(train, { opacity: 1, duration: 0.6, ease: 'power2.out' }, 0)
      .to(progress, { pct: 4, duration: 0.6, ease: 'power2.out' }, 0)
      .to(progress, { pct: 22, duration: 1.3, ease: 'power2.out' }, '>')
      .to(progress, { pct: 84, duration: 4.6, ease: 'none' }, '>')
      .to(progress, { pct: 96, duration: 1.2, ease: 'power2.in' }, '>')
      .to(progress, { pct: TRAVEL_END, duration: 0.4, ease: 'power1.out' }, '>')
      .to(check, { opacity: 1, scale: 1, rotate: 0, duration: 0.45, ease: 'back.out(2.2)' }, '<')
      .to({}, { duration: 0.9 })
      .to([train, check], { opacity: 0, duration: 0.5, ease: 'power1.in' }, '>')
      .set(progress, { pct: TRAVEL_START })
      .set(check, { scale: 0.4, rotate: -12 })
      .set(glow, { width: '0%' })
      .call(() => {
        waypointRefs.current.forEach((el) => el && gsap.set(el, { opacity: 0.3, scale: 1 }))
        passedState.fill(false)
      })

    const totalTravel = TRAVEL_END - TRAVEL_START
    let lastPct = TRAVEL_START
    let lastTime = tl.time()
    const passedState = WAYPOINT_PCTS.map(() => false)

    tl.eventCallback('onUpdate', () => {
      const currentPct = progress.pct
      const clampedRight = gsap.utils.clamp(0, 100 - TRAVEL_START, 100 - currentPct)
      gsap.set(train, { right: `${clampedRight}%` })

      const now = tl.time()
      const dt = Math.max(now - lastTime, 0.001)
      const velocity = Math.abs(currentPct - lastPct) / dt
      lastPct = currentPct
      lastTime = now

      // Directional motion blur on the train only — an SVG feGaussianBlur
      // with stdDeviation "<x> 0" blurs purely along the horizontal axis
      // of travel (unlike CSS filter: blur(), which is omnidirectional),
      // so it reads as real motion blur rather than a soft-focus effect.
      const blurX = Math.min(3.2, (velocity / 24) * 3.2)
      if (motionBlurRef.current) motionBlurRef.current.setAttribute('stdDeviation', `${blurX.toFixed(2)} 0`)

      // Speed streaks trailing the train, staggered so they read as several
      // thin motion lines rather than one blob — length + opacity scale
      // with velocity.
      streakRefs.current.forEach((el, i) => {
        if (!el) return
        const falloff = 1 - i * 0.28
        gsap.set(el, {
          opacity: gsap.utils.clamp(0, 0.85 * falloff, (velocity / 20) * falloff),
          scaleX: gsap.utils.clamp(0.5, 1.6, velocity / 18),
        })
      })

      // Link ambient parallax + rail-tie scroll speed to real velocity so
      // the background genuinely speeds up/slows down with the train,
      // while the base infinite loop guarantees it's never fully static.
      const speedScale = gsap.utils.clamp(0.15, 2.6, velocity / 14)
      farTween?.play().timeScale(speedScale * 0.4)
      nearTween?.play().timeScale(speedScale * 0.9)
      tiesTween?.play().timeScale(speedScale)

      const travelFrac = gsap.utils.clamp(0, 1, (currentPct - TRAVEL_START) / totalTravel)
      const travelPct = Math.max(0, currentPct)
      gsap.set(glow, { width: `${gsap.utils.clamp(0, 100, travelPct)}%` })
      if (hotPointRef.current) gsap.set(hotPointRef.current, { opacity: travelFrac > 0.01 && travelFrac < 0.995 ? 1 : 0 })

      waypointRefs.current.forEach((el, i) => {
        if (!el) return
        const passed = travelPct >= WAYPOINT_PCTS[i]
        gsap.to(el, { opacity: passed ? 1 : 0.3, scale: passed ? 1.3 : 1, duration: 0.25, overwrite: 'auto' })
        if (passed && !passedState[i]) {
          passedState[i] = true
          const ring = waypointRingRefs.current[i]
          if (ring) {
            gsap.fromTo(ring, { opacity: 0.8, scale: 1 }, { opacity: 0, scale: 2.6, duration: 0.6, ease: 'power1.out' })
          }
        } else if (!passed && passedState[i]) {
          passedState[i] = false
        }
      })
    })

    return () => {
      tl.kill()
      wheelTween?.kill()
      headlightTween?.kill()
      pulseTween?.kill()
      farTween?.kill()
      nearTween?.kill()
      tiesTween?.kill()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Real, internally-consistent math for the next-stop tiles: the fraction
  // is a fixed illustrative constant, but the distance/time it implies are
  // derived from the row's actual total distance and real scheduled/
  // predicted times, so "covered" + "to go" always sum to the real total.
  const distanceCovered = Math.round(result.distanceKm * NEXT_STOP_FRAC)
  const distanceToGo = Math.max(0, Math.round(result.distanceKm - distanceCovered))
  const nextStopEta = addMinutesToTime(result.scheduledDeparture, Math.round(result.delayMin * NEXT_STOP_FRAC))

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-white via-slate-50 to-white shadow-lg ring-1 ring-slate-200 dark:from-[#0a0e1a] dark:via-[#0d1526] dark:to-[#080c16] dark:ring-white/10">
      {/* Directional (horizontal-only) motion-blur filter for the train,
          referenced via CSS `filter: url(#hiwMotionBlur)` — driven live by
          real per-frame velocity, not a constant/decorative blur. */}
      <svg width="0" height="0" className="absolute" aria-hidden="true">
        <defs>
          <filter id="hiwMotionBlur" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur ref={motionBlurRef} in="SourceGraphic" stdDeviation="0 0" />
          </filter>
        </defs>
      </svg>

      {/* -------------------------------------------------------------- */}
      {/* Route scene                                                    */}
      {/* -------------------------------------------------------------- */}
      <div className="relative h-56 w-full overflow-hidden bg-gradient-to-b from-sky-100 via-sky-50 to-white dark:from-transparent dark:via-transparent dark:to-transparent">
        {/* Cinematic vignette (dark mode only) */}
        <div
          className="pointer-events-none absolute inset-0 z-10 hidden dark:block"
          style={{ background: 'radial-gradient(ellipse at 50% 30%, transparent 45%, rgba(0,0,0,0.35) 100%)' }}
        />

        {/* Sun / moon */}
        <div className="absolute right-6 top-3 h-6 w-6 rounded-full bg-yellow-200 shadow-[0_0_24px_6px_rgba(250,204,21,0.4)] dark:bg-slate-100/90 dark:shadow-[0_0_26px_6px_rgba(226,232,240,0.3)]" />
        {/* Stars (dark mode only) */}
        <div className="absolute inset-x-0 top-2 hidden h-16 opacity-70 dark:block">
          {[12, 28, 46, 62, 78, 90].map((left, i) => (
            <span
              key={left}
              className="absolute h-[3px] w-[3px] rounded-full bg-white"
              style={{ left: `${left}%`, top: `${(i % 3) * 10 + 4}px`, opacity: 0.4 + (i % 3) * 0.2 }}
            />
          ))}
        </div>

        {/* Origin / destination labels — kept clear of the train, pinned near the top */}
        <div className="absolute inset-x-5 top-2.5 flex items-start justify-between text-[10px]">
          <div className="max-w-[100px]">
            <p className="truncate font-semibold text-slate-800 dark:text-slate-100">{result.origin}</p>
            <p className="text-[9px] text-slate-500 dark:text-slate-400">{result.scheduledDeparture}</p>
          </div>
          <div className="max-w-[100px] text-right">
            <p className="truncate font-semibold text-slate-800 dark:text-slate-100">{result.destination}</p>
            <p className="text-[9px] text-slate-500 dark:text-slate-400">{result.predictedArrival}</p>
          </div>
        </div>

        {/* Parallax far mountains (seamless loop: two copies side by side) */}
        <div ref={farMountainsRef} className="absolute bottom-16 left-0 flex h-16 w-[200%] opacity-60 will-change-transform dark:opacity-45">
          {[0, 1].map((rep) => (
            <div
              key={rep}
              className="h-full w-1/2 flex-none bg-gradient-to-b from-indigo-200 to-indigo-300 dark:from-[#1e293b] dark:to-[#0f172a]"
              style={{
                clipPath: 'polygon(0% 100%, 0% 60%, 10% 40%, 22% 58%, 34% 30%, 48% 55%, 60% 35%, 74% 60%, 88% 42%, 100% 60%, 100% 100%)',
              }}
            />
          ))}
        </div>

        {/* Parallax near hills, with trees + a soft skyline (seamless loop) */}
        <div ref={nearHillsRef} className="absolute bottom-11 left-0 flex h-10 w-[200%] opacity-90 will-change-transform dark:opacity-75">
          {[0, 1].map((rep) => (
            <div key={rep} className="relative h-full w-1/2 flex-none">
              <div
                className="absolute inset-0 bg-gradient-to-b from-emerald-200 to-emerald-300 dark:from-[#111c33] dark:to-[#0a1222]"
                style={{
                  clipPath: 'polygon(0% 100%, 0% 55%, 15% 40%, 30% 55%, 45% 35%, 62% 52%, 78% 38%, 100% 55%, 100% 100%)',
                }}
              />
              {/* Trees */}
              {[9, 18, 27].map((left, i) => (
                <svg key={left} className="absolute bottom-1" style={{ left: `${left}%`, height: '18px', width: '12px' }} viewBox="0 0 12 18">
                  <line x1="6" y1="12" x2="6" y2="18" className="stroke-emerald-800/60 dark:stroke-slate-700" strokeWidth="1" />
                  <path d={`M6,${1 + i} L11,13 L1,13 Z`} className="fill-emerald-700/70 dark:fill-slate-600/70" />
                </svg>
              ))}
              {/* Skyline */}
              {[
                { left: 66, h: 10 },
                { left: 72, h: 16 },
                { left: 78, h: 8 },
                { left: 84, h: 13 },
              ].map((b) => (
                <span
                  key={b.left}
                  className="absolute bottom-0 w-[5px] bg-slate-400/50 dark:bg-slate-600/60"
                  style={{ left: `${b.left}%`, height: `${b.h}px` }}
                />
              ))}
            </div>
          ))}
        </div>

        {/* Track + route line + nodes */}
        <div className="absolute inset-x-5 bottom-9">
          {/* Rail ties scrolling under the track for a speed cue */}
          <div className="absolute inset-x-0 -bottom-1.5 h-1 overflow-hidden opacity-40">
            <div ref={tiesRef} className="flex h-full w-[200%]">
              {[0, 1].map((rep) => (
                <div
                  key={rep}
                  className="h-full w-1/2 flex-none"
                  style={{
                    backgroundImage: 'repeating-linear-gradient(90deg, #64748b 0px, #64748b 2px, transparent 2px, transparent 10px)',
                  }}
                />
              ))}
            </div>
          </div>

          <div className="relative h-[2px] w-full rounded-full bg-slate-300 dark:bg-white/15">
            <div
              ref={glowRef}
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                background: 'linear-gradient(90deg, rgba(16,185,129,0) 0%, rgba(16,185,129,0.4) 55%, rgba(52,211,153,0.95) 100%)',
                boxShadow: '0 0 12px 2px rgba(16,185,129,0.5)',
              }}
            >
              <div
                ref={hotPointRef}
                className="absolute right-0 top-1/2 h-2.5 w-2.5 -translate-y-1/2 translate-x-1/2 rounded-full bg-emerald-400"
                style={{ boxShadow: '0 0 10px 4px rgba(52,211,153,0.8)' }}
              />
            </div>
          </div>

          {/* Origin node */}
          <span className="absolute left-0 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-500 shadow-[0_0_8px_2px_rgba(16,185,129,0.6)] dark:bg-emerald-400" />

          {/* Waypoint dots with one-shot "ping" ring on pass */}
          {WAYPOINT_PCTS.map((pos, i) => (
            <div key={pos} className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2" style={{ left: `${pos}%` }}>
              <span
                ref={(el) => {
                  waypointRingRefs.current[i] = el
                }}
                className="absolute inset-0 -m-1 rounded-full bg-emerald-400/70 opacity-0"
              />
              <span
                ref={(el) => {
                  waypointRefs.current[i] = el
                }}
                className="relative block h-1.5 w-1.5 rounded-full bg-slate-400 dark:bg-white/40"
              />
            </div>
          ))}

          {/* Destination node + arrival check */}
          <span className="absolute right-0 top-1/2 h-2 w-2 -translate-y-1/2 translate-x-1/2 rounded-full bg-emerald-500 shadow-[0_0_8px_2px_rgba(16,185,129,0.6)] dark:bg-emerald-400" />
          <CheckCircle2 ref={checkRef} size={13} className="absolute text-emerald-500 dark:text-emerald-400" style={{ right: '-6px', top: '-20px' }} />

          {/* Train reflection on the platform surface — same coordinate
              space as the line/nodes above so nose position lines up exactly */}
          <div
            ref={reflectionRef}
            className="pointer-events-none absolute bottom-0 opacity-[0.15] blur-[1.5px] dark:opacity-[0.12]"
            style={{ transform: 'scaleY(-1)', width: '92%', right: 0 }}
          >
            <TrainSvgBody withDetail={false} />
          </div>

          {/* Train — anchored by its nose via `right`, body trails backward.
              filter: url(#hiwMotionBlur) applies a directional (horizontal-
              only) motion blur, driven by real velocity each frame — only
              this element is blurred, nothing else in the scene. */}
          <div
            ref={trainRef}
            data-testid="hiw-train"
            className="absolute bottom-0 will-change-transform"
            style={{ right: `${100 - TRAVEL_START}%`, width: '92%', filter: 'url(#hiwMotionBlur)' }}
          >
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                ref={(el) => {
                  streakRefs.current[i] = el
                }}
                className="pointer-events-none absolute right-[97%] -translate-y-1/2 rounded-full bg-gradient-to-l from-sky-400/60 to-transparent dark:from-sky-300/70"
                style={{ top: `${50 + (i - 1) * 14}%`, height: `${2.5 - i * 0.5}px`, width: `${14 - i * 3}px` }}
              />
            ))}
            <TrainGlyph wheelRefs={wheelRefs} headlightRef={headlightRef} />
          </div>
        </div>
      </div>

      {/* -------------------------------------------------------------- */}
      {/* Stats panel                                                    */}
      {/* -------------------------------------------------------------- */}
      <div className="relative border-t border-slate-200 bg-slate-50/60 p-3 dark:border-white/10 dark:bg-black/25">
        <div className="mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <CheckCircle2 size={14} className="text-emerald-600 dark:text-emerald-400" />
            <span className="text-[10px] font-bold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">Pipeline complete</span>
          </div>
          <span className="font-mono text-[9px] text-slate-500 dark:text-slate-400">Train {result.trainNo}</span>
        </div>

        <div className="flex items-end justify-between">
          <span className="text-2xl font-black leading-none text-slate-900 dark:text-white">{result.predictedArrival}</span>
          <span className="text-xs font-bold text-orange-500 dark:text-orange-400">+{result.delayMin}m delay</span>
        </div>

        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
          <div className="h-full rounded-full bg-emerald-500 dark:bg-emerald-400" style={{ width: `${result.confidence}%` }} />
        </div>
        <p className="mt-1 text-right text-[10px] font-bold text-emerald-600 dark:text-emerald-400">{result.confidence}% confident</p>

        <div className="mt-3 grid grid-cols-2 gap-2">
          <StatTile icon={Timer} label="ETA (next stop)" value={nextStopEta} sub={result.nextStopName} />
          <StatTile icon={Milestone} label="Distance covered" value={`${distanceCovered} km`} sub={`${Math.round(NEXT_STOP_FRAC * 100)}% of journey`} />
          <StatTile icon={Gauge} label="Speed" value={`${result.speed} km/h`} sub="Current reading" />
          <StatTile icon={MapPin} label="Distance to go" value={`${distanceToGo} km`} sub={`${100 - Math.round(NEXT_STOP_FRAC * 100)}% remaining`} />
        </div>
      </div>
    </div>
  )
}

function addMinutesToTime(timeStr: string, minutesToAdd: number): string {
  const match = timeStr.match(/(\d{1,2}):(\d{2})\s*(AM|PM)/i)
  if (!match) return timeStr
  let hours = Number.parseInt(match[1], 10) % 12
  const minutes = Number.parseInt(match[2], 10)
  const isPM = match[3].toUpperCase() === 'PM'
  if (isPM) hours += 12

  const total = hours * 60 + minutes + minutesToAdd
  const wrapped = ((total % (24 * 60)) + 24 * 60) % (24 * 60)
  const outHours24 = Math.floor(wrapped / 60)
  const outMinutes = wrapped % 60
  const outPeriod = outHours24 >= 12 ? 'PM' : 'AM'
  const outHours12 = outHours24 % 12 === 0 ? 12 : outHours24 % 12
  return `${outHours12}:${String(outMinutes).padStart(2, '0')} ${outPeriod}`
}

function StatTile({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: typeof Gauge
  label: string
  value: string
  sub: string
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-2 dark:border-white/10 dark:bg-white/[0.03]">
      <div className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
        <Icon size={11} />
        <span className="text-[9px] font-medium uppercase tracking-wide">{label}</span>
      </div>
      <p className="mt-0.5 truncate text-xs font-bold text-slate-900 dark:text-slate-100">{value}</p>
      <p className="truncate text-[8.5px] text-slate-400 dark:text-slate-500">{sub}</p>
    </div>
  )
}

/**
 * Realistic multi-car high-speed train: an aerodynamic engine car up front
 * plus three connected trailer cars (visible coupler gaps between each),
 * a continuous window band and livery stripe, and wheels mostly hidden
 * beneath an undercarriage skirt with only a subtle, continuous rotation —
 * not full exposed cartoon wheels.
 */
const CAR_SEGMENTS = [
  { x: 12, w: 104 }, // rear trailer
  { x: 121, w: 104 }, // trailer
  { x: 230, w: 104 }, // trailer
] as const
const ENGINE_X = 339
const ENGINE_NOSE_X = 472
const BODY_TOP = 32
const BODY_BOTTOM = 78

function TrainSvgBody({ withDetail }: { withDetail: boolean }) {
  return (
    <svg width="100%" viewBox="0 0 480 110" fill="none" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="hiwTrainBody" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" className="[stop-color:#f1f5f9] dark:[stop-color:#e2e8f0]" />
          <stop offset="35%" className="[stop-color:#cbd5e1] dark:[stop-color:#94a3b8]" />
          <stop offset="70%" className="[stop-color:#94a3b8] dark:[stop-color:#475569]" />
          <stop offset="100%" className="[stop-color:#64748b] dark:[stop-color:#1e293b]" />
        </linearGradient>
        <linearGradient id="hiwTrainGlass" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" className="[stop-color:#bae6fd] dark:[stop-color:#a5f3fc]" />
          <stop offset="100%" className="[stop-color:#0284c7] dark:[stop-color:#0369a1]" />
        </linearGradient>
      </defs>

      {/* Continuous livery stripe running under all cars */}
      <rect x="12" y="78" width="418" height="5" className="fill-sky-500 dark:fill-sky-400" opacity="0.9" />
      <rect x="12" y="83" width="418" height="1.6" className="fill-orange-500 dark:fill-orange-400" opacity="0.9" />

      {/* Trailer cars */}
      {CAR_SEGMENTS.map((seg, i) => (
        <g key={seg.x}>
          <rect x={seg.x} y={BODY_TOP} width={seg.w} height={BODY_BOTTOM - BODY_TOP} rx="6" fill="url(#hiwTrainBody)" />
          {withDetail && (
            <>
              <rect x={seg.x + 6} y={BODY_TOP + 8} width={seg.w - 12} height="3" rx="1.5" className="fill-white" opacity="0.22" />
              {[0, 1, 2].map((wi) => (
                <rect
                  key={wi}
                  x={seg.x + 10 + wi * 32}
                  y={BODY_TOP + 12}
                  width="20"
                  height="14"
                  rx="3"
                  fill="url(#hiwTrainGlass)"
                />
              ))}
            </>
          )}
          {/* Coupler to the next segment */}
          {i < CAR_SEGMENTS.length - 1 && (
            <rect x={seg.x + seg.w} y={BODY_TOP + 18} width="5" height="14" rx="1.5" className="fill-slate-700 dark:fill-slate-900" />
          )}
        </g>
      ))}

      {/* Coupler between last trailer and the engine */}
      <rect x={CAR_SEGMENTS[2].x + CAR_SEGMENTS[2].w} y={BODY_TOP + 18} width="5" height="14" rx="1.5" className="fill-slate-700 dark:fill-slate-900" />

      {/* Engine / nose car */}
      <path
        d={`M${ENGINE_X},${BODY_TOP} L${ENGINE_X + 91},${BODY_TOP} C${ENGINE_X + 111},${BODY_TOP} ${ENGINE_X + 123},${BODY_TOP + 8} ${ENGINE_NOSE_X},${55} C${ENGINE_X + 123},${BODY_BOTTOM - 8} ${ENGINE_X + 111},${BODY_BOTTOM} ${ENGINE_X + 91},${BODY_BOTTOM} L${ENGINE_X},${BODY_BOTTOM} Z`}
        fill="url(#hiwTrainBody)"
      />

      {withDetail && (
        <>
          {/* Roofline + pantographs */}
          <path d="M28,32 L420,32" className="stroke-slate-400 dark:stroke-slate-300" strokeWidth="1" opacity="0.5" />
          <path d="M75,32 L75,20 L87,20 L87,32" className="stroke-slate-500 dark:stroke-slate-400" strokeWidth="1.6" fill="none" />
          <path d="M270,32 L270,20 L282,20 L282,32" className="stroke-slate-500 dark:stroke-slate-400" strokeWidth="1.6" fill="none" />
          {/* Nose sensor antenna */}
          <path d="M410,32 L410,18" className="stroke-slate-500 dark:stroke-slate-400" strokeWidth="1.4" />
          <circle cx="410" cy="16" r="2" className="fill-slate-400 dark:fill-slate-300" />

          {/* Cab windshield near the nose */}
          <path d="M405,44 L452,53 L452,64 L405,72 Z" fill="url(#hiwTrainGlass)" opacity="0.92" />

          {/* Tail light (rear, dim) */}
          <ellipse cx="16" cy="55" rx="2.4" ry="3.2" className="fill-red-800/70 dark:fill-red-900/70" />
        </>
      )}
    </svg>
  )
}

function TrainGlyph({
  wheelRefs,
  headlightRef,
}: {
  wheelRefs: React.MutableRefObject<(SVGGElement | null)[]>
  headlightRef: React.RefObject<HTMLDivElement | null>
}) {
  return (
    <div className="relative">
      <TrainSvgBody withDetail />

      {/* Wheels — separate overlay SVG so refs stay simple to spin. The
          skirt is drawn AFTER the wheels (last = on top) so it occludes
          their top ~70%, leaving only a subtle peek visible beneath it. */}
      <svg width="100%" viewBox="0 0 480 110" fill="none" className="absolute inset-0" preserveAspectRatio="xMidYMid meet">
        {[40, 90, 150, 200, 255, 305, 365, 410].map((cx, i) => (
          <g
            key={cx}
            ref={(el) => {
              wheelRefs.current[i] = el
            }}
          >
            <circle cx={cx} cy="90" r="7" className="fill-slate-950 dark:fill-[#0f172a]" opacity="0.9" />
            <circle cx={cx} cy="90" r="2.2" className="fill-slate-500 dark:fill-slate-600" />
          </g>
        ))}
        <rect x="10" y="83" width="452" height="10" rx="4" className="fill-slate-700 dark:fill-[#1e293b]" />
        <rect x="10" y="83" width="452" height="10" rx="4" className="fill-slate-950/55 dark:fill-[#020617]/70" />
      </svg>

      {/* Headlight glow at the nose tip */}
      <div
        ref={headlightRef}
        className="absolute h-2.5 w-2.5 rounded-full bg-amber-200 shadow-[0_0_14px_4px_rgba(252,211,77,0.8)] dark:bg-amber-100 dark:shadow-[0_0_14px_4px_rgba(254,243,199,0.9)]"
        style={{ right: '1%', top: '50%' }}
      />
    </div>
  )
}
