import { useEffect, useRef, useState } from 'react'
import {
  Check,
  Download,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  AmbientAudioSynthesizer,
  calculateTotalDuration,
  exportVideoFile,
  preloadImages,
  renderVideoFrame,
  type RenderScene,
  type VideoRenderConfig,
} from '@/lib/videoRenderer'
import { cn } from '@/lib/utils'

interface VideoPlayerModalProps {
  isOpen: boolean
  onClose: () => void
  images: {
    id: string
    previewUrl: string
    title?: string
    script?: string
    action?: string
    label: string
  }[]
  script: string
  templateKey: string
  transitionsOn: boolean
  transitionType: 'Fade' | 'Cut' | 'Slide' | 'Wipe' | 'Zoom Blur'
  transitionDuration: number
  cameraOn: boolean
  cameraMove: 'Zoom In' | 'Zoom Out' | 'Pan Left' | 'Pan Right' | 'Static'
  cameraIntensity: number
  animationOn: boolean
  animationStyle: 'None' | 'Parallax' | 'Kinetic Burst' | 'Subtle Float'
  animationIntensity: number
  timingOn: boolean
  sceneDuration: number
  captionsOn: boolean
  captionStyle: 'None' | 'Minimal' | 'Bold' | 'Subtitle-style'
  captionPosition: 'Top' | 'Center' | 'Bottom'
  voiceOn: boolean
  musicVolume: number
  narrationFile?: File | null
  musicFile?: File | null
}

export function VideoPlayerModal({
  isOpen,
  onClose,
  images,
  script,
  templateKey,
  transitionsOn,
  transitionType,
  transitionDuration,
  cameraOn,
  cameraMove,
  cameraIntensity,
  animationOn,
  animationStyle,
  animationIntensity,
  timingOn,
  sceneDuration,
  captionsOn,
  captionStyle,
  captionPosition,
  voiceOn,
  musicVolume,
}: VideoPlayerModalProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animFrameRef = useRef<number | null>(null)
  const synthRef = useRef<AmbientAudioSynthesizer | null>(null)

  const [loadedScenes, setLoadedScenes] = useState<RenderScene[]>([])
  const [loadingImages, setLoadingImages] = useState(true)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [playbackRate, setPlaybackRate] = useState(1.0)
  const [isMuted, setIsMuted] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isLooping, setIsLooping] = useState(true)

  // Export State
  const [isExporting, setIsExporting] = useState(false)
  const [exportProgress, setExportProgress] = useState(0)
  const [exportedUrl, setExportedUrl] = useState<string | null>(null)

  // Total Duration
  const totalDuration = calculateTotalDuration(
    loadedScenes,
    transitionsOn,
    transitionDuration,
  )

  // Preload images when modal opens
  useEffect(() => {
    if (!isOpen || images.length === 0) return

    let isMounted = true
    setLoadingImages(true)
    setCurrentTime(0)
    setIsPlaying(false)
    setExportedUrl(null)

    preloadImages(images.map((img) => img.previewUrl))
      .then((htmlImages) => {
        if (!isMounted) return
        const scenes: RenderScene[] = htmlImages.map((img, i) => ({
          id: images[i].id,
          image: img,
          title: images[i].title || `Scene ${i + 1}`,
          script: images[i].script || (i === 0 ? script : undefined),
          action: images[i].action,
          duration: timingOn ? sceneDuration : 3.5,
        }))
        setLoadedScenes(scenes)
        setLoadingImages(false)
      })
      .catch((err) => {
        console.error('Failed to preload images', err)
        setLoadingImages(false)
      })

    return () => {
      isMounted = false
    }
  }, [isOpen, images, script, timingOn, sceneDuration])

  // Initialize synthesizer
  useEffect(() => {
    if (!isOpen) return
    const synth = new AmbientAudioSynthesizer()
    synth.init()
    synthRef.current = synth

    return () => {
      synth.destroy()
      synthRef.current = null
    }
  }, [isOpen])

  // Volume & Mute Sync
  useEffect(() => {
    if (!synthRef.current) return
    if (isMuted) {
      synthRef.current.setVolume(0)
    } else {
      synthRef.current.setVolume(voiceOn ? musicVolume : 40)
    }
  }, [isMuted, musicVolume, voiceOn])

  // Render loop
  useEffect(() => {
    if (!isOpen || loadingImages || loadedScenes.length === 0) return

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const config: VideoRenderConfig = {
      width: canvas.width,
      height: canvas.height,
      scenes: loadedScenes,
      transitionType,
      transitionDuration,
      transitionsOn,
      cameraMove,
      cameraIntensity,
      cameraOn,
      animationStyle,
      animationIntensity,
      animationOn,
      captionStyle,
      captionPosition,
      captionsOn,
      templateKey,
      musicVolume,
      voiceOn,
    }

    let lastTimestamp = performance.now()

    function loop(now: number) {
      const delta = (now - lastTimestamp) / 1000
      lastTimestamp = now

      if (isPlaying) {
        setCurrentTime((prev) => {
          const next = prev + delta * playbackRate
          if (next >= totalDuration) {
            if (isLooping) {
              return 0
            } else {
              setIsPlaying(false)
              if (synthRef.current) synthRef.current.stop()
              return totalDuration
            }
          }
          return next
        })
      }

      renderVideoFrame(ctx!, canvas!.width, canvas!.height, currentTime, config)
      animFrameRef.current = requestAnimationFrame(loop)
    }

    animFrameRef.current = requestAnimationFrame(loop)

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [
    isOpen,
    loadingImages,
    loadedScenes,
    isPlaying,
    currentTime,
    playbackRate,
    isLooping,
    totalDuration,
    transitionType,
    transitionDuration,
    transitionsOn,
    cameraMove,
    cameraIntensity,
    cameraOn,
    animationStyle,
    animationIntensity,
    animationOn,
    captionStyle,
    captionPosition,
    captionsOn,
    templateKey,
    musicVolume,
    voiceOn,
  ])

  function handlePlayPause() {
    if (isPlaying) {
      setIsPlaying(false)
      if (synthRef.current) synthRef.current.stop()
    } else {
      if (currentTime >= totalDuration) {
        setCurrentTime(0)
      }
      setIsPlaying(true)
      if (synthRef.current && !isMuted) {
        synthRef.current.start(templateKey)
      }
    }
  }

  function handleScrub(e: React.ChangeEvent<HTMLInputElement>) {
    const val = parseFloat(e.target.value)
    setCurrentTime(val)
  }

  function handleRestart() {
    setCurrentTime(0)
    if (!isPlaying) {
      setIsPlaying(true)
      if (synthRef.current && !isMuted) synthRef.current.start(templateKey)
    }
  }

  function toggleFullscreen() {
    if (!containerRef.current) return
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => console.error(err))
      setIsFullscreen(true)
    } else {
      document.exitFullscreen().catch((err) => console.error(err))
      setIsFullscreen(false)
    }
  }

  async function handleExportVideo() {
    if (loadedScenes.length === 0) return
    setIsExporting(true)
    setExportProgress(0)

    const config: VideoRenderConfig = {
      width: 1280,
      height: 720,
      scenes: loadedScenes,
      transitionType,
      transitionDuration,
      transitionsOn,
      cameraMove,
      cameraIntensity,
      cameraOn,
      animationStyle,
      animationIntensity,
      animationOn,
      captionStyle,
      captionPosition,
      captionsOn,
      templateKey,
      musicVolume,
      voiceOn,
    }

    try {
      const videoBlob = await exportVideoFile(config, totalDuration, (percent) => {
        setExportProgress(percent)
      })

      const url = URL.createObjectURL(videoBlob)
      setExportedUrl(url)

      // Automatically trigger download
      const a = document.createElement('a')
      a.href = url
      a.download = `story-video-${templateKey}-${Date.now()}.webm`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    } catch (err) {
      console.error('Export failed', err)
      window.alert('Video rendering export failed. Check browser permissions.')
    } finally {
      setIsExporting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div
        ref={containerRef}
        className="relative flex w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-page shadow-2xl"
      >
        {/* Top Header Bar */}
        <div className="flex items-center justify-between border-b px-5 py-3.5 bg-surface">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-bento-magenta to-bento-violet text-white">
              <Sparkles size={16} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-ink-primary">Story to Video Studio</h2>
              <p className="text-[11px] text-ink-muted">
                {loadedScenes.length} scenes · {totalDuration.toFixed(1)}s runtime · {templateKey.toUpperCase()} style
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              size="sm"
              onClick={handleExportVideo}
              disabled={loadingImages || isExporting}
              className="bg-bento-blue hover:bg-bento-blue/90 text-white font-semibold text-xs h-8 cursor-pointer shadow"
            >
              {isExporting ? (
                <>
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                  Rendering Video ({exportProgress}%)…
                </>
              ) : exportedUrl ? (
                <>
                  <Check size={14} />
                  Download Again
                </>
              ) : (
                <>
                  <Download size={14} />
                  Export &amp; Download Video
                </>
              )}
            </Button>

            <button
              type="button"
              onClick={() => {
                if (isPlaying && synthRef.current) synthRef.current.stop()
                onClose()
              }}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-muted hover:bg-black/[0.04] dark:hover:bg-white/[0.06] hover:text-ink-primary cursor-pointer transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Video Canvas Viewport */}
        <div className="relative flex aspect-video w-full items-center justify-center bg-black overflow-hidden">
          {loadingImages ? (
            <div className="flex flex-col items-center gap-3 text-white">
              <div className="h-10 w-10 animate-spin rounded-full border-3 border-white/20 border-t-bento-blue" />
              <p className="text-xs font-medium">Preparing and rendering scenes…</p>
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              width={1280}
              height={720}
              className="h-full w-full object-contain cursor-pointer"
              onClick={handlePlayPause}
            />
          )}

          {/* Big Play Overlay if paused */}
          {!isPlaying && !loadingImages && (
            <div
              onClick={handlePlayPause}
              className="absolute inset-0 flex items-center justify-center bg-black/30 cursor-pointer transition-all hover:bg-black/20"
            >
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 text-black shadow-2xl backdrop-blur transition-transform hover:scale-110">
                <Play size={28} className="ml-1" />
              </div>
            </div>
          )}

          {/* Exporting Progress Overlay */}
          {isExporting && (
            <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm p-6 text-white text-center">
              <div className="h-12 w-12 animate-spin rounded-full border-4 border-white/20 border-t-bento-blue mb-4" />
              <h3 className="text-base font-bold">Rendering &amp; Encoding Full HD Video</h3>
              <p className="text-xs text-white/70 mt-1 max-w-sm">
                Composing camera motions, crossfades, subtitles, and ambient audio track...
              </p>
              <div className="w-64 bg-white/20 rounded-full h-2.5 mt-4 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-bento-violet to-bento-blue h-2.5 rounded-full transition-all duration-150"
                  style={{ width: `${exportProgress}%` }}
                />
              </div>
              <span className="text-xs font-semibold text-bento-blue mt-2">{exportProgress}%</span>
            </div>
          )}
        </div>

        {/* Player Controls Bar */}
        <div className="flex flex-col gap-2 p-4 bg-surface border-t">
          {/* Timeline Scrubber */}
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-ink-muted w-10 text-right">
              {currentTime.toFixed(1)}s
            </span>
            <input
              type="range"
              min={0}
              max={totalDuration || 1}
              step={0.05}
              value={currentTime}
              onChange={handleScrub}
              className="h-1.5 flex-1 cursor-pointer appearance-none rounded-lg bg-black/10 dark:bg-white/15 accent-bento-blue"
            />
            <span className="text-[11px] font-mono text-ink-muted w-10">
              {totalDuration.toFixed(1)}s
            </span>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRestart}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-secondary hover:bg-black/[0.04] dark:hover:bg-white/[0.06] hover:text-ink-primary transition-colors cursor-pointer"
                title="Restart from beginning"
              >
                <RotateCcw size={15} />
              </button>

              <button
                type="button"
                onClick={handlePlayPause}
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-bento-blue text-white shadow-sm hover:opacity-90 transition-all cursor-pointer"
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <Pause size={17} /> : <Play size={17} className="ml-0.5" />}
              </button>

              {/* Sound & Music toggle */}
              <button
                type="button"
                onClick={() => setIsMuted(!isMuted)}
                className={cn(
                  'flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium transition-colors cursor-pointer',
                  !isMuted
                    ? 'text-bento-blue bg-bento-blue/10'
                    : 'text-ink-muted hover:bg-black/[0.04] dark:hover:bg-white/[0.06]',
                )}
                title="Toggle ambient background soundtrack"
              >
                {isMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}
                <span className="hidden sm:inline">Ambient Music</span>
              </button>

              {/* Loop toggle */}
              <button
                type="button"
                onClick={() => setIsLooping(!isLooping)}
                className={cn(
                  'flex h-8 items-center gap-1 rounded-lg px-2 text-xs font-medium transition-colors cursor-pointer',
                  isLooping
                    ? 'text-bento-violet bg-bento-violet/10 font-semibold'
                    : 'text-ink-muted hover:bg-black/[0.04] dark:hover:bg-white/[0.06]',
                )}
                title="Loop playback"
              >
                Loop
              </button>
            </div>

            <div className="flex items-center gap-2">
              {/* Speed toggle */}
              <div className="flex items-center rounded-lg border bg-page p-0.5 text-xs">
                {[0.5, 1.0, 1.5, 2.0].map((rate) => (
                  <button
                    key={rate}
                    type="button"
                    onClick={() => setPlaybackRate(rate)}
                    className={cn(
                      'rounded-md px-2 py-0.5 text-[11px] font-medium transition-colors cursor-pointer',
                      playbackRate === rate
                        ? 'bg-bento-blue text-white font-bold'
                        : 'text-ink-secondary hover:text-ink-primary',
                    )}
                  >
                    {rate}x
                  </button>
                ))}
              </div>

              {/* Fullscreen button */}
              <button
                type="button"
                onClick={toggleFullscreen}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-secondary hover:bg-black/[0.04] dark:hover:bg-white/[0.06] hover:text-ink-primary transition-colors cursor-pointer"
                title="Fullscreen"
              >
                {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
