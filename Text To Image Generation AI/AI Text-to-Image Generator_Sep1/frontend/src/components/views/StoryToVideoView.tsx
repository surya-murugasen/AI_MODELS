import { useEffect, useRef, useState } from 'react'
import {
  Camera,
  Captions,
  Clapperboard,
  Film,
  GripVertical,
  Music2,
  PackageOpen,
  Plus,
  Sparkles,
  Timer,
  Trash2,
  Wand2,
  Workflow,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { VideoPlayerModal } from '@/components/views/VideoPlayerModal'
import type { ImagePack } from '@/lib/types'
import { cn } from '@/lib/utils'

const TEMPLATES = [
  { key: 'cinematic', label: 'Cinematic' },
  { key: 'slideshow', label: 'Slideshow' },
  { key: 'documentary', label: 'Documentary' },
  { key: 'social-reel', label: 'Social Reel' },
  { key: 'music-video', label: 'Music Video' },
] as const
type TemplateKey = (typeof TEMPLATES)[number]['key']

const TRANSITIONS = ['Fade', 'Cut', 'Slide', 'Wipe', 'Zoom Blur'] as const
const CAMERA_MOVES = ['Zoom In', 'Zoom Out', 'Pan Left', 'Pan Right', 'Static'] as const
const ANIMATION_STYLES = ['None', 'Parallax', 'Kinetic Burst', 'Subtle Float'] as const
const NARRATIVE_STRUCTURES = ['Linear', 'Flashback', 'Montage', 'Build-up'] as const
const PACING_OPTIONS = ['Slow', 'Medium', 'Fast'] as const
const CAPTION_STYLES = ['None', 'Minimal', 'Bold', 'Subtitle-style'] as const
const CAPTION_POSITIONS = ['Top', 'Center', 'Bottom'] as const

interface StoryImage {
  id: string
  previewUrl: string
  label: string
  title?: string
  script?: string
  action?: string
  /** Object URLs (from local uploads) must be revoked on removal/unmount;
   * data URLs (from a pack, already owned by the generation record) must not. */
  isObjectUrl: boolean
}

function OptionPills<T extends string>({ options, value, onChange }: { options: readonly T[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => (
        <button
          key={opt}
          type="button"
          onClick={() => onChange(opt)}
          className={cn(
            'rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors',
            opt === value
              ? 'border-bento-blue bg-bento-blue/10 text-bento-blue'
              : 'text-ink-secondary hover:bg-black/[0.03] dark:hover:bg-white/[0.05]',
          )}
        >
          {opt}
        </button>
      ))}
    </div>
  )
}

function LabeledSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 10,
  step = 1,
  suffix = '',
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  suffix?: string
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">{label}</p>
        <span className="text-xs font-semibold text-ink-primary">
          {value}
          {suffix}
        </span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={([v]) => onChange(v)} />
    </div>
  )
}

function PluginCard({
  icon,
  accent,
  title,
  description,
  enabled,
  onToggle,
  children,
}: {
  icon: React.ReactNode
  accent: string
  title: string
  description: string
  enabled: boolean
  onToggle: (v: boolean) => void
  children: React.ReactNode
}) {
  return (
    <Card className={cn('transition-opacity', !enabled && 'opacity-70')}>
      <CardHeader className="flex-row items-start justify-between space-y-0">
        <div className="flex items-start gap-3">
          <div
            className="flex h-9 w-9 flex-none items-center justify-center rounded-lg"
            style={{ backgroundColor: `${accent}1a`, color: accent }}
          >
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm">{title}</CardTitle>
            <CardDescription className="mt-0.5 text-xs">{description}</CardDescription>
          </div>
        </div>
        <Switch checked={enabled} onCheckedChange={onToggle} aria-label={`Toggle ${title}`} />
      </CardHeader>
      {enabled && <CardContent className="flex flex-col gap-4 pt-4">{children}</CardContent>}
    </Card>
  )
}

export function StoryToVideoView({ packs }: { packs: ImagePack[] }) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [images, setImages] = useState<StoryImage[]>([])
  const [script, setScript] = useState('')
  const [template, setTemplate] = useState<TemplateKey>('cinematic')

  // Plugin: Scene-to-scene transitions
  const [transitionsOn, setTransitionsOn] = useState(true)
  const [transitionType, setTransitionType] = useState<(typeof TRANSITIONS)[number]>('Fade')
  const [transitionDuration, setTransitionDuration] = useState(0.6)

  // Plugin: Camera movement & zoom
  const [cameraOn, setCameraOn] = useState(true)
  const [cameraMove, setCameraMove] = useState<(typeof CAMERA_MOVES)[number]>('Zoom In')
  const [cameraIntensity, setCameraIntensity] = useState(5)

  // Plugin: Image animation
  const [animationOn, setAnimationOn] = useState(false)
  const [animationStyle, setAnimationStyle] = useState<(typeof ANIMATION_STYLES)[number]>('Parallax')
  const [animationIntensity, setAnimationIntensity] = useState(4)

  // Plugin: Scene timing
  const [timingOn, setTimingOn] = useState(true)
  const [sceneDuration, setSceneDuration] = useState(3)

  // Plugin: Story progression
  const [progressionOn, setProgressionOn] = useState(false)
  const [narrative, setNarrative] = useState<(typeof NARRATIVE_STRUCTURES)[number]>('Linear')
  const [pacing, setPacing] = useState<(typeof PACING_OPTIONS)[number]>('Medium')

  // Plugin: Text / captions
  const [captionsOn, setCaptionsOn] = useState(false)
  const [captionStyle, setCaptionStyle] = useState<(typeof CAPTION_STYLES)[number]>('Minimal')
  const [captionPosition, setCaptionPosition] = useState<(typeof CAPTION_POSITIONS)[number]>('Bottom')

  // Plugin: Voice / music
  const [voiceOn, setVoiceOn] = useState(false)
  const [narrationFile, setNarrationFile] = useState<File | null>(null)
  const [musicFile, setMusicFile] = useState<File | null>(null)
  const [musicVolume, setMusicVolume] = useState(60)

  useEffect(() => {
    return () => {
      images.forEach((img) => img.isObjectUrl && URL.revokeObjectURL(img.previewUrl))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleAddImages(fileList: FileList | null) {
    if (!fileList) return
    const added: StoryImage[] = Array.from(fileList).map((file) => ({
      id: crypto.randomUUID(),
      previewUrl: URL.createObjectURL(file),
      label: file.name,
      isObjectUrl: true,
    }))
    setImages((prev) => [...prev, ...added])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  function handleLoadPack(pack: ImagePack) {
    const added: StoryImage[] = pack.images.map((pi) => ({
      id: crypto.randomUUID(),
      previewUrl: pi.imageSrc,
      label: pi.title || pi.prompt,
      title: pi.title,
      script: pi.script,
      action: pi.action,
      isObjectUrl: false,
    }))
    setImages((prev) => [...prev, ...added])
    if (pack.script && !script.trim()) {
      setScript(pack.script)
    }
    if (pack.templateKey && (TEMPLATES.map((t) => t.key) as string[]).includes(pack.templateKey)) {
      setTemplate(pack.templateKey as TemplateKey)
    }
  }

  function moveImage(index: number, direction: -1 | 1) {
    setImages((prev) => {
      const next = [...prev]
      const target = index + direction
      if (target < 0 || target >= next.length) return prev
      ;[next[index], next[target]] = [next[target], next[index]]
      return next
    })
  }

  function removeImage(id: string) {
    setImages((prev) => {
      const found = prev.find((img) => img.id === id)
      if (found?.isObjectUrl) URL.revokeObjectURL(found.previewUrl)
      return prev.filter((img) => img.id !== id)
    })
  }

  const [isStudioOpen, setIsStudioOpen] = useState(false)

  const estimatedSeconds = images.length * (timingOn ? sceneDuration : 3)
  const enabledPluginCount = [transitionsOn, cameraOn, animationOn, timingOn, progressionOn, captionsOn, voiceOn].filter(Boolean).length

  function handleGenerateClick() {
    if (images.length === 0) return
    setIsStudioOpen(true)
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <header className="mb-6 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-magenta to-bento-violet text-white">
          <Clapperboard size={20} />
        </div>
        <div>
          <h1 className="text-xl font-bold text-ink-primary md:text-2xl">Story to Video</h1>
          <p className="text-sm text-ink-secondary">
            Turn a sequence of images, a script, and a template into a coherent animated video.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left column: inputs */}
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>1. Images</CardTitle>
              <CardDescription>Add images in the order they should appear in the video.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={(e) => handleAddImages(e.target.files)}
              />

              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" size="sm" onClick={() => fileInputRef.current?.click()}>
                  <Plus size={14} />
                  Add images
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="secondary" size="sm">
                      <PackageOpen size={14} />
                      Load from pack
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-64">
                    {packs.length === 0 ? (
                      <div className="px-3 py-2 text-xs text-ink-muted">
                        No packs yet — select images in Image Generate and create one.
                      </div>
                    ) : (
                      packs.map((pack) => (
                        <DropdownMenuItem
                          key={pack.id}
                          onClick={() => handleLoadPack(pack)}
                          className="flex items-center justify-between gap-2"
                        >
                          <span className="truncate">{pack.name}</span>
                          <span className="flex-none text-xs text-ink-muted">
                            {pack.images.length} img{pack.images.length === 1 ? '' : 's'}
                          </span>
                        </DropdownMenuItem>
                      ))
                    )}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {images.length === 0 ? (
                <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed py-8 text-ink-muted">
                  <Plus size={22} />
                  <span className="text-sm">No images yet — add files or load a pack</span>
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  {images.map((img, i) => (
                    <div key={img.id} className="flex flex-col gap-2 rounded-xl border p-2.5 bg-page">
                      <div className="flex items-center gap-3">
                        <GripVertical size={16} className="flex-none text-ink-muted" />
                        <img src={img.previewUrl} alt="" className="h-14 w-14 flex-none rounded-lg object-cover border" />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-semibold text-ink-primary">
                            Scene {i + 1} — {img.title || img.label}
                          </p>
                          <p className="text-[11px] text-ink-muted">{img.isObjectUrl ? 'Uploaded image' : 'From template pack'}</p>
                        </div>
                        <div className="flex flex-none items-center gap-1">
                          <button
                            type="button"
                            disabled={i === 0}
                            onClick={() => moveImage(i, -1)}
                            className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-secondary transition-colors hover:bg-black/[0.04] disabled:opacity-30 dark:hover:bg-white/[0.06] cursor-pointer"
                            title="Move earlier"
                          >
                            ↑
                          </button>
                          <button
                            type="button"
                            disabled={i === images.length - 1}
                            onClick={() => moveImage(i, 1)}
                            className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-secondary transition-colors hover:bg-black/[0.04] disabled:opacity-30 dark:hover:bg-white/[0.06] cursor-pointer"
                            title="Move later"
                          >
                            ↓
                          </button>
                          <button
                            type="button"
                            onClick={() => removeImage(img.id)}
                            className="flex h-7 w-7 items-center justify-center rounded-lg text-ink-secondary transition-colors hover:bg-status-critical/10 hover:text-status-critical cursor-pointer"
                            title="Remove"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </div>

                      {(img.script || img.action) && (
                        <div className="mt-1 flex flex-col gap-1 rounded-lg bg-surface p-2 text-xs border border-dashed">
                          {img.script && (
                            <p className="text-ink-primary leading-tight">
                              <span className="font-semibold text-bento-violet">Script: </span>
                              {img.script}
                            </p>
                          )}
                          {img.action && (
                            <p className="text-ink-secondary leading-tight">
                              <span className="font-semibold text-bento-blue">Video Action: </span>
                              {img.action}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>2. Storyline / Script</CardTitle>
              <CardDescription>Describe how the story unfolds across your images.</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={script}
                onChange={(e) => setScript(e.target.value)}
                placeholder="e.g. Scene 1: the hero sets out at dawn. Scene 2: a storm rolls in..."
                rows={5}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>3. Video Template</CardTitle>
              <CardDescription>Pick a starting style — every plugin below can still be customized.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {TEMPLATES.map((t) => (
                  <button
                    key={t.key}
                    type="button"
                    onClick={() => setTemplate(t.key)}
                    className={cn(
                      'rounded-xl border px-3 py-2.5 text-center text-xs font-medium transition-colors',
                      template === t.key
                        ? 'border-bento-blue bg-bento-blue/10 text-bento-blue'
                        : 'text-ink-secondary hover:bg-black/[0.03] dark:hover:bg-white/[0.05]',
                    )}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right column: plugins + summary */}
        <div className="flex flex-col gap-6">
          <Card className="border-bento-blue/40 bg-bento-blue/5 shadow-sm">
            <CardContent className="flex flex-col gap-3 p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-bold text-ink-primary">
                    {images.length} scene{images.length === 1 ? '' : 's'} ready
                  </p>
                  <p className="text-xs text-ink-secondary">
                    ~{estimatedSeconds}s estimated · {enabledPluginCount}/7 plugins active · {template.toUpperCase()}
                  </p>
                </div>
                <Button
                  size="default"
                  onClick={handleGenerateClick}
                  disabled={images.length === 0}
                  className="bg-gradient-to-r from-bento-violet via-bento-magenta to-bento-blue text-white hover:opacity-90 font-semibold shadow-md cursor-pointer"
                >
                  <Wand2 size={16} />
                  Generate &amp; Play Video
                </Button>
              </div>
            </CardContent>
          </Card>

          <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">Plugins — enable and customize each stage</p>

          <PluginCard
            icon={<Film size={16} />}
            accent="#2a78d6"
            title="Scene-to-Scene Transitions"
            description="How each scene blends into the next."
            enabled={transitionsOn}
            onToggle={setTransitionsOn}
          >
            <OptionPills options={TRANSITIONS} value={transitionType} onChange={setTransitionType} />
            <LabeledSlider label="Duration" value={transitionDuration} onChange={setTransitionDuration} min={0.2} max={2} step={0.1} suffix="s" />
          </PluginCard>

          <PluginCard
            icon={<Camera size={16} />}
            accent="#1baf7a"
            title="Camera Movement & Zoom"
            description="Ken Burns–style pan and zoom applied to still images."
            enabled={cameraOn}
            onToggle={setCameraOn}
          >
            <OptionPills options={CAMERA_MOVES} value={cameraMove} onChange={setCameraMove} />
            <LabeledSlider label="Intensity" value={cameraIntensity} onChange={setCameraIntensity} min={1} max={10} />
          </PluginCard>

          <PluginCard
            icon={<Sparkles size={16} />}
            accent="#eda100"
            title="Image Animation"
            description="Motion effects layered on top of each still image."
            enabled={animationOn}
            onToggle={setAnimationOn}
          >
            <OptionPills options={ANIMATION_STYLES} value={animationStyle} onChange={setAnimationStyle} />
            <LabeledSlider label="Intensity" value={animationIntensity} onChange={setAnimationIntensity} min={1} max={10} />
          </PluginCard>

          <PluginCard
            icon={<Timer size={16} />}
            accent="#e34948"
            title="Scene Timing"
            description="How long each scene stays on screen."
            enabled={timingOn}
            onToggle={setTimingOn}
          >
            <LabeledSlider label="Seconds per scene" value={sceneDuration} onChange={setSceneDuration} min={1} max={10} suffix="s" />
          </PluginCard>

          <PluginCard
            icon={<Workflow size={16} />}
            accent="#4a3aa7"
            title="Story Progression"
            description="The narrative arc and pacing across scenes."
            enabled={progressionOn}
            onToggle={setProgressionOn}
          >
            <OptionPills options={NARRATIVE_STRUCTURES} value={narrative} onChange={setNarrative} />
            <OptionPills options={PACING_OPTIONS} value={pacing} onChange={setPacing} />
          </PluginCard>

          <PluginCard
            icon={<Captions size={16} />}
            accent="#e87ba4"
            title="Text / Captions"
            description="On-screen text overlaid on top of scenes."
            enabled={captionsOn}
            onToggle={setCaptionsOn}
          >
            <OptionPills options={CAPTION_STYLES} value={captionStyle} onChange={setCaptionStyle} />
            <OptionPills options={CAPTION_POSITIONS} value={captionPosition} onChange={setCaptionPosition} />
          </PluginCard>

          <PluginCard
            icon={<Music2 size={16} />}
            accent="#199e70"
            title="Voice / Music"
            description="Narration and background music tracks."
            enabled={voiceOn}
            onToggle={setVoiceOn}
          >
            <div className="flex flex-col gap-2">
              <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">Narration</p>
              {narrationFile ? (
                <div className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm text-ink-primary">
                  <span className="truncate">{narrationFile.name}</span>
                  <button type="button" onClick={() => setNarrationFile(null)} className="text-ink-muted hover:text-status-critical">
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed py-2 text-xs text-ink-muted hover:bg-black/[0.02] dark:hover:bg-white/[0.03]">
                  <Plus size={14} />
                  Upload narration audio
                  <input type="file" accept="audio/*" className="hidden" onChange={(e) => setNarrationFile(e.target.files?.[0] ?? null)} />
                </label>
              )}
            </div>
            <div className="flex flex-col gap-2">
              <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">Background music</p>
              {musicFile ? (
                <div className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm text-ink-primary">
                  <span className="truncate">{musicFile.name}</span>
                  <button type="button" onClick={() => setMusicFile(null)} className="text-ink-muted hover:text-status-critical">
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed py-2 text-xs text-ink-muted hover:bg-black/[0.02] dark:hover:bg-white/[0.03]">
                  <Plus size={14} />
                  Upload background music
                  <input type="file" accept="audio/*" className="hidden" onChange={(e) => setMusicFile(e.target.files?.[0] ?? null)} />
                </label>
              )}
            </div>
            <LabeledSlider label="Music volume" value={musicVolume} onChange={setMusicVolume} min={0} max={100} suffix="%" />
          </PluginCard>
        </div>
      </div>

      {/* Real-time Video Studio & Player Modal */}
      <VideoPlayerModal
        isOpen={isStudioOpen}
        onClose={() => setIsStudioOpen(false)}
        images={images}
        script={script}
        templateKey={template}
        transitionsOn={transitionsOn}
        transitionType={transitionType}
        transitionDuration={transitionDuration}
        cameraOn={cameraOn}
        cameraMove={cameraMove}
        cameraIntensity={cameraIntensity}
        animationOn={animationOn}
        animationStyle={animationStyle}
        animationIntensity={animationIntensity}
        timingOn={timingOn}
        sceneDuration={sceneDuration}
        captionsOn={captionsOn}
        captionStyle={captionStyle}
        captionPosition={captionPosition}
        voiceOn={voiceOn}
        musicVolume={musicVolume}
        narrationFile={narrationFile}
        musicFile={musicFile}
      />
    </div>
  )
}
