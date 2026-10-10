import { useEffect, useState } from 'react'
import { Sidebar } from '@/components/Sidebar'
import { GenerateView } from '@/components/views/GenerateView'
import { GalleryView } from '@/components/views/GalleryView'
import { HistoryView } from '@/components/views/HistoryView'
import { PredictiveAIView } from '@/components/views/PredictiveAIView'
import { StoryToVideoView } from '@/components/views/StoryToVideoView'
import { HowItWorksView } from '@/components/views/HowItWorksView'
import { SettingsView } from '@/components/views/SettingsView'
import { AboutView } from '@/components/views/AboutView'
import { CloudLearningView } from '@/components/views/CloudLearningView'
import { SecurityLearningView } from '@/components/views/SecurityLearningView'
import { applyTheme, getInitialTheme, type Theme } from '@/lib/theme'
import { predictNextAccuracy, scoreGeneration, type SettingsRecommendation } from '@/lib/predictive'
import {
  IMAGE_SIZES,
  type GenerationRecord,
  type HealthInfo,
  type ImagePack,
  type ImageSizeKey,
  type ModelsInfo,
  type PackedImage,
  type ViewKey,
} from '@/lib/types'

type GenerationState = 'idle' | 'loading' | 'error'

export default function App() {
  const [view, setView] = useState<ViewKey>('generate')
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  const [prompt, setPrompt] = useState('a watercolor painting of a fox in a snowy forest')
  const [imageSizeKey, setImageSizeKey] = useState<ImageSizeKey>('square')
  const [steps, setSteps] = useState(1)
  const [guidanceScale, setGuidanceScale] = useState(0)

  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [models, setModels] = useState<ModelsInfo | null>(null)
  const [modelId, setModelId] = useState<string>('')
  const [status, setStatus] = useState<GenerationState>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const [generations, setGenerations] = useState<GenerationRecord[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [packs, setPacks] = useState<ImagePack[]>([])

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data: HealthInfo) => {
        setHealth(data)
        setSteps(data.default_steps)
        setGuidanceScale(data.default_guidance_scale)
      })
      .catch(() => setHealth(null))

    fetch('/api/models')
      .then((res) => res.json())
      .then((data: ModelsInfo) => {
        setModels(data)
        setModelId(data.current)
      })
      .catch(() => setModels(null))
  }, [])

  async function handleGenerate(customPrompt?: string): Promise<GenerationRecord | null> {
    const trimmed = (customPrompt !== undefined ? customPrompt : prompt).trim()
    if (!trimmed) return null

    setStatus('loading')
    setErrorMessage('')

    const { width, height } = IMAGE_SIZES[imageSizeKey]

    // Captured from history as it exists right now, before this generation
    // exists — so it's a real "before" prediction to compare against the
    // "after" outcome, not one computed with hindsight.
    const predictedAccuracy =
      predictNextAccuracy(generations, {
        modelId: modelId || models?.current || '',
        steps,
        guidanceScale,
        width,
        height,
        prompt: trimmed,
      })?.breakdown.overall ?? null

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: trimmed,
          steps,
          guidance_scale: guidanceScale,
          width,
          height,
          model_id: modelId || undefined,
        }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        const detail = Array.isArray(body?.detail)
          ? body.detail.map((d: { msg?: string }) => d.msg).join(', ')
          : body?.detail
        throw new Error(detail || `Request failed with status ${res.status}`)
      }

      const data = await res.json()
      const id = crypto.randomUUID()
      const record: GenerationRecord = {
        id,
        prompt: trimmed,
        imageSrc: `data:image/png;base64,${data.image_base64}`,
        steps,
        guidanceScale,
        width: data.width,
        height: data.height,
        seconds: data.seconds,
        createdAt: Date.now(),
        modelId: data.model_id,
        accuracy: scoreGeneration({
          modelId: data.model_id,
          steps,
          guidanceScale,
          width: data.width,
          height: data.height,
          prompt: trimmed,
          seed: id,
        }),
        predictedAccuracy,
      }
      setGenerations((prev) => [record, ...prev])
      setSelectedId(record.id)
      setStatus('idle')
      // Reflect the model actually used (in case it just switched) in
      // Settings/stat tiles without waiting for another /api/health poll.
      setHealth((prev) => (prev ? { ...prev, model_id: data.model_id } : prev))
      setModels((prev) => (prev ? { ...prev, current: data.model_id } : prev))
      return record
    } catch (err) {
      setStatus('error')
      setErrorMessage((err as Error).message)
      return null
    }
  }

  function handleNewPrompt() {
    setPrompt('')
    setSelectedId(null)
    setStatus('idle')
    setErrorMessage('')
  }

  function handleApplyRecommended(rec: SettingsRecommendation) {
    setModelId(rec.modelId)
    setSteps(rec.steps)
    setGuidanceScale(rec.guidanceScale)
    const matchingSize = (Object.entries(IMAGE_SIZES) as [ImageSizeKey, (typeof IMAGE_SIZES)[ImageSizeKey]][]).find(
      ([, size]) => size.width === rec.width && size.height === rec.height,
    )
    if (matchingSize) setImageSizeKey(matchingSize[0])
    setView('generate')
  }

  function handleDeleteGeneration(id: string) {
    setGenerations((prev) => {
      const next = prev.filter((g) => g.id !== id)
      if (selectedId === id) {
        setSelectedId(next[0]?.id ?? null)
      }
      return next
    })
  }

  function handleCreatePack(
    name: string,
    recordsOrPacked: (GenerationRecord | PackedImage)[],
    options?: { script?: string; templateKey?: string },
  ) {
    const images: PackedImage[] = recordsOrPacked.map((item) => {
      const packed = item as PackedImage
      return {
        id: item.id,
        imageSrc: item.imageSrc,
        prompt: item.prompt,
        title: packed.title,
        script: packed.script,
        action: packed.action,
      }
    })

    const pack: ImagePack = {
      id: crypto.randomUUID(),
      name,
      createdAt: Date.now(),
      script: options?.script,
      templateKey: options?.templateKey,
      images,
    }
    setPacks((prev) => [pack, ...prev])
  }

  return (
    <div className="flex h-screen bg-page">
      <Sidebar view={view} onNavigate={setView} theme={theme} onThemeChange={setTheme} />

      {view === 'generate' && (
        <GenerateView
          prompt={prompt}
          onPromptChange={setPrompt}
          imageSizeKey={imageSizeKey}
          onImageSizeChange={setImageSizeKey}
          steps={steps}
          onStepsChange={setSteps}
          guidanceScale={guidanceScale}
          onGuidanceScaleChange={setGuidanceScale}
          models={models}
          modelId={modelId}
          onModelChange={setModelId}
          health={health}
          status={status}
          errorMessage={errorMessage}
          generations={generations}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onGenerate={handleGenerate}
          onNewPrompt={handleNewPrompt}
          onOpenAbout={() => setView('about')}
          onCreatePack={handleCreatePack}
          onNavigateToStoryVideo={() => setView('story-video')}
        />
      )}
      {view === 'gallery' && (
        <GalleryView
          generations={generations}
          onSelect={(id) => {
            setSelectedId(id)
            setView('generate')
          }}
          onDelete={handleDeleteGeneration}
        />
      )}
      {view === 'history' && (
        <HistoryView
          generations={generations}
          onSelect={(id) => {
            setSelectedId(id)
            setView('generate')
          }}
          onDelete={handleDeleteGeneration}
        />
      )}
      {view === 'predictive' && (
        <PredictiveAIView
          generations={generations}
          models={models}
          draftPrompt={prompt}
          draftModelId={modelId || models?.current || ''}
          draftSteps={steps}
          draftGuidanceScale={guidanceScale}
          draftWidth={IMAGE_SIZES[imageSizeKey].width}
          draftHeight={IMAGE_SIZES[imageSizeKey].height}
          onApplyRecommended={handleApplyRecommended}
          onOpenGenerate={() => setView('generate')}
        />
      )}
      {view === 'story-video' && <StoryToVideoView packs={packs} />}
      {view === 'how-it-works' && <HowItWorksView />}
      {view === 'settings' && <SettingsView health={health} theme={theme} onThemeChange={setTheme} />}
      {view === 'cloud-learning' && <CloudLearningView />}
      {view === 'security-learning' && <SecurityLearningView />}
      {view === 'about' && <AboutView />}
    </div>
  )
}
