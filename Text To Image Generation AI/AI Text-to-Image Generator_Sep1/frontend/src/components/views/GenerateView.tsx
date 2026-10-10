import { useRef, useState } from 'react'
import {
  ArrowRight,
  Bot,
  Camera,
  Check,
  ChevronDown,
  Clapperboard,
  Copy,
  Cpu,
  Download,
  Eraser,
  FileText,
  HelpCircle,
  ImageIcon,
  Layers,
  MoreVertical,
  PackageCheck,
  PackagePlus,
  Play,
  Plus,
  Share2,
  Sparkles,
  Timer,
  Trash2,
  TriangleAlert,
  Wand2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { StatTile } from '@/components/StatTile'
import { enhancePrompt } from '@/lib/enhancePrompt'
import {
  IMAGE_SIZES,
  type GenerateMode,
  type GenerationRecord,
  type HealthInfo,
  type ImageSizeKey,
  type ModelsInfo,
  type PackedImage,
  type VideoTemplatePreset,
} from '@/lib/types'
import { cn } from '@/lib/utils'

type GenerationState = 'idle' | 'loading' | 'error'

const VIDEO_TEMPLATE_PRESETS: VideoTemplatePreset[] = [
  {
    id: 'cinematic-epic',
    name: 'Cinematic Epic',
    description: 'Heroic storyline with dramatic lighting, sweeping vistas, and atmospheric camera moves',
    videoStyle: 'Cinematic',
    templateKey: 'cinematic',
    suggestedScript:
      'Act I: The first light of dawn strikes the ancient fortress as slumbering bells awake the kingdom.\n\nAct II: Armed with ancestral courage, the traveler ventures deep into the glowing enchanted forest.\n\nAct III: Thunder shatters the heavens as two destinies collide at the edge of storm peak.\n\nAct IV: The tempest clears, and peace is restored across the golden realm.',
    scenes: [
      {
        title: 'Scene 1: Dawn at the Fortress',
        script: 'The first rays of sunlight strike the towering stone ramparts of the ancient fortress, awakening the slumbering kingdom.',
        action: 'Slow aerial crane down towards main gate, volumetric fog drift, 4.0s hold, slow fade-in transition',
        prompt:
          'cinematic aerial establishing shot of an ancient towering stone mountain fortress at sunrise, golden sunbeams piercing misty valley, photorealistic, 8k, majestic scale',
      },
      {
        title: 'Scene 2: Into the Whispering Forest',
        script: 'Grasping his grandfather’s blade, the hero leaves the walls behind and ventures deep into the enchanted woods.',
        action: 'Tracking medium shot following hero forward, subtle parallax depth on glowing tree trunks, 3.5s, wipe transition',
        prompt:
          'heroic traveler in weathered cloaked armor walking away into glowing enchanted ancient forest, bioluminescent moss, beams of light, cinematic depth of field',
      },
      {
        title: 'Scene 3: Climax at Storm Peak',
        script: 'At the edge of the world, thunder crashes as the dark tempest gathers for the ultimate confrontation.',
        action: 'Dynamic low-angle camera tilt upward, fast kinetic zoom burst with simulated lightning flash, 3.0s, zoom blur transition',
        prompt:
          'dramatic wide angle shot of two warriors clashing on cliff edge under dark purple lightning storm clouds, rain flying, intense atmospheric lighting, masterwork',
      },
      {
        title: 'Scene 4: The Golden Horizon',
        script: 'The storm clears. Gazing upon the vast reclaimed lands, the warrior stands victorious under the golden evening sky.',
        action: 'Slow push-in zoom with gentle pan right across horizon, warm sunset lens flare, 4.0s, slow fade to black',
        prompt:
          'triumphant warrior in silhouette standing on high mountain peak looking out over vast sunny valley at golden sunset, warm glow, masterpiece composition',
      },
    ],
  },
  {
    id: 'cyberpunk-odyssey',
    name: 'Cyberpunk Odyssey',
    description: 'Futuristic sci-fi world with neon reflections, holographic nodes, and high-speed motion',
    videoStyle: 'Social Reel',
    templateKey: 'social-reel',
    suggestedScript:
      'Scene 1: Neon rain slicks the pavement of Neo-Shinjuku as holographic ads tower into the smog.\n\nScene 2: In a hidden mainframe, the neural deck breaches government firewalls.\n\nScene 3: Sirens blare as high-speed cyber-bikes rip through the neon underbelly.\n\nScene 4: High above the city on a rain-soaked ledge, the stolen data is broadcast to the world.',
    scenes: [
      {
        title: 'Scene 1: Neon Megacity Rain',
        script: 'Tears in the rain. Towering neon spires pierce the heavy smog of Neo-Shinjuku at midnight.',
        action: 'Dolly shot drifting past neon sign reflections in wet street puddles, subtle chromatic aberration, 3.0s, fast cut transition',
        prompt:
          'cyberpunk metropolis street at night in rain, towering holographic billboards, neon purple and cyan reflections on wet asphalt, flying hovercars, octane render 8k',
      },
      {
        title: 'Scene 2: The Neural Breach',
        script: 'Fingers dance across holographic nodes as the neural decryptor cracks the mainframe.',
        action: 'Extreme close-up fast zoom into eyes with glowing holographic interface reflections, pulsing glow animation, 3.0s, digital glitch transition',
        prompt:
          'cyberpunk female netrunner with cybernetic eye glowing blue, interacting with floating translucent data screens, dark server room, volumetric smoke, high detail',
      },
      {
        title: 'Scene 3: Lower District Pursuit',
        script: 'Red alert triggers. A modified light-cycle tears down the narrow neon market corridors.',
        action: 'High-speed tracking side pan with neon light trails, motion blur burst, 2.5s, slide transition',
        prompt:
          'futuristic sleek cyber motorcycle racing through crowded underground neon market alleyway, neon light trails, motion blur, cinematic dynamic framing, 8k',
      },
      {
        title: 'Scene 4: Skyline Transmission',
        script: 'On the rooftop overlook, the transmitter links up. The truth is finally free.',
        action: 'Slow wide pull back revealing the entire sprawling glowing megacity horizon, 4.0s, fade out',
        prompt:
          'solitary cyber runner on skyscraper ledge holding glowing transmitter antenna overlooking endless illuminated neon metropolis at night, dramatic cyberpunk view',
      },
    ],
  },
  {
    id: 'nature-documentary',
    name: 'Nature Documentary',
    description: 'Stunning wildlife & pristine landscapes with smooth cinematic drone & telephoto perspectives',
    videoStyle: 'Documentary',
    templateKey: 'documentary',
    suggestedScript:
      'Scene 1: Morning mist lifts from the primeval rainforest canopy as birds take flight.\n\nScene 2: Across the golden savanna, a lion pride begins their morning patrol.\n\nScene 3: Deep in the canyon, roaring glacial waters carve through ancient granite cliffs.\n\nScene 4: Under the celestial brilliance of the Milky Way, the wilderness rests in silent wonder.',
    scenes: [
      {
        title: 'Scene 1: Rainforest Canopy Sunrise',
        script: 'Dawn breaks over the emerald canopy of the Amazon basin, lifting the veil of night with golden light.',
        action: 'Smooth drone glide over tree canopy, rising mist layer animation, 4.0s, crossfade transition',
        prompt:
          'lush tropical rainforest canopy viewed from above at sunrise, golden sun rays breaking through morning mist, exotic birds flying, National Geographic 8k photography',
      },
      {
        title: 'Scene 2: Pride of the Savanna',
        script: 'Across the vast Serengeti plains, the pride moves with silent grace under the awakening sun.',
        action: 'Slow telephoto pan left following the lions through tall grass, heat shimmer effect, 3.5s, cut transition',
        prompt:
          'majestic lion pride moving across golden African savanna tall grass at dawn, warm morning sunlight, crisp wildlife telephoto photography, ultra high definition',
      },
      {
        title: 'Scene 3: Cascades of the Canyon',
        script: 'Glacial meltwaters plunge hundreds of feet into turquoise lagoons, a testament to nature’s raw power.',
        action: 'Vertical camera tilt down from mountain peaks to roaring waterfall basin, water mist dispersion, 3.5s, wipe transition',
        prompt:
          'huge powerful waterfall roaring down mossy volcanic cliff into crystal clear turquoise pool, rainbow mist, lush fern vegetation, award winning landscape photo',
      },
      {
        title: 'Scene 4: Starlit Wilderness',
        script: 'As night falls, the infinite cosmos reflects upon the tranquil mirror of the alpine lake.',
        action: 'Static long-exposure timelapse rotation of star trails reflecting on still water, 4.5s, slow fade to black',
        prompt:
          'milky way galaxy stretching across starry night sky perfectly reflected in calm mirror-like mountain lake, pristine wilderness, long exposure astrophotography',
      },
    ],
  },
  {
    id: 'social-reel',
    name: 'Social Media Reel',
    description: 'Vibrant, fast-paced commercial aesthetics made for mobile reels and product showcases',
    videoStyle: 'Social Reel',
    templateKey: 'social-reel',
    suggestedScript:
      'Scene 1: Hook the viewer with an ultra-clean, aesthetic studio macro product reveal.\n\nScene 2: Showcase dynamic real-world lifestyle utility in a modern environment.\n\nScene 3: Deliver the final high-energy visual call-to-action.',
    scenes: [
      {
        title: 'Scene 1: The Aesthetic Hook',
        script: 'Stop scrolling — meet the design that changes everything.',
        action: 'Fast 360 rotation snap zoom into product surface with soft studio bounce light, 2.0s, quick whip-pan transition',
        prompt:
          'modern aesthetic luxury product macro shot, floating against soft pastel peach and lavender gradient background, studio rim lighting, 8k commercial quality',
      },
      {
        title: 'Scene 2: In-Use Lifestyle',
        script: 'Crafted for creators who demand performance anywhere.',
        action: 'Medium lifestyle tracking pan right, warm natural sunlight flare, 2.5s, slide transition',
        prompt:
          'stylish creator holding sleek modern gadget in sunlit modern plant-filled cafe, natural candid smile, organic warm lifestyle photography, high resolution',
      },
      {
        title: 'Scene 3: Minimal Flatlay Call-to-Action',
        script: 'Upgrade your workflow today. Link in bio.',
        action: 'Top-down kinetic punch-in zoom, clean typographic overlay space, 3.0s, bounce finish',
        prompt:
          'aesthetic organized flatlay on white Italian marble desk, notebook, coffee cup, modern gadget, soft morning shadow, Pinterest aesthetic, minimalist 8k',
      },
    ],
  },
  {
    id: 'anime-adventure',
    name: 'Anime Adventure',
    description: 'Vivid Makoto Shinkai & Ghibli-inspired art with emotional story arcs and floating particles',
    videoStyle: 'Music Video',
    templateKey: 'music-video',
    suggestedScript:
      'Scene 1: Cherry blossom petals swirl across the quiet countryside train platform.\n\nScene 2: An ancient glowing crystal pulses with forgotten magical energy.\n\nScene 3: The clouds part to reveal a legendary floating castle island in the sky.\n\nScene 4: Thousand glowing lanterns ascend into the twilight sky at the summer festival.',
    scenes: [
      {
        title: 'Scene 1: Sakura Station Dawn',
        script: 'The spring wind carries soft cherry petals across the lonely station platform as the morning express arrives.',
        action: 'Pan right along train tracks with floating sakura petal particles in wind, 3.5s, soft fade transition',
        prompt:
          'anime style scenic Japanese rural train platform in spring, pink cherry blossom petals swirling in wind, cumulus clouds in bright blue sky, Makoto Shinkai style art',
      },
      {
        title: 'Scene 2: The Crystal Awakens',
        script: 'Held aloft in trembling hands, the ancestral pendant ignites with brilliant celestial starlight.',
        action: 'Close-up slow zoom in on glowing gem with radial sparkle emission and hair wind motion, 3.0s, flash transition',
        prompt:
          'anime protagonist holding glowing magical sapphire crystal radiating bright blue sparkle aura, wind blowing hair, vivid saturated colors, Studio Ghibli inspired',
      },
      {
        title: 'Scene 3: Sky Kingdom Laputa',
        script: 'Bursting through the thunderheads, they behold the mythical floating fortress hovering above the world.',
        action: 'Grand upward crane reveal through parting clouds showing massive floating island, 4.0s, slide transition',
        prompt:
          'breathtaking anime scenery of massive floating castle island surrounded by sea of white clouds, waterfalls pouring into sky, airships flying, fantasy anime concept art',
      },
      {
        title: 'Scene 4: Night of Ten Thousand Lanterns',
        script: 'Together by the shrine steps, they watch countless dreams float up into the starlit twilight.',
        action: 'Slow upward tilt following glowing lanterns rising into the deep violet starry sky, 4.0s, fade out',
        prompt:
          'traditional Japanese anime festival at dusk, thousands of warm glowing paper lanterns floating into purple starry night sky, magical romantic atmosphere, masterpiece',
      },
    ],
  },
  {
    id: 'custom',
    name: 'Custom Storyboard',
    description: 'Create your own bespoke multi-scene sequence with customizable prompts, scripts, and video actions',
    videoStyle: 'Cinematic',
    templateKey: 'cinematic',
    suggestedScript: 'Scene 1: Introduction / Establishing.\n\nScene 2: Conflict / Escalation.\n\nScene 3: Resolution / Ending.',
    scenes: [
      {
        title: 'Scene 1: Establishing Shot',
        script: 'Introduce the environment and set the tone for the journey.',
        action: 'Slow wide establishing push-in, 3.5s, fade-in transition',
        prompt: 'wide cinematic establishing shot of a picturesque landscape, atmospheric lighting, high detail, 8k',
      },
      {
        title: 'Scene 2: Dynamic Action',
        script: 'The pivotal moment where the central conflict or development unfolds.',
        action: 'Dynamic tracking camera move with focus pull, 3.0s, cut transition',
        prompt: 'close dynamic shot of the central subject with intense dramatic lighting and mood, cinematic framing',
      },
      {
        title: 'Scene 3: Final Resolution',
        script: 'The concluding thought that leaves a lasting emotional impact.',
        action: 'Slow pullback to wide shot, warm golden hour glow, 4.0s, fade to black',
        prompt: 'poetic closing frame with golden hour sunlight and serene composition, masterpiece quality',
      },
    ],
  },
]

interface TemplateSceneItem {
  id: string
  title: string
  prompt: string
  script: string
  action: string
  record?: GenerationRecord | null
}

export function GenerateView({
  prompt,
  onPromptChange,
  imageSizeKey,
  onImageSizeChange,
  steps,
  onStepsChange,
  guidanceScale,
  onGuidanceScaleChange,
  models,
  modelId,
  onModelChange,
  health,
  status,
  errorMessage,
  generations,
  selectedId,
  onSelect,
  onGenerate,
  onNewPrompt,
  onOpenAbout,
  onCreatePack,
  onNavigateToStoryVideo,
}: {
  prompt: string
  onPromptChange: (value: string) => void
  imageSizeKey: ImageSizeKey
  onImageSizeChange: (key: ImageSizeKey) => void
  steps: number
  onStepsChange: (value: number) => void
  guidanceScale: number
  onGuidanceScaleChange: (value: number) => void
  models: ModelsInfo | null
  modelId: string
  onModelChange: (id: string) => void
  health: HealthInfo | null
  status: GenerationState
  errorMessage: string
  generations: GenerationRecord[]
  selectedId: string | null
  onSelect: (id: string) => void
  onGenerate: (customPrompt?: string) => Promise<GenerationRecord | null>
  onNewPrompt: () => void
  onOpenAbout: () => void
  onCreatePack: (
    name: string,
    recordsOrPacked: (GenerationRecord | PackedImage)[],
    options?: { script?: string; templateKey?: string },
  ) => void
  onNavigateToStoryVideo?: () => void
}) {
  const [mode, setMode] = useState<GenerateMode>('image')
  const switchingModel = models !== null && modelId !== '' && modelId !== models.current
  const downloadRef = useRef<HTMLAnchorElement>(null)
  const selected = generations.find((g) => g.id === selectedId) ?? null

  // Template generation state
  const [selectedPresetId, setSelectedPresetId] = useState<string>('cinematic-epic')
  const [templateScenes, setTemplateScenes] = useState<TemplateSceneItem[]>(() =>
    VIDEO_TEMPLATE_PRESETS[0].scenes.map((s) => ({
      id: crypto.randomUUID(),
      title: s.title,
      prompt: s.prompt,
      script: s.script,
      action: s.action,
      record: null,
    })),
  )
  const [activeInspectorIndex, setActiveInspectorIndex] = useState<number>(0)
  const [generatingSceneIndex, setGeneratingSceneIndex] = useState<number | null>(null)
  const [selectedTemplateForPack, setSelectedTemplateForPack] = useState<Set<string>>(new Set())
  const [packDialogOpen, setPackDialogOpen] = useState(false)
  const [packName, setPackName] = useState('')
  const [packNotice, setPackNotice] = useState('')
  const [lastCreatedPackName, setLastCreatedPackName] = useState<string | null>(null)

  function handleSelectPreset(preset: VideoTemplatePreset) {
    setSelectedPresetId(preset.id)
    setTemplateScenes(
      preset.scenes.map((s) => ({
        id: crypto.randomUUID(),
        title: s.title,
        prompt: s.prompt,
        script: s.script,
        action: s.action,
        record: null,
      })),
    )
    setActiveInspectorIndex(0)
    setSelectedTemplateForPack(new Set())
  }

  function handleUpdateScenePrompt(index: number, newPrompt: string) {
    setTemplateScenes((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], prompt: newPrompt }
      return next
    })
  }

  function handleUpdateSceneScript(index: number, newScript: string) {
    setTemplateScenes((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], script: newScript }
      return next
    })
  }

  function handleUpdateSceneAction(index: number, newAction: string) {
    setTemplateScenes((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], action: newAction }
      return next
    })
  }

  function handleUpdateSceneTitle(index: number, newTitle: string) {
    setTemplateScenes((prev) => {
      const next = [...prev]
      next[index] = { ...next[index], title: newTitle }
      return next
    })
  }

  function handleAddScene() {
    setTemplateScenes((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        title: `Scene ${prev.length + 1}: Continuation`,
        script: 'Describe the story action unfolding in this scene...',
        action: 'Medium camera tracking pan, 3.5s hold, dissolve transition',
        prompt: 'cinematic establishing shot with dramatic lighting, 8k resolution',
        record: null,
      },
    ])
  }

  function handleRemoveScene(index: number) {
    setTemplateScenes((prev) => prev.filter((_, i) => i !== index))
    if (activeInspectorIndex >= templateScenes.length - 1 && activeInspectorIndex > 0) {
      setActiveInspectorIndex(activeInspectorIndex - 1)
    }
  }

  async function handleGenerateSingleScene(index: number) {
    const scene = templateScenes[index]
    if (!scene || !scene.prompt.trim()) return
    setGeneratingSceneIndex(index)
    setActiveInspectorIndex(index)
    const rec = await onGenerate(scene.prompt)
    if (rec) {
      setTemplateScenes((prev) => {
        const next = [...prev]
        next[index] = { ...next[index], record: rec }
        return next
      })
      setSelectedTemplateForPack((prev) => new Set([...prev, rec.id]))
    }
    setGeneratingSceneIndex(null)
  }

  async function handleGenerateAllTemplateScenes() {
    for (let i = 0; i < templateScenes.length; i++) {
      setGeneratingSceneIndex(i)
      setActiveInspectorIndex(i)
      const scene = templateScenes[i]
      if (scene.prompt.trim()) {
        const rec = await onGenerate(scene.prompt)
        if (rec) {
          setTemplateScenes((prev) => {
            const next = [...prev]
            next[i] = { ...next[i], record: rec }
            return next
          })
          setSelectedTemplateForPack((prev) => new Set([...prev, rec.id]))
        }
      }
    }
    setGeneratingSceneIndex(null)
  }

  const generatedTemplateRecords = templateScenes
    .map((s) => s.record)
    .filter((r): r is GenerationRecord => r !== null && r !== undefined)

  function toggleTemplateImageForPack(id: string) {
    setSelectedTemplateForPack((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  function handleSelectAllTemplateForPack() {
    const allIds = generatedTemplateRecords.map((r) => r.id)
    setSelectedTemplateForPack(new Set(allIds))
  }

  function handleDeselectAllTemplateForPack() {
    setSelectedTemplateForPack(new Set())
  }

  function openPackDialog(all = false) {
    if (all) {
      handleSelectAllTemplateForPack()
    }
    const currentPreset = VIDEO_TEMPLATE_PRESETS.find((p) => p.id === selectedPresetId)
    const defaultName = currentPreset ? `${currentPreset.name} Pack` : `Video Template Pack`
    setPackName(defaultName)
    setPackDialogOpen(true)
  }

  function handleConfirmCreatePack() {
    const currentPreset = VIDEO_TEMPLATE_PRESETS.find((p) => p.id === selectedPresetId)
    const scenesToPack = templateScenes
      .filter((s) => s.record && (selectedTemplateForPack.size > 0 ? selectedTemplateForPack.has(s.record.id) : true))
      .map((s) => ({
        id: s.record!.id,
        imageSrc: s.record!.imageSrc,
        prompt: s.record!.prompt,
        title: s.title,
        script: s.script,
        action: s.action,
      }))

    if (scenesToPack.length === 0) return

    const combinedScript = scenesToPack.map((s) => `${s.title}: ${s.script}`).join('\n\n')
    const name = packName.trim() || `${currentPreset?.name ?? 'Video'} Pack ${new Date().toLocaleDateString()}`

    onCreatePack(name, scenesToPack, {
      script: combinedScript,
      templateKey: currentPreset?.templateKey ?? 'cinematic',
    })

    setPackDialogOpen(false)
    setLastCreatedPackName(name)
    setPackNotice(`Pack "${name}" created with ${scenesToPack.length} video scenes, scripts & action cues ready for Story to Video!`)
    window.setTimeout(() => setPackNotice(''), 8000)
  }

  async function copyToClipboard(text: string, label = 'Copied') {
    try {
      await navigator.clipboard.writeText(text)
      window.alert(`${label} to clipboard!`)
    } catch {
      window.alert('Could not copy to clipboard.')
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      onGenerate()
    }
  }

  function handleDownload() {
    if (!selected || !downloadRef.current) return
    downloadRef.current.href = selected.imageSrc
    downloadRef.current.download = `${selected.prompt.slice(0, 40).replace(/[^a-z0-9]+/gi, '-') || 'generated-image'}.png`
    downloadRef.current.click()
  }

  async function handleShare() {
    if (!selected) return
    if (!navigator.share) {
      window.alert('Sharing isn’t supported in this browser. Use Download instead.')
      return
    }
    try {
      const res = await fetch(selected.imageSrc)
      const blob = await res.blob()
      const file = new File([blob], 'generated-image.png', { type: 'image/png' })
      await navigator.share({ files: [file], title: 'AI-generated image', text: selected.prompt })
    } catch {
      // User cancelled the share sheet
    }
  }

  async function handleCopyPrompt() {
    if (!selected) return
    copyToClipboard(selected.prompt, 'Prompt copied')
  }

  const activeInspectorScene = templateScenes[activeInspectorIndex] ?? templateScenes[0]

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <a ref={downloadRef} className="hidden" aria-hidden="true" />

      {/* Top Header with Mode Selector */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b pb-5">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-violet to-bento-blue text-white shadow-sm">
            {mode === 'image' ? <Sparkles size={20} /> : <Clapperboard size={20} />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-ink-primary md:text-2xl">
                {mode === 'image' ? 'Image Generate' : 'Generate Template'}
              </h1>
              <span className="rounded-full bg-bento-blue/10 px-2.5 py-0.5 text-xs font-semibold text-bento-blue">
                {mode === 'image' ? 'Single Visual' : 'For Story to Video'}
              </span>
            </div>
            <p className="text-sm text-ink-secondary">
              {mode === 'image'
                ? 'Generate standalone visuals directly from single descriptive prompts'
                : 'Create themed storyboard templates with script, video camera actions, and visual prompts for Story to Video'}
            </p>
          </div>
        </div>

        {/* Mode Switcher Segmented Control */}
        <div className="flex items-center gap-3">
          <div className="flex items-center rounded-xl border bg-surface p-1 shadow-sm">
            <button
              type="button"
              onClick={() => setMode('image')}
              className={cn(
                'flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all cursor-pointer',
                mode === 'image'
                  ? 'bg-bento-blue text-white shadow'
                  : 'text-ink-secondary hover:text-ink-primary hover:bg-black/[0.04] dark:hover:bg-white/[0.06]',
              )}
            >
              <Wand2 size={14} />
              Generate Image
            </button>
            <button
              type="button"
              onClick={() => setMode('template')}
              className={cn(
                'flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-all cursor-pointer',
                mode === 'template'
                  ? 'bg-bento-blue text-white shadow'
                  : 'text-ink-secondary hover:text-ink-primary hover:bg-black/[0.04] dark:hover:bg-white/[0.06]',
              )}
            >
              <Clapperboard size={14} />
              Generate Template
            </button>
          </div>

          <Button variant="secondary" size="sm" onClick={onOpenAbout} title="How this app works">
            <HelpCircle size={16} />
          </Button>
          {mode === 'image' && (
            <Button size="sm" onClick={onNewPrompt}>
              <Plus size={16} />
              New prompt
            </Button>
          )}
        </div>
      </header>

      {/* ========================================================================= */}
      {/* MODE 1: GENERATE IMAGE                                                    */}
      {/* ========================================================================= */}
      {mode === 'image' && (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Left column: prompt + settings */}
            <div className="flex flex-col gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Prompt</CardTitle>
                  <CardDescription>Describe the image you want to generate.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  <div className="relative">
                    <Textarea
                      value={prompt}
                      onChange={(e) => onPromptChange(e.target.value.slice(0, 500))}
                      onKeyDown={handleKeyDown}
                      placeholder="e.g. a neon-lit cyberpunk city street in the rain"
                      rows={6}
                    />
                    <span className="pointer-events-none absolute bottom-2 right-3 text-xs text-ink-muted">
                      {prompt.length}/500
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="secondary" size="sm" onClick={() => onPromptChange('')}>
                      <Eraser size={14} />
                      Clear
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => onPromptChange(enhancePrompt(prompt))}
                      disabled={!prompt.trim()}
                      title="Appends quality keywords locally — no extra AI call involved"
                    >
                      <Wand2 size={14} />
                      Enhance prompt
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Model &amp; settings</CardTitle>
                  <CardDescription>Steps and guidance are sent straight to the pipeline.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-5">
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-muted">Model</p>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <button
                          type="button"
                          disabled={!models}
                          className="flex w-full items-center justify-between gap-2 rounded-xl border bg-page px-3.5 py-2.5 text-left text-sm text-ink-primary transition-all hover:bg-black/[0.02] dark:hover:bg-white/[0.03] outline-none focus:ring-2 focus:ring-bento-blue/40 disabled:opacity-60 cursor-pointer"
                        >
                          <div className="flex items-center gap-2.5 min-w-0">
                            <Bot
                              size={16}
                              className={cn(
                                'flex-none',
                                modelId === 'custom-scratch-model' ? 'text-bento-violet' : 'text-bento-orange',
                              )}
                            />
                            <span className="truncate font-medium">
                              {models?.available.find((m) => m.id === modelId)?.label ?? modelId ?? 'Loading models…'}
                            </span>
                          </div>
                          <ChevronDown size={15} className="flex-none text-ink-muted" />
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent
                        align="start"
                        className="w-[340px] max-w-[90vw] p-1.5 shadow-2xl border bg-surface rounded-2xl"
                      >
                        {!models && <div className="px-3 py-2 text-xs text-ink-muted">Loading models…</div>}
                        {models?.available.map((m) => {
                          const isSelected = m.id === modelId
                          const isCustom = m.id === 'custom-scratch-model'
                          return (
                            <DropdownMenuItem
                              key={m.id}
                              onClick={() => onModelChange(m.id)}
                              className={cn(
                                'flex items-center justify-between gap-2.5 rounded-xl px-3 py-2.5 text-sm transition-colors cursor-pointer',
                                isSelected
                                  ? 'bg-bento-blue/10 font-semibold text-bento-blue dark:bg-bento-blue/20'
                                  : 'text-ink-primary hover:bg-black/[0.04] dark:hover:bg-white/[0.06]',
                              )}
                            >
                              <div className="flex items-start gap-2.5 min-w-0">
                                <Bot
                                  size={16}
                                  className={cn(
                                    'flex-none mt-0.5',
                                    isCustom
                                      ? 'text-bento-violet'
                                      : isSelected
                                      ? 'text-bento-blue'
                                      : 'text-ink-muted',
                                  )}
                                />
                                <div className="flex flex-col min-w-0 text-left">
                                  <span className="truncate leading-snug">{m.label}</span>
                                  {isCustom ? (
                                    <span className="text-[11px] font-normal text-bento-violet">
                                      Step 1: Problem Definition • From Scratch
                                    </span>
                                  ) : (
                                    <span className="text-[11px] font-normal text-ink-muted">
                                      Hugging Face Diffusers Pipeline
                                    </span>
                                  )}
                                </div>
                              </div>
                              {isSelected && <Check size={16} className="flex-none text-bento-blue" />}
                            </DropdownMenuItem>
                          )
                        })}
                      </DropdownMenuContent>
                    </DropdownMenu>
                    {switchingModel && (
                      <p className="mt-1.5 text-xs text-ink-muted">
                        Switches on your next Generate — the new model needs to load first, which can take a while.
                      </p>
                    )}
                  </div>

                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-muted">Image size</p>
                    <div className="grid grid-cols-3 gap-2">
                      {(Object.entries(IMAGE_SIZES) as [ImageSizeKey, (typeof IMAGE_SIZES)[ImageSizeKey]][]).map(
                        ([key, size]) => (
                          <button
                            key={key}
                            type="button"
                            onClick={() => onImageSizeChange(key)}
                            className={cn(
                              'rounded-xl border px-2 py-2.5 text-center text-xs font-medium transition-colors cursor-pointer',
                              imageSizeKey === key
                                ? 'border-bento-blue bg-bento-blue/10 text-bento-blue font-semibold'
                                : 'text-ink-secondary hover:bg-black/[0.03] dark:hover:bg-white/[0.05]',
                            )}
                          >
                            <div>{size.label}</div>
                            <div className="text-ink-muted">
                              {size.width}×{size.height}
                            </div>
                          </button>
                        ),
                      )}
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">Inference steps</p>
                      <span className="text-xs font-semibold text-ink-primary">{steps}</span>
                    </div>
                    <Slider min={1} max={50} step={1} value={[steps]} onValueChange={([v]) => onStepsChange(v)} />
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">Guidance scale</p>
                      <span className="text-xs font-semibold text-ink-primary">{guidanceScale.toFixed(1)}</span>
                    </div>
                    <Slider
                      min={0}
                      max={15}
                      step={0.5}
                      value={[guidanceScale]}
                      onValueChange={([v]) => onGuidanceScaleChange(v)}
                    />
                  </div>

                  <Button
                    size="lg"
                    className="w-full bg-gradient-to-r from-bento-violet via-bento-magenta to-bento-blue hover:opacity-90 cursor-pointer"
                    onClick={() => onGenerate()}
                    disabled={status === 'loading' || !prompt.trim()}
                  >
                    {status === 'loading' ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                        Generating…
                      </>
                    ) : (
                      <>
                        <Sparkles size={16} />
                        Generate image
                      </>
                    )}
                  </Button>

                  {status === 'error' && (
                    <div className="flex items-center gap-2 rounded-lg bg-status-critical/10 px-3 py-2 text-sm text-status-critical">
                      <TriangleAlert size={16} />
                      <span>{errorMessage}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Right column: result */}
            <Card className="flex flex-col">
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <div>
                  <CardTitle>Generated result</CardTitle>
                  <CardDescription>
                    {status === 'loading'
                      ? switchingModel
                        ? 'Loading the new model, then running inference…'
                        : 'Running inference…'
                      : 'Your generated image appears here.'}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="secondary" size="sm" onClick={handleDownload} disabled={!selected}>
                    <Download size={14} />
                    Download
                  </Button>
                  <Button variant="secondary" size="sm" onClick={handleShare} disabled={!selected}>
                    <Share2 size={14} />
                    Share
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="secondary" size="sm" disabled={!selected}>
                        <MoreVertical size={14} />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={handleCopyPrompt}>Copy prompt</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </CardHeader>
              <CardContent className="flex flex-1 flex-col gap-4">
                <div className="flex aspect-square w-full items-center justify-center overflow-hidden rounded-xl border bg-page">
                  {status === 'loading' ? (
                    <div className="flex flex-col items-center gap-3 text-ink-muted">
                      <div className="h-8 w-8 animate-spin rounded-full border-2 border-ink-muted border-t-bento-blue" />
                      <span className="text-xs">Generating…</span>
                    </div>
                  ) : selected ? (
                    <img src={selected.imageSrc} alt={selected.prompt} className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex flex-col items-center gap-2 text-ink-muted">
                      <ImageIcon size={28} />
                      <span className="text-xs">No image yet</span>
                    </div>
                  )}
                </div>

                {generations.length > 0 && (
                  <div className="flex flex-col gap-2">
                    <p className="text-xs font-medium uppercase tracking-wide text-ink-muted">Recent generations</p>
                    <div className="flex gap-3 overflow-x-auto pb-1">
                      {generations.map((g) => (
                        <button
                          key={g.id}
                          type="button"
                          onClick={() => onSelect(g.id)}
                          className={cn(
                            'relative h-16 w-16 flex-none overflow-hidden rounded-lg border-2 transition-all cursor-pointer',
                            g.id === selectedId
                              ? 'border-bento-blue ring-2 ring-bento-blue/30 scale-105'
                              : 'border-transparent opacity-80 hover:opacity-100',
                          )}
                          title={g.prompt}
                        >
                          <img src={g.imageSrc} alt={g.prompt} className="h-full w-full object-cover" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatTile icon={<Bot size={18} />} accent="#eb6834" label="Model" value={selected?.modelId ?? health?.model_id ?? '—'} />
            <StatTile icon={<Cpu size={18} />} accent="#1baf7a" label="Device" value={health?.device ?? '—'} />
            <StatTile icon={<Layers size={18} />} accent="#eda100" label="Inference steps" value={selected ? String(selected.steps) : String(steps)} />
            <StatTile
              icon={<Timer size={18} />}
              accent="var(--accent-violet)"
              label="Generated at"
              value={selected ? new Date(selected.createdAt).toLocaleString() : '—'}
            />
          </div>
        </>
      )}

      {/* ========================================================================= */}
      {/* MODE 2: GENERATE TEMPLATE (FOR VIDEO)                                     */}
      {/* ========================================================================= */}
      {mode === 'template' && (
        <div className="flex flex-col gap-6">
          {packNotice && (
            <div className="flex items-center justify-between rounded-xl border border-bento-blue/40 bg-bento-blue/10 p-4 text-sm font-medium text-bento-blue shadow-sm">
              <div className="flex items-center gap-2">
                <PackageCheck size={18} />
                <span>{packNotice}</span>
              </div>
              {onNavigateToStoryVideo && (
                <Button size="sm" onClick={onNavigateToStoryVideo} className="bg-bento-blue hover:bg-bento-blue/90 text-white cursor-pointer">
                  Open in Story to Video
                  <ArrowRight size={14} />
                </Button>
              )}
            </div>
          )}

          {/* Section 1: Pick Preset Theme */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-base">1. Video Storyboard Presets</CardTitle>
                  <CardDescription>
                    Select a curated cinematic theme equipped with story script, video camera directions, and visual image prompts.
                  </CardDescription>
                </div>
                <span className="text-xs font-semibold text-bento-blue bg-bento-blue/10 px-2.5 py-1 rounded-full">
                  {VIDEO_TEMPLATE_PRESETS.find((p) => p.id === selectedPresetId)?.videoStyle} Style
                </span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {VIDEO_TEMPLATE_PRESETS.map((preset) => {
                  const isSelected = preset.id === selectedPresetId
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() => handleSelectPreset(preset)}
                      className={cn(
                        'flex flex-col items-start gap-1 rounded-xl border p-3.5 text-left transition-all cursor-pointer',
                        isSelected
                          ? 'border-bento-blue bg-bento-blue/10 ring-1 ring-bento-blue shadow-sm'
                          : 'border-transparent bg-page hover:border-black/10 dark:hover:border-white/10 hover:bg-black/[0.02] dark:hover:bg-white/[0.04]',
                      )}
                    >
                      <div className="flex w-full items-center justify-between">
                        <span className={cn('text-sm font-bold', isSelected ? 'text-bento-blue' : 'text-ink-primary')}>
                          {preset.name}
                        </span>
                        {isSelected && <Check size={16} className="text-bento-blue" />}
                      </div>
                      <p className="text-xs text-ink-secondary line-clamp-2">{preset.description}</p>
                      <div className="mt-2 text-[11px] font-medium text-ink-muted">
                        {preset.scenes.length} Storyboard Scenes · {preset.videoStyle}
                      </div>
                    </button>
                  )
                })}
              </div>
            </CardContent>
          </Card>

          {/* Section 2: Two Column Scene Builder & Storyboard Studio */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Left Column (7 cols): Scene Prompts, Scripts & Camera Actions */}
            <div className="flex flex-col gap-6 lg:col-span-7">
              <Card>
                <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
                  <div>
                    <CardTitle className="text-base">2. Storyboard Scene Script &amp; Action Builder</CardTitle>
                    <CardDescription>
                      Each scene contains a narrative script, video camera/motion action, and diffusion image prompt.
                    </CardDescription>
                  </div>
                  <Button variant="secondary" size="sm" onClick={handleAddScene}>
                    <Plus size={14} />
                    Add scene
                  </Button>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                  {templateScenes.map((scene, idx) => {
                    const isGenerating = generatingSceneIndex === idx
                    const isInspecting = activeInspectorIndex === idx
                    return (
                      <div
                        key={scene.id}
                        onClick={() => setActiveInspectorIndex(idx)}
                        className={cn(
                          'rounded-xl border p-4 transition-all cursor-pointer flex flex-col gap-3',
                          isGenerating
                            ? 'border-bento-blue bg-bento-blue/5 ring-1 ring-bento-blue/40'
                            : isInspecting
                            ? 'border-bento-blue ring-1 ring-bento-blue shadow-sm bg-surface'
                            : scene.record
                            ? 'border-status-success/30 bg-page hover:border-black/20 dark:hover:border-white/20'
                            : 'bg-page hover:border-black/20 dark:hover:border-white/20',
                        )}
                      >
                        {/* Scene Header */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 flex-1">
                            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-bento-blue/10 text-[11px] font-bold text-bento-blue">
                              {idx + 1}
                            </span>
                            <input
                              value={scene.title}
                              onChange={(e) => handleUpdateSceneTitle(idx, e.target.value)}
                              className="bg-transparent text-xs font-bold text-ink-primary focus:outline-none focus:underline flex-1"
                              placeholder="Scene title"
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            {scene.record ? (
                              <span className="flex items-center gap-1 rounded-full bg-status-success/10 px-2 py-0.5 text-[11px] font-medium text-status-success">
                                <Check size={12} />
                                Rendered
                              </span>
                            ) : (
                              <span className="text-[11px] text-ink-muted">Ready to render</span>
                            )}
                            {templateScenes.length > 1 && (
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleRemoveScene(idx)
                                }}
                                className="text-ink-muted hover:text-status-critical p-1 cursor-pointer"
                                title="Remove scene"
                              >
                                <Trash2 size={13} />
                              </button>
                            )}
                          </div>
                        </div>

                        {/* Scene Script / Voiceover Line */}
                        <div className="flex flex-col gap-1 rounded-lg bg-page p-2.5 border">
                          <div className="flex items-center gap-1.5 text-[11px] font-bold text-bento-violet">
                            <FileText size={13} />
                            <span>Narrative Script / Storyline Line</span>
                          </div>
                          <textarea
                            value={scene.script}
                            onChange={(e) => handleUpdateSceneScript(idx, e.target.value)}
                            rows={2}
                            placeholder="What happens or what narration is spoken in this scene..."
                            className="bg-transparent text-xs text-ink-primary placeholder:text-ink-muted focus:outline-none resize-none"
                          />
                        </div>

                        {/* Video Camera & Motion Action */}
                        <div className="flex flex-col gap-1 rounded-lg bg-bento-blue/5 p-2.5 border border-bento-blue/20">
                          <div className="flex items-center gap-1.5 text-[11px] font-bold text-bento-blue">
                            <Camera size={13} />
                            <span>Video Camera &amp; Motion Action Directive</span>
                          </div>
                          <input
                            value={scene.action}
                            onChange={(e) => handleUpdateSceneAction(idx, e.target.value)}
                            placeholder="e.g. Slow zoom in, pan right, volumetric smoke, 3.5s hold, fade transition"
                            className="bg-transparent text-xs text-ink-primary placeholder:text-ink-muted focus:outline-none"
                          />
                        </div>

                        {/* Visual Image Generation Prompt */}
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center justify-between text-[11px] font-medium text-ink-muted">
                            <span className="flex items-center gap-1">
                              <Sparkles size={12} className="text-bento-orange" />
                              Visual Image Prompt (Diffusion Model)
                            </span>
                            <button
                              type="button"
                              onClick={() => handleUpdateScenePrompt(idx, enhancePrompt(scene.prompt))}
                              disabled={!scene.prompt.trim() || isGenerating}
                              className="text-bento-blue hover:underline text-[11px] flex items-center gap-1 disabled:opacity-40"
                            >
                              <Wand2 size={11} />
                              Enhance prompt
                            </button>
                          </div>
                          <Textarea
                            value={scene.prompt}
                            onChange={(e) => handleUpdateScenePrompt(idx, e.target.value)}
                            rows={2}
                            placeholder="Describe visual appearance for AI image generation..."
                            className="text-xs"
                          />
                        </div>

                        {/* Scene Action Footer */}
                        <div className="flex items-center justify-between pt-1">
                          <span className="text-[11px] text-ink-muted">
                            Click card to preview in Storyboard Inspector
                          </span>
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleGenerateSingleScene(idx)
                            }}
                            disabled={isGenerating || generatingSceneIndex !== null || !scene.prompt.trim()}
                            className="text-[11px] h-7 cursor-pointer"
                          >
                            {isGenerating ? (
                              <>
                                <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                                Rendering…
                              </>
                            ) : (
                              <>
                                <Play size={12} />
                                {scene.record ? 'Re-render Scene' : 'Render Scene Image'}
                              </>
                            )}
                          </Button>
                        </div>
                      </div>
                    )
                  })}

                  {/* Batch Generation & Settings Panel */}
                  <div className="mt-2 rounded-xl border bg-surface p-4 flex flex-col gap-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-ink-muted">
                          Video Aspect Ratio
                        </p>
                        <div className="grid grid-cols-3 gap-1.5">
                          {(Object.entries(IMAGE_SIZES) as [ImageSizeKey, (typeof IMAGE_SIZES)[ImageSizeKey]][]).map(
                            ([key, size]) => (
                              <button
                                key={key}
                                type="button"
                                onClick={() => onImageSizeChange(key)}
                                className={cn(
                                  'rounded-lg border px-1.5 py-1.5 text-center text-[11px] font-medium transition-colors cursor-pointer',
                                  imageSizeKey === key
                                    ? 'border-bento-blue bg-bento-blue/10 text-bento-blue font-semibold'
                                    : 'text-ink-secondary hover:bg-black/[0.03] dark:hover:bg-white/[0.05]',
                                )}
                              >
                                {size.label}
                              </button>
                            ),
                          )}
                        </div>
                      </div>

                      <div>
                        <div className="mb-1 flex items-center justify-between">
                          <p className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">Steps</p>
                          <span className="text-xs font-semibold text-ink-primary">{steps}</span>
                        </div>
                        <Slider min={1} max={50} step={1} value={[steps]} onValueChange={([v]) => onStepsChange(v)} />
                      </div>
                    </div>

                    <Button
                      size="lg"
                      className="w-full bg-gradient-to-r from-bento-violet via-bento-magenta to-bento-blue hover:opacity-90 cursor-pointer shadow-md"
                      onClick={handleGenerateAllTemplateScenes}
                      disabled={generatingSceneIndex !== null || templateScenes.every((s) => !s.prompt.trim())}
                    >
                      {generatingSceneIndex !== null ? (
                        <>
                          <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                          Rendering Scene {generatingSceneIndex + 1} of {templateScenes.length}…
                        </>
                      ) : (
                        <>
                          <Sparkles size={16} />
                          Generate All {templateScenes.length} Template Scenes
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Column (5 cols): Storyboard Inspector & Set for Package */}
            <div className="flex flex-col gap-6 lg:col-span-5">
              {/* Storyboard Grid & Inspector */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">3. Storyboard Gallery &amp; Inspector</CardTitle>
                    <span className="text-xs font-semibold text-ink-muted">
                      {generatedTemplateRecords.length} / {templateScenes.length} rendered
                    </span>
                  </div>
                  <CardDescription>Inspect scene script, visual prompt, and video actions.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                  {/* Visual Thumbnails Grid */}
                  <div className="grid grid-cols-2 gap-2.5">
                    {templateScenes.map((scene, i) => {
                      const isInspecting = activeInspectorIndex === i
                      const isChecked = scene.record ? selectedTemplateForPack.has(scene.record.id) : false
                      return (
                        <div
                          key={scene.id}
                          onClick={() => {
                            setActiveInspectorIndex(i)
                            if (scene.record) toggleTemplateImageForPack(scene.record.id)
                          }}
                          className={cn(
                            'group relative aspect-video overflow-hidden rounded-xl border-2 transition-all cursor-pointer flex flex-col justify-end',
                            isInspecting
                              ? 'border-bento-blue ring-2 ring-bento-blue/40'
                              : scene.record
                              ? 'border-transparent opacity-90 hover:opacity-100'
                              : 'border-dashed bg-page/60',
                          )}
                        >
                          {scene.record ? (
                            <img
                              src={scene.record.imageSrc}
                              alt={scene.title}
                              className="absolute inset-0 h-full w-full object-cover"
                            />
                          ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-2 text-ink-muted">
                              <Clapperboard size={20} className="opacity-40 mb-1" />
                              <span className="text-[11px] font-medium leading-tight">Scene {i + 1}</span>
                              <span className="text-[9px] opacity-60">Not rendered</span>
                            </div>
                          )}

                          <div className="relative z-10 bg-gradient-to-t from-black/85 via-black/40 to-transparent p-1.5 text-white">
                            <p className="text-[10px] font-bold truncate leading-tight">{scene.title}</p>
                          </div>

                          {scene.record && (
                            <span
                              className={cn(
                                'absolute right-1.5 top-1.5 z-20 flex h-4 w-4 items-center justify-center rounded-full border shadow-sm transition-all',
                                isChecked
                                  ? 'border-bento-blue bg-bento-blue text-white'
                                  : 'border-white/80 bg-black/50 text-transparent',
                              )}
                            >
                              {isChecked && <Check size={10} />}
                            </span>
                          )}
                        </div>
                      )
                    })}
                  </div>

                  {/* Active Scene Inspector Card */}
                  {activeInspectorScene && (
                    <div className="flex flex-col gap-2 rounded-xl border bg-page p-3.5 shadow-sm text-xs">
                      <div className="flex items-center justify-between border-b pb-2">
                        <span className="font-bold text-ink-primary truncate">
                          {activeInspectorScene.title}
                        </span>
                        <span className="text-[10px] rounded-full bg-bento-blue/10 px-2 py-0.5 font-semibold text-bento-blue">
                          Active Scene
                        </span>
                      </div>

                      {/* Script Display */}
                      <div className="flex flex-col gap-0.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-bento-violet flex items-center gap-1">
                            <FileText size={11} /> Scene Script
                          </span>
                          <button
                            type="button"
                            onClick={() => copyToClipboard(activeInspectorScene.script, 'Script copied')}
                            className="text-ink-muted hover:text-ink-primary"
                            title="Copy script"
                          >
                            <Copy size={11} />
                          </button>
                        </div>
                        <p className="text-ink-secondary text-[11px] leading-snug bg-surface rounded-lg p-2 border">
                          {activeInspectorScene.script}
                        </p>
                      </div>

                      {/* Video Action Display */}
                      <div className="flex flex-col gap-0.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-bento-blue flex items-center gap-1">
                            <Camera size={11} /> Video Action &amp; Motion
                          </span>
                          <button
                            type="button"
                            onClick={() => copyToClipboard(activeInspectorScene.action, 'Action copied')}
                            className="text-ink-muted hover:text-ink-primary"
                            title="Copy action"
                          >
                            <Copy size={11} />
                          </button>
                        </div>
                        <p className="text-ink-secondary text-[11px] leading-snug bg-bento-blue/5 rounded-lg p-2 border border-bento-blue/20">
                          {activeInspectorScene.action}
                        </p>
                      </div>

                      {/* Prompt Display */}
                      <div className="flex flex-col gap-0.5">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-ink-muted flex items-center gap-1">
                            <Sparkles size={11} /> Image Prompt
                          </span>
                          <button
                            type="button"
                            onClick={() => copyToClipboard(activeInspectorScene.prompt, 'Prompt copied')}
                            className="text-ink-muted hover:text-ink-primary"
                            title="Copy prompt"
                          >
                            <Copy size={11} />
                          </button>
                        </div>
                        <p className="text-ink-muted text-[10px] leading-tight line-clamp-2 bg-surface rounded-lg p-2 border">
                          {activeInspectorScene.prompt}
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Set for Package — EXCLUSIVE TO GENERATE TEMPLATE */}
              <Card className="border-bento-blue/40 bg-bento-blue/5 shadow-sm">
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <PackagePlus size={18} className="text-bento-blue" />
                    <CardTitle className="text-base text-bento-blue">Set for Package</CardTitle>
                  </div>
                  <CardDescription>
                    Bundle these generated template scenes, scripts, and video actions into an Image Pack ready to load into
                    Story to Video.
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-ink-primary">
                      {selectedTemplateForPack.size > 0
                        ? `${selectedTemplateForPack.size} scene${selectedTemplateForPack.size === 1 ? '' : 's'} selected`
                        : `${generatedTemplateRecords.length} rendered scenes available`}
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleSelectAllTemplateForPack}
                        disabled={generatedTemplateRecords.length === 0}
                        className="text-bento-blue hover:underline text-xs disabled:opacity-40 cursor-pointer"
                      >
                        Select all
                      </button>
                      <span className="text-ink-muted">·</span>
                      <button
                        type="button"
                        onClick={handleDeselectAllTemplateForPack}
                        disabled={selectedTemplateForPack.size === 0}
                        className="text-ink-muted hover:underline text-xs disabled:opacity-40 cursor-pointer"
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2">
                    <Button
                      size="default"
                      onClick={() => openPackDialog(true)}
                      disabled={generatedTemplateRecords.length === 0}
                      className="w-full bg-bento-blue text-white hover:bg-bento-blue/90 font-semibold shadow-sm cursor-pointer"
                    >
                      <PackagePlus size={16} />
                      Package All Scenes ({generatedTemplateRecords.length}) for Video
                    </Button>

                    {selectedTemplateForPack.size > 0 &&
                      selectedTemplateForPack.size < generatedTemplateRecords.length && (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => openPackDialog(false)}
                          className="w-full text-xs cursor-pointer"
                        >
                          Create Pack from Selected ({selectedTemplateForPack.size})
                        </Button>
                      )}
                  </div>

                  {lastCreatedPackName && onNavigateToStoryVideo && (
                    <div className="mt-2 rounded-xl border border-bento-blue/30 bg-surface p-3 flex items-center justify-between">
                      <div>
                        <p className="text-xs font-bold text-ink-primary truncate">Pack "{lastCreatedPackName}" ready!</p>
                        <p className="text-[11px] text-ink-secondary">Includes scene images, scripts &amp; video actions</p>
                      </div>
                      <Button size="sm" onClick={onNavigateToStoryVideo} className="text-xs h-7 cursor-pointer">
                        Open Story to Video
                        <ArrowRight size={12} />
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* Package Name Confirmation Dialog */}
      <AlertDialog open={packDialogOpen} onOpenChange={setPackDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Create Video Image Pack</AlertDialogTitle>
            <AlertDialogDescription>
              {selectedTemplateForPack.size > 0 ? selectedTemplateForPack.size : generatedTemplateRecords.length} template
              scene
              {(selectedTemplateForPack.size > 0 ? selectedTemplateForPack.size : generatedTemplateRecords.length) === 1
                ? ''
                : 's'}{' '}
              with scripts and video action directions will be packed together for instant sequencing in Story to Video.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="flex flex-col gap-1.5 mt-2">
            <label className="text-xs font-semibold text-ink-muted">Package Name</label>
            <input
              autoFocus
              value={packName}
              onChange={(e) => setPackName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleConfirmCreatePack()}
              placeholder="e.g. Cinematic Epic Pack"
              className="w-full rounded-xl border bg-page px-3.5 py-2.5 text-sm text-ink-primary placeholder:text-ink-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
            />
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setPackName('')}>Cancel</AlertDialogCancel>
            <Button
              onClick={handleConfirmCreatePack}
              disabled={generatedTemplateRecords.length === 0}
              className="bg-bento-blue text-white hover:bg-bento-blue/90"
            >
              Create Video Pack
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

