export interface HealthInfo {
  status: string
  device: string
  model_id: string
  default_steps: number
  default_guidance_scale: number
  default_width: number
  default_height: number
}

export interface ModelOption {
  id: string
  label: string
}

export interface ModelsInfo {
  available: ModelOption[]
  current: string
}

export type ImageSizeKey = 'square' | 'landscape' | 'portrait'

export const IMAGE_SIZES: Record<ImageSizeKey, { label: string; width: number; height: number }> = {
  square: { label: 'Square', width: 512, height: 512 },
  landscape: { label: 'Landscape', width: 768, height: 512 },
  portrait: { label: 'Portrait', width: 512, height: 768 },
}

export interface GenerationRecord {
  id: string
  prompt: string
  imageSrc: string
  steps: number
  guidanceScale: number
  width: number
  height: number
  seconds: number
  createdAt: number
  modelId: string
  /** Estimated 0-100 quality score for this generation, see lib/predictive.ts. */
  accuracy: number
  /**
   * What predictNextAccuracy() estimated *before* this generation ran, using
   * only the history that existed at that moment — null when there wasn't
   * enough prior history to predict from (e.g. the very first generation).
   */
  predictedAccuracy: number | null
}

export type ViewKey =
  | 'generate'
  | 'gallery'
  | 'history'
  | 'predictive'
  | 'story-video'
  | 'how-it-works'
  | 'settings'
  | 'about'
  | 'cloud-learning'
  | 'security-learning'

export type GenerateMode = 'image' | 'template'

export interface VideoTemplateScene {
  id: string
  title: string
  prompt: string
  script: string
  action: string
  generatedRecord?: GenerationRecord | null
}

export interface VideoTemplatePreset {
  id: string
  name: string
  description: string
  videoStyle: string
  templateKey?: 'cinematic' | 'slideshow' | 'documentary' | 'social-reel' | 'music-video'
  suggestedScript: string
  scenes: {
    title: string
    prompt: string
    script: string
    action: string
  }[]
}

/** A single image captured into a pack — a stable snapshot at pack-creation
 * time, not a live reference, so deleting the original generation later
 * (Gallery/History) can never silently break or empty out an existing pack. */
export interface PackedImage {
  id: string
  imageSrc: string
  prompt: string
  title?: string
  script?: string
  action?: string
}

/** A named group of generated images, created via multi-select in Image
 * Generate, for reuse as a ready-made image sequence in Story to Video. */
export interface ImagePack {
  id: string
  name: string
  createdAt: number
  script?: string
  templateKey?: string
  images: PackedImage[]
}

export interface LearningTopic {
  id: string
  title: string
  description: string
  category: string
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  estimatedTime: string
  completed: boolean
  content: string
}
