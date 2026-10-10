/**
 * videoRenderer.ts
 * ----------------
 * Pure client-side high-performance video rendering engine for Story to Video.
 *
 * Features:
 *  - Ken Burns camera moves (Zoom In, Zoom Out, Pan Left, Pan Right, Static)
 *  - Cinematic transitions (Fade, Cut, Slide, Wipe, Zoom Blur)
 *  - Particle atmospheric effects (Parallax, Kinetic Burst, Subtle Float)
 *  - Styled on-screen subtitles and captions
 *  - Web Audio ambient background soundtrack synthesis & audio file mixing
 *  - Live Canvas playback & MediaRecorder MP4/WebM export
 */

export interface RenderScene {
  id: string
  image: HTMLImageElement
  title?: string
  script?: string
  action?: string
  duration: number
}

export interface VideoRenderConfig {
  width: number
  height: number
  scenes: RenderScene[]
  transitionType: 'Fade' | 'Cut' | 'Slide' | 'Wipe' | 'Zoom Blur'
  transitionDuration: number
  transitionsOn: boolean
  cameraMove: 'Zoom In' | 'Zoom Out' | 'Pan Left' | 'Pan Right' | 'Static'
  cameraIntensity: number
  cameraOn: boolean
  animationStyle: 'None' | 'Parallax' | 'Kinetic Burst' | 'Subtle Float'
  animationIntensity: number
  animationOn: boolean
  captionStyle: 'None' | 'Minimal' | 'Bold' | 'Subtitle-style'
  captionPosition: 'Top' | 'Center' | 'Bottom'
  captionsOn: boolean
  templateKey: string
  musicVolume: number
  narrationAudio?: HTMLAudioElement | null
  musicAudio?: HTMLAudioElement | null
  voiceOn: boolean
}

export function preloadImages(urls: string[]): Promise<HTMLImageElement[]> {
  return Promise.all(
    urls.map(
      (url) =>
        new Promise<HTMLImageElement>((resolve) => {
          const img = new Image()
          img.crossOrigin = 'anonymous'
          img.onload = () => resolve(img)
          img.onerror = () => {
            // fallback in case of CORS or broken image
            const fallback = new Image()
            fallback.src = url
            resolve(fallback)
          }
          img.src = url
        }),
    ),
  )
}

export function calculateTotalDuration(
  scenes: { duration: number }[],
  transitionsOn: boolean,
  transitionDuration: number,
): number {
  if (scenes.length === 0) return 0
  const sceneTotal = scenes.reduce((acc, s) => acc + s.duration, 0)
  if (!transitionsOn || scenes.length <= 1) return sceneTotal
  // With crossfades, scenes overlap by transitionDuration
  return Math.max(1, sceneTotal - (scenes.length - 1) * (transitionDuration * 0.5))
}

/**
 * Find the active scene index and scene-local progress for any given timestamp.
 */
export function getSceneAtTime(
  time: number,
  scenes: RenderScene[],
  transitionsOn: boolean,
  transitionDuration: number,
): {
  currentIndex: number
  nextIndex: number | null
  sceneTime: number
  sceneDuration: number
  transitionProgress: number // 0 to 1 if in transition, otherwise 0
} {
  if (scenes.length === 0) {
    return { currentIndex: 0, nextIndex: null, sceneTime: 0, sceneDuration: 1, transitionProgress: 0 }
  }

  const effectiveTransition = transitionsOn ? Math.min(transitionDuration, 1.5) : 0
  const overlap = effectiveTransition * 0.5

  let accumulatedTime = 0
  for (let i = 0; i < scenes.length; i++) {
    const scene = scenes[i]
    const sceneStart = accumulatedTime
    const sceneEnd = sceneStart + scene.duration

    if (time >= sceneStart && (time < sceneEnd || i === scenes.length - 1)) {
      const sceneTime = time - sceneStart
      const timeLeft = sceneEnd - time

      let nextIndex: number | null = null
      let transitionProgress = 0

      if (i < scenes.length - 1 && timeLeft <= effectiveTransition && effectiveTransition > 0) {
        nextIndex = i + 1
        transitionProgress = 1 - timeLeft / effectiveTransition
      }

      return {
        currentIndex: i,
        nextIndex,
        sceneTime,
        sceneDuration: scene.duration,
        transitionProgress: Math.min(1, Math.max(0, transitionProgress)),
      }
    }

    accumulatedTime += scene.duration - overlap
  }

  const lastIdx = scenes.length - 1
  return {
    currentIndex: lastIdx,
    nextIndex: null,
    sceneTime: scenes[lastIdx].duration,
    sceneDuration: scenes[lastIdx].duration,
    transitionProgress: 0,
  }
}

/**
 * Draw a single scene with camera movement and animation
 */
function drawSceneImage(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  width: number,
  height: number,
  progress: number, // 0 to 1
  config: VideoRenderConfig,
  time: number,
) {
  if (!img.complete || img.naturalWidth === 0) return

  ctx.save()

  // Calculate cover aspect ratio
  const imgAspect = img.naturalWidth / img.naturalHeight
  const canvasAspect = width / height

  let baseW = width
  let baseH = height
  if (imgAspect > canvasAspect) {
    baseW = height * imgAspect
  } else {
    baseH = width / imgAspect
  }

  // Camera Motion (Ken Burns)
  let scale = 1.0
  let offsetX = 0
  let offsetY = 0

  if (config.cameraOn) {
    const intensity = (config.cameraIntensity / 10) * 0.25 // max 25% zoom / pan
    switch (config.cameraMove) {
      case 'Zoom In':
        scale = 1.0 + progress * intensity
        break
      case 'Zoom Out':
        scale = 1.0 + (1 - progress) * intensity
        break
      case 'Pan Left':
        scale = 1.0 + intensity * 0.5
        offsetX = (progress - 0.5) * intensity * baseW
        break
      case 'Pan Right':
        scale = 1.0 + intensity * 0.5
        offsetX = -(progress - 0.5) * intensity * baseW
        break
      case 'Static':
      default:
        scale = 1.0 + Math.sin(progress * Math.PI) * (intensity * 0.15)
        break
    }
  }

  // Image Animation (Parallax / Kinetic / Float)
  if (config.animationOn) {
    const animIntensity = (config.animationIntensity / 10) * 8
    switch (config.animationStyle) {
      case 'Parallax':
        offsetX += Math.sin(time * 1.5) * animIntensity
        offsetY += Math.cos(time * 1.2) * (animIntensity * 0.5)
        break
      case 'Kinetic Burst': {
        const pulse = 1.0 + Math.sin(time * 4) * 0.02 * (config.animationIntensity / 5)
        scale *= pulse
        break
      }
      case 'Subtle Float':
        offsetY += Math.sin(time * 1.0) * animIntensity
        break
      case 'None':
      default:
        break
    }
  }

  const finalW = baseW * scale
  const finalH = baseH * scale
  const dx = (width - finalW) / 2 + offsetX
  const dy = (height - finalH) / 2 + offsetY

  ctx.drawImage(img, dx, dy, finalW, finalH)

  // Floating particles atmospheric overlay for specific animation styles
  if (config.animationOn && config.animationStyle === 'Subtle Float') {
    drawParticles(ctx, width, height, time, config.templateKey)
  }

  ctx.restore()
}

function drawParticles(ctx: CanvasRenderingContext2D, width: number, height: number, time: number, template: string) {
  ctx.save()
  const isAnime = template === 'music-video'
  const isCyber = template === 'social-reel'
  const count = 18

  for (let i = 0; i < count; i++) {
    const seed = i * 137.5
    const speed = 0.3 + (i % 5) * 0.15
    const x = ((seed + time * 30 * speed) % (width + 60)) - 30
    const y = ((seed * 1.7 + Math.sin(time + i) * 40 + time * 15 * speed) % (height + 60)) - 30
    const radius = 1.5 + (i % 3) * 1.5

    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)

    if (isAnime) {
      // Pink sakura petal glow
      ctx.fillStyle = `rgba(255, 182, 193, ${0.35 + Math.sin(time * 2 + i) * 0.2})`
    } else if (isCyber) {
      // Cyan/magenta cyber particle
      ctx.fillStyle = i % 2 === 0 ? 'rgba(0, 240, 255, 0.45)' : 'rgba(255, 0, 128, 0.45)'
    } else {
      // Warm golden dust ember
      ctx.fillStyle = `rgba(255, 230, 160, ${0.3 + Math.sin(time * 1.5 + i) * 0.15})`
    }

    ctx.fill()
  }
  ctx.restore()
}

/**
 * Render on-screen captions & subtitles
 */
function drawCaptions(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  scene: RenderScene,
  config: VideoRenderConfig,
) {
  if (!config.captionsOn || config.captionStyle === 'None') return
  const text = scene.script?.trim() || scene.title?.trim()
  if (!text) return

  ctx.save()

  const fontSize = Math.max(16, Math.floor(width * 0.032))
  ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  // Word wrap
  const maxLineWidth = width * 0.8
  const words = text.split(' ')
  const lines: string[] = []
  let currentLine = ''

  for (const word of words) {
    const testLine = currentLine ? `${currentLine} ${word}` : word
    const metrics = ctx.measureText(testLine)
    if (metrics.width > maxLineWidth && currentLine) {
      lines.push(currentLine)
      currentLine = word
    } else {
      currentLine = testLine
    }
  }
  if (currentLine) lines.push(currentLine)

  const lineHeight = fontSize * 1.35
  const totalTextHeight = lines.length * lineHeight

  // Position calculation
  let startY = height * 0.85
  if (config.captionPosition === 'Top') {
    startY = height * 0.12
  } else if (config.captionPosition === 'Center') {
    startY = height * 0.5 - totalTextHeight / 2
  }

  // Draw background box or shadow depending on caption style
  if (config.captionStyle === 'Subtitle-style' || config.captionStyle === 'Bold') {
    const paddingX = 20
    const paddingY = 10
    let maxWidth = 0
    for (const line of lines) {
      const w = ctx.measureText(line).width
      if (w > maxWidth) maxWidth = w
    }

    ctx.fillStyle = config.captionStyle === 'Bold' ? 'rgba(0, 0, 0, 0.85)' : 'rgba(0, 0, 0, 0.65)'
    const boxX = (width - maxWidth) / 2 - paddingX
    const boxY = startY - lineHeight / 2 - paddingY / 2
    const boxW = maxWidth + paddingX * 2
    const boxH = totalTextHeight + paddingY

    ctx.beginPath()
    ctx.roundRect(boxX, boxY, boxW, boxH, 8)
    ctx.fill()
  }

  // Draw text lines
  for (let i = 0; i < lines.length; i++) {
    const lineY = startY + i * lineHeight

    if (config.captionStyle === 'Minimal') {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.9)'
      ctx.shadowBlur = 6
      ctx.shadowOffsetX = 0
      ctx.shadowOffsetY = 2
    }

    ctx.fillStyle = '#ffffff'
    ctx.fillText(lines[i], width / 2, lineY)
  }

  ctx.restore()
}

/**
 * Main frame rendering function
 */
export function renderVideoFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  time: number,
  config: VideoRenderConfig,
) {
  ctx.clearRect(0, 0, width, height)

  // Black background
  ctx.fillStyle = '#000000'
  ctx.fillRect(0, 0, width, height)

  const scenes = config.scenes
  if (scenes.length === 0) return

  const { currentIndex, nextIndex, sceneTime, sceneDuration, transitionProgress } = getSceneAtTime(
    time,
    scenes,
    config.transitionsOn,
    config.transitionDuration,
  )

  const currentScene = scenes[currentIndex]
  const currentProgress = Math.min(1, Math.max(0, sceneTime / sceneDuration))

  if (!nextIndex || transitionProgress === 0 || !config.transitionsOn) {
    // Single scene active
    drawSceneImage(ctx, currentScene.image, width, height, currentProgress, config, time)
    drawCaptions(ctx, width, height, currentScene, config)
  } else {
    // Transition between currentScene and nextScene
    const nextScene = scenes[nextIndex]
    const nextProgress = transitionProgress * 0.3

    switch (config.transitionType) {
      case 'Fade': {
        // Draw base scene
        drawSceneImage(ctx, currentScene.image, width, height, currentProgress, config, time)
        // Draw next scene over with alpha
        ctx.save()
        ctx.globalAlpha = transitionProgress
        drawSceneImage(ctx, nextScene.image, width, height, nextProgress, config, time)
        ctx.restore()
        break
      }
      case 'Slide': {
        const slideOffset = transitionProgress * width
        ctx.save()
        ctx.translate(-slideOffset, 0)
        drawSceneImage(ctx, currentScene.image, width, height, currentProgress, config, time)
        ctx.restore()

        ctx.save()
        ctx.translate(width - slideOffset, 0)
        drawSceneImage(ctx, nextScene.image, width, height, nextProgress, config, time)
        ctx.restore()
        break
      }
      case 'Wipe': {
        drawSceneImage(ctx, currentScene.image, width, height, currentProgress, config, time)
        ctx.save()
        ctx.beginPath()
        ctx.rect(0, 0, width * transitionProgress, height)
        ctx.clip()
        drawSceneImage(ctx, nextScene.image, width, height, nextProgress, config, time)
        ctx.restore()
        break
      }
      case 'Zoom Blur': {
        const blurScale = 1.0 + transitionProgress * 0.3
        ctx.save()
        ctx.globalAlpha = 1 - transitionProgress
        ctx.translate(width / 2, height / 2)
        ctx.scale(blurScale, blurScale)
        ctx.translate(-width / 2, -height / 2)
        drawSceneImage(ctx, currentScene.image, width, height, currentProgress, config, time)
        ctx.restore()

        ctx.save()
        ctx.globalAlpha = transitionProgress
        drawSceneImage(ctx, nextScene.image, width, height, nextProgress, config, time)
        ctx.restore()
        break
      }
      case 'Cut':
      default: {
        const active = transitionProgress < 0.5 ? currentScene : nextScene
        const prog = transitionProgress < 0.5 ? currentProgress : nextProgress
        drawSceneImage(ctx, active.image, width, height, prog, config, time)
        break
      }
    }

    const captionScene = transitionProgress < 0.5 ? currentScene : nextScene
    drawCaptions(ctx, width, height, captionScene, config)
  }
}

/**
 * Web Audio Ambient Soundtrack Generator
 * Produces smooth cinematic ambient pads, chords, and basslines in real-time.
 */
export class AmbientAudioSynthesizer {
  private ctx: AudioContext | null = null
  private gainNode: GainNode | null = null
  private isPlaying = false
  private oscillators: OscillatorNode[] = []

  init(destination?: MediaStreamAudioDestinationNode) {
    if (this.ctx) return
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    this.ctx = new AudioCtx()
    this.gainNode = this.ctx.createGain()
    this.gainNode.gain.setValueAtTime(0.08, this.ctx.currentTime)

    if (destination) {
      this.gainNode.connect(destination)
    }
    this.gainNode.connect(this.ctx.destination)
  }

  setVolume(volPercent: number) {
    if (this.gainNode && this.ctx) {
      const vol = (volPercent / 100) * 0.12
      this.gainNode.gain.setTargetAtTime(vol, this.ctx.currentTime, 0.1)
    }
  }

  start(template = 'cinematic') {
    if (this.isPlaying || !this.ctx || !this.gainNode) return
    this.isPlaying = true

    if (this.ctx.state === 'suspended') {
      this.ctx.resume()
    }

    // Determine chord frequencies based on template
    let freqs = [130.81, 196.0, 261.63, 329.63] // C major/minor cinematic chord
    if (template === 'social-reel') {
      freqs = [146.83, 220.0, 293.66, 370.0] // D synthwave
    } else if (template === 'music-video') {
      freqs = [174.61, 261.63, 329.63, 392.0] // F majestic anime
    } else if (template === 'documentary') {
      freqs = [110.0, 164.81, 220.0, 329.63] // A ambient nature
    }

    this.oscillators = freqs.map((freq, idx) => {
      const osc = this.ctx!.createOscillator()
      const filter = this.ctx!.createBiquadFilter()

      osc.type = idx === 0 ? 'triangle' : 'sine'
      osc.frequency.setValueAtTime(freq, this.ctx!.currentTime)

      // Add gentle chorus vibrato
      const lfo = this.ctx!.createOscillator()
      const lfoGain = this.ctx!.createGain()
      lfo.frequency.setValueAtTime(0.2 + idx * 0.1, this.ctx!.currentTime)
      lfoGain.gain.setValueAtTime(1.5, this.ctx!.currentTime)
      lfo.connect(osc.frequency)
      lfo.start()

      filter.type = 'lowpass'
      filter.frequency.setValueAtTime(800 + idx * 200, this.ctx!.currentTime)

      osc.connect(filter)
      filter.connect(this.gainNode!)
      osc.start()

      return osc
    })
  }

  stop() {
    if (!this.isPlaying) return
    this.isPlaying = false
    this.oscillators.forEach((osc) => {
      try {
        osc.stop()
        osc.disconnect()
      } catch {
        // ignore
      }
    })
    this.oscillators = []
  }

  destroy() {
    this.stop()
    if (this.ctx) {
      this.ctx.close()
      this.ctx = null
    }
  }
}

/**
 * Record the entire composition to a high-quality WebM/MP4 video file via MediaRecorder
 */
export async function exportVideoFile(
  config: VideoRenderConfig,
  totalDuration: number,
  onProgress: (percent: number) => void,
): Promise<Blob> {
  const canvas = document.createElement('canvas')
  canvas.width = config.width
  canvas.height = config.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get canvas context')

  // Set up audio destination
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const audioCtx = new AudioCtx()
  const audioDest = audioCtx.createMediaStreamDestination()

  // Synthesize background audio stream
  const synth = new AmbientAudioSynthesizer()
  synth.init(audioDest)
  synth.setVolume(config.voiceOn ? config.musicVolume : 40)
  synth.start(config.templateKey)

  // Mix streams
  const canvasStream = canvas.captureStream(30) // 30 fps
  const combinedStream = new MediaStream([
    ...canvasStream.getVideoTracks(),
    ...audioDest.stream.getAudioTracks(),
  ])

  let mimeType = 'video/webm;codecs=vp9,opus'
  if (!MediaRecorder.isTypeSupported(mimeType)) {
    mimeType = 'video/webm'
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = 'video/mp4'
    }
  }

  const mediaRecorder = new MediaRecorder(combinedStream, {
    mimeType,
    videoBitsPerSecond: 4_000_000,
  })

  const chunks: Blob[] = []
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data)
  }

  return new Promise<Blob>((resolve, reject) => {
    mediaRecorder.onstop = () => {
      synth.stop()
      synth.destroy()
      audioCtx.close()
      const videoBlob = new Blob(chunks, { type: mimeType })
      resolve(videoBlob)
    }

    mediaRecorder.onerror = (err) => {
      synth.stop()
      synth.destroy()
      audioCtx.close()
      reject(err)
    }

    mediaRecorder.start(100)

    const fps = 30
    const totalFrames = Math.ceil(totalDuration * fps)
    let currentFrame = 0

    function renderNextFrame() {
      if (currentFrame >= totalFrames) {
        onProgress(100)
        mediaRecorder.stop()
        return
      }

      const currentTime = (currentFrame / totalFrames) * totalDuration
      renderVideoFrame(ctx!, config.width, config.height, currentTime, config)

      currentFrame++
      const progress = Math.min(99, Math.round((currentFrame / totalFrames) * 100))
      onProgress(progress)

      setTimeout(renderNextFrame, 1000 / fps)
    }

    renderNextFrame()
  })
}
