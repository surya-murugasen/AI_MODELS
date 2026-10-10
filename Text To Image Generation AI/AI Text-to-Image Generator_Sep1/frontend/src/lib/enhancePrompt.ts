// A purely local, rule-based "prompt enhancer" — no LLM call involved. It
// just appends a few quality-boosting keywords that tend to help diffusion
// models, which is the same trick many text-to-image tools use under a
// fancier name. Idempotent: running it twice doesn't double up the suffix.
const QUALITY_SUFFIX = 'highly detailed, sharp focus, vibrant colors, professional digital art'

export function enhancePrompt(prompt: string): string {
  const trimmed = prompt.trim()
  if (!trimmed) return trimmed
  if (trimmed.toLowerCase().includes(QUALITY_SUFFIX)) return trimmed
  return `${trimmed}, ${QUALITY_SUFFIX}`
}
