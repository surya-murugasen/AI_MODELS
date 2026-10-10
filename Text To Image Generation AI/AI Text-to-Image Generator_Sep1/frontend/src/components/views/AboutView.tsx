import { ArrowRight, Info } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const STEPS = [
  { label: 'User', detail: 'You type a prompt and click Generate.' },
  { label: 'Web UI', detail: 'This React app sends POST /api/generate as JSON.' },
  { label: 'FastAPI', detail: 'Validates the request, calls the model wrapper.' },
  { label: 'Diffusion model', detail: 'A Hugging Face Diffusers pipeline runs inference on PyTorch.' },
  { label: 'Generated image', detail: 'Encoded as base64 PNG and returned in the JSON response.' },
  { label: 'UI', detail: 'Decoded into an <img> and shown in the Result panel.' },
]

export function AboutView() {
  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <header className="mb-6 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-violet to-bento-magenta text-white">
          <Info size={20} />
        </div>
        <div>
          <h1 className="text-xl font-bold text-ink-primary md:text-2xl">About this project</h1>
          <p className="text-sm text-ink-secondary">A learning project for AI system architecture.</p>
        </div>
      </header>

      <div className="flex max-w-2xl flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>How a request flows through the system</CardTitle>
            <CardDescription>The whole point of this app is to make each of these boxes visible.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {STEPS.map((step, i) => (
              <div key={step.label} className="flex items-start gap-3">
                <div className="flex h-7 w-7 flex-none items-center justify-center rounded-full bg-bento-blue/10 text-xs font-semibold text-bento-blue">
                  {i + 1}
                </div>
                <div>
                  <p className="text-sm font-semibold text-ink-primary">{step.label}</p>
                  <p className="text-sm text-ink-secondary">{step.detail}</p>
                </div>
                {i < STEPS.length - 1 && <ArrowRight size={14} className="mt-2 flex-none text-ink-muted" />}
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Deliberate simplifications</CardTitle>
            <CardDescription>What's real, and what's kept simple on purpose.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-sm text-ink-secondary">
            <p>
              <strong className="text-ink-primary">Gallery &amp; History</strong> are session-only, kept in the
              browser's memory — there is intentionally no database. Refreshing the page clears them.
            </p>
            <p>
              <strong className="text-ink-primary">Enhance prompt</strong> is a local, rule-based helper that appends
              a few quality keywords — it does not call another AI model.
            </p>
            <p>
              <strong className="text-ink-primary">Model</strong> is switchable from a small curated dropdown (see
              Generate), but only one model is ever kept in memory at a time — picking a different one unloads the
              current model and loads the new one on your next Generate, which can take a while (longer still the
              first time that model is used, since it's also downloaded). This keeps memory use bounded on a laptop
              instead of holding every model in RAM/VRAM at once.
            </p>
            <p>There is no authentication, rate limiting, or cloud deployment — this is meant to run on your own machine.</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
