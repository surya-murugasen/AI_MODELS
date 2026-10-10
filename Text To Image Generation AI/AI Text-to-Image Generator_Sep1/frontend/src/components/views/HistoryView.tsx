import { useState } from 'react'
import { History, Trash2 } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import type { GenerationRecord } from '@/lib/types'

export function HistoryView({
  generations,
  onSelect,
  onDelete,
}: {
  generations: GenerationRecord[]
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}) {
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const pending = generations.find((g) => g.id === pendingDeleteId) ?? null

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8">
      <header className="mb-6 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-yellow to-bento-orange text-white">
          <History size={20} />
        </div>
        <div>
          <h1 className="text-xl font-bold text-ink-primary md:text-2xl">History</h1>
          <p className="text-sm text-ink-secondary">The prompt and settings behind every generation this session.</p>
        </div>
      </header>

      {generations.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-16 text-center text-ink-muted">
            <History size={28} />
            <p className="text-sm">No generations yet.</p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="divide-y p-0">
            {generations.map((g) => (
              <div
                key={g.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelect(g.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') onSelect(g.id)
                }}
                className="flex w-full cursor-pointer items-center gap-4 px-5 py-4 text-left transition-colors hover:bg-black/[0.02] dark:hover:bg-white/[0.03]"
              >
                <img src={g.imageSrc} alt={g.prompt} className="h-14 w-14 flex-none rounded-lg object-cover" />
                <div className="flex-1 overflow-hidden">
                  <p className="truncate text-sm font-medium text-ink-primary">{g.prompt}</p>
                  <p className="text-xs text-ink-muted">
                    {new Date(g.createdAt).toLocaleString()} · {g.modelId} · {g.steps} steps · guidance{' '}
                    {g.guidanceScale.toFixed(1)} · {g.width}×{g.height} · {g.seconds}s
                  </p>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    setPendingDeleteId(g.id)
                  }}
                  title="Delete this generation"
                  className="flex h-8 w-8 flex-none items-center justify-center rounded-lg text-ink-muted transition-colors hover:bg-status-critical/10 hover:text-status-critical"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <p className="mt-6 text-xs text-ink-muted">
        Session-only, like the Gallery — there's no database in this project, so history resets on page refresh.
      </p>

      <AlertDialog open={pendingDeleteId !== null} onOpenChange={(open: boolean) => !open && setPendingDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this generation?</AlertDialogTitle>
            <AlertDialogDescription>
              {pending ? `"${pending.prompt}"` : 'This entry'} will be removed from History and the Gallery for this
              session. This can't be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (pendingDeleteId) onDelete(pendingDeleteId)
                setPendingDeleteId(null)
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
