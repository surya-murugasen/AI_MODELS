import { useState } from 'react'
import { Images, Trash2 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
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

export function GalleryView({
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
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-bento-aqua to-bento-blue text-white">
          <Images size={20} />
        </div>
        <div>
          <h1 className="text-xl font-bold text-ink-primary md:text-2xl">Gallery</h1>
          <p className="text-sm text-ink-secondary">Every image generated in this session, most recent first.</p>
        </div>
      </header>

      {generations.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-16 text-center text-ink-muted">
            <Images size={28} />
            <p className="text-sm">Nothing generated yet — head to Generate to create your first image.</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {generations.map((g) => (
            <Card key={g.id} className="overflow-hidden">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => onSelect(g.id)}
                  className="block aspect-square w-full overflow-hidden"
                  title={g.prompt}
                >
                  <img src={g.imageSrc} alt={g.prompt} className="h-full w-full object-cover transition-transform hover:scale-105" />
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    setPendingDeleteId(g.id)
                  }}
                  title="Delete this image"
                  className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-lg bg-black/60 text-white backdrop-blur transition-colors hover:bg-status-critical"
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <CardHeader className="p-3 pb-0">
                <CardDescription className="line-clamp-2 text-xs">{g.prompt}</CardDescription>
              </CardHeader>
              <CardContent className="p-3 pt-2 text-xs text-ink-muted">
                {new Date(g.createdAt).toLocaleTimeString()} · {g.width}×{g.height}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <p className="mt-6 text-xs text-ink-muted">
        This gallery lives in memory for the current browser tab only — there's no database in this project (by
        design, see the README), so refreshing the page clears it.
      </p>

      <AlertDialog open={pendingDeleteId !== null} onOpenChange={(open: boolean) => !open && setPendingDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this image?</AlertDialogTitle>
            <AlertDialogDescription>
              {pending ? `"${pending.prompt}"` : 'This image'} will be removed from the Gallery and History for this
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
