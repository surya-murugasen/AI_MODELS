import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { LearningTopic } from '@/lib/types'

interface TopicModalProps {
  topic?: LearningTopic | null
  isOpen: boolean
  onClose: () => void
  onSave: (topic: LearningTopic) => void
}

export function TopicModal({ topic, isOpen, onClose, onSave }: TopicModalProps) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [category, setCategory] = useState('')
  const [difficulty, setDifficulty] = useState<'Beginner' | 'Intermediate' | 'Advanced'>('Beginner')
  const [estimatedTime, setEstimatedTime] = useState('')
  const [content, setContent] = useState('')

  useEffect(() => {
    if (topic) {
      setTitle(topic.title)
      setDescription(topic.description)
      setCategory(topic.category)
      setDifficulty(topic.difficulty)
      setEstimatedTime(topic.estimatedTime)
      setContent(topic.content)
    } else {
      setTitle('')
      setDescription('')
      setCategory('')
      setDifficulty('Beginner')
      setEstimatedTime('')
      setContent('')
    }
  }, [topic, isOpen])

  if (!isOpen) return null

  const handleSave = () => {
    onSave({
      id: topic?.id || crypto.randomUUID(),
      title,
      description,
      category,
      difficulty,
      estimatedTime,
      content,
      completed: topic?.completed || false,
    })
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="flex w-full max-w-2xl flex-col rounded-2xl border bg-page shadow-2xl overflow-hidden max-h-[90vh]">
        <div className="flex items-center justify-between border-b bg-surface px-6 py-4">
          <h2 className="text-lg font-bold text-ink-primary">
            {topic ? 'Edit Topic' : 'Add New Topic'}
          </h2>
          <button onClick={onClose} className="rounded-full p-2 hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
            <X size={20} className="text-ink-secondary" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Title</label>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink-primary placeholder:text-ink-muted outline-none focus:border-bento-blue"
                placeholder="e.g. AWS EC2 Basics"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Category</label>
              <input
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink-primary placeholder:text-ink-muted outline-none focus:border-bento-blue"
                placeholder="e.g. Compute"
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Description</label>
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink-primary placeholder:text-ink-muted outline-none focus:border-bento-blue"
              placeholder="Short description of the topic..."
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as any)}
                className="w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink-primary outline-none focus:border-bento-blue"
              >
                <option value="Beginner">Beginner</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Advanced">Advanced</option>
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Estimated Time</label>
              <input
                value={estimatedTime}
                onChange={(e) => setEstimatedTime(e.target.value)}
                className="w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink-primary placeholder:text-ink-muted outline-none focus:border-bento-blue"
                placeholder="e.g. 2 hours"
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Learning Content (Markdown)</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={8}
              className="w-full rounded-lg border bg-surface px-3 py-2 text-sm text-ink-primary placeholder:text-ink-muted outline-none focus:border-bento-blue resize-none"
              placeholder="Write the learning materials here..."
            />
          </div>
        </div>
        <div className="flex justify-end gap-3 border-t bg-surface px-6 py-4">
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave} disabled={!title.trim() || !category.trim()}>
            {topic ? 'Save Changes' : 'Create Topic'}
          </Button>
        </div>
      </div>
    </div>
  )
}
