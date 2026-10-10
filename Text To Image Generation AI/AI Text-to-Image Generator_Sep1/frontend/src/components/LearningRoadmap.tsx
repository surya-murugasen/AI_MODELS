import { useState, useMemo } from 'react'
import { Plus, CheckCircle2, Circle, Edit2, Trash2, Clock, BookOpen, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { TopicModal } from './TopicModal'
import type { LearningTopic } from '@/lib/types'
import { cn } from '@/lib/utils'

interface LearningRoadmapProps {
  title: string
  icon: React.ReactNode
  description: string
  initialTopics: LearningTopic[]
  colorTone: 'blue' | 'critical'
}

export function LearningRoadmap({ title, icon, description, initialTopics, colorTone }: LearningRoadmapProps) {
  const [topics, setTopics] = useState<LearningTopic[]>(initialTopics)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingTopic, setEditingTopic] = useState<LearningTopic | null>(null)

  const completedCount = topics.filter(t => t.completed).length
  const progressPercent = topics.length > 0 ? Math.round((completedCount / topics.length) * 100) : 0

  const handleSaveTopic = (topic: LearningTopic) => {
    setTopics(prev => {
      const exists = prev.find(t => t.id === topic.id)
      if (exists) {
        return prev.map(t => t.id === topic.id ? topic : t)
      }
      return [...prev, topic]
    })
  }

  const handleDeleteTopic = (id: string) => {
    setTopics(prev => prev.filter(t => t.id !== id))
  }

  const handleToggleComplete = (id: string) => {
    setTopics(prev => prev.map(t => t.id === id ? { ...t, completed: !t.completed } : t))
  }

  // Find the first incomplete topic
  const nextIncompleteId = useMemo(() => {
    return topics.find(t => !t.completed)?.id || null
  }, [topics])

  const accentColorClass = colorTone === 'blue' ? 'text-bento-blue' : 'text-status-critical'
  const accentBgSoftClass = colorTone === 'blue' ? 'bg-bento-blue/10' : 'bg-status-critical/10'

  return (
    <div className="flex h-full flex-col bg-page">
      <div className="flex items-center justify-between border-b bg-surface px-8 py-5 flex-none">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-bold text-ink-primary">
            <span className={accentColorClass}>{icon}</span>
            {title}
          </h1>
          <p className="mt-1 text-sm text-ink-secondary">{description}</p>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <span className="text-xs font-bold uppercase tracking-wider text-ink-muted">Progress</span>
            <span className="text-sm font-bold text-ink-primary">{completedCount} / {topics.length} ({progressPercent}%)</span>
          </div>
          <Button onClick={() => { setEditingTopic(null); setIsModalOpen(true); }} className="gap-2">
            <Plus size={16} /> Add Topic
          </Button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-8">
        <div className="relative mx-auto max-w-4xl">
          
          {topics.length === 0 ? (
             <div className="py-20 text-center text-ink-muted">No topics yet. Start building your roadmap!</div>
          ) : (
            <div className="relative space-y-0 py-4 pl-4">
              {topics.map((topic, index) => {
                const isNext = topic.id === nextIncompleteId
                const isLast = index === topics.length - 1
                
                return (
                  <div key={topic.id} className="group relative flex gap-8 pb-10">
                    {/* Timeline Line */}
                    {!isLast && (
                      <div 
                        className={cn(
                          "absolute bottom-0 left-4 top-8 w-1 -translate-x-1/2",
                          topic.completed ? "bg-status-good" : "bg-black/5 dark:bg-white/5"
                        )}
                      />
                    )}
                    
                    {/* Node Icon */}
                    <div className="relative z-10 flex-none self-start">
                      <button
                        onClick={() => handleToggleComplete(topic.id)}
                        className={cn(
                          "flex h-8 w-8 items-center justify-center rounded-full border-4 border-page shadow-sm transition-colors",
                          topic.completed ? "bg-status-good text-page" : "bg-surface text-ink-muted hover:text-ink-primary",
                          isNext && "animate-pulse bg-emerald-500/10 text-emerald-500 ring-4 ring-emerald-500/20"
                        )}
                      >
                        {topic.completed ? <CheckCircle2 size={20} /> : <Circle size={14} />}
                      </button>
                    </div>

                    {/* Card */}
                    <div 
                      className={cn(
                        "flex-1 rounded-2xl border bg-surface p-5 shadow-sm transition-all duration-300",
                        topic.completed && "border-status-good/30 bg-status-good/[0.02]",
                        isNext && `border-emerald-500/50 shadow-md`
                      )}
                    >
                      <div className="mb-3 flex items-start justify-between">
                        <div>
                          <div className="mb-1.5 flex items-center gap-2">
                            <span className={cn(
                              "rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                              accentBgSoftClass, accentColorClass
                            )}>
                              {topic.category}
                            </span>
                            <span className="rounded-md bg-black/5 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-ink-muted dark:bg-white/5">
                              {topic.difficulty}
                            </span>
                          </div>
                          <h3 className="text-lg font-bold text-ink-primary">{topic.title}</h3>
                        </div>
                        <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                           <button onClick={() => { setEditingTopic(topic); setIsModalOpen(true); }} className="rounded-md p-1.5 text-ink-muted transition-colors hover:bg-black/5 hover:text-bento-blue dark:hover:bg-white/5"><Edit2 size={16} /></button>
                           <button onClick={() => handleDeleteTopic(topic.id)} className="rounded-md p-1.5 text-ink-muted transition-colors hover:bg-black/5 hover:text-status-critical dark:hover:bg-white/5"><Trash2 size={16} /></button>
                        </div>
                      </div>
                      
                      <p className="mb-4 leading-relaxed text-sm text-ink-secondary">{topic.description}</p>
                      
                      <div className="mt-auto flex items-center justify-between border-t border-black/5 pt-4 dark:border-white/5">
                        <div className="flex items-center gap-4 text-xs font-medium text-ink-muted">
                          <span className="flex items-center gap-1.5"><Clock size={14} /> {topic.estimatedTime}</span>
                          {topic.content && <span className="flex items-center gap-1.5"><BookOpen size={14} /> Content Available</span>}
                        </div>
                        
                        <Button variant="secondary" size="sm" className="h-8 gap-1 rounded-lg text-xs" onClick={() => handleToggleComplete(topic.id)}>
                          {topic.completed ? "Mark Incomplete" : "Mark Complete"} <ChevronRight size={14} />
                        </Button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
      
      <TopicModal 
        isOpen={isModalOpen} 
        topic={editingTopic} 
        onClose={() => setIsModalOpen(false)} 
        onSave={handleSaveTopic} 
      />
    </div>
  )
}
