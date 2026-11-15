import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router'
import { CornerDownLeft, Search } from 'lucide-react'
import { modules, pad, phases } from '../data/curriculum'
import { projects } from '../data/projects'
import { visualizers } from '../data/visualizers'
import {
  isPublished,
  lessons,
  loadLesson,
  loadProject,
  projectPublished,
  publishedProjects,
  searchableText,
} from '../lib/content'

type Kind = 'Phase' | 'Module' | 'Lesson' | 'Project' | 'Visualiser'

interface Entry {
  kind: Kind
  title: string
  context: string
  path: string
  published: boolean
}

const base: Entry[] = [
  ...phases.map<Entry>((p) => ({
    kind: 'Phase',
    title: p.title,
    context: `Phase ${pad(p.n)} · ${p.blurb}`,
    path: `/roadmap#phase-${p.n}`,
    published: true,
  })),
  ...modules.flatMap<Entry>((m) => [
    { kind: 'Module', title: m.title, context: `Module ${pad(m.n)} · ${m.blurb}`, path: `/learn/${m.slug}`, published: true },
    ...m.topics.map<Entry>((t) => {
      const published = isPublished(m.slug, t.slug)
      return {
        kind: 'Lesson',
        title: t.title,
        context: m.title,
        path: published ? `/learn/${m.slug}/${t.slug}` : `/learn/${m.slug}`,
        published,
      }
    }),
  ]),
  ...projects.map<Entry>((p) => ({
    kind: 'Project',
    title: p.title,
    context: `${p.level} project · ${p.stack.join(', ')}`,
    path: projectPublished(p) ? `/projects/${p.slug}` : '/projects',
    published: projectPublished(p),
  })),
  ...visualizers.map<Entry>((v) => ({
    kind: 'Visualiser',
    title: v.title,
    context: v.blurb,
    path: `/playground/${v.slug}`,
    published: true,
  })),
]

let bodies: Promise<Record<string, string>> | null = null
const loadBodies = () =>
  (bodies ??= Promise.all([
    ...lessons.map(async (l) => [l.path, searchableText(await loadLesson(l.module.slug, l.slug))] as const),
    ...publishedProjects.map(async (p) => [`/projects/${p.slug}`, searchableText(await loadProject(p.slug))] as const),
  ]).then((pairs) => Object.fromEntries(pairs.map(([path, text]) => [path, text]))))

interface Hit extends Entry {
  snippet?: string
}

/** Splits a query the same way lesson text is normalised, so `count_tokens` finds "count tokens". */
const queryTerms = (q: string) =>
  q
    .toLowerCase()
    .replace(/[#>*_`$|\\[\]]/g, ' ')
    .split(/\s+/)
    .filter(Boolean)

const KIND_BOOST: Record<Kind, number> = { Lesson: 0.5, Module: 0.4, Project: 0.3, Visualiser: 0.2, Phase: 0.1 }

function search(q: string, text: Record<string, string>, lower: Record<string, string>): Hit[] {
  const terms = queryTerms(q)
  if (!terms.length) return []
  const scored: [number, Hit][] = []
  for (const e of base) {
    const title = e.title.toLowerCase()
    const ctx = e.context.toLowerCase()
    const body = e.published ? (lower[e.path] ?? '') : ''
    let score = 0
    let ok = true
    let snippet: string | undefined
    for (const t of terms) {
      if (title.startsWith(t)) score += 12
      else if (title.includes(t)) score += 8
      else if (ctx.includes(t)) score += 3
      else if (body.includes(t)) {
        score += 1
        if (!snippet) {
          const at = body.indexOf(t)
          const from = Math.max(0, at - 50)
          snippet = (from > 0 ? '…' : '') + text[e.path].slice(from, at + 90).trim() + '…'
        }
      } else {
        ok = false
        break
      }
    }
    if (ok) scored.push([score + (e.published ? KIND_BOOST[e.kind] : 0), { ...e, snippet }])
  }
  return scored
    .sort((a, b) => b[0] - a[0])
    .slice(0, 14)
    .map(([, h]) => h)
}

/** Wraps every occurrence of a search term in <mark>. */
function Highlight({ text, terms }: { text: string; terms: string[] }): ReactNode {
  if (!terms.length) return text
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  return text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : part))
}

export default function SearchDialog({ onClose }: { onClose: () => void }) {
  const [q, setQ] = useState('')
  const [text, setText] = useState<Record<string, string>>({})
  const [sel, setSel] = useState(0)
  const navigate = useNavigate()
  const listRef = useRef<HTMLUListElement>(null)

  useEffect(() => {
    let live = true
    loadBodies().then((t) => live && setText(t))
    return () => {
      live = false
    }
  }, [])

  const lower = useMemo(
    () => Object.fromEntries(Object.entries(text).map(([k, v]) => [k, v.toLowerCase()])),
    [text],
  )
  const terms = useMemo(() => queryTerms(q), [q])
  const hits = useMemo(() => search(q, text, lower), [q, text, lower])

  useEffect(() => setSel(0), [q])

  useEffect(() => {
    listRef.current?.children[sel]?.scrollIntoView({ block: 'nearest' })
  }, [sel])

  const go = (h: Hit) => {
    navigate(h.path)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]" role="dialog" aria-modal="true" aria-label="Search">
      <div className="absolute inset-0 bg-black/45 backdrop-blur-[2px]" onClick={onClose} />
      <div className="relative w-full max-w-xl overflow-hidden rounded-2xl border border-line bg-surface shadow-2xl">
        <div className="flex items-center gap-3 border-b border-line px-4">
          <Search size={17} className="text-faint" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
              else if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSel((s) => Math.min(s + 1, hits.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSel((s) => Math.max(s - 1, 0))
              } else if (e.key === 'Enter' && hits[sel]) go(hits[sel])
            }}
            placeholder="Search lessons, modules, projects and code…"
            aria-label="Search the curriculum"
            className="h-13 flex-1 bg-transparent text-[15px] outline-none placeholder:text-faint"
          />
          <kbd className="rounded border border-line bg-sunken px-1.5 font-mono text-[11px] text-faint">esc</kbd>
        </div>

        {q.trim() === '' ? (
          <div className="px-5 py-7 text-center text-sm text-muted">
            <p>
              Search all {phases.length} phases, {modules.length} modules, {projects.length} projects and{' '}
              {visualizers.length} visualisers.
            </p>
            <p className="mt-3 flex flex-wrap justify-center gap-2">
              {['BM25', 'chunk overlap', 'cosine', 'HNSW', 'reranking', 'prompt injection'].map((s) => (
                <button
                  key={s}
                  onClick={() => setQ(s)}
                  className="rounded-full border border-line px-2.5 py-0.5 text-[13px] hover:border-accent hover:text-accent"
                >
                  {s}
                </button>
              ))}
            </p>
          </div>
        ) : hits.length === 0 ? (
          <div className="px-5 py-8 text-center text-sm text-muted">
            <p>
              No results for “<span className="font-medium text-fg">{q}</span>”.
            </p>
            <p className="mt-1.5">Try a shorter or more general term, or browse the curriculum roadmap.</p>
          </div>
        ) : (
          <ul ref={listRef} className="thin-scroll max-h-[56vh] overflow-y-auto p-2">
            {hits.map((h, i) => (
              <li key={h.kind + h.path + h.title}>
                <button
                  onClick={() => go(h)}
                  onMouseMove={() => setSel(i)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left ${i === sel ? 'bg-accent-soft' : ''}`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2 text-sm font-medium">
                      <span className="truncate">
                        <Highlight text={h.title} terms={terms} />
                      </span>
                      <span className="shrink-0 rounded-full border border-line px-1.5 text-[10px] font-normal text-faint">
                        {h.published ? h.kind : 'soon'}
                      </span>
                    </span>
                    <span className="block truncate text-xs text-muted">
                      <Highlight text={h.snippet ?? h.context} terms={terms} />
                    </span>
                  </span>
                  {i === sel && <CornerDownLeft size={14} className="shrink-0 text-accent" />}
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center gap-4 border-t border-line bg-sunken px-4 py-2 text-[11px] text-faint">
          <span>
            <kbd className="font-mono">↑↓</kbd> navigate
          </span>
          <span>
            <kbd className="font-mono">↵</kbd> open
          </span>
          <span className="ml-auto">Searches titles, descriptions and full lesson text</span>
        </div>
      </div>
    </div>
  )
}
