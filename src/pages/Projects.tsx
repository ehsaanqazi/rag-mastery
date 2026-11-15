import { Link } from 'react-router'
import { ArrowRight, Check, Clock } from 'lucide-react'
import { Eyebrow } from '../components/Article'
import { LEVELS, projects, type Project } from '../data/projects'
import { projectId, projectPublished } from '../lib/content'
import { useStore, useTitle } from '../lib/store'

const LEVEL_NOTE: Record<string, string> = {
  Beginner: 'After Phases 1 to 5. One retriever, one index, one prompt.',
  Intermediate: 'After Phases 6 to 9. Better retrieval, measured.',
  Advanced: 'After Phases 10 to 15. Graphs, agents, tenants and production.',
}

export default function Projects() {
  useTitle('Projects', 'Fifteen hands-on RAG projects, from a PDF question-answering system to a production platform.')
  const { done } = useStore()

  return (
    <div className="mx-auto max-w-4xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Eyebrow>Hands-on projects</Eyebrow>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">Build it to understand it</h1>
      <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted">
        {projects.length} projects in three levels. Each has an architecture, a folder structure, complete code, tests
        and expected results. Every project keeps the language model behind one small function, so retrieval can be
        built and tested without an API key.
      </p>

      {LEVELS.map((level) => (
        <section key={level} className="mt-12">
          <div className="flex flex-wrap items-baseline gap-x-3 border-b border-line pb-2">
            <h2 className="font-serif text-2xl font-medium">{level}</h2>
            <p className="text-sm text-muted">{LEVEL_NOTE[level]}</p>
          </div>
          <ul className="mt-4 grid gap-3 sm:grid-cols-2">
            {projects
              .filter((p) => p.level === level)
              .map((p) => (
                <li key={p.slug}>
                  <Card project={p} done={!!done[projectId(p.slug)]} />
                </li>
              ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

function Card({ project: p, done }: { project: Project; done: boolean }) {
  const published = projectPublished(p)
  const body = (
    <>
      <span className="flex items-center gap-2">
        <span className="font-mono text-xs text-faint">{String(p.n).padStart(2, '0')}</span>
        {p.complete && (
          <span className="rounded-full bg-accent-soft px-2 py-px text-[10.5px] font-semibold text-accent">
            Full implementation
          </span>
        )}
        {!published && <span className="rounded-full border border-line px-1.5 text-[10px] text-faint">soon</span>}
        {done && <Check size={14} className="ml-auto text-accent" />}
      </span>
      <span className={`mt-2 block font-medium leading-snug ${published ? 'group-hover:text-accent' : ''}`}>{p.title}</span>
      <span className="mt-1.5 block flex-1 text-[13.5px] leading-snug text-muted">{p.blurb}</span>
      <span className="mt-3 flex flex-wrap gap-1.5">
        {p.stack.map((s) => (
          <span key={s} className="rounded border border-line bg-sunken px-1.5 py-px font-mono text-[11px] text-muted">
            {s}
          </span>
        ))}
      </span>
      <span className="mt-3 flex items-center gap-1.5 text-xs text-faint">
        <Clock size={12} /> {p.hours}
        {published && <ArrowRight size={14} className="ml-auto transition-transform group-hover:translate-x-0.5 group-hover:text-accent" />}
      </span>
    </>
  )
  const cls = 'flex h-full flex-col rounded-xl border bg-surface p-4'
  return published ? (
    <Link to={`/projects/${p.slug}`} className={`group ${cls} border-line transition-colors hover:border-accent`}>
      {body}
    </Link>
  ) : (
    <div className={`${cls} border-dashed border-line`}>{body}</div>
  )
}
