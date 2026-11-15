import { Link } from 'react-router'
import { ClipboardCheck } from 'lucide-react'
import { Eyebrow } from '../components/Article'
import { modules, pad, phases, TOTAL_WEEKS, totalTopics } from '../data/curriculum'
import { hasExam, modulePublished, phaseLessons } from '../lib/content'
import { useStore, useTitle } from '../lib/store'

export default function Roadmap() {
  useTitle('Curriculum roadmap', 'All 16 phases and 57 modules of the RAG Engineering Mastery curriculum, in order.')
  const { done } = useStore()

  return (
    <div className="mx-auto max-w-3xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Eyebrow>Curriculum roadmap</Eyebrow>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">The {TOTAL_WEEKS}-week path</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">
        {phases.length} phases, {modules.length} modules and {totalTopics} lessons that take you from understanding what
        RAG is to architecting and deploying production retrieval systems. The estimate assumes 8 to 10 hours a week;
        go faster or slower as you need.
      </p>

      <ol className="mt-12">
        {phases.map((phase, i) => {
          const mine = phaseLessons(phase)
          const completed = mine.filter((l) => done[l.id]).length
          const complete = mine.length > 0 && completed === mine.length
          const started = completed > 0
          const topics = phase.modules.reduce((n, m) => n + m.topics.length, 0)
          return (
            <li key={phase.n} id={`phase-${phase.n}`} className="relative grid scroll-mt-20 grid-cols-[2.5rem_1fr] gap-x-5 pb-10">
              {i < phases.length - 1 && <span className="absolute bottom-0 left-5 top-10 w-px -translate-x-1/2 bg-line" />}
              <span
                className={`relative grid h-10 w-10 place-items-center rounded-full border font-mono text-sm font-semibold ${
                  complete
                    ? 'border-accent bg-accent text-accent-fg'
                    : started
                      ? 'border-accent bg-accent-soft text-accent'
                      : 'border-line bg-surface text-muted'
                }`}
              >
                {pad(phase.n)}
              </span>
              <div className="pt-1.5">
                <p className="font-mono text-xs text-faint">
                  {phase.weeks} · {phase.modules.length} modules · {topics} lessons
                  {mine.length > 0 && ` · ${completed}/${mine.length} completed`}
                </p>
                <h2 className="mt-0.5 font-serif text-2xl font-medium leading-snug">{phase.title}</h2>
                <p className="mt-1.5 leading-relaxed text-muted">{phase.blurb}</p>
                <ul className="mt-4 flex flex-wrap gap-2">
                  {phase.modules.map((m) => (
                    <li key={m.slug}>
                      <Link
                        to={`/learn/${m.slug}`}
                        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[13px] transition-colors hover:border-accent hover:text-accent ${
                          modulePublished(m) ? 'border-line bg-surface' : 'border-dashed border-line text-muted'
                        }`}
                      >
                        <span className="font-mono text-[11px] text-faint">{m.n}</span>
                        {m.title}
                      </Link>
                    </li>
                  ))}
                  {hasExam(phase.n) && (
                    <li>
                      <Link
                        to={`/exam/${phase.n}`}
                        className="inline-flex items-center gap-1.5 rounded-full border border-accent bg-accent-soft px-3 py-1 text-[13px] font-medium text-accent"
                      >
                        <ClipboardCheck size={13} /> Phase exam
                      </Link>
                    </li>
                  )}
                </ul>
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
