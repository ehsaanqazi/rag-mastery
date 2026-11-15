import { Link } from 'react-router'
import { ArrowRight, Check, FlaskConical, Hammer } from 'lucide-react'
import { Eyebrow, ProgressBar } from '../components/Article'
import { modules, pad, phases, TOTAL_WEEKS, totalTopics } from '../data/curriculum'
import { projects } from '../data/projects'
import { visualizers } from '../data/visualizers'
import { lessons, modulePublished, projectPublished } from '../lib/content'
import { useStore, useTitle } from '../lib/store'

const OUTCOMES = [
  ['Ingest anything', 'Parse PDFs, HTML, tables and scans into clean, structured, versioned documents.'],
  ['Chunk and embed well', 'Choose chunking strategies and embedding models from evidence, not habit.'],
  ['Search properly', 'Build dense, sparse and hybrid retrieval, then rerank the candidates.'],
  ['Ground the answer', 'Engineer context and prompts that produce cited answers and honest refusals.'],
  ['Go beyond naive RAG', 'Design corrective, graph, hierarchical, agentic and multimodal systems.'],
  ['Measure everything', 'Evaluate retrieval and generation, and catch regressions before users do.'],
  ['Ship it', 'Deploy behind an API with tracing, caching, CI/CD and versioned indexes.'],
  ['Keep it safe', 'Defend against prompt injection and enforce permissions at retrieval time.'],
]

const PATH = [
  ['Understand', 'Phases 1–2', 'What RAG is, how language models work, and the Python you need.'],
  ['Index', 'Phases 3–6', 'Ingestion, chunking, embeddings and vector databases.'],
  ['Retrieve', 'Phases 7–9', 'Search, reranking, and turning context into grounded answers.'],
  ['Advance', 'Phases 10–12', 'Graph and agentic architectures, frameworks, and evaluation.'],
  ['Deploy', 'Phases 13–16', 'Observability, production, security, and capstone systems.'],
]

export default function Home() {
  useTitle()
  const { done, last } = useStore()
  const completed = lessons.filter((l) => done[l.id]).length
  const resume = lessons.find((l) => l.id === last) ?? lessons.find((l) => !done[l.id]) ?? lessons[0]
  const started = completed > 0 || !!last
  const featured = projects.filter((p) => p.complete)

  return (
    <div className="mx-auto max-w-5xl px-5 pb-24 sm:px-8">
      <section className="pb-14 pt-14 sm:pt-20">
        <Eyebrow>RAG Engineering Mastery</Eyebrow>
        <h1 className="mt-4 max-w-3xl font-serif text-[2.5rem] font-medium leading-[1.08] tracking-tight sm:text-[3.5rem]">
          Master Retrieval-Augmented Generation.
          <span className="block text-muted">From your first document to production-ready AI systems.</span>
        </h1>
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">
          A comprehensive, free, hands-on RAG engineering curriculum covering document processing, embeddings, vector
          databases, retrieval, reranking, Graph RAG, Agentic RAG and production deployment.
        </p>

        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Link
            to={resume ? resume.path : '/roadmap'}
            className="inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-[15px] font-semibold text-accent-fg transition-opacity hover:opacity-90"
          >
            {started ? 'Continue learning' : 'Start learning'} <ArrowRight size={16} />
          </Link>
          <Link
            to="/roadmap"
            className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-5 py-2.5 text-[15px] font-medium hover:border-faint"
          >
            View the curriculum
          </Link>
        </div>

        <dl className="mt-12 grid max-w-3xl grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
          {[
            [phases.length, 'phases'],
            [modules.length, 'modules'],
            [totalTopics, 'lessons'],
            [`~${TOTAL_WEEKS}`, 'weeks at 8–10 h/week'],
          ].map(([n, label]) => (
            <div key={label} className="bg-surface px-5 py-4">
              <dt className="font-serif text-3xl font-medium">{n}</dt>
              <dd className="text-sm text-muted">{label}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-faint">
          {lessons.length >= totalTopics
            ? `All ${totalTopics} lessons are published, with ${modules.length} module assessments, ${phases.length} phase exams and ${projects.length} projects.`
            : `${lessons.length} lessons and assessments are published so far. The rest of the syllabus is listed and marked "soon"; lessons appear as they are written.`}
        </p>
      </section>

      {started && resume && (
        <section className="mb-14 flex flex-wrap items-center gap-x-8 gap-y-4 rounded-xl border border-line bg-surface p-5">
          <div className="min-w-0 flex-1 basis-64">
            <p className="text-xs font-semibold uppercase tracking-wider text-faint">Pick up where you left off</p>
            <Link to={resume.path} className="mt-1 block truncate font-serif text-xl font-medium hover:text-accent">
              {resume.title}
            </Link>
            <p className="text-sm text-muted">{resume.module.title}</p>
          </div>
          <div className="basis-56">
            <div className="flex justify-between text-xs text-muted">
              <span>Published lessons completed</span>
              <span className="font-mono">
                {completed}/{lessons.length}
              </span>
            </div>
            <ProgressBar value={completed} total={lessons.length} className="mt-2" />
          </div>
        </section>
      )}

      <section className="border-t border-line pt-10">
        <h2 className="font-serif text-2xl font-medium">A textbook, a course and an engineering handbook</h2>
        <p className="mt-2 max-w-3xl leading-relaxed text-muted">
          Every lesson follows the same structure: the intuition first, then the mechanism, the mathematics where there
          is some, working Python with its real output, a production scenario, the mistakes people make, and exercises
          with hidden solutions. Diagrams and interactive visualisers sit next to the ideas they explain. Nothing needs
          an account, and nothing is behind a paywall.
        </p>

        <h3 className="mt-10 text-xs font-semibold uppercase tracking-wider text-faint">What you will be able to do</h3>
        <ul className="mt-4 grid gap-x-8 gap-y-5 sm:grid-cols-2">
          {OUTCOMES.map(([title, text]) => (
            <li key={title} className="flex gap-3">
              <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent-soft text-accent">
                <Check size={12} strokeWidth={3} />
              </span>
              <span>
                <span className="block font-medium">{title}</span>
                <span className="block text-[15px] leading-relaxed text-muted">{text}</span>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-16 border-t border-line pt-10">
        <h2 className="font-serif text-2xl font-medium">The learning path</h2>
        <p className="mt-1 text-muted">Five stages across sixteen phases. Each one builds on the last.</p>
        <ol className="mt-6 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-5">
          {PATH.map(([title, range, text], i) => (
            <li key={title} className="bg-surface p-4">
              <p className="font-mono text-[11px] font-semibold tracking-wider text-accent">
                {pad(i + 1)} · {range.toUpperCase()}
              </p>
              <p className="mt-1.5 font-serif text-lg font-medium">{title}</p>
              <p className="mt-1 text-[13px] leading-snug text-muted">{text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-16 border-t border-line pt-10">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-serif text-2xl font-medium">Build real systems</h2>
            <p className="mt-1 text-muted">
              {projects.length} projects from a first PDF assistant to a production platform, and {visualizers.length}{' '}
              interactive visualisers.
            </p>
          </div>
          <div className="flex gap-2 text-sm">
            <Link to="/projects" className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 font-medium hover:border-accent hover:text-accent">
              <Hammer size={14} /> All projects
            </Link>
            <Link to="/playground" className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-3 py-1.5 font-medium hover:border-accent hover:text-accent">
              <FlaskConical size={14} /> Playground
            </Link>
          </div>
        </div>
        <ul className="mt-6 grid gap-3 sm:grid-cols-3">
          {featured.map((p) => (
            <li key={p.slug}>
              <Link
                to={projectPublished(p) ? `/projects/${p.slug}` : '/projects'}
                className="group flex h-full flex-col rounded-xl border border-line bg-surface p-4 transition-colors hover:border-accent"
              >
                <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-accent">{p.level}</span>
                <span className="mt-1.5 font-medium leading-snug group-hover:text-accent">{p.title}</span>
                <span className="mt-1.5 flex-1 text-[13px] leading-snug text-muted">{p.blurb}</span>
                <span className="mt-3 text-xs text-faint">{p.stack.join(' · ')}</span>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-16 border-t border-line pt-10">
        <h2 className="font-serif text-2xl font-medium">The curriculum</h2>
        <p className="mt-1 text-muted">Sixteen phases, taken in order.</p>

        <div className="mt-8 space-y-10">
          {phases.map((phase) => (
            <div key={phase.n} className="grid gap-x-10 gap-y-4 border-t border-line pt-6 md:grid-cols-[15rem_1fr]">
              <div>
                <p className="font-mono text-xs font-semibold tracking-wider text-accent">
                  PHASE {pad(phase.n)} <span className="text-faint">· {phase.weeks}</span>
                </p>
                <h3 className="mt-1.5 font-serif text-xl font-medium leading-snug">{phase.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-muted">{phase.blurb}</p>
              </div>
              <ul className="grid content-start gap-2.5 sm:grid-cols-2">
                {phase.modules.map((m) => {
                  const published = modulePublished(m)
                  const mine = lessons.filter((l) => l.module === m)
                  const allDone = mine.length > 0 && mine.every((l) => done[l.id])
                  return (
                    <li key={m.slug}>
                      <Link
                        to={`/learn/${m.slug}`}
                        className="group flex h-full gap-3 rounded-xl border border-line bg-surface p-4 transition-colors hover:border-accent"
                      >
                        <span className="font-mono text-xs text-faint">{pad(m.n)}</span>
                        <span className="min-w-0 flex-1">
                          <span className="flex items-center gap-2">
                            <span className="font-medium leading-snug group-hover:text-accent">{m.title}</span>
                            {allDone && <Check size={14} className="shrink-0 text-accent" />}
                          </span>
                          <span className="mt-1 block text-[13px] leading-snug text-muted">{m.blurb}</span>
                          <span className="mt-2 block text-xs text-faint">
                            {m.topics.length} lessons{!published && ' · coming soon'}
                          </span>
                        </span>
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-16 rounded-xl border border-line bg-surface p-6 text-center sm:p-8">
        <p className="font-serif text-2xl font-medium">Learn. Retrieve. Reason. Build. Deploy.</p>
        <p className="mx-auto mt-2 max-w-xl text-muted">
          Our mission is to make world-class RAG engineering education free and accessible to everyone.
        </p>
        <Link
          to={resume ? resume.path : '/roadmap'}
          className="mt-5 inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-[15px] font-semibold text-accent-fg hover:opacity-90"
        >
          {started ? 'Continue learning' : 'Start with lesson one'} <ArrowRight size={16} />
        </Link>
      </section>
    </div>
  )
}
