import { lazy, Suspense } from 'react'
import { Link, useParams } from 'react-router'
import { ArrowLeft, ArrowRight, BookOpen, ChevronRight } from 'lucide-react'
import { Eyebrow } from '../components/Article'
import { findVisualizer, visualizers, VIZ_GROUPS } from '../data/visualizers'
import { isPublished } from '../lib/content'
import { useTitle } from '../lib/store'
import NotFound from './NotFound'

const Visualizer = lazy(() => import('../components/visualizers/Embedded').then((m) => ({ default: m.Visualizer })))

export function PlaygroundIndex() {
  useTitle('Playground', 'Twenty interactive visualisers for embeddings, chunking, BM25, hybrid search, reranking, HNSW and more.')
  return (
    <div className="mx-auto max-w-4xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Eyebrow>Playground</Eyebrow>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">See it move</h1>
      <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted">
        {visualizers.length} interactive visualisers. Each one computes the real algorithm in your browser on a small
        example: change a parameter and the scores, rankings and explanations update.
      </p>

      {VIZ_GROUPS.map((group) => (
        <section key={group} className="mt-10">
          <h2 className="border-b border-line pb-2 text-xs font-semibold uppercase tracking-wider text-faint">{group}</h2>
          <ul className="mt-4 grid gap-3 sm:grid-cols-2">
            {visualizers
              .filter((v) => v.group === group)
              .map((v) => (
                <li key={v.slug}>
                  <Link
                    to={`/playground/${v.slug}`}
                    className="group flex h-full gap-3 rounded-xl border border-line bg-surface p-4 transition-colors hover:border-accent"
                  >
                    <span className="font-mono text-xs text-faint">{String(v.n).padStart(2, '0')}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block font-medium leading-snug group-hover:text-accent">{v.title}</span>
                      <span className="mt-1 block text-[13.5px] leading-snug text-muted">{v.blurb}</span>
                    </span>
                  </Link>
                </li>
              ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

export function PlaygroundPage() {
  const { slug } = useParams()
  const v = findVisualizer(slug)
  useTitle(v?.title, v?.blurb)
  if (!v) return <NotFound />

  const at = visualizers.indexOf(v)
  const prev = visualizers[at - 1]
  const next = visualizers[at + 1]
  const [mod, topic] = v.lesson?.split('/') ?? []
  const lessonLive = !!v.lesson && isPublished(mod, topic)

  return (
    <div className="mx-auto max-w-4xl px-5 pb-24 pt-8 sm:px-8 sm:pt-10">
      <nav className="flex flex-wrap items-center gap-1.5 text-[13px] text-muted" aria-label="Breadcrumb">
        <Link to="/playground" className="hover:text-accent">
          Playground
        </Link>
        <ChevronRight size={13} className="text-faint" />
        <span>{v.group}</span>
      </nav>
      <h1 className="mt-4 font-serif text-[2.1rem] font-medium leading-[1.12] tracking-tight sm:text-[2.6rem]">{v.title}</h1>
      <p className="mt-2 text-lg leading-relaxed text-muted">{v.blurb}</p>

      <div className="mt-8">
        <Suspense fallback={<div className="h-80 animate-pulse rounded-xl bg-sunken" />}>
          <Visualizer slug={v.slug} />
        </Suspense>
      </div>

      {v.lesson && (
        <p className="mt-6 flex items-center gap-2 text-sm text-muted">
          <BookOpen size={15} className="text-accent" />
          {lessonLive ? (
            <>
              Read the lesson:{' '}
              <Link to={`/learn/${v.lesson}`} className="font-medium text-accent hover:underline">
                {topic.replace(/-/g, ' ')}
              </Link>
            </>
          ) : (
            <>
              The lesson that teaches this is in{' '}
              <Link to={`/learn/${mod}`} className="font-medium text-accent hover:underline">
                {mod.replace(/-/g, ' ')}
              </Link>
              .
            </>
          )}
        </p>
      )}

      <nav className="mt-10 grid gap-4 sm:grid-cols-2" aria-label="Visualiser navigation">
        {prev ? (
          <Link to={`/playground/${prev.slug}`} className="group rounded-xl border border-line bg-surface p-4 hover:border-accent">
            <span className="flex items-center gap-1.5 text-xs text-faint">
              <ArrowLeft size={13} /> Previous
            </span>
            <span className="mt-1 block font-medium group-hover:text-accent">{prev.title}</span>
          </Link>
        ) : (
          <span />
        )}
        {next && (
          <Link to={`/playground/${next.slug}`} className="group rounded-xl border border-line bg-surface p-4 text-right hover:border-accent">
            <span className="flex items-center justify-end gap-1.5 text-xs text-faint">
              Next <ArrowRight size={13} />
            </span>
            <span className="mt-1 block font-medium group-hover:text-accent">{next.title}</span>
          </Link>
        )}
      </nav>
    </div>
  )
}
