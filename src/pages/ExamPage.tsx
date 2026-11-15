import { Link, useParams } from 'react-router'
import { ChevronRight } from 'lucide-react'
import Markdown from '../components/Markdown'
import { Skeleton, useSource } from '../components/Article'
import { pad, phases } from '../data/curriculum'
import { examId, hasExam, loadExam } from '../lib/content'
import { useStore, useTitle } from '../lib/store'
import NotFound from './NotFound'

export default function ExamPage() {
  const { n } = useParams()
  const phase = phases.find((p) => String(p.n) === n)
  const available = !!phase && hasExam(phase.n)
  const { source } = useSource(available ? `exam-${phase.n}` : null, () => loadExam(phase!.n))
  const best = useStore().quiz[phase ? examId(phase.n) : '']
  useTitle(phase ? `Phase ${pad(phase.n)} exam` : undefined)

  if (!phase) return <NotFound />
  if (!available)
    return (
      <div className="mx-auto max-w-xl px-5 py-24 text-center">
        <h1 className="font-serif text-3xl font-medium">This exam isn't published yet</h1>
        <p className="mt-3 text-muted">The Phase {pad(phase.n)} exam will appear once the phase's lessons are written.</p>
        <Link to="/assessments" className="mt-6 inline-block font-medium text-accent hover:underline">
          Back to assessments
        </Link>
      </div>
    )

  return (
    <div className="mx-auto max-w-[46rem] px-5 pb-24 pt-8 sm:px-8 sm:pt-10">
      <nav className="flex flex-wrap items-center gap-1.5 text-[13px] text-muted" aria-label="Breadcrumb">
        <Link to="/assessments" className="hover:text-accent">
          Assessments
        </Link>
        <ChevronRight size={13} className="text-faint" />
        <span className="font-mono text-xs text-faint">PHASE {pad(phase.n)}</span>
      </nav>
      <h1 className="mt-4 font-serif text-[2.2rem] font-medium leading-[1.12] tracking-tight sm:text-[2.8rem]">
        Phase {pad(phase.n)} exam
      </h1>
      <p className="mt-2 text-lg text-muted">{phase.title}</p>
      {best && (
        <p className="mt-3 text-sm text-muted">
          Best score so far:{' '}
          <span className="font-mono font-medium text-fg">
            {best.score}/{best.total}
          </span>{' '}
          ({Math.round((best.score / best.total) * 100)}%)
        </p>
      )}
      <div className="mt-8 min-h-[50vh] border-t border-line pt-8">
        {source ? <Markdown source={source} id={examId(phase.n)} /> : <Skeleton />}
      </div>
    </div>
  )
}
