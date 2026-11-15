import { Link } from 'react-router'
import { ArrowRight, ClipboardCheck, GraduationCap } from 'lucide-react'
import { Eyebrow } from '../components/Article'
import { pad, phases } from '../data/curriculum'
import { ASSESSMENT, examId, hasAssessment, hasExam, lessonId } from '../lib/content'
import { useStore, useTitle, type QuizScore } from '../lib/store'

function Score({ s }: { s?: QuizScore }) {
  if (!s) return <span className="text-xs text-faint">Not attempted</span>
  const pct = Math.round((s.score / s.total) * 100)
  return (
    <span className={`font-mono text-xs font-medium ${pct >= 80 ? 'text-good' : pct >= 60 ? 'text-warn' : 'text-danger'}`}>
      {s.score}/{s.total} · {pct}%
    </span>
  )
}

export default function Assessments() {
  useTitle('Assessments', 'Module assessments and phase exams, with your best scores.')
  const { quiz } = useStore()
  const scores = Object.entries(quiz).filter(([id]) => id.endsWith(`/${ASSESSMENT}`) || id.startsWith('exam/'))
  const mean = scores.length
    ? Math.round((scores.reduce((n, [, s]) => n + s.score / s.total, 0) / scores.length) * 100)
    : null
  const anyPublished = phases.some((p) => hasExam(p.n) || p.modules.some(hasAssessment))

  return (
    <div className="mx-auto max-w-3xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Eyebrow>Assessments</Eyebrow>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">Test what you know</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">
        Each module ends with an assessment: a scored quiz, conceptual questions, coding exercises, an architecture
        design challenge and troubleshooting scenarios, all with hidden solutions. Each phase ends with an exam. Your
        best scores are kept in this browser.
      </p>

      {mean !== null && (
        <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line">
          <div className="bg-surface px-5 py-4">
            <dt className="font-serif text-3xl font-medium">{scores.length}</dt>
            <dd className="text-sm text-muted">assessments attempted</dd>
          </div>
          <div className="bg-surface px-5 py-4">
            <dt className="font-serif text-3xl font-medium">{mean}%</dt>
            <dd className="text-sm text-muted">average best score</dd>
          </div>
        </dl>
      )}

      {!anyPublished && (
        <p className="mt-10 rounded-xl border border-dashed border-line px-4 py-8 text-center text-sm text-muted">
          Assessments appear here as each module is published.
        </p>
      )}

      {phases.map((phase) => {
        const mods = phase.modules.filter(hasAssessment)
        const exam = hasExam(phase.n)
        if (!mods.length && !exam) return null
        return (
          <section key={phase.n} className="mt-10">
            <h2 className="flex items-baseline gap-2">
              <span className="font-mono text-xs font-semibold tracking-wider text-accent">PHASE {pad(phase.n)}</span>
              <span className="font-serif text-xl font-medium">{phase.title}</span>
            </h2>
            <ul className="mt-3 overflow-hidden rounded-xl border border-line bg-surface">
              {mods.map((m) => (
                <li key={m.slug} className="border-b border-line last:border-0">
                  <Link to={`/learn/${m.slug}/${ASSESSMENT}`} className="group flex items-center gap-3.5 px-4 py-3 hover:bg-sunken">
                    <ClipboardCheck size={17} className="shrink-0 text-faint group-hover:text-accent" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium">{m.title}</span>
                      <span className="block text-xs text-faint">Module {pad(m.n)} assessment</span>
                    </span>
                    <Score s={quiz[lessonId(m.slug, ASSESSMENT)]} />
                    <ArrowRight size={15} className="text-faint group-hover:text-accent" />
                  </Link>
                </li>
              ))}
              {exam && (
                <li className="border-b border-line last:border-0">
                  <Link to={`/exam/${phase.n}`} className="group flex items-center gap-3.5 bg-accent-soft/40 px-4 py-3 hover:bg-accent-soft">
                    <GraduationCap size={17} className="shrink-0 text-accent" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium">Phase {pad(phase.n)} exam</span>
                      <span className="block text-xs text-faint">Covers every module in this phase</span>
                    </span>
                    <Score s={quiz[examId(phase.n)]} />
                    <ArrowRight size={15} className="text-faint group-hover:text-accent" />
                  </Link>
                </li>
              )}
            </ul>
          </section>
        )
      })}
    </div>
  )
}
