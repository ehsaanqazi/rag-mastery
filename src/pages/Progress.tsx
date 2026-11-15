import { useRef, useState } from 'react'
import { Link } from 'react-router'
import { Download, RotateCcw, Upload } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Eyebrow, ProgressBar } from '../components/Article'
import { modules, pad, phases, totalTopics } from '../data/curriculum'
import { projects } from '../data/projects'
import { lessons, moduleLessons, phaseLessons, projectId } from '../lib/content'
import { actions, useStore, useTitle } from '../lib/store'

const DAY = 86_400_000

export default function Progress() {
  useTitle('Progress dashboard')
  const { done, quiz, bookmarks, notes, last } = useStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)

  const completed = lessons.filter((l) => done[l.id])
  const modulesDone = modules.filter((m) => {
    const mine = moduleLessons(m)
    return mine.length > 0 && mine.every((l) => done[l.id])
  }).length
  const projectsBuilt = projects.filter((p) => done[projectId(p.slug)]).length
  const scores = Object.values(quiz)
  const mean = scores.length ? Math.round((scores.reduce((n, s) => n + s.score / s.total, 0) / scores.length) * 100) : null
  const resume = lessons.find((l) => l.id === last) ?? lessons.find((l) => !done[l.id])

  // completions per day for the last 14 days
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const days = Array.from({ length: 14 }, (_, i) => {
    const start = today.getTime() - (13 - i) * DAY
    return {
      day: new Date(start).toLocaleDateString(undefined, { day: 'numeric', month: 'short' }),
      lessons: Object.values(done).filter((t) => t >= start && t < start + DAY).length,
    }
  })
  const activeDays = new Set(Object.values(done).map((t) => new Date(t).toDateString())).size

  const download = () => {
    const url = URL.createObjectURL(new Blob([actions.export()], { type: 'application/json' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `rag-mastery-progress-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
    setMessage({ ok: true, text: 'Progress exported.' })
  }

  const upload = async (file: File | undefined) => {
    if (!file) return
    if (!confirm('Importing replaces the progress, bookmarks and notes in this browser. Continue?')) return
    const ok = actions.import(await file.text())
    setMessage(ok ? { ok: true, text: 'Progress imported.' } : { ok: false, text: 'That file is not a RAG Mastery progress export.' })
    if (fileRef.current) fileRef.current.value = ''
  }

  const reset = () => {
    if (confirm('Erase all progress, bookmarks, notes and scores in this browser? This cannot be undone.')) {
      actions.reset()
      setMessage({ ok: true, text: 'Progress reset.' })
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Eyebrow>Progress dashboard</Eyebrow>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">Your learning</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">
        Stored in this browser only: no account, nothing sent anywhere.
        {resume && (
          <>
            {' '}
            <Link to={resume.path} className="font-medium text-accent hover:underline">
              {completed.length ? 'Continue with' : 'Start with'} “{resume.title}”
            </Link>
            .
          </>
        )}
      </p>

      <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
        {[
          [`${completed.length}/${lessons.length}`, 'published lessons completed'],
          [`${Math.round((completed.length / totalTopics) * 100)}%`, `of the full ${totalTopics}-lesson curriculum`],
          [`${modulesDone}/${modules.length}`, 'modules completed'],
          [`${projectsBuilt}/${projects.length}`, 'projects built'],
          [scores.length, 'quizzes taken'],
          [mean === null ? '–' : `${mean}%`, 'average best quiz score'],
          [Object.keys(bookmarks).length, 'bookmarks'],
          [activeDays, activeDays === 1 ? 'day studied' : 'days studied'],
        ].map(([n, label]) => (
          <div key={label} className="bg-surface px-5 py-4">
            <dt className="font-serif text-[1.7rem] font-medium leading-tight">{n}</dt>
            <dd className="mt-0.5 text-[13px] leading-snug text-muted">{label}</dd>
          </div>
        ))}
      </dl>

      <section className="mt-12">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">Lessons completed, last 14 days</h2>
        <div className="mt-3 h-48 rounded-xl border border-line bg-surface p-3 pl-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={days} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="day" tickLine={false} axisLine={false} interval={1} />
              <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: 'var(--sunken)' }}
                contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, fontSize: 13 }}
              />
              <Bar dataKey="lessons" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="mt-12">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">By phase</h2>
        <ul className="mt-3 overflow-hidden rounded-xl border border-line bg-surface">
          {phases.map((p) => {
            const mine = phaseLessons(p)
            const n = mine.filter((l) => done[l.id]).length
            return (
              <li key={p.n} className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-line px-4 py-3 last:border-0">
                <span className="font-mono text-xs font-semibold text-accent">{pad(p.n)}</span>
                <Link to={`/roadmap#phase-${p.n}`} className="min-w-0 flex-1 basis-56 truncate text-[15px] hover:text-accent">
                  {p.title}
                </Link>
                {mine.length ? (
                  <>
                    <ProgressBar value={n} total={mine.length} className="w-32" />
                    <span className="w-12 text-right font-mono text-xs text-muted">
                      {n}/{mine.length}
                    </span>
                  </>
                ) : (
                  <span className="text-xs text-faint">not yet published</span>
                )}
              </li>
            )
          })}
        </ul>
      </section>

      {scores.length > 0 && (
        <section className="mt-12">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">Quiz scores</h2>
          <ul className="mt-3 overflow-hidden rounded-xl border border-line bg-surface">
            {Object.entries(quiz)
              .sort((a, b) => (b[1].at ?? 0) - (a[1].at ?? 0))
              .map(([id, s]) => {
                const lesson = lessons.find((l) => l.id === id)
                const exam = /^exam\/phase-(\d+)$/.exec(id)
                const title = lesson
                  ? lesson.slug === 'assessment'
                    ? `${lesson.module.title}: assessment`
                    : lesson.title
                  : exam
                    ? `Phase ${exam[1]} exam`
                    : id
                const path = lesson?.path ?? (exam ? `/exam/${Number(exam[1])}` : null)
                return (
                  <li key={id} className="flex items-center gap-4 border-b border-line px-4 py-2.5 text-[15px] last:border-0">
                    {path ? (
                      <Link to={path} className="min-w-0 flex-1 truncate hover:text-accent">
                        {title}
                      </Link>
                    ) : (
                      <span className="min-w-0 flex-1 truncate">{title}</span>
                    )}
                    {s.attempts && s.attempts > 1 && <span className="text-xs text-faint">{s.attempts} attempts</span>}
                    <span className="font-mono text-xs text-muted">
                      {s.score}/{s.total}
                    </span>
                  </li>
                )
              })}
          </ul>
        </section>
      )}

      <section className="mt-12 rounded-xl border border-line bg-surface p-5">
        <h2 className="font-medium">Back up or move your progress</h2>
        <p className="mt-1 text-sm leading-relaxed text-muted">
          Export a JSON file with your completed lessons, bookmarks, notes ({Object.keys(notes).length}) and quiz scores,
          and import it in another browser or device.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-sm">
          <button onClick={download} className="inline-flex items-center gap-2 rounded-lg bg-accent px-3.5 py-2 font-semibold text-accent-fg hover:opacity-90">
            <Download size={15} /> Export progress
          </button>
          <button
            onClick={() => fileRef.current?.click()}
            className="inline-flex items-center gap-2 rounded-lg border border-line px-3.5 py-2 font-medium hover:border-faint"
          >
            <Upload size={15} /> Import progress
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => upload(e.target.files?.[0])}
          />
          <button
            onClick={reset}
            className="ml-auto inline-flex items-center gap-2 rounded-lg border border-line px-3.5 py-2 font-medium text-danger hover:border-danger"
          >
            <RotateCcw size={15} /> Reset
          </button>
        </div>
        {message && (
          <p role="status" className={`mt-3 text-sm ${message.ok ? 'text-good' : 'text-danger'}`}>
            {message.text}
          </p>
        )}
      </section>
    </div>
  )
}
