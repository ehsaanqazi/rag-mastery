import { useEffect, useMemo } from 'react'
import { Link, useParams } from 'react-router'
import { ArrowLeft, ArrowRight, Bookmark, Check, ChevronRight, Clock } from 'lucide-react'
import Markdown from '../components/Markdown'
import { InlineToc, Notes, ProgressBar, Rail, ReadingBar, Skeleton, useSource } from '../components/Article'
import { findModule, pad, phaseOf } from '../data/curriculum'
import { ASSESSMENT, extractHeadings, lessons, loadLesson, moduleLessons, readingMinutes } from '../lib/content'
import { actions, useStore, useTitle } from '../lib/store'
import NotFound from './NotFound'

export default function LessonPage() {
  const { moduleSlug = '', topicSlug = '' } = useParams()
  const index = lessons.findIndex((l) => l.module.slug === moduleSlug && l.slug === topicSlug)
  const lesson = lessons[index]
  const { done, bookmarks } = useStore()

  const { source } = useSource(lesson?.id ?? null, () => loadLesson(lesson.module.slug, lesson.slug))
  const headings = useMemo(() => (source ? extractHeadings(source) : []), [source])

  useEffect(() => {
    if (lesson) actions.setLast(lesson.id)
  }, [lesson])

  const isQuiz = lesson?.slug === ASSESSMENT
  useTitle(
    lesson ? (isQuiz ? `${lesson.module.title}: assessment` : lesson.title) : undefined,
    lesson ? `${lesson.title}, a lesson in ${lesson.module.title}. ${lesson.module.blurb}` : undefined,
  )

  if (!lesson) return findModule(moduleSlug) ? <Unpublished moduleSlug={moduleSlug} /> : <NotFound />

  const m = lesson.module
  const phase = phaseOf(m)
  const siblings = moduleLessons(m)
  const position = siblings.indexOf(lesson) + 1
  const completedInModule = siblings.filter((l) => done[l.id]).length
  const prev = lessons[index - 1]
  const next = lessons[index + 1]
  const isDone = !!done[lesson.id]
  const isBookmarked = !!bookmarks[lesson.id]

  return (
    <>
      <ReadingBar />
      <div className="mx-auto flex max-w-6xl justify-center gap-12 px-5 pb-24 pt-8 sm:px-8 sm:pt-10">
        <article className="min-w-0 max-w-[46rem] flex-1">
          <nav className="flex flex-wrap items-center gap-1.5 text-[13px] text-muted" aria-label="Breadcrumb">
            <Link to={`/roadmap#phase-${phase.n}`} className="font-mono text-xs text-faint hover:text-accent">
              PHASE {pad(phase.n)}
            </Link>
            <ChevronRight size={13} className="text-faint" />
            <Link to={`/learn/${m.slug}`} className="hover:text-accent">
              {m.title}
            </Link>
          </nav>

          <header className="mt-4">
            <h1 className="font-serif text-[2.2rem] font-medium leading-[1.12] tracking-tight sm:text-[2.9rem]">
              {isQuiz ? `${m.title}: assessment` : lesson.title}
            </h1>
            <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-muted">
              <span>
                Lesson {position} of {siblings.length}
              </span>
              {source && (
                <span className="flex items-center gap-1.5">
                  <Clock size={14} /> {readingMinutes(source)} min {isQuiz ? '' : 'read'}
                </span>
              )}
              {isDone && (
                <span className="flex items-center gap-1.5 text-accent">
                  <Check size={14} /> Completed
                </span>
              )}
              <button
                onClick={() => actions.toggleBookmark(lesson.id)}
                aria-pressed={isBookmarked}
                className={`ml-auto flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[13px] transition-colors ${
                  isBookmarked ? 'border-accent bg-accent-soft text-accent' : 'border-line hover:border-faint hover:text-fg'
                }`}
              >
                <Bookmark size={14} fill={isBookmarked ? 'currentColor' : 'none'} />
                {isBookmarked ? 'Bookmarked' : 'Bookmark'}
              </button>
            </div>
          </header>

          <div className="mt-8 min-h-[50vh] border-t border-line pt-8">
            {source ? (
              <>
                {!isQuiz && <InlineToc headings={headings} />}
                <Markdown source={source} id={lesson.id} />
              </>
            ) : (
              <Skeleton />
            )}
          </div>

          {source && (
            <>
              <Notes id={lesson.id} />

              <div className="mt-8 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line bg-surface p-5">
                <div>
                  <p className="font-medium">{isDone ? 'Lesson completed' : 'Finished this lesson?'}</p>
                  <p className="text-sm text-muted">Your progress is saved in this browser.</p>
                </div>
                <button
                  onClick={() => actions.toggleDone(lesson.id)}
                  aria-pressed={isDone}
                  className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                    isDone ? 'border border-accent bg-accent-soft text-accent' : 'bg-accent text-accent-fg hover:opacity-90'
                  }`}
                >
                  <Check size={15} /> {isDone ? 'Completed' : 'Mark as complete'}
                </button>
              </div>

              <nav className="mt-4 grid gap-4 sm:grid-cols-2" aria-label="Lesson navigation">
                {prev ? (
                  <Link to={prev.path} className="group rounded-xl border border-line bg-surface p-4 hover:border-accent">
                    <span className="flex items-center gap-1.5 text-xs text-faint">
                      <ArrowLeft size={13} /> Previous
                    </span>
                    <span className="mt-1 block font-medium group-hover:text-accent">{prev.title}</span>
                  </Link>
                ) : (
                  <span />
                )}
                {next && (
                  <Link
                    to={next.path}
                    onClick={() => actions.markDone(lesson.id)}
                    className="group rounded-xl border border-line bg-surface p-4 text-right hover:border-accent"
                  >
                    <span className="flex items-center justify-end gap-1.5 text-xs text-faint">
                      Next <ArrowRight size={13} />
                    </span>
                    <span className="mt-1 block font-medium group-hover:text-accent">{next.title}</span>
                  </Link>
                )}
              </nav>
            </>
          )}
        </article>

        <aside className="hidden w-56 shrink-0 xl:block">
          <Rail headings={headings}>
            <div className="mb-5">
              <div className="flex justify-between text-[11px] font-semibold uppercase tracking-wider text-faint">
                <span>Module progress</span>
                <span className="font-mono">
                  {completedInModule}/{siblings.length}
                </span>
              </div>
              <ProgressBar value={completedInModule} total={siblings.length} className="mt-2 !h-1" />
            </div>
          </Rail>
        </aside>
      </div>
    </>
  )
}

function Unpublished({ moduleSlug }: { moduleSlug: string }) {
  return (
    <div className="mx-auto max-w-xl px-5 py-24 text-center">
      <h1 className="font-serif text-3xl font-medium">This lesson isn't published yet</h1>
      <p className="mt-3 text-muted">It's on the syllabus and will appear here once it's written.</p>
      <Link to={`/learn/${moduleSlug}`} className="mt-6 inline-block font-medium text-accent hover:underline">
        Back to the module
      </Link>
    </div>
  )
}
