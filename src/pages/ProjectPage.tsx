import { useMemo } from 'react'
import { Link, useParams } from 'react-router'
import { ArrowLeft, ArrowRight, Bookmark, Check, ChevronRight, Clock } from 'lucide-react'
import Markdown from '../components/Markdown'
import { InlineToc, Notes, Rail, ReadingBar, Skeleton, useSource } from '../components/Article'
import { findProject, projects } from '../data/projects'
import { extractHeadings, loadProject, projectId, projectPublished, readingMinutes } from '../lib/content'
import { actions, useStore, useTitle } from '../lib/store'
import NotFound from './NotFound'

export default function ProjectPage() {
  const { slug } = useParams()
  const project = findProject(slug)
  const published = !!project && projectPublished(project)
  const { done, bookmarks } = useStore()
  const { source } = useSource(published ? project.slug : null, () => loadProject(project!.slug))
  const headings = useMemo(() => (source ? extractHeadings(source) : []), [source])
  useTitle(project?.title, project?.blurb)

  if (!project) return <NotFound />
  if (!published)
    return (
      <div className="mx-auto max-w-xl px-5 py-24 text-center">
        <h1 className="font-serif text-3xl font-medium">This project isn't published yet</h1>
        <p className="mt-3 text-muted">{project.blurb}</p>
        <Link to="/projects" className="mt-6 inline-block font-medium text-accent hover:underline">
          Back to projects
        </Link>
      </div>
    )

  const id = projectId(project.slug)
  const isDone = !!done[id]
  const isBookmarked = !!bookmarks[id]
  const live = projects.filter(projectPublished)
  const at = live.indexOf(project)
  const prev = live[at - 1]
  const next = live[at + 1]

  return (
    <>
      <ReadingBar />
      <div className="mx-auto flex max-w-6xl justify-center gap-12 px-5 pb-24 pt-8 sm:px-8 sm:pt-10">
        <article className="min-w-0 max-w-[46rem] flex-1">
          <nav className="flex flex-wrap items-center gap-1.5 text-[13px] text-muted" aria-label="Breadcrumb">
            <Link to="/projects" className="hover:text-accent">
              Projects
            </Link>
            <ChevronRight size={13} className="text-faint" />
            <span>{project.level}</span>
          </nav>

          <header className="mt-4">
            <p className="font-mono text-xs font-semibold uppercase tracking-wider text-accent">
              Project {String(project.n).padStart(2, '0')} · {project.level}
            </p>
            <h1 className="mt-2 font-serif text-[2.2rem] font-medium leading-[1.12] tracking-tight sm:text-[2.8rem]">
              {project.title}
            </h1>
            <p className="mt-3 text-lg leading-relaxed text-muted">{project.blurb}</p>
            <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-muted">
              <span className="flex items-center gap-1.5">
                <Clock size={14} /> {project.hours} to build
              </span>
              {source && <span>{readingMinutes(source)} min read</span>}
              {isDone && (
                <span className="flex items-center gap-1.5 text-accent">
                  <Check size={14} /> Built
                </span>
              )}
              <button
                onClick={() => actions.toggleBookmark(id)}
                aria-pressed={isBookmarked}
                className={`ml-auto flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[13px] transition-colors ${
                  isBookmarked ? 'border-accent bg-accent-soft text-accent' : 'border-line hover:border-faint hover:text-fg'
                }`}
              >
                <Bookmark size={14} fill={isBookmarked ? 'currentColor' : 'none'} />
                {isBookmarked ? 'Bookmarked' : 'Bookmark'}
              </button>
            </div>
            <ul className="mt-4 flex flex-wrap gap-1.5">
              {project.stack.map((s) => (
                <li key={s} className="rounded border border-line bg-sunken px-1.5 py-px font-mono text-[11.5px] text-muted">
                  {s}
                </li>
              ))}
            </ul>
          </header>

          <div className="mt-8 min-h-[50vh] border-t border-line pt-8">
            {source ? (
              <>
                <InlineToc headings={headings} />
                <Markdown source={source} id={id} />
              </>
            ) : (
              <Skeleton />
            )}
          </div>

          {source && (
            <>
              <Notes id={id} />
              <div className="mt-8 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line bg-surface p-5">
                <div>
                  <p className="font-medium">{isDone ? 'Project built' : 'Built and tested it?'}</p>
                  <p className="text-sm text-muted">Mark it so it counts in your progress dashboard.</p>
                </div>
                <button
                  onClick={() => actions.toggleDone(id)}
                  aria-pressed={isDone}
                  className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                    isDone ? 'border border-accent bg-accent-soft text-accent' : 'bg-accent text-accent-fg hover:opacity-90'
                  }`}
                >
                  <Check size={15} /> {isDone ? 'Built' : 'Mark as built'}
                </button>
              </div>
              <nav className="mt-4 grid gap-4 sm:grid-cols-2" aria-label="Project navigation">
                {prev ? (
                  <Link to={`/projects/${prev.slug}`} className="group rounded-xl border border-line bg-surface p-4 hover:border-accent">
                    <span className="flex items-center gap-1.5 text-xs text-faint">
                      <ArrowLeft size={13} /> Previous project
                    </span>
                    <span className="mt-1 block font-medium group-hover:text-accent">{prev.title}</span>
                  </Link>
                ) : (
                  <span />
                )}
                {next && (
                  <Link to={`/projects/${next.slug}`} className="group rounded-xl border border-line bg-surface p-4 text-right hover:border-accent">
                    <span className="flex items-center justify-end gap-1.5 text-xs text-faint">
                      Next project <ArrowRight size={13} />
                    </span>
                    <span className="mt-1 block font-medium group-hover:text-accent">{next.title}</span>
                  </Link>
                )}
              </nav>
            </>
          )}
        </article>

        <aside className="hidden w-56 shrink-0 xl:block">
          <Rail headings={headings} />
        </aside>
      </div>
    </>
  )
}
