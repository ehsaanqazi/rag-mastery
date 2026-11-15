import { Link } from 'react-router'
import { Bookmark, NotebookPen, X } from 'lucide-react'
import { Empty, Eyebrow } from '../components/Article'
import { projects } from '../data/projects'
import { lessons, projectId } from '../lib/content'
import { actions, useStore, useTitle } from '../lib/store'

interface Item {
  id: string
  title: string
  context: string
  path: string
}

const items: Item[] = [
  ...lessons.map((l) => ({ id: l.id, title: l.title, context: l.module.title, path: l.path })),
  ...projects.map((p) => ({ id: projectId(p.slug), title: p.title, context: `${p.level} project`, path: `/projects/${p.slug}` })),
]

export default function Library() {
  useTitle('Bookmarks and notes')
  const { bookmarks, notes } = useStore()
  const saved = items.filter((l) => bookmarks[l.id])
  const noted = items.filter((l) => notes[l.id])

  return (
    <div className="mx-auto max-w-3xl px-5 pb-24 pt-10 sm:px-8 sm:pt-14">
      <Eyebrow>My library</Eyebrow>
      <h1 className="mt-3 font-serif text-4xl font-medium tracking-tight sm:text-5xl">Bookmarks and notes</h1>
      <p className="mt-4 text-lg leading-relaxed text-muted">
        Everything here is stored in this browser only. You can back it up or move it to another device from the{' '}
        <Link to="/progress" className="font-medium text-accent hover:underline">
          progress dashboard
        </Link>
        .
      </p>

      <h2 className="mt-12 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-faint">
        <Bookmark size={14} /> Bookmarks {saved.length > 0 && `· ${saved.length}`}
      </h2>
      {saved.length === 0 ? (
        <Empty>Bookmark a lesson or project to find it here later.</Empty>
      ) : (
        <ul className="mt-3 overflow-hidden rounded-xl border border-line bg-surface">
          {saved.map((l) => (
            <li key={l.id} className="flex items-center border-b border-line last:border-0">
              <Link to={l.path} className="min-w-0 flex-1 px-4 py-3 hover:bg-sunken">
                <span className="block truncate font-medium">{l.title}</span>
                <span className="block text-sm text-muted">{l.context}</span>
              </Link>
              <button
                onClick={() => actions.toggleBookmark(l.id)}
                aria-label={`Remove bookmark for ${l.title}`}
                className="p-3 text-faint hover:text-danger"
              >
                <X size={16} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <h2 className="mt-12 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-faint">
        <NotebookPen size={14} /> Notes {noted.length > 0 && `· ${noted.length}`}
      </h2>
      {noted.length === 0 ? (
        <Empty>Notes you write at the bottom of a lesson or project are collected here.</Empty>
      ) : (
        <ul className="mt-3 space-y-3">
          {noted.map((l) => (
            <li key={l.id} className="rounded-xl border border-line bg-surface p-4">
              <Link to={l.path} className="font-medium hover:text-accent">
                {l.title}
              </Link>
              <span className="ml-2 text-sm text-faint">{l.context}</span>
              <p className="mt-2 whitespace-pre-wrap text-[15px] leading-relaxed text-muted">{notes[l.id]}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
