import { useEffect, useState, type ReactNode } from 'react'
import { ChevronRight, ListTree, NotebookPen } from 'lucide-react'
import type { Heading } from '../lib/content'
import { actions, useStore } from '../lib/store'

/** Loads a Markdown source and forgets it when the key changes. */
export function useSource(key: string | null, load: () => Promise<string>): { source: string | null; failed: boolean } {
  const [loaded, setLoaded] = useState<{ key: string; source: string | null } | null>(null)

  useEffect(() => {
    if (!key) return
    let cancelled = false
    load().then(
      (source) => !cancelled && setLoaded({ key, source }),
      () => !cancelled && setLoaded({ key, source: null }),
    )
    return () => {
      cancelled = true
    }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps

  const current = loaded && loaded.key === key ? loaded : null
  return { source: current?.source ?? null, failed: !!current && current.source === null }
}

/** Fraction of the page scrolled, 0 to 1. */
function useScrollProgress(): number {
  const [p, setP] = useState(0)
  useEffect(() => {
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      setP(max > 0 ? Math.min(1, window.scrollY / max) : 0)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [])
  return p
}

/** Thin reading-progress bar pinned under the header. */
export function ReadingBar() {
  const p = useScrollProgress()
  return (
    <div className="pointer-events-none sticky top-14 z-10 h-0.5 w-full" aria-hidden="true">
      <div className="h-full origin-left bg-accent" style={{ transform: `scaleX(${p})` }} />
    </div>
  )
}

function useActiveHeading(headings: Heading[]) {
  const [active, setActive] = useState(headings[0]?.id)
  useEffect(() => {
    const onScroll = () => {
      let current = headings[0]?.id
      for (const h of headings) {
        const el = document.getElementById(h.id)
        if (el && el.getBoundingClientRect().top < 120) current = h.id
      }
      setActive(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [headings])
  return active
}

function TocList({ headings, active, onPick }: { headings: Heading[]; active?: string; onPick?: () => void }) {
  return (
    <ul className="border-l border-line text-[13px]">
      {headings.map((h) => (
        <li key={h.id}>
          <a
            href={`#${h.id}`}
            onClick={onPick}
            className={`-ml-px block border-l py-1 leading-snug transition-colors ${h.depth === 3 ? 'pl-7' : 'pl-4'} ${
              active === h.id ? 'border-accent font-medium text-accent' : 'border-transparent text-muted hover:text-fg'
            }`}
          >
            {h.text}
          </a>
        </li>
      ))}
    </ul>
  )
}

/** Right-hand rail: table of contents, reading progress and anything passed as children. */
export function Rail({ headings, children }: { headings: Heading[]; children?: ReactNode }) {
  const active = useActiveHeading(headings)
  const p = useScrollProgress()
  return (
    <div className="thin-scroll sticky top-24 max-h-[calc(100vh-7rem)] overflow-y-auto pb-4">
      {children}
      <div className="mb-5">
        <div className="flex justify-between text-[11px] font-semibold uppercase tracking-wider text-faint">
          <span>Reading progress</span>
          <span className="font-mono">{Math.round(p * 100)}%</span>
        </div>
        <div className="mt-2 h-1 overflow-hidden rounded-full bg-sunken">
          <div className="h-full rounded-full bg-accent" style={{ width: `${p * 100}%` }} />
        </div>
      </div>
      {headings.length > 1 && (
        <nav aria-label="On this page">
          <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-faint">On this page</p>
          <TocList headings={headings.filter((h) => h.depth === 2 || headings.length < 26)} active={active} />
        </nav>
      )}
    </div>
  )
}

/** Collapsible table of contents shown above the article on screens without the rail. */
export function InlineToc({ headings }: { headings: Heading[] }) {
  const [open, setOpen] = useState(false)
  const h2 = headings.filter((h) => h.depth === 2)
  if (h2.length < 2) return null
  return (
    <div className="mb-8 rounded-xl border border-line bg-surface xl:hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 px-4 py-3 text-left text-sm font-medium"
      >
        <ListTree size={16} className="text-accent" />
        On this page
        <span className="text-xs font-normal text-faint">· {h2.length} sections</span>
        <ChevronRight size={15} className={`ml-auto text-faint transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-line px-4 py-3">
          <TocList headings={h2} onPick={() => setOpen(false)} />
        </div>
      )}
    </div>
  )
}

export function Notes({ id }: { id: string }) {
  const saved = useStore().notes[id] ?? ''
  const [open, setOpen] = useState(!!saved)

  useEffect(() => setOpen(!!saved), [id]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="mt-14 rounded-xl border border-line bg-surface">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2.5 px-5 py-3.5 text-left text-sm font-medium"
        aria-expanded={open}
      >
        <NotebookPen size={16} className="text-accent" />
        My notes
        {saved && !open && <span className="text-xs font-normal text-faint">· saved</span>}
        <ChevronRight size={15} className={`ml-auto text-faint transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="border-t border-line p-3">
          <textarea
            key={id}
            defaultValue={saved}
            onChange={(e) => actions.setNote(id, e.target.value)}
            placeholder="Write down what clicked, what confused you, or questions to revisit. Saved automatically."
            aria-label="My notes for this page"
            rows={5}
            className="block w-full resize-y rounded-lg bg-sunken p-3 text-[15px] leading-relaxed outline-none placeholder:text-faint focus:ring-1 focus:ring-accent"
          />
        </div>
      )}
    </section>
  )
}

export function Skeleton() {
  return (
    <div className="animate-pulse space-y-4" aria-label="Loading">
      {[100, 92, 96, 60].map((w, i) => (
        <div key={i} className="h-4 rounded bg-sunken" style={{ width: `${w}%` }} />
      ))}
      <div className="h-40 rounded-xl bg-sunken" />
    </div>
  )
}

export function ProgressBar({ value, total, className = '' }: { value: number; total: number; className?: string }) {
  const pct = total ? (value / total) * 100 : 0
  return (
    <div
      className={`h-1.5 overflow-hidden rounded-full bg-sunken ${className}`}
      role="progressbar"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="font-mono text-xs font-semibold uppercase tracking-[0.14em] text-accent">{children}</p>
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 rounded-xl border border-dashed border-line px-4 py-8 text-center text-sm text-muted">{children}</p>
  )
}
