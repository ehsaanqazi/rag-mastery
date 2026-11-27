import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router'
import { Menu, Moon, Search, Star, Sun, X } from 'lucide-react'
import Sidebar from './Sidebar'
import SearchDialog from './SearchDialog'
import Logo from './Logo'
import { toggleTheme, useTheme } from '../lib/store'

const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

export default function Layout() {
  const [drawer, setDrawer] = useState(false)
  const [search, setSearch] = useState(false)
  const theme = useTheme()
  const location = useLocation()

  useEffect(() => {
    setDrawer(false)
    setSearch(false)
    if (!location.hash) window.scrollTo(0, 0)
  }, [location.pathname, location.hash])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setSearch((s) => !s)
      } else if (e.key === '/' && !typing) {
        e.preventDefault()
        setSearch(true)
      } else if (e.key === 'Escape') {
        setDrawer(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // stop the page scrolling behind the mobile drawer
  useEffect(() => {
    document.body.style.overflow = drawer ? 'hidden' : ''
    return () => {
      document.body.style.overflow = ''
    }
  }, [drawer])

  return (
    <div className="min-h-screen">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-accent-fg"
      >
        Skip to content
      </a>

      {/* desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[19rem] border-r border-line bg-surface lg:block">
        <Sidebar onSearch={() => setSearch(true)} />
      </aside>

      {/* mobile drawer */}
      {drawer && (
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-black/50" onClick={() => setDrawer(false)} />
          <aside className="absolute inset-y-0 left-0 w-[19rem] max-w-[86vw] border-r border-line bg-surface shadow-2xl">
            <Sidebar
              onSearch={() => {
                setDrawer(false)
                setSearch(true)
              }}
            />
            <button
              onClick={() => setDrawer(false)}
              aria-label="Close menu"
              className="absolute right-3 top-3.5 rounded-md p-1.5 text-muted hover:bg-sunken hover:text-fg"
            >
              <X size={18} />
            </button>
          </aside>
        </div>
      )}

      <div className="lg:pl-[19rem]">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-line bg-bg/85 px-4 backdrop-blur sm:px-6">
          <button
            onClick={() => setDrawer(true)}
            aria-label="Open menu"
            className="-ml-1.5 rounded-md p-1.5 text-muted hover:bg-sunken hover:text-fg lg:hidden"
          >
            <Menu size={20} />
          </button>
          <Link to="/" className="lg:hidden" aria-label="RAG Engineering Mastery home">
            <Logo />
          </Link>

          <button
            onClick={() => setSearch(true)}
            aria-label="Search the curriculum"
            className="ml-auto flex h-9 items-center gap-2 rounded-lg border border-line bg-surface px-3 text-sm text-muted transition-colors hover:border-faint hover:text-fg sm:ml-0 sm:w-72"
          >
            <Search size={15} />
            <span className="hidden sm:inline">Search the curriculum</span>
            <kbd className="ml-auto hidden rounded border border-line bg-sunken px-1.5 font-mono text-[11px] text-faint sm:inline">
              {isMac ? '⌘K' : 'Ctrl K'}
            </kbd>
          </button>

          <nav className="ml-auto hidden items-center gap-1 text-sm md:flex" aria-label="Sections">
            {[
              ['/roadmap', 'Roadmap'],
              ['/projects', 'Projects'],
              ['/playground', 'Playground'],
              ['/progress', 'Progress'],
            ].map(([to, label]) => (
              <Link key={to} to={to} className="rounded-md px-3 py-1.5 text-muted hover:bg-sunken hover:text-fg">
                {label}
              </Link>
            ))}
          </nav>
          <a
            href="https://github.com/ehsaanqazi/rag-mastery"
            target="_blank"
            rel="noreferrer"
            aria-label="Star RAG Engineering Mastery on GitHub"
            className="flex h-9 items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 text-sm text-muted transition-colors hover:border-faint hover:text-fg"
          >
            <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
            <Star size={14} className="hidden sm:block" />
            <span className="hidden sm:inline">Star</span>
          </a>
          <button
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            className="rounded-lg border border-line bg-surface p-2 text-muted transition-colors hover:border-faint hover:text-fg"
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </header>

        <main id="main">
          <Outlet />
        </main>

        <footer className="border-t border-line px-5 py-8 text-center text-[13px] text-faint sm:px-8">
          <p className="font-medium text-muted">Learn. Retrieve. Reason. Build. Deploy.</p>
          <p className="mt-1">
            RAG Engineering Mastery is free and needs no account. Your progress is stored only in this browser.
          </p>
        </footer>
      </div>

      {search && <SearchDialog onClose={() => setSearch(false)} />}
    </div>
  )
}
