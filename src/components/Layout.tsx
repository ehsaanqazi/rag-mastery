import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router'
import { Menu, Moon, Search, Sun, X } from 'lucide-react'
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
