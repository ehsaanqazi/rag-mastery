import { useEffect, useSyncExternalStore } from 'react'

export interface QuizScore {
  score: number
  total: number
  /** When the best score was set (ms since epoch). */
  at?: number
  attempts?: number
}

export interface State {
  /** Lesson id → completion time (ms since epoch). */
  done: Record<string, number>
  bookmarks: Record<string, true>
  notes: Record<string, string>
  /** Best quiz score per lesson, assessment or exam id. */
  quiz: Record<string, QuizScore>
  last?: string
}

const KEY = 'rag-mastery:v1'
const empty: State = { done: {}, bookmarks: {}, notes: {}, quiz: {} }

function read(): State {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? { ...empty, ...JSON.parse(raw) } : empty
  } catch {
    return empty
  }
}

let state = read()
const listeners = new Set<() => void>()

function set(next: State) {
  state = next
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    /* storage unavailable: keep in memory */
  }
  listeners.forEach((l) => l())
}

const subscribe = (l: () => void) => {
  listeners.add(l)
  return () => listeners.delete(l)
}

// keep several open tabs in step
window.addEventListener('storage', (e) => {
  if (e.key === KEY) {
    state = read()
    listeners.forEach((l) => l())
  }
})

export function useStore(): State {
  return useSyncExternalStore(subscribe, () => state)
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/** Accepts only the shape this app writes; anything else is dropped. */
function sanitise(input: unknown): State | null {
  if (!isRecord(input)) return null
  const out: State = { done: {}, bookmarks: {}, notes: {}, quiz: {} }
  if (isRecord(input.done))
    for (const [k, v] of Object.entries(input.done)) if (v) out.done[k] = typeof v === 'number' ? v : Date.now()
  if (isRecord(input.bookmarks)) for (const [k, v] of Object.entries(input.bookmarks)) if (v) out.bookmarks[k] = true
  if (isRecord(input.notes))
    for (const [k, v] of Object.entries(input.notes)) if (typeof v === 'string' && v.trim()) out.notes[k] = v
  if (isRecord(input.quiz))
    for (const [k, v] of Object.entries(input.quiz))
      if (isRecord(v) && typeof v.score === 'number' && typeof v.total === 'number' && v.total > 0)
        out.quiz[k] = {
          score: v.score,
          total: v.total,
          at: typeof v.at === 'number' ? v.at : undefined,
          attempts: typeof v.attempts === 'number' ? v.attempts : undefined,
        }
  if (typeof input.last === 'string') out.last = input.last
  return out
}

function toggle<T>(map: Record<string, T>, id: string, value: T): Record<string, T> {
  const next = { ...map }
  if (next[id]) delete next[id]
  else next[id] = value
  return next
}

export const actions = {
  toggleDone: (id: string) => set({ ...state, done: toggle(state.done, id, Date.now()) }),
  markDone: (id: string) => {
    if (!state.done[id]) set({ ...state, done: { ...state.done, [id]: Date.now() } })
  },
  toggleBookmark: (id: string) => set({ ...state, bookmarks: toggle(state.bookmarks, id, true as const) }),
  setNote: (id: string, text: string) => {
    const notes = { ...state.notes }
    if (text.trim()) notes[id] = text
    else delete notes[id]
    set({ ...state, notes })
  },
  /** Records a finished attempt; keeps the best score. */
  recordQuiz: (id: string, score: number, total: number) => {
    const prev = state.quiz[id]
    const attempts = (prev?.attempts ?? (prev ? 1 : 0)) + 1
    const better = !prev || score / total > prev.score / prev.total
    set({
      ...state,
      quiz: {
        ...state.quiz,
        [id]: better ? { score, total, at: Date.now(), attempts } : { ...prev, attempts },
      },
    })
  },
  setLast: (id: string) => {
    if (state.last !== id) set({ ...state, last: id })
  },
  reset: () => set(empty),
  export: (): string => JSON.stringify({ app: 'rag-mastery', version: 1, exported: new Date().toISOString(), data: state }, null, 2),
  /** Replaces local progress with an exported file. Returns false if the file is not one of ours. */
  import: (json: string): boolean => {
    try {
      const parsed: unknown = JSON.parse(json)
      if (!isRecord(parsed) || parsed.app !== 'rag-mastery') return false
      const next = sanitise(parsed.data)
      if (!next) return false
      set(next)
      return true
    } catch {
      return false
    }
  },
}

/* ---------- theme ---------- */

type Theme = 'light' | 'dark'
const THEME_KEY = 'rag-mastery:theme'
let theme: Theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light'
const themeListeners = new Set<() => void>()

export function useTheme(): Theme {
  return useSyncExternalStore(
    (l) => {
      themeListeners.add(l)
      return () => themeListeners.delete(l)
    },
    () => theme,
  )
}

export function toggleTheme() {
  theme = theme === 'dark' ? 'light' : 'dark'
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem(THEME_KEY, theme)
  } catch {
    /* ignore */
  }
  themeListeners.forEach((l) => l())
}

/* ---------- page title and description (SEO) ---------- */

const SITE = 'RAG Engineering Mastery'

export function useTitle(title?: string, description?: string) {
  useEffect(() => {
    document.title = title ? `${title} · ${SITE}` : `${SITE}: from fundamentals to production-ready retrieval systems`
    if (description) document.querySelector('meta[name="description"]')?.setAttribute('content', description)
  }, [title, description])
}
