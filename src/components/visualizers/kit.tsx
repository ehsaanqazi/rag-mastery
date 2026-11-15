import type { ReactNode } from 'react'
import { Lightbulb } from 'lucide-react'

/** Outer card every visualiser sits in. */
export function Frame({ title, children, footer }: { title: string; children: ReactNode; footer?: ReactNode }) {
  return (
    <section className="viz not-prose overflow-hidden rounded-xl border border-line bg-surface text-[15px] leading-normal">
      <header className="flex items-center justify-between border-b border-line bg-sunken px-5 py-2.5">
        <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-accent">Interactive · {title}</span>
      </header>
      <div className="space-y-5 p-4 sm:p-5">{children}</div>
      {footer}
    </section>
  )
}

/** "What just changed" explanation, updated as the inputs move. */
export function Explain({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3 rounded-lg bg-accent-soft px-3.5 py-3 text-[14.5px] leading-relaxed" role="status" aria-live="polite">
      <Lightbulb size={17} className="mt-0.5 shrink-0 text-accent" />
      <div className="min-w-0">{children}</div>
    </div>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = String,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
  hint?: string
}) {
  return (
    <label className="block min-w-0 flex-1 basis-48">
      <span className="flex items-baseline justify-between gap-2 text-[13px]">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-accent">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full"
      />
      {hint && <span className="block text-xs text-faint">{hint}</span>}
    </label>
  )
}

export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: readonly (T | { value: T; label: string })[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="min-w-0">
      <span className="block text-[13px] font-medium">{label}</span>
      <div className="mt-1.5 inline-flex flex-wrap overflow-hidden rounded-lg border border-line" role="group" aria-label={label}>
        {options.map((o) => {
          const v = typeof o === 'string' ? o : o.value
          const text = typeof o === 'string' ? o : o.label
          return (
            <button
              key={v}
              onClick={() => onChange(v)}
              aria-pressed={v === value}
              className={`border-r border-line px-3 py-1 text-[13px] last:border-r-0 ${
                v === value ? 'bg-accent font-medium text-accent-fg' : 'text-muted hover:bg-sunken hover:text-fg'
              }`}
            >
              {text}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-[13.5px]">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="h-4 w-4 accent-[var(--accent)]" />
      {label}
    </label>
  )
}

export function TextInput({
  label,
  value,
  onChange,
  presets,
  placeholder,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  presets?: string[]
  placeholder?: string
}) {
  return (
    <div>
      <label className="block">
        <span className="text-[13px] font-medium">{label}</span>
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          spellCheck={false}
          className="mt-1 block w-full rounded-lg border border-line bg-bg px-3 py-2 text-[15px] outline-none focus:border-accent"
        />
      </label>
      {presets && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          <span className="text-xs text-faint">Try:</span>
          {presets.map((p) => (
            <button
              key={p}
              onClick={() => onChange(p)}
              className={`rounded-full border px-2 py-px text-xs ${
                p === value ? 'border-accent text-accent' : 'border-line text-muted hover:border-accent hover:text-accent'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: 'good' | 'warn' | 'danger' }) {
  const colour = tone === 'good' ? 'text-good' : tone === 'warn' ? 'text-warn' : tone === 'danger' ? 'text-danger' : 'text-fg'
  return (
    <div className="min-w-0 bg-surface px-3.5 py-2.5">
      <dt className="truncate text-[11px] font-semibold uppercase tracking-wider text-faint">{label}</dt>
      <dd className={`mt-0.5 font-mono text-lg ${colour}`}>{value}</dd>
    </div>
  )
}

export function Stats({ children }: { children: ReactNode }) {
  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-4">{children}</dl>
  )
}

/** A horizontal score bar; `max` is the value that fills the track. */
export function Bar({ value, max, tone = 'accent' }: { value: number; max: number; tone?: 'accent' | 'info' | 'warn' | 'muted' }) {
  const bg = tone === 'info' ? 'bg-info' : tone === 'warn' ? 'bg-warn' : tone === 'muted' ? 'bg-faint' : 'bg-accent'
  return (
    <span className="block h-1.5 w-full overflow-hidden rounded-full bg-sunken">
      <span className={`block h-full rounded-full ${bg} transition-all`} style={{ width: `${max > 0 ? Math.max(0, Math.min(1, value / max)) * 100 : 0}%` }} />
    </span>
  )
}

export interface RankRow {
  id: string
  label: string
  score: number
  note?: ReactNode
  dim?: boolean
  highlight?: boolean
}

/** A ranked list with score bars. */
export function RankList({
  title,
  rows,
  digits = 3,
  tone = 'accent',
  max,
}: {
  title?: string
  rows: RankRow[]
  digits?: number
  tone?: 'accent' | 'info' | 'warn' | 'muted'
  max?: number
}) {
  const top = max ?? Math.max(...rows.map((r) => r.score), 1e-9)
  return (
    <div className="min-w-0">
      {title && <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-faint">{title}</h4>}
      <ol className="space-y-1.5">
        {rows.map((r, i) => (
          <li
            key={r.id}
            className={`rounded-md border px-2.5 py-1.5 ${r.highlight ? 'border-accent bg-accent-soft' : 'border-line'} ${r.dim ? 'opacity-45' : ''}`}
          >
            <div className="flex items-baseline gap-2 text-[13.5px]">
              <span className="w-4 shrink-0 font-mono text-[11px] text-faint">{i + 1}</span>
              <span className="min-w-0 flex-1 truncate">
                <span className="font-mono text-[11.5px] text-muted">{r.id}</span> {r.label}
              </span>
              <span className="shrink-0 font-mono text-[12.5px]">{r.score.toFixed(digits)}</span>
            </div>
            <div className="mt-1 pl-6">
              <Bar value={r.score} max={top} tone={tone} />
              {r.note && <div className="mt-1 text-xs text-muted">{r.note}</div>}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}

export function Formula({ children }: { children: ReactNode }) {
  return (
    <p className="thin-scroll overflow-x-auto rounded-lg border border-line bg-code px-3 py-2 font-mono text-[13px] leading-relaxed">
      {children}
    </p>
  )
}

export const SVG_COLORS = {
  accent: 'var(--accent)',
  info: 'var(--info)',
  warn: 'var(--warn)',
  danger: 'var(--danger)',
  good: 'var(--good)',
  fg: 'var(--fg)',
  muted: 'var(--muted)',
  faint: 'var(--faint)',
  line: 'var(--line)',
  surface: 'var(--surface)',
  soft: 'var(--accent-soft)',
}
