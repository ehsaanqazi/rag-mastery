export default function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-accent-fg">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="10.5" cy="10.5" r="5.5" />
          <path d="m14.7 14.7 5 5" />
          <circle cx="10.5" cy="10.5" r="1" fill="currentColor" />
        </svg>
      </span>
      <span className="font-serif text-[1.2rem] font-semibold tracking-tight text-fg">RAG Mastery</span>
    </span>
  )
}
