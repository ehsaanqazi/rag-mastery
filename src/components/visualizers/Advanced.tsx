import { useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'
import { CORPUS, denseScores, ranked, rng } from '../../lib/ragmath'
import { Explain, Frame, Segmented, Slider, Stat, Stats, SVG_COLORS as C, Toggle } from './kit'

function Stepper({ step, last, onStep, label }: { step: number; last: number; onStep: (n: number) => void; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <button onClick={() => onStep(0)} disabled={step === 0} aria-label="Restart" className="rounded-md border border-line p-1.5 text-muted hover:text-fg disabled:opacity-40">
        <RotateCcw size={15} />
      </button>
      <button onClick={() => onStep(step - 1)} disabled={step === 0} aria-label="Previous step" className="rounded-md border border-line p-1.5 text-muted hover:text-fg disabled:opacity-40">
        <ChevronLeft size={15} />
      </button>
      <button onClick={() => onStep(step + 1)} disabled={step >= last} className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-fg disabled:opacity-40">
        Next <ChevronRight size={15} />
      </button>
      <span className="ml-2 text-xs text-faint">
        {label} {step + 1} of {last + 1}
      </span>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 17. HNSW                                                             */
/* ------------------------------------------------------------------ */

type P = [number, number]
const d2 = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1])

function buildHnsw(m: number) {
  const r = rng(7)
  const pts: P[] = Array.from({ length: 30 }, () => [0.06 + r() * 0.88, 0.08 + r() * 0.84])
  const level = pts.map(() => (r() < 0.32 ? (r() < 0.34 ? 2 : 1) : 0))
  if (!level.includes(2)) level[0] = 2
  // per layer: link every node to its m nearest neighbours in that layer (made symmetric)
  const layers = [0, 1, 2].map((l) => {
    const ids = pts.map((_, i) => i).filter((i) => level[i] >= l)
    const adj = new Map<number, Set<number>>(ids.map((i) => [i, new Set<number>()]))
    for (const i of ids) {
      const near = ids.filter((j) => j !== i).sort((a, b) => d2(pts[i], pts[a]) - d2(pts[i], pts[b])).slice(0, l === 0 ? m : Math.max(2, m - 1))
      for (const j of near) {
        adj.get(i)!.add(j)
        adj.get(j)!.add(i)
      }
    }
    return { ids, adj }
  })
  return { pts, layers, entry: level.indexOf(2) }
}

interface HStep {
  layer: number
  at: number
  checked: number[]
  note: string
}

function searchHnsw(g: ReturnType<typeof buildHnsw>, q: P): { steps: HStep[]; computed: number } {
  const steps: HStep[] = []
  const seen = new Set<number>([g.entry])
  let at = g.entry
  steps.push({ layer: 2, at, checked: [], note: `Start at the entry point, node ${at}, on the sparse top layer.` })
  for (let l = 2; l >= 0; l--) {
    for (;;) {
      const nbrs = [...g.layers[l].adj.get(at)!]
      nbrs.forEach((n) => seen.add(n))
      const best = nbrs.sort((a, b) => d2(q, g.pts[a]) - d2(q, g.pts[b]))[0]
      if (best !== undefined && d2(q, g.pts[best]) < d2(q, g.pts[at])) {
        at = best
        steps.push({ layer: l, at, checked: nbrs, note: `Layer ${l}: neighbour ${at} is closer to the query, so move there.` })
      } else {
        steps.push({
          layer: l,
          at,
          checked: nbrs,
          note:
            l > 0
              ? `Layer ${l}: no neighbour of ${at} is closer. Drop down to layer ${l - 1} and continue from the same node.`
              : `Layer 0: no neighbour of ${at} is closer. Search stops and returns node ${at}.`,
        })
        break
      }
    }
  }
  return { steps, computed: seen.size }
}

export function Hnsw() {
  const [m, setM] = useState(3)
  const [q, setQ] = useState<P>([0.72, 0.3])
  const [step, setStep] = useState(0)
  const g = useMemo(() => buildHnsw(m), [m])
  const { steps, computed } = useMemo(() => searchHnsw(g, q), [g, q])
  useEffect(() => setStep(0), [m, q])
  const s = steps[Math.min(step, steps.length - 1)]
  const path = steps.slice(0, step + 1)
  const truth = g.pts.map((_, i) => i).sort((a, b) => d2(q, g.pts[a]) - d2(q, g.pts[b]))[0]
  const found = steps[steps.length - 1].at
  const done = step >= steps.length - 1
  const W = 220
  const H = 170

  return (
    <Frame title="HNSW: greedy search through a layered graph">
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Slider label="M (links per node)" value={m} min={2} max={6} onChange={setM} hint="More links: better recall, more memory, slower build" />
        <p className="basis-56 text-sm text-muted">Click in any layer to move the query.</p>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {[2, 1, 0].map((l) => {
          const layer = g.layers[l]
          const active = s.layer === l
          const visited = new Set(path.filter((p) => p.layer === l).map((p) => p.at))
          return (
            <div key={l} className={`rounded-lg border ${active ? 'border-accent' : 'border-line'} bg-bg`}>
              <p className={`border-b px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider ${active ? 'border-accent text-accent' : 'border-line text-faint'}`}>
                Layer {l} · {layer.ids.length} nodes{l === 0 ? ' (all)' : ''}
              </p>
              <svg
                viewBox={`0 0 ${W} ${H}`}
                className="w-full cursor-crosshair touch-none"
                onPointerDown={(e) => {
                  const r = e.currentTarget.getBoundingClientRect()
                  setQ([Math.min(0.97, Math.max(0.03, (e.clientX - r.left) / r.width)), Math.min(0.97, Math.max(0.03, (e.clientY - r.top) / r.height))])
                }}
                role="img"
                aria-label={`HNSW layer ${l}`}
              >
                {layer.ids.flatMap((i) =>
                  [...layer.adj.get(i)!].filter((j) => j > i).map((j) => (
                    <line key={`${i}-${j}`} x1={g.pts[i][0] * W} y1={g.pts[i][1] * H} x2={g.pts[j][0] * W} y2={g.pts[j][1] * H} stroke={C.line} strokeWidth="1" />
                  )),
                )}
                {path
                  .filter((p) => p.layer === l)
                  .map((p, i, arr) =>
                    i > 0 ? (
                      <line key={i} x1={g.pts[arr[i - 1].at][0] * W} y1={g.pts[arr[i - 1].at][1] * H} x2={g.pts[p.at][0] * W} y2={g.pts[p.at][1] * H} stroke={C.accent} strokeWidth="2.5" />
                    ) : null,
                  )}
                {layer.ids.map((i) => {
                  const here = active && s.at === i
                  const chk = active && s.checked.includes(i)
                  return (
                    <g key={i}>
                      <circle
                        cx={g.pts[i][0] * W}
                        cy={g.pts[i][1] * H}
                        r={here ? 7 : visited.has(i) ? 5.5 : 4}
                        fill={here ? C.accent : visited.has(i) ? C.accent : chk ? C.info : C.faint}
                        opacity={here || visited.has(i) || chk ? 1 : 0.55}
                      />
                      {done && l === 0 && i === truth && <circle cx={g.pts[i][0] * W} cy={g.pts[i][1] * H} r="10" fill="none" stroke={C.good} strokeWidth="2" />}
                    </g>
                  )
                })}
                <g>
                  <line x1={q[0] * W - 6} y1={q[1] * H - 6} x2={q[0] * W + 6} y2={q[1] * H + 6} stroke={C.warn} strokeWidth="2.5" />
                  <line x1={q[0] * W - 6} y1={q[1] * H + 6} x2={q[0] * W + 6} y2={q[1] * H - 6} stroke={C.warn} strokeWidth="2.5" />
                </g>
              </svg>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-muted">
        <span className="text-warn">✕</span> query · <span className="text-accent">●</span> search path · <span className="text-info">●</span>{' '}
        neighbours just compared · <span className="text-good">○</span> true nearest neighbour (shown at the end)
      </p>
      <p className="min-h-10 rounded-lg border border-line bg-bg px-3 py-2 text-[14px]">{s.note}</p>
      <Stepper step={step} last={steps.length - 1} onStep={setStep} label="Step" />
      <Stats>
        <Stat label="Distances computed" value={`${computed} of 30`} tone={computed < 20 ? 'good' : undefined} />
        <Stat label="Brute force would need" value="30" />
        <Stat label="Hops" value={steps.filter((x, i) => i > 0 && x.at !== steps[i - 1].at).length} />
        <Stat label="Found the true nearest?" value={done ? (found === truth ? 'yes' : 'no') : '…'} tone={done ? (found === truth ? 'good' : 'danger') : undefined} />
      </Stats>
      <Explain>
        The top layer holds a few nodes with long links, so the search crosses the space in a couple of hops; each lower layer
        adds nodes and shorter links to refine the position. On 30 points the saving is small; on millions, the search touches
        a few hundred vectors instead of all of them.{' '}
        {done && found !== truth
          ? 'Here the greedy walk got stuck in a local minimum and missed the true nearest neighbour: this is why HNSW is approximate. Real implementations keep a list of the best ef candidates instead of one, which raises recall at the cost of more distance computations.'
          : 'Move the query or lower M to find a case where the greedy walk misses the true nearest neighbour: that is the "approximate" in approximate nearest-neighbour search.'}{' '}
        This demo links each node to its M nearest in the layer, a simplification of the real insertion algorithm.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 18. Graph RAG traversal                                              */
/* ------------------------------------------------------------------ */

const NODES = [
  { id: 'Priya Shah', kind: 'person', x: 60, y: 46 },
  { id: 'Tom Reyes', kind: 'person', x: 60, y: 150 },
  { id: 'Platform team', kind: 'team', x: 200, y: 96 },
  { id: 'AskMeridian', kind: 'system', x: 350, y: 96 },
  { id: 'Qdrant cluster', kind: 'system', x: 350, y: 208 },
  { id: 'eu-west region', kind: 'place', x: 200, y: 250 },
  { id: 'Employee handbook', kind: 'document', x: 500, y: 46 },
  { id: 'IT runbooks', kind: 'document', x: 500, y: 150 },
  { id: 'HR team', kind: 'team', x: 620, y: 96 },
  { id: 'Dana Okoye', kind: 'person', x: 620, y: 208 },
] as const
type NodeId = (typeof NODES)[number]['id']

const EDGES: [NodeId, string, NodeId][] = [
  ['Priya Shah', 'leads', 'Platform team'],
  ['Tom Reyes', 'is a member of', 'Platform team'],
  ['Platform team', 'owns', 'AskMeridian'],
  ['AskMeridian', 'runs on', 'Qdrant cluster'],
  ['Tom Reyes', 'is on call for', 'Qdrant cluster'],
  ['Qdrant cluster', 'is hosted in', 'eu-west region'],
  ['AskMeridian', 'indexes', 'Employee handbook'],
  ['AskMeridian', 'indexes', 'IT runbooks'],
  ['HR team', 'owns', 'Employee handbook'],
  ['Dana Okoye', 'leads', 'HR team'],
]

const GRAPH_QUESTIONS: { q: string; start: NodeId; needs: number[]; answer: string }[] = [
  { q: 'Who leads the team that owns AskMeridian?', start: 'AskMeridian', needs: [2, 0], answer: 'Priya Shah' },
  { q: 'Who is on call for the database AskMeridian runs on?', start: 'AskMeridian', needs: [3, 4], answer: 'Tom Reyes' },
  { q: 'Who leads the team that owns a document AskMeridian indexes?', start: 'AskMeridian', needs: [6, 8, 9], answer: 'Dana Okoye' },
  { q: 'In which region does the system owned by Priya Shah’s team run?', start: 'Priya Shah', needs: [0, 2, 3, 5], answer: 'eu-west region' },
]

export function GraphRag() {
  const [qi, setQi] = useState(0)
  const [hops, setHops] = useState(1)
  const [start, setStart] = useState<NodeId>(GRAPH_QUESTIONS[0].start)
  const question = GRAPH_QUESTIONS[qi]
  useEffect(() => setStart(GRAPH_QUESTIONS[qi].start), [qi])

  // breadth-first traversal, edges treated as undirected
  const dist = new Map<NodeId, number>([[start, 0]])
  const edgeHop = new Map<number, number>()
  let frontier: NodeId[] = [start]
  for (let h = 1; h <= hops; h++) {
    const next: NodeId[] = []
    EDGES.forEach(([a, , b], i) => {
      for (const [from, to] of [[a, b], [b, a]] as const) {
        if (frontier.includes(from)) {
          if (!edgeHop.has(i)) edgeHop.set(i, h)
          if (!dist.has(to)) {
            dist.set(to, h)
            next.push(to)
          }
        }
      }
    })
    frontier = next
  }
  const covered = question.needs.filter((i) => edgeHop.has(i)).length
  const answerable = covered === question.needs.length && start === question.start
  const pos = (id: NodeId) => NODES.find((n) => n.id === id)!
  const HOP = [C.accent, C.info, C.warn, C.good]

  return (
    <Frame title="Graph RAG: collecting facts by traversal">
      <div>
        <span className="block text-[13px] font-medium">Question</span>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {GRAPH_QUESTIONS.map((g, i) => (
            <button
              key={g.q}
              onClick={() => setQi(i)}
              className={`rounded-lg border px-2.5 py-1 text-left text-[13px] ${i === qi ? 'border-accent bg-accent-soft text-accent' : 'border-line text-muted hover:border-accent'}`}
            >
              {g.q}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Slider label="Hops from the start entity" value={hops} min={1} max={4} onChange={setHops} />
        <p className="basis-56 text-sm text-muted">
          Start entity: <strong className="text-fg">{start}</strong>. Click a node to start elsewhere.
        </p>
      </div>
      <div className="thin-scroll overflow-x-auto rounded-lg border border-line bg-bg">
        <svg viewBox="0 0 690 290" className="min-w-[560px]" role="img" aria-label="Knowledge graph of Meridian entities">
          {EDGES.map(([a, label, b], i) => {
            const A = pos(a)
            const B = pos(b)
            const h = edgeHop.get(i)
            const need = question.needs.includes(i)
            return (
              <g key={i}>
                <line x1={A.x} y1={A.y} x2={B.x} y2={B.y} stroke={h ? HOP[h - 1] : C.line} strokeWidth={h ? 2.2 : 1} strokeDasharray={need && !h ? '5 4' : undefined} />
                <text x={(A.x + B.x) / 2} y={(A.y + B.y) / 2 - 5} textAnchor="middle" fontSize="10" fill={h ? C.fg : C.faint}>
                  {label}
                </text>
              </g>
            )
          })}
          {NODES.map((n) => {
            const h = dist.get(n.id)
            const isAnswer = answerable && n.id === question.answer
            return (
              <g key={n.id} onClick={() => setStart(n.id)} className="cursor-pointer">
                <rect
                  x={n.x - 56}
                  y={n.y - 15}
                  width="112"
                  height="30"
                  rx="8"
                  fill={h === 0 ? C.accent : h !== undefined ? C.soft : C.surface}
                  stroke={isAnswer ? C.good : h !== undefined ? C.accent : C.line}
                  strokeWidth={isAnswer ? 3 : 1.2}
                />
                <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="11.5" fontWeight="600" fill={h === 0 ? 'var(--accent-fg)' : C.fg}>
                  {n.id}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <div className="grid gap-5 md:grid-cols-[1.4fr_1fr]">
        <div className="min-w-0">
          <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-faint">Facts collected ({edgeHop.size})</h4>
          <ul className="space-y-1 text-[13.5px]">
            {[...edgeHop.entries()]
              .sort((a, b) => a[1] - b[1])
              .map(([i, h]) => (
                <li key={i} className={`flex gap-2 rounded-md border px-2.5 py-1 ${question.needs.includes(i) ? 'border-accent bg-accent-soft' : 'border-line'}`}>
                  <span className="font-mono text-[11px] text-faint">hop {h}</span>
                  <span>
                    {EDGES[i][0]} <span className="text-muted">{EDGES[i][1]}</span> {EDGES[i][2]}
                  </span>
                </li>
              ))}
          </ul>
        </div>
        <Stats>
          <Stat label="Entities reached" value={dist.size} />
          <Stat label="Facts needed" value={`${covered}/${question.needs.length}`} tone={answerable ? 'good' : 'warn'} />
          <Stat label="Answerable?" value={answerable ? 'yes' : 'not yet'} tone={answerable ? 'good' : 'warn'} />
          <Stat label="Unneeded facts" value={edgeHop.size - covered} tone={edgeHop.size - covered > 5 ? 'warn' : undefined} />
        </Stats>
      </div>
      <Explain>
        {answerable ? (
          <>
            With {hops} hop{hops === 1 ? '' : 's'} the traversal has collected every fact in the chain, and the answer is{' '}
            <strong>{question.answer}</strong>. No single fact contains it: it only exists once the facts are joined.
          </>
        ) : start !== question.start ? (
          <>The question mentions “{question.start}”, so entity linking would start the traversal there. Click that node.</>
        ) : (
          <>
            The dashed edges are facts the answer needs that are still out of reach. Each one lives in a different document, and
            none of them mentions the question's wording, so vector search over chunks tends to retrieve only the first link.
            Increase the hops.
          </>
        )}{' '}
        More hops reach more of the graph: the context fills with facts the question does not need
        {edgeHop.size - covered > 0 ? ` (${edgeHop.size - covered} here)` : ''}, which is why real systems limit hops and filter by
        relationship type.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 19. Query decomposition                                              */
/* ------------------------------------------------------------------ */

const COMPOUND = [
  {
    q: 'How many days of annual leave do I get, and what is the home office budget?',
    subs: ['How many days of annual leave do I get?', 'What is the home office equipment budget?'],
    needs: ['D1', 'D2'],
  },
  {
    q: 'Do I need the VPN in a café, and when will my laptop be replaced?',
    subs: ['Is the VPN required on public Wi-Fi?', 'How often are laptops replaced?'],
    needs: ['D5', 'D6'],
  },
  {
    q: 'Can I claim travel to a client, do I need a receipt, and who do I phone if the expenses portal is down?',
    subs: ['Are travel expenses for client visits reimbursed?', 'When does an expense need a receipt?', 'How do I contact the support desk about an urgent incident?'],
    needs: ['D9', 'D3', 'D7'],
  },
]

export function QueryDecomposition() {
  const [i, setI] = useState(0)
  const [on, setOn] = useState(true)
  const [k, setK] = useState(2)
  const c = COMPOUND[i]
  const single = ranked(denseScores(c.q)).slice(0, k)
  const per = c.subs.map((s) => ({ s, top: ranked(denseScores(s))[0] }))
  const retrieved = on ? [...new Set(per.map((p) => p.top.doc.id))] : single.map((s) => s.doc.id)
  const covered = c.needs.filter((n) => retrieved.includes(n))

  return (
    <Frame title="query decomposition">
      <div>
        <span className="block text-[13px] font-medium">Compound question</span>
        <div className="mt-1.5 flex flex-col gap-1.5">
          {COMPOUND.map((x, j) => (
            <button
              key={x.q}
              onClick={() => setI(j)}
              className={`rounded-lg border px-2.5 py-1.5 text-left text-[13.5px] ${j === i ? 'border-accent bg-accent-soft text-accent' : 'border-line text-muted hover:border-accent'}`}
            >
              {x.q}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Toggle label="Decompose into sub-questions" checked={on} onChange={setOn} />
        {!on && <Slider label="K for the single query" value={k} min={1} max={5} onChange={setK} />}
      </div>

      {on ? (
        <ol className="space-y-2">
          {per.map((p, j) => (
            <li key={p.s} className="rounded-lg border border-line p-3">
              <p className="text-[13.5px]">
                <span className="mr-2 font-mono text-[11px] text-faint">sub-question {j + 1}</span>
                {p.s}
              </p>
              <p className="mt-1.5 rounded-md bg-accent-soft px-2.5 py-1 text-[13.5px]">
                <span className="font-mono text-[11.5px] text-muted">
                  → {p.top.doc.id} · {p.top.score.toFixed(2)}
                </span>{' '}
                {p.top.doc.text}
              </p>
            </li>
          ))}
        </ol>
      ) : (
        <ol className="space-y-1.5">
          {single.map((s, j) => (
            <li key={s.doc.id} className={`rounded-md border px-3 py-1.5 text-[13.5px] ${c.needs.includes(s.doc.id) ? 'border-accent bg-accent-soft' : 'border-line'}`}>
              <span className="mr-2 font-mono text-[11px] text-faint">{j + 1}</span>
              <span className="font-mono text-[11.5px] text-muted">
                {s.doc.id} · {s.score.toFixed(2)}
              </span>{' '}
              {s.doc.text}
            </li>
          ))}
        </ol>
      )}

      <Stats>
        <Stat label="Evidence needed" value={c.needs.join(', ')} />
        <Stat label="Evidence found" value={`${covered.length}/${c.needs.length}`} tone={covered.length === c.needs.length ? 'good' : 'danger'} />
        <Stat label="Searches run" value={on ? c.subs.length : 1} />
        <Stat label="Chunks sent to the model" value={retrieved.length} />
      </Stats>
      <Explain>
        {on ? (
          <>
            Each sub-question is about one thing, so its embedding points cleanly at one topic and retrieves the right chunk.
            The cost is {c.subs.length} searches instead of one, plus the model call that wrote the sub-questions.
          </>
        ) : covered.length === c.needs.length ? (
          <>
            With K = {k} the single query happens to retrieve all the evidence. Lower K to see how fragile that is: one embedding
            has to represent {c.subs.length} topics at once.
          </>
        ) : (
          <>
            The single query's embedding is a blend of {c.subs.length} topics, so it sits between them. The top {k} results cover
            only {covered.length} of the {c.needs.length} facts the answer needs
            {covered.length < c.needs.length ? ` (missing ${c.needs.filter((n) => !retrieved.includes(n)).join(', ')})` : ''}, and
            the model would answer part of the question and guess or refuse the rest.
          </>
        )}{' '}
        In a real system an LLM writes the sub-questions; here they are fixed so that the retrieval can be computed in your
        browser.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 20. Agentic RAG workflow                                             */
/* ------------------------------------------------------------------ */

type NodeKey = 'agent' | 'need' | 'plan' | 'search' | 'grade' | 'enough' | 'answer'

const FLOW: Record<NodeKey, { x: number; y: number; label: string; diamond?: boolean }> = {
  agent: { x: 70, y: 36, label: 'Agent' },
  need: { x: 230, y: 36, label: 'Need retrieval?', diamond: true },
  answer: { x: 395, y: 36, label: 'Answer' },
  plan: { x: 230, y: 110, label: 'Plan query' },
  search: { x: 395, y: 110, label: 'Search tool' },
  grade: { x: 560, y: 110, label: 'Grade context' },
  enough: { x: 560, y: 184, label: 'Sufficient?', diamond: true },
}

/** from, to, SVG path, optional label and its position */
const ARROWS: [NodeKey, NodeKey, string, string?, number?, number?][] = [
  ['agent', 'need', 'M122,36 L165,36'],
  ['need', 'plan', 'M230,58 L230,89', 'yes', 244, 78],
  ['need', 'answer', 'M292,36 L340,36', 'no', 316, 29],
  ['plan', 'search', 'M282,110 L340,110'],
  ['search', 'grade', 'M447,110 L505,110'],
  ['grade', 'enough', 'M560,128 L560,159'],
  ['enough', 'plan', 'M498,184 L230,184 L230,131', 'no: search again', 364, 177],
  ['enough', 'answer', 'M622,184 L650,184 L650,36 L450,36', 'yes', 636, 110],
]

interface AStep {
  node: NodeKey
  text: string
}

const hit = (query: string) => {
  const t = ranked(denseScores(query))[0]
  return `Top result: ${t.doc.id} (similarity ${t.score.toFixed(2)}) “${t.doc.text}”`
}

function scenario(which: string, budget: number): AStep[] {
  if (which === 'chat')
    return [
      { node: 'agent', text: 'User: “Thanks, that answers it!”' },
      { node: 'need', text: 'This is a pleasantry, not a question about Meridian. Decision: no retrieval needed.' },
      { node: 'answer', text: '“You’re welcome. Ask me anything else about the handbook.” Zero searches, no wasted latency or cost.' },
    ]
  if (which === 'simple')
    return [
      { node: 'agent', text: 'User: “How many days a week can I work remotely?”' },
      { node: 'need', text: 'This asks for a Meridian policy the model cannot know. Decision: retrieve.' },
      { node: 'plan', text: 'Search query written by the agent: “remote work days per week”.' },
      { node: 'search', text: hit('remote work days per week') },
      { node: 'grade', text: 'The chunk states the number of remote days directly. Graded: relevant.' },
      { node: 'enough', text: 'One relevant chunk answers the whole question. Decision: sufficient.' },
      { node: 'answer', text: '“Up to 3 days a week [D2].” One search: the agent behaved exactly like plain RAG, which is the right outcome for a simple question.' },
    ]
  const steps: AStep[] = [
    { node: 'agent', text: 'User: “Can I use my home-office budget to replace my laptop early?”' },
    { node: 'need', text: 'Needs Meridian policy on two things: the budget and laptop replacement. Decision: retrieve.' },
    { node: 'plan', text: 'Search 1: “home office equipment budget”.' },
    { node: 'search', text: hit('home office equipment budget') },
    { node: 'grade', text: 'Relevant to the budget (£400), but says nothing about how laptops are replaced. Graded: partly relevant.' },
  ]
  if (budget < 2)
    return [
      ...steps,
      { node: 'enough', text: 'Not sufficient, but the search budget (1) is spent. Decision: stop and say what is missing.' },
      {
        node: 'answer',
        text: '“The home-office budget is £400 [D2]. I could not find the laptop replacement policy within my search limit, so I can’t say whether the budget covers it.” An honest partial answer beats a guess.',
      },
    ]
  return [
    ...steps,
    { node: 'enough', text: 'Half the question is uncovered and budget remains. Decision: not sufficient, search again.' },
    { node: 'plan', text: 'Search 2, rewritten to target the gap: “laptop replacement cycle”.' },
    { node: 'search', text: hit('laptop replacement cycle') },
    { node: 'grade', text: 'States the 3-year refresh cycle and the IT portal route. Graded: relevant.' },
    { node: 'enough', text: 'Both halves are now covered by retrieved text. Decision: sufficient.' },
    {
      node: 'answer',
      text: '“The budget is £400 for home-office equipment [D2]. Laptops are replaced on a 3-year cycle through the IT portal [D6]. The handbook does not say the budget can fund an early replacement.” Two searches, both cited, and no claim beyond the sources.',
    },
  ]
}

export function AgenticRag() {
  const [which, setWhich] = useState('multi')
  const [budget, setBudget] = useState(3)
  const [step, setStep] = useState(0)
  const steps = useMemo(() => scenario(which, budget), [which, budget])
  useEffect(() => setStep(0), [which, budget])
  const s = steps[Math.min(step, steps.length - 1)]
  const searches = steps.slice(0, step + 1).filter((x) => x.node === 'search').length
  const prev = step > 0 ? steps[step - 1].node : null

  return (
    <Frame title="an agent deciding when and how to retrieve">
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Segmented
          label="Scenario"
          options={[
            { value: 'chat', label: 'No retrieval needed' },
            { value: 'simple', label: 'One search is enough' },
            { value: 'multi', label: 'Needs a second search' },
          ]}
          value={which}
          onChange={setWhich}
        />
        <Slider label="Search budget (max tool calls)" value={budget} min={1} max={3} onChange={setBudget} />
      </div>
      <div className="thin-scroll overflow-x-auto rounded-lg border border-line bg-bg">
        <svg viewBox="0 0 670 216" className="min-w-[540px]" role="img" aria-label="Agentic RAG control flow">
          <defs>
            <marker id="agent-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" />
            </marker>
          </defs>
          {ARROWS.map(([a, b, d, label, lx, ly]) => {
            const live = prev === a && s.node === b
            return (
              <g key={a + b}>
                <path d={d} fill="none" stroke={live ? C.accent : C.faint} strokeWidth={live ? 2.5 : 1.2} strokeLinejoin="round" markerEnd="url(#agent-arrow)" />
                {label && (
                  <text x={lx} y={ly} textAnchor="middle" fontSize="10" fill={live ? C.accent : C.muted}>
                    {label}
                  </text>
                )}
              </g>
            )
          })}
          {(Object.keys(FLOW) as NodeKey[]).map((key) => {
            const n = FLOW[key]
            const active = s.node === key
            return (
              <g key={key}>
                {n.diamond ? (
                  <polygon points={`${n.x},${n.y - 22} ${n.x + 62},${n.y} ${n.x},${n.y + 22} ${n.x - 62},${n.y}`} fill={active ? C.accent : C.surface} stroke={active ? C.accent : C.line} strokeWidth="1.4" />
                ) : (
                  <rect x={n.x - 52} y={n.y - 18} width="104" height="36" rx="9" fill={active ? C.accent : C.surface} stroke={active ? C.accent : C.line} strokeWidth="1.4" />
                )}
                <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="11.5" fontWeight="600" fill={active ? 'var(--accent-fg)' : C.fg}>
                  {n.label}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <ol className="space-y-1.5">
        {steps.slice(0, step + 1).map((x, i) => (
          <li key={i} className={`flex gap-3 rounded-md border px-3 py-1.5 text-[13.5px] ${i === step ? 'border-accent bg-accent-soft' : 'border-line text-muted'}`}>
            <span className="w-24 shrink-0 font-mono text-[11px] uppercase text-faint">{FLOW[x.node].label}</span>
            <span className="min-w-0">{x.text}</span>
          </li>
        ))}
      </ol>
      <Stepper step={step} last={steps.length - 1} onStep={setStep} label="Step" />
      <Stats>
        <Stat label="Searches so far" value={`${searches} / ${budget}`} tone={searches >= budget && which === 'multi' ? 'warn' : undefined} />
        <Stat label="Model decisions" value={steps.slice(0, step + 1).filter((x) => ['need', 'plan', 'grade', 'enough', 'answer'].includes(x.node)).length} />
        <Stat label="Current step" value={FLOW[s.node].label} />
        <Stat label="Outcome" value={step < steps.length - 1 ? '…' : which === 'multi' && budget < 2 ? 'partial' : 'answered'} tone={step < steps.length - 1 ? undefined : which === 'multi' && budget < 2 ? 'warn' : 'good'} />
      </Stats>
      <Explain>
        In plain RAG the pipeline is fixed: always retrieve once, then answer. Here the model makes the control-flow decisions:
        whether to search, what to search for, whether the results are good enough, and when to stop. That buys flexibility on
        questions like the third scenario and costs extra model calls on every question, including the easy ones. The search
        budget is the safety rail: without a hard limit, an agent that never judges its context “sufficient” loops forever.
        The decisions in this demo are scripted; the retrieval results are computed.
      </Explain>
    </Frame>
  )
}
