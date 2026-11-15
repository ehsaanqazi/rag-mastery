import { useMemo, useRef, useState } from 'react'
import { CONCEPTS, CORPUS, cosine, dot, embed, euclid, norm, project2d, rng, round } from '../../lib/ragmath'
import { Explain, Frame, Formula, RankList, Segmented, Slider, Stat, Stats, SVG_COLORS as C, TextInput, Toggle } from './kit'

/* ------------------------------------------------------------------ */
/* 1. How embeddings represent text                                     */
/* ------------------------------------------------------------------ */

const MAP_SENTENCES = [
  'How many holiday days do I get?',
  'Can I carry over unused annual leave?',
  'Am I allowed to work from home?',
  'What is the budget for home office equipment?',
  'How do I claim travel costs?',
  'Do I need a receipt for expenses?',
  'How often must I change my password?',
  'Is the VPN required on public Wi-Fi?',
  'When will my laptop be replaced?',
  'What are the support desk hours?',
  'Who do I phone about an urgent incident?',
]

export function EmbeddingsMap() {
  const [text, setText] = useState('I want to take a vacation in March')
  const points = useMemo(() => MAP_SENTENCES.map((s) => ({ s, v: embed(s), p: project2d(embed(s)) })), [])
  const qv = embed(text)
  const known = qv.some((x) => x !== 0)
  const qp = project2d(qv)
  const near = points
    .map((pt) => ({ ...pt, sim: cosine(qv, pt.v) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, 3)
  const S = 150
  const at = ([x, y]: [number, number]) => [170 + x * S, 170 + y * S] as const
  const top = qv.map((w, i) => ({ c: CONCEPTS[i], w })).sort((a, b) => b.w - a.w)

  return (
    <Frame title="embeddings as a map of meaning">
      <TextInput
        label="Type a sentence"
        value={text}
        onChange={setText}
        presets={['I want to take a vacation in March', 'my computer is old', 'reimbursement for a client trip', 'forgot my login', 'working remotely on public wifi']}
      />
      <div className="grid gap-5 md:grid-cols-[340px_1fr]">
        <svg viewBox="0 0 340 340" className="mx-auto w-full max-w-[340px]" role="img" aria-label="Two-dimensional map of sentence embeddings">
          <circle cx="170" cy="170" r={S} fill="none" stroke={C.line} strokeDasharray="3 4" />
          {CONCEPTS.map((c, i) => {
            const a = (i * Math.PI) / 3 - Math.PI / 2
            const [x, y] = [170 + Math.cos(a) * (S + 4), 170 + Math.sin(a) * (S + 4)]
            return (
              <g key={c}>
                <line x1="170" y1="170" x2={170 + Math.cos(a) * S} y2={170 + Math.sin(a) * S} stroke={C.line} />
                <text x={x} y={y + (Math.sin(a) > 0.5 ? 12 : Math.sin(a) < -0.5 ? -4 : 4)} textAnchor={Math.cos(a) > 0.3 ? 'end' : Math.cos(a) < -0.3 ? 'start' : 'middle'} fontSize="10.5" fontWeight="600" fill={C.faint}>
                  {c.toUpperCase()}
                </text>
              </g>
            )
          })}
          {points.map((pt) => {
            const [x, y] = at(pt.p)
            const isNear = known && near.some((n) => n.s === pt.s)
            return (
              <circle key={pt.s} cx={x} cy={y} r={isNear ? 6 : 4.5} fill={isNear ? C.info : C.faint} opacity={isNear ? 1 : 0.7}>
                <title>{pt.s}</title>
              </circle>
            )
          })}
          {known &&
            near.map((n) => {
              const [x1, y1] = at(qp)
              const [x2, y2] = at(n.p)
              return <line key={n.s} x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.info} strokeDasharray="2 3" />
            })}
          {known && <circle cx={at(qp)[0]} cy={at(qp)[1]} r="7.5" fill={C.accent} stroke={C.surface} strokeWidth="2" />}
        </svg>
        <div className="min-w-0 space-y-4">
          <div>
            <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-faint">Your sentence as a vector</h4>
            <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-1 text-[13px]">
              {qv.map((w, i) => (
                <div key={i} className="contents">
                  <span className="text-muted">{CONCEPTS[i]}</span>
                  <span className="h-1.5 overflow-hidden rounded-full bg-sunken">
                    <span className="block h-full rounded-full bg-accent" style={{ width: `${w * 100}%` }} />
                  </span>
                  <span className="font-mono text-xs">{w.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
          {known ? (
            <RankList
              title="Nearest sentences (cosine similarity)"
              tone="info"
              max={1}
              rows={near.map((n, i) => ({ id: `#${i + 1}`, label: n.s, score: n.sim }))}
            />
          ) : (
            <p className="text-sm text-muted">None of these words is in the toy model's vocabulary, so the vector is all zeros.</p>
          )}
        </div>
      </div>
      <Explain>
        {known ? (
          <>
            The model placed your sentence closest to the <strong>{top[0].c}</strong> direction. Its nearest neighbour is “{near[0].s}”
            even where the two share no words: sentences are near each other because they mean similar things, not because they
            spell similarly. This toy uses six hand-built dimensions; real embedding models learn hundreds from data.
          </>
        ) : (
          <>Try a word the handbook might use: holiday, laptop, receipt, password, remote, support.</>
        )}
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 2. Cosine similarity                                                 */
/* ------------------------------------------------------------------ */

export function CosineSimilarity() {
  const [a, setA] = useState<[number, number]>([4, 1.5])
  const [b, setB] = useState<[number, number]>([2, 3.5])
  const ref = useRef<SVGSVGElement>(null)
  const drag = useRef<'a' | 'b' | null>(null)
  const U = 34 // pixels per unit
  const O = [40, 240]
  const px = ([x, y]: [number, number]) => [O[0] + x * U, O[1] - y * U]

  const move = (e: React.PointerEvent) => {
    if (!drag.current || !ref.current) return
    const r = ref.current.getBoundingClientRect()
    const sx = 320 / r.width
    const x = Math.max(0, Math.min(8, ((e.clientX - r.left) * sx - O[0]) / U))
    const y = Math.max(0, Math.min(6.5, (O[1] - (e.clientY - r.top) * sx) / U))
    const v: [number, number] = [round(x, 1), round(y, 1)]
    if (v[0] === 0 && v[1] === 0) return
    ;(drag.current === 'a' ? setA : setB)(v)
  }

  const d = dot(a, b)
  const cos = cosine(a, b)
  const angle = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI
  const [ax, ay] = px(a)
  const [bx, by] = px(b)
  const a1 = Math.atan2(a[1], a[0])
  const a2 = Math.atan2(b[1], b[0])
  const arc = (r: number) => {
    const [s, e] = a1 < a2 ? [a1, a2] : [a2, a1]
    return `M ${O[0] + r * Math.cos(s)} ${O[1] - r * Math.sin(s)} A ${r} ${r} 0 0 0 ${O[0] + r * Math.cos(e)} ${O[1] - r * Math.sin(e)}`
  }

  return (
    <Frame title="cosine similarity">
      <div className="grid gap-5 md:grid-cols-[320px_1fr]">
        <svg
          ref={ref}
          viewBox="0 0 320 260"
          className="mx-auto w-full max-w-[320px] touch-none select-none"
          onPointerMove={move}
          onPointerUp={() => (drag.current = null)}
          onPointerLeave={() => (drag.current = null)}
          role="img"
          aria-label="Two draggable vectors"
        >
          {Array.from({ length: 9 }, (_, i) => (
            <line key={`v${i}`} x1={O[0] + i * U} y1="14" x2={O[0] + i * U} y2={O[1]} stroke={C.line} strokeWidth={i ? 0.6 : 1.4} />
          ))}
          {Array.from({ length: 7 }, (_, i) => (
            <line key={`h${i}`} x1={O[0]} y1={O[1] - i * U} x2="312" y2={O[1] - i * U} stroke={C.line} strokeWidth={i ? 0.6 : 1.4} />
          ))}
          <path d={arc(30)} fill="none" stroke={C.warn} strokeWidth="2" />
          <line x1={ax} y1={ay} x2={bx} y2={by} stroke={C.faint} strokeDasharray="3 3" />
          <line x1={O[0]} y1={O[1]} x2={ax} y2={ay} stroke={C.accent} strokeWidth="2.5" />
          <line x1={O[0]} y1={O[1]} x2={bx} y2={by} stroke={C.info} strokeWidth="2.5" />
          <circle cx={ax} cy={ay} r="9" fill={C.accent} className="cursor-grab" onPointerDown={() => (drag.current = 'a')} />
          <circle cx={bx} cy={by} r="9" fill={C.info} className="cursor-grab" onPointerDown={() => (drag.current = 'b')} />
          <text x={ax + 11} y={ay - 8} fontSize="12" fontWeight="600" fill={C.accent}>a</text>
          <text x={bx + 11} y={by - 8} fontSize="12" fontWeight="600" fill={C.info}>b</text>
        </svg>
        <div className="min-w-0 space-y-4">
          <p className="text-sm text-muted">Drag the two points, or use the sliders.</p>
          <div className="flex flex-wrap gap-x-5 gap-y-3">
            <Slider label="a: x" value={a[0]} min={0} max={8} step={0.1} onChange={(x) => (x || a[1]) && setA([x, a[1]])} />
            <Slider label="a: y" value={a[1]} min={0} max={6.5} step={0.1} onChange={(y) => (y || a[0]) && setA([a[0], y])} />
            <Slider label="b: x" value={b[0]} min={0} max={8} step={0.1} onChange={(x) => (x || b[1]) && setB([x, b[1]])} />
            <Slider label="b: y" value={b[1]} min={0} max={6.5} step={0.1} onChange={(y) => (y || b[0]) && setB([b[0], y])} />
          </div>
          <Formula>
            cos θ = (a · b) / (‖a‖ ‖b‖) = {d.toFixed(2)} / ({norm(a).toFixed(2)} × {norm(b).toFixed(2)}) ={' '}
            <strong>{cos.toFixed(3)}</strong>
          </Formula>
        </div>
      </div>
      <Stats>
        <Stat label="Cosine similarity" value={cos.toFixed(3)} tone={cos > 0.9 ? 'good' : undefined} />
        <Stat label="Angle θ" value={`${angle.toFixed(1)}°`} />
        <Stat label="Dot product" value={d.toFixed(2)} />
        <Stat label="Euclidean distance" value={euclid(a, b).toFixed(2)} />
      </Stats>
      <Explain>
        Cosine depends only on the <strong>angle</strong> ({angle.toFixed(0)}°). Drag a point straight away from the origin: the
        dot product and the distance change, the cosine does not. That is why cosine suits embeddings, where direction carries
        the meaning and length mostly does not. {cos > 0.98 && 'These two point almost the same way, so they count as near-identical in meaning.'}
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 3. Vector search                                                     */
/* ------------------------------------------------------------------ */

type Metric = 'cosine' | 'dot' | 'euclidean'

export function VectorSearch() {
  const pts = useMemo(() => {
    const r = rng(11)
    return Array.from({ length: 26 }, (_, i) => {
      const angle = r() * (Math.PI / 2)
      const len = 1.2 + r() * 6.3
      return { id: i + 1, v: [Math.cos(angle) * len, Math.sin(angle) * len] as [number, number] }
    })
  }, [])
  const [q, setQ] = useState<[number, number]>([3.2, 2.4])
  const [k, setK] = useState(4)
  const [metric, setMetric] = useState<Metric>('cosine')
  const U = 36
  const O = [28, 292]
  const px = ([x, y]: [number, number]) => [O[0] + x * U, O[1] - y * U]

  const score = (v: [number, number], m: Metric) => (m === 'cosine' ? cosine(q, v) : m === 'dot' ? dot(q, v) : -euclid(q, v))
  const rank = (m: Metric) => [...pts].sort((a, b) => score(b.v, m) - score(a.v, m)).slice(0, k)
  const top = rank(metric)
  const ids = new Set(top.map((p) => p.id))
  const overlap = (m: Metric) => rank(m).filter((p) => ids.has(p.id)).length
  const others = (['cosine', 'dot', 'euclidean'] as Metric[]).filter((m) => m !== metric)

  const place = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const s = 320 / r.width
    const x = ((e.clientX - r.left) * s - O[0]) / U
    const y = (O[1] - (e.clientY - r.top) * s) / U
    if (x > 0.1 && y > 0.1 && x <= 8 && y <= 7.6) setQ([round(x, 2), round(y, 2)])
  }
  const [qx, qy] = px(q)

  return (
    <Frame title="nearest-neighbour vector search">
      <div className="grid gap-5 md:grid-cols-[320px_1fr]">
        <svg viewBox="0 0 320 300" className="mx-auto w-full max-w-[320px] cursor-crosshair touch-none" onPointerDown={place} role="img" aria-label="Document vectors and a query point">
          <line x1={O[0]} y1={O[1]} x2="316" y2={O[1]} stroke={C.line} strokeWidth="1.4" />
          <line x1={O[0]} y1={O[1]} x2={O[0]} y2="8" stroke={C.line} strokeWidth="1.4" />
          {metric === 'euclidean' && <circle cx={qx} cy={qy} r={euclid(q, top[top.length - 1].v) * U} fill={C.soft} stroke={C.accent} strokeDasharray="3 3" opacity="0.7" />}
          {metric === 'cosine' && <line x1={O[0]} y1={O[1]} x2={O[0] + (q[0] / norm(q)) * 330} y2={O[1] - (q[1] / norm(q)) * 330} stroke={C.accent} strokeDasharray="3 3" opacity="0.6" />}
          {pts.map((p) => {
            const [x, y] = px(p.v)
            const hit = ids.has(p.id)
            return (
              <g key={p.id}>
                {hit && <line x1={qx} y1={qy} x2={x} y2={y} stroke={C.info} strokeWidth="1" />}
                <circle cx={x} cy={y} r={hit ? 6 : 4} fill={hit ? C.info : C.faint} opacity={hit ? 1 : 0.6} />
                {hit && (
                  <text x={x + 8} y={y + 4} fontSize="10.5" fill={C.fg}>
                    {top.findIndex((t) => t.id === p.id) + 1}
                  </text>
                )}
              </g>
            )
          })}
          <circle cx={qx} cy={qy} r="7.5" fill={C.accent} stroke={C.surface} strokeWidth="2" />
          <text x={qx + 10} y={qy - 8} fontSize="12" fontWeight="600" fill={C.accent}>query</text>
        </svg>
        <div className="min-w-0 space-y-4">
          <p className="text-sm text-muted">Click anywhere in the plot to move the query.</p>
          <Segmented label="Similarity metric" options={['cosine', 'dot', 'euclidean'] as const} value={metric} onChange={setMetric} />
          <Slider label="K (neighbours returned)" value={k} min={1} max={10} onChange={setK} />
          <Stats>
            <Stat label="Vectors compared" value={pts.length} />
            <Stat label={`Same as ${others[0]}`} value={`${overlap(others[0])}/${k}`} />
            <Stat label={`Same as ${others[1]}`} value={`${overlap(others[1])}/${k}`} />
            <Stat label="Best score" value={Math.abs(score(top[0].v, metric)).toFixed(3)} />
          </Stats>
        </div>
      </div>
      <Explain>
        {metric === 'cosine' && <>Cosine picks the points closest to the query's <strong>direction</strong> (the dashed ray), however long they are.</>}
        {metric === 'dot' && <>The dot product rewards <strong>long</strong> vectors: far-out points win even when their direction is a little off.</>}
        {metric === 'euclidean' && <>Euclidean distance picks the points inside the smallest <strong>circle</strong> around the query, so length matters as much as direction.</>}{' '}
        These vectors have different lengths, so the three metrics disagree: only {overlap(others[0])} and {overlap(others[1])} of
        the {k} results are shared with the other two. Normalise every vector to length 1 and all three give the same ranking.
        This is exact search: all {pts.length} vectors were compared, which is what an index such as HNSW avoids.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 11. Top-K retrieval                                                  */
/* ------------------------------------------------------------------ */

export function TopK() {
  const [query, setQuery] = useState('How much annual leave can I carry over?')
  const [k, setK] = useState(3)
  const [tokens, setTokens] = useState(180)
  const [exact, setExact] = useState(false)
  const qv = embed(query)
  const scored = CORPUS.map((doc, i) => ({ doc, score: cosine(qv, embed(doc.text)), i })).sort((a, b) => b.score - a.score)
  // "relevant" = shares the query's strongest concept
  const main = qv.indexOf(Math.max(...qv))
  const relevant = (text: string) => {
    const v = embed(text)
    return Math.max(...qv) > 0 && v[main] >= 0.5
  }
  const totalRel = scored.filter((s) => relevant(s.doc.text)).length
  const got = scored.slice(0, k).filter((s) => relevant(s.doc.text)).length
  const precision = got / k
  const recall = totalRel ? got / totalRel : 0

  return (
    <Frame title="top-K retrieval">
      <TextInput
        label="Query"
        value={query}
        onChange={setQuery}
        presets={['How much annual leave can I carry over?', 'expenses and receipts', 'laptop', 'is my password and VPN setup secure?']}
      />
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Slider label="K (chunks passed to the model)" value={k} min={1} max={9} onChange={setK} />
        <Slider label="Tokens per chunk" value={tokens} min={60} max={600} step={20} onChange={setTokens} />
        <Toggle label="Show scores to 3 decimals" checked={exact} onChange={setExact} />
      </div>
      <ol className="space-y-1.5">
        {scored.map((s, i) => {
          const rel = relevant(s.doc.text)
          const inK = i < k
          return (
            <li key={s.doc.id} className={`flex items-center gap-3 rounded-md border px-3 py-1.5 text-[13.5px] ${inK ? 'border-accent bg-accent-soft' : 'border-line opacity-55'}`}>
              <span className="w-4 font-mono text-[11px] text-faint">{i + 1}</span>
              <span className="min-w-0 flex-1 truncate">
                <span className="font-mono text-[11.5px] text-muted">{s.doc.id}</span> {s.doc.text}
              </span>
              <span className={`shrink-0 rounded-full px-1.5 text-[10.5px] font-semibold ${rel ? 'bg-good-soft text-good' : 'bg-sunken text-faint'}`}>
                {rel ? 'relevant' : 'not relevant'}
              </span>
              <span className="w-12 shrink-0 text-right font-mono text-xs">{s.score.toFixed(exact ? 3 : 2)}</span>
            </li>
          )
        })}
      </ol>
      <Stats>
        <Stat label={`Precision@${k}`} value={precision.toFixed(2)} tone={precision >= 0.67 ? 'good' : precision < 0.4 ? 'warn' : undefined} />
        <Stat label={`Recall@${k}`} value={totalRel ? recall.toFixed(2) : 'n/a'} tone={recall === 1 ? 'good' : undefined} />
        <Stat label="Context tokens" value={(k * tokens).toLocaleString()} />
        <Stat label="Irrelevant chunks sent" value={k - got} tone={k - got > 1 ? 'warn' : undefined} />
      </Stats>
      <Explain>
        With K = {k} the model receives {got} of the {totalRel} relevant chunk{totalRel === 1 ? '' : 's'} and {k - got} irrelevant
        one{k - got === 1 ? '' : 's'}, costing {(k * tokens).toLocaleString()} context tokens.{' '}
        {recall < 1 && totalRel > 0
          ? 'Raise K to recover the relevant chunks still below the line.'
          : k - got > 0
            ? 'Everything relevant is already in; the extra chunks only add noise and cost. Lower K, or rerank and cut.'
            : 'Every chunk sent is relevant and nothing relevant is missing: this is the K to aim for.'}{' '}
        Raising K can only raise recall and usually lowers precision: that tension is why K is tuned, not guessed.
      </Explain>
    </Frame>
  )
}
