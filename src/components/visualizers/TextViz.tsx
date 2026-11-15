import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { bm25Scores, CONCEPTS, cosine, embed, INDEX, ranked, tfidfScores } from '../../lib/ragmath'
import { Explain, Frame, Formula, RankList, Slider, Stat, Stats, TextInput, Toggle } from './kit'

/* ------------------------------------------------------------------ */
/* 4. Chunk size and overlap                                            */
/* ------------------------------------------------------------------ */

const HANDBOOK =
  'Annual leave at Meridian is 25 days per year for full-time staff. Up to 5 unused days carry over into the next year. Carried-over days must be used by 31 March or they are lost. Staff may work remotely up to 3 days a week with their manager’s agreement. The home-office equipment budget is £400 per person. Equipment bought with the budget remains company property. Expenses must be submitted within 30 days of the purchase. Any expense over £75 needs a receipt. Late claims need approval from a director.'

const TINTS = ['bg-accent-soft', 'bg-info-soft', 'bg-warn-soft', 'bg-good-soft', 'bg-danger-soft']

export function Chunking() {
  const [size, setSize] = useState(24)
  const [overlap, setOverlap] = useState(6)
  const [sentences, setSentences] = useState(false)
  const words = useMemo(() => HANDBOOK.split(' '), [])
  const ov = Math.min(overlap, size - 1)

  // chunk boundaries as [start, end) word indexes
  const chunks = useMemo(() => {
    const out: [number, number][] = []
    if (!sentences) {
      for (let s = 0; s < words.length; s += size - ov) {
        out.push([s, Math.min(s + size, words.length)])
        if (s + size >= words.length) break
      }
      return out
    }
    // greedy sentence packing: never cut inside a sentence; overlap by whole sentences
    const ends: number[] = []
    words.forEach((w, i) => /[.!?]$/.test(w) && ends.push(i + 1))
    let start = 0
    while (start < words.length) {
      let end = ends.find((e) => e > start)!
      for (const e of ends) if (e > start && e - start <= size) end = e
      out.push([start, end])
      if (end >= words.length) break
      // step back whole sentences while they fit in the overlap budget
      let next = end
      for (const e of [...ends].reverse()) if (e < end && e > start && end - e <= ov) next = e
      start = next
    }
    return out
  }, [words, size, ov, sentences])

  const total = chunks.reduce((n, [s, e]) => n + (e - s), 0)
  const dup = total - words.length
  const midSentence = chunks.filter(([, e]) => e < words.length && !/[.!?]$/.test(words[e - 1])).length

  return (
    <Frame title="chunk size and overlap">
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Slider label="Chunk size (words)" value={size} min={8} max={60} onChange={setSize} />
        <Slider label="Overlap (words)" value={ov} min={0} max={Math.min(30, size - 1)} onChange={setOverlap} />
        <Toggle label="Respect sentence boundaries" checked={sentences} onChange={setSentences} />
      </div>
      <ol className="space-y-2">
        {chunks.map(([s, e], i) => {
          const sharedPrev = i > 0 ? Math.max(0, chunks[i - 1][1] - s) : 0
          return (
            <li key={i} className={`rounded-lg px-3 py-2 text-[14px] leading-relaxed ${TINTS[i % TINTS.length]}`}>
              <span className="mr-2 font-mono text-[11px] font-semibold text-muted">
                chunk {i + 1} · {e - s} words
              </span>
              {sharedPrev > 0 && (
                <span className="underline decoration-accent decoration-2 underline-offset-4">{words.slice(s, s + sharedPrev).join(' ')} </span>
              )}
              {words.slice(s + sharedPrev, e).join(' ')}
              {e < words.length && !/[.!?]$/.test(words[e - 1]) && <span className="ml-1 font-mono text-xs text-danger">✂ mid-sentence</span>}
            </li>
          )
        })}
      </ol>
      <Stats>
        <Stat label="Chunks" value={chunks.length} />
        <Stat label="Words stored" value={`${total} (+${Math.round((dup / words.length) * 100)}%)`} tone={dup / words.length > 0.5 ? 'warn' : undefined} />
        <Stat label="Mid-sentence cuts" value={midSentence} tone={midSentence ? 'danger' : 'good'} />
        <Stat label="Step" value={sentences ? 'by sentence' : `${size - ov} words`} />
      </Stats>
      <Explain>
        Underlined words are the <strong>overlap</strong>: they are repeated from the previous chunk so that a fact cut in two
        still appears whole somewhere. Overlap costs storage and embedding calls (here {Math.round((dup / words.length) * 100)}%
        more text) and can return near-duplicate chunks.{' '}
        {midSentence > 0
          ? `${midSentence} chunk${midSentence === 1 ? ' ends' : 's end'} in the middle of a sentence; the "31 March" deadline can be separated from the rule it belongs to. Switch on sentence boundaries to see the cuts move.`
          : 'No chunk ends mid-sentence, so every chunk can be read on its own.'}
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 5. Semantic chunking                                                 */
/* ------------------------------------------------------------------ */

const SENTENCES = [
  'Annual leave is 25 days per year.',
  'Up to 5 unused leave days carry over.',
  'Carried-over holiday must be used by 31 March.',
  'Staff may work remotely up to 3 days a week.',
  'Working from home needs your manager’s agreement.',
  'The home office equipment budget is £400.',
  'Expenses must be submitted within 30 days.',
  'Any expense over £75 needs a receipt.',
  'Travel costs for client visits are reimbursed.',
  'Passwords must be rotated every 90 days.',
  'The VPN is required on public Wi-Fi.',
]

export function SemanticChunking() {
  const [threshold, setThreshold] = useState(0.5)
  const vecs = useMemo(() => SENTENCES.map(embed), [])
  const sims = useMemo(() => vecs.slice(1).map((v, i) => cosine(vecs[i], v)), [vecs])
  const groups: number[][] = [[0]]
  sims.forEach((s, i) => (s < threshold ? groups.push([i + 1]) : groups[groups.length - 1].push(i + 1)))
  const topic = (g: number[]) => {
    const sum = [0, 0, 0, 0, 0, 0]
    g.forEach((i) => vecs[i].forEach((w, d) => (sum[d] += w)))
    return CONCEPTS[sum.indexOf(Math.max(...sum))]
  }
  const mixed = groups.filter((g) => new Set(g.map((i) => vecs[i].indexOf(Math.max(...vecs[i])))).size > 1).length

  return (
    <Frame title="semantic chunking">
      <Slider
        label="Similarity threshold (split when adjacent sentences fall below it)"
        value={threshold}
        min={0}
        max={1}
        step={0.01}
        onChange={setThreshold}
        format={(v) => v.toFixed(2)}
      />
      <ol>
        {SENTENCES.map((s, i) => {
          const g = groups.findIndex((x) => x.includes(i))
          return (
            <li key={i}>
              {i > 0 && (
                <div className="flex items-center gap-2 py-1 pl-3 text-xs">
                  <span className={`h-px flex-1 ${sims[i - 1] < threshold ? 'bg-danger' : 'bg-line'}`} />
                  <span className={`font-mono ${sims[i - 1] < threshold ? 'font-semibold text-danger' : 'text-faint'}`}>
                    similarity {sims[i - 1].toFixed(2)} {sims[i - 1] < threshold ? '→ split here' : ''}
                  </span>
                  <span className={`h-px w-6 ${sims[i - 1] < threshold ? 'bg-danger' : 'bg-line'}`} />
                </div>
              )}
              <p className={`rounded-md px-3 py-1.5 text-[14px] ${TINTS[g % TINTS.length]}`}>
                <span className="mr-2 font-mono text-[11px] text-muted">chunk {g + 1}</span>
                {s}
              </p>
            </li>
          )
        })}
      </ol>
      <Stats>
        <Stat label="Chunks" value={groups.length} />
        <Stat label="Largest chunk" value={`${Math.max(...groups.map((g) => g.length))} sentences`} />
        <Stat label="Chunks mixing topics" value={mixed} tone={mixed ? 'warn' : 'good'} />
        <Stat label="Topics found" value={[...new Set(groups.map(topic))].length} />
      </Stats>
      <Explain>
        Each sentence is embedded and compared with the one before it. Where the similarity drops below {threshold.toFixed(2)},
        the topic has probably changed, so a new chunk starts.{' '}
        {groups.length <= 2
          ? 'The threshold is so low that unrelated policies are merged into one chunk; a query about expenses would retrieve leave rules too.'
          : groups.length >= 8
            ? 'The threshold is so high that almost every sentence is its own chunk; each one loses the context of its neighbours.'
            : `Around here the chunks line up with the handbook's topics (${[...new Set(groups.map(topic))].join(', ')}).`}{' '}
        Production implementations usually set the threshold from a percentile of the observed similarities rather than a fixed number.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 6. TF-IDF                                                            */
/* ------------------------------------------------------------------ */

const QUERY_PRESETS = ['annual leave carry over', 'expenses receipt', 'vpn', 'days', 'laptop replacement']

function TermTable({ rows, scored }: { rows: ReturnType<typeof tfidfScores>; scored: 'tf-idf' | 'bm25' }) {
  const termsUsed = rows[0]?.parts.map((p) => p.term) ?? []
  if (!termsUsed.length) return null
  return (
    <div className="thin-scroll overflow-x-auto rounded-lg border border-line">
      <table className="w-full min-w-[30rem] border-collapse text-[13px]">
        <thead>
          <tr className="bg-sunken text-left text-[11px] uppercase tracking-wider text-muted">
            <th className="px-3 py-2 font-semibold">Document</th>
            {termsUsed.map((t) => (
              <th key={t} className="px-3 py-2 font-mono font-semibold normal-case">
                {t}
                <span className="block font-sans text-[10px] font-normal text-faint">idf {rows[0].parts.find((p) => p.term === t)!.idf.toFixed(2)}</span>
              </th>
            ))}
            <th className="px-3 py-2 text-right font-semibold">{scored} score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.doc.id} className={`border-t border-line ${r.score === 0 ? 'opacity-45' : ''}`}>
              <td className="px-3 py-1.5">
                <span className="font-mono text-[11.5px] text-muted">{r.doc.id}</span> {r.doc.title}
              </td>
              {r.parts.map((p) => (
                <td key={p.term} className="px-3 py-1.5 font-mono">
                  {p.tf === 0 ? <span className="text-faint">–</span> : <>tf {p.tf} → {p.score.toFixed(2)}</>}
                </td>
              ))}
              <td className="px-3 py-1.5 text-right font-mono font-semibold">{r.score.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function TfIdf() {
  const [query, setQuery] = useState('annual leave carry over')
  const rows = ranked(tfidfScores(query))
  const parts = rows[0]?.parts ?? []
  const rare = [...parts].sort((a, b) => b.idf - a.idf)[0]
  const common = [...parts].sort((a, b) => a.idf - b.idf)[0]

  return (
    <Frame title="TF-IDF scoring">
      <TextInput label="Query" value={query} onChange={setQuery} presets={QUERY_PRESETS} />
      <Formula>
        score(q, d) = Σ<sub>t ∈ q</sub> tf(t, d) × idf(t) &nbsp;&nbsp; idf(t) = ln((1 + N) / (1 + df(t))) + 1 &nbsp;&nbsp; N = {INDEX.docs.length}
      </Formula>
      {parts.length ? (
        <>
          <div className="flex flex-wrap gap-2">
            {parts.map((p) => (
              <span key={p.term} className="rounded-lg border border-line px-2.5 py-1 text-[13px]">
                <span className="font-mono font-semibold">{p.term}</span>{' '}
                <span className="text-muted">
                  in {INDEX.df.get(p.term) ?? 0} of {INDEX.docs.length} docs · idf {p.idf.toFixed(2)}
                </span>
              </span>
            ))}
          </div>
          <TermTable rows={rows} scored="tf-idf" />
        </>
      ) : (
        <p className="text-sm text-muted">Type at least one word that is not a stop word.</p>
      )}
      <Explain>
        {parts.length === 0 ? (
          <>Stop words such as “the” and “is” are removed before scoring.</>
        ) : rows[0].score === 0 ? (
          <>No document contains any query term, so every score is zero. Keyword search cannot match on meaning: try “holiday”, which the handbook calls “leave”.</>
        ) : (
          <>
            <strong>{rows[0].doc.id}</strong> wins with {rows[0].score.toFixed(3)}.{' '}
            {parts.length > 1 && rare.term !== common.term ? (
              <>
                “{rare.term}” appears in only {INDEX.df.get(rare.term) ?? 0} document{(INDEX.df.get(rare.term) ?? 0) === 1 ? '' : 's'}, so its
                IDF is high ({rare.idf.toFixed(2)}) and a match on it counts for more than a match on “{common.term}” (in{' '}
                {INDEX.df.get(common.term) ?? 0} documents, IDF {common.idf.toFixed(2)}).
              </>
            ) : (
              <>The score is the term's count in the document multiplied by how rare the term is across the corpus.</>
            )}{' '}
            Rare words identify documents; common words barely discriminate.
          </>
        )}
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 7. BM25                                                              */
/* ------------------------------------------------------------------ */

export function Bm25() {
  const [query, setQuery] = useState('annual leave days')
  const [k1, setK1] = useState(1.5)
  const [b, setB] = useState(0.75)
  const rows = ranked(bm25Scores(query, k1, b))
  const lens = INDEX.tokens.map((t) => t.length)
  const curve = Array.from({ length: 11 }, (_, f) => ({
    tf: f,
    'this k1': f === 0 ? 0 : (f * (k1 + 1)) / (f + k1),
    'raw tf (no saturation)': f,
  }))
  const cap = k1 + 1

  return (
    <Frame title="BM25 scoring">
      <TextInput label="Query" value={query} onChange={setQuery} presets={['annual leave days', 'vpn', 'laptop', 'expenses receipts 30 days']} />
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Slider label="k1 (term-frequency saturation)" value={k1} min={0} max={3} step={0.1} onChange={setK1} format={(v) => v.toFixed(1)} hint="0 ignores repeats; typical 1.2 to 2.0" />
        <Slider label="b (length normalisation)" value={b} min={0} max={1} step={0.05} onChange={setB} format={(v) => v.toFixed(2)} hint="0 ignores length; 1 normalises fully" />
      </div>
      <Formula>
        score = Σ idf(t) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × |d| / avgdl)) &nbsp;&nbsp; avgdl = {INDEX.avgLen.toFixed(1)} terms
      </Formula>
      <div className="grid gap-5 lg:grid-cols-2">
        <RankList
          title="Ranking"
          rows={rows.slice(0, 6).map((r) => ({
            id: r.doc.id,
            label: r.doc.title,
            score: r.score,
            dim: r.score === 0,
            note: `${INDEX.tokens[INDEX.docs.indexOf(r.doc)].length} terms · ${r.parts.filter((p) => p.tf).map((p) => `${p.term}×${p.tf}`).join(', ') || 'no match'}`,
          }))}
        />
        <div className="min-w-0">
          <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-faint">Contribution of one term as it repeats</h4>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve} margin={{ top: 6, right: 12, bottom: 0, left: -22 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="tf" tickLine={false} />
                <YAxis tickLine={false} domain={[0, 5]} allowDataOverflow />
                <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, fontSize: 12 }} formatter={(v) => Number(v).toFixed(2)} />
                <ReferenceLine y={cap} stroke="var(--warn)" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="raw tf (no saturation)" stroke="var(--faint)" dot={false} strokeDasharray="4 3" />
                <Line type="monotone" dataKey="this k1" stroke="var(--accent)" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-muted">
            Document of average length, before multiplying by IDF. The curve can never pass k1 + 1 = {cap.toFixed(1)} (dashed).
          </p>
        </div>
      </div>
      <Explain>
        <strong>k1 = {k1.toFixed(1)}</strong>:{' '}
        {k1 === 0
          ? 'term frequency is ignored entirely; a document either contains the term or does not.'
          : `the second occurrence of a term adds less than the first and the curve flattens towards ${cap.toFixed(1)}, so stuffing a keyword cannot win.`}{' '}
        <strong>b = {b.toFixed(2)}</strong>:{' '}
        {b === 0
          ? 'document length is ignored, so long documents that mention everything are favoured.'
          : `a match in a short document counts for more than the same match in a long one (documents here run from ${Math.min(...lens)} to ${Math.max(...lens)} terms).`}{' '}
        TF-IDF has neither control: its score grows linearly with repetition.
      </Explain>
    </Frame>
  )
}
