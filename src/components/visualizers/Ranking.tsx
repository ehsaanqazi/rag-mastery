import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Minus } from 'lucide-react'
import { bm25Scores, CORPUS, cosine, denseScores, DOC_VECTORS, embed, minmax, mmr, ranked, rrf, stem, terms, tokenize } from '../../lib/ragmath'
import { Explain, Frame, Formula, RankList, Slider, Stat, Stats, TextInput } from './kit'

const title = (id: string) => CORPUS.find((d) => d.id === id)!.title

const HYBRID_PRESETS = ['holiday allowance', 'VPN', 'reimbursement for a client trip', 'how often do I change my password', 'computer replacement portal']

/* ------------------------------------------------------------------ */
/* 8. Hybrid retrieval (weighted score fusion)                          */
/* ------------------------------------------------------------------ */

export function HybridRetrieval() {
  const [query, setQuery] = useState('holiday allowance')
  const [alpha, setAlpha] = useState(0.5)
  const dense = denseScores(query)
  const sparse = bm25Scores(query)
  const dn = minmax(dense.map((d) => d.score))
  const sn = minmax(sparse.map((d) => d.score))
  const fused = ranked(CORPUS.map((doc, i) => ({ doc, dense: dn[i], sparse: sn[i], score: alpha * dn[i] + (1 - alpha) * sn[i] })))
  const dTop = ranked(dense)[0]
  const sTop = ranked(sparse)[0]
  const sparseEmpty = sTop.score === 0
  const denseEmpty = dTop.score === 0

  return (
    <Frame title="hybrid retrieval with weighted score fusion">
      <TextInput label="Query" value={query} onChange={setQuery} presets={HYBRID_PRESETS} />
      <Slider
        label="α: weight on dense (semantic) scores"
        value={alpha}
        min={0}
        max={1}
        step={0.05}
        onChange={setAlpha}
        format={(v) => `${v.toFixed(2)} dense · ${(1 - v).toFixed(2)} sparse`}
      />
      <Formula>hybrid = α × norm(dense) + (1 − α) × norm(BM25), each min-max scaled to [0, 1] over the candidates</Formula>
      <div className="grid gap-5 md:grid-cols-3">
        <RankList title="Dense (toy embeddings)" tone="info" max={1} rows={ranked(dense).slice(0, 5).map((d) => ({ id: d.doc.id, label: d.doc.title, score: d.score, dim: d.score === 0 }))} />
        <RankList title="Sparse (BM25)" tone="warn" rows={ranked(sparse).slice(0, 5).map((d) => ({ id: d.doc.id, label: d.doc.title, score: d.score, dim: d.score === 0 }))} />
        <RankList
          title="Hybrid"
          max={1}
          rows={fused.slice(0, 5).map((d, i) => ({
            id: d.doc.id,
            label: d.doc.title,
            score: d.score,
            highlight: i === 0,
            dim: d.score === 0,
            note: `${alpha.toFixed(2)}×${d.dense.toFixed(2)} + ${(1 - alpha).toFixed(2)}×${d.sparse.toFixed(2)}`,
          }))}
        />
      </div>
      <Explain>
        {sparseEmpty && !denseEmpty ? (
          <>
            BM25 found <strong>nothing</strong>: none of the query's words appear in the handbook. Dense retrieval still finds{' '}
            {dTop.doc.id} because it matches on meaning. This is the vocabulary-mismatch problem that hybrid search exists to cover.
          </>
        ) : denseEmpty && !sparseEmpty ? (
          <>
            The toy embedding model does not know these words, so dense retrieval is blind. BM25 finds {sTop.doc.id} by exact match:
            identifiers, product codes and rare names are where keyword search earns its place.
          </>
        ) : dTop.doc.id === sTop.doc.id ? (
          <>Both retrievers agree on {dTop.doc.id}, so the weight barely matters for the top result. Agreement between independent signals is strong evidence.</>
        ) : (
          <>
            The retrievers disagree: dense prefers {dTop.doc.id}, BM25 prefers {sTop.doc.id}. Slide α to see the hybrid winner flip
            (currently {fused[0].doc.id}).
          </>
        )}{' '}
        Scores must be normalised before mixing because cosine lives in [−1, 1] and BM25 is unbounded; Reciprocal Rank Fusion
        avoids that problem by using ranks instead.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 9. Reciprocal Rank Fusion                                            */
/* ------------------------------------------------------------------ */

export function Rrf() {
  const [query, setQuery] = useState('carry over unused holiday')
  const [k, setK] = useState(60)
  const depth = 5
  const dense = ranked(denseScores(query)).filter((d) => d.score > 0).slice(0, depth).map((d) => d.doc.id)
  const sparse = ranked(bm25Scores(query)).filter((d) => d.score > 0).slice(0, depth).map((d) => d.doc.id)
  const fused = rrf([dense, sparse], k)
  const both = fused.filter((f) => f.ranks.every((r) => r !== null)).length

  const list = (name: string, ids: string[]) => (
    <div className="min-w-0">
      <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-faint">{name}</h4>
      {ids.length ? (
        <ol className="space-y-1">
          {ids.map((id, i) => (
            <li key={id} className="flex gap-2 rounded-md border border-line px-2.5 py-1 text-[13.5px]">
              <span className="font-mono text-[11px] text-faint">{i + 1}</span>
              <span className="truncate">
                <span className="font-mono text-[11.5px] text-muted">{id}</span> {title(id)}
              </span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="rounded-md border border-dashed border-line px-2.5 py-2 text-[13px] text-muted">No matches</p>
      )}
    </div>
  )

  return (
    <Frame title="Reciprocal Rank Fusion">
      <TextInput label="Query" value={query} onChange={setQuery} presets={['carry over unused holiday', 'expenses receipt', 'remote work equipment budget', 'VPN']} />
      <Slider label="k (rank constant)" value={k} min={1} max={100} onChange={setK} hint="60 is the value from the original paper and the usual default" />
      <Formula>RRF(d) = Σ over lists of 1 / (k + rank(d)) &nbsp;·&nbsp; a document missing from a list contributes 0 for that list</Formula>
      <div className="grid gap-5 md:grid-cols-[1fr_1fr_1.5fr]">
        {list('Dense ranking', dense)}
        {list('BM25 ranking', sparse)}
        <RankList
          title="Fused"
          digits={4}
          rows={fused.map((f, i) => ({
            id: f.id,
            label: title(f.id),
            score: f.score,
            highlight: i === 0,
            note: f.ranks.map((r) => (r === null ? '0' : `1/(${k}+${r})`)).join(' + '),
          }))}
        />
      </div>
      <Explain>
        RRF never looks at the scores, only at positions, so it needs no normalisation and works across retrievers whose scores
        are not comparable. {both} document{both === 1 ? ' appears' : 's appear'} in both lists and collect two terms, which is why
        agreement is rewarded.{' '}
        {k <= 5
          ? `With k = ${k} the top rank dominates: 1/(${k}+1) is much larger than 1/(${k}+2), so a first place in one list can beat a document that is second in both.`
          : `With k = ${k} the gap between neighbouring ranks is small (1/${k + 1} versus 1/${k + 2}), so appearing in both lists matters more than being first in one.`}
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 10. Reranking                                                        */
/* ------------------------------------------------------------------ */

/**
 * A toy second-stage scorer that reads the query and the document together:
 * the share of query terms the document covers exactly, plus a bonus for query
 * terms appearing next to each other. A real cross-encoder is a transformer
 * that attends across both texts; this stand-in only reproduces the property
 * that matters here (it sees word-level interactions the bi-encoder cannot).
 */
function pairScore(query: string, text: string): number {
  const q = [...new Set(terms(query))]
  if (!q.length) return 0
  const d = tokenize(text).map(stem)
  const covered = q.filter((t) => d.includes(t)).length / q.length
  let adjacent = 0
  for (let i = 0; i < d.length - 1; i++) if (q.includes(d[i]) && q.includes(d[i + 1]) && d[i] !== d[i + 1]) adjacent++
  const semantic = cosine(embed(query), embed(text))
  return Math.min(1, 0.55 * covered + 0.15 * Math.min(1, adjacent / 2) + 0.3 * semantic)
}

export function Reranking() {
  const [query, setQuery] = useState('how many unused leave days carry over')
  const [n, setN] = useState(6)
  const [keep, setKeep] = useState(3)
  const [perPair, setPerPair] = useState(15)
  const first = ranked(denseScores(query)).slice(0, n)
  const second = ranked(first.map((f) => ({ doc: f.doc, score: pairScore(query, f.doc.text) })))
  const k = Math.min(keep, n)
  const moved = second.filter((s, i) => first.findIndex((f) => f.doc.id === s.doc.id) !== i).length
  const promoted = second.slice(0, k).filter((s) => first.findIndex((f) => f.doc.id === s.doc.id) >= k)

  return (
    <Frame title="two-stage retrieval with reranking">
      <TextInput
        label="Query"
        value={query}
        onChange={setQuery}
        presets={['how many unused leave days carry over', 'receipt needed for expenses over £75', 'VPN on public wifi', 'remote work days a week']}
      />
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Slider label="N candidates from stage 1" value={n} min={2} max={9} onChange={setN} />
        <Slider label="K kept after reranking" value={k} min={1} max={n} onChange={setKeep} />
        <Slider label="Reranker cost per pair (ms, assumed)" value={perPair} min={2} max={60} onChange={setPerPair} />
      </div>
      <div className="grid gap-5 md:grid-cols-2">
        <RankList
          title="Stage 1 · bi-encoder (fast, approximate)"
          tone="info"
          max={1}
          rows={first.map((f, i) => ({ id: f.doc.id, label: f.doc.title, score: f.score, dim: i >= k }))}
        />
        <div className="min-w-0">
          <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-faint">Stage 2 · reranker (slow, precise)</h4>
          <ol className="space-y-1.5">
            {second.map((s, i) => {
              const was = first.findIndex((f) => f.doc.id === s.doc.id)
              return (
                <li key={s.doc.id} className={`flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-[13.5px] ${i < k ? 'border-accent bg-accent-soft' : 'border-line opacity-45'}`}>
                  <span className="w-4 font-mono text-[11px] text-faint">{i + 1}</span>
                  <span className="min-w-0 flex-1 truncate">
                    <span className="font-mono text-[11.5px] text-muted">{s.doc.id}</span> {s.doc.title}
                  </span>
                  <span className={`flex items-center gap-0.5 font-mono text-[11px] ${was > i ? 'text-good' : was < i ? 'text-danger' : 'text-faint'}`}>
                    {was > i ? <ArrowUp size={12} /> : was < i ? <ArrowDown size={12} /> : <Minus size={12} />}
                    {was !== i && Math.abs(was - i)}
                  </span>
                  <span className="w-11 text-right font-mono text-xs">{s.score.toFixed(3)}</span>
                </li>
              )
            })}
          </ol>
        </div>
      </div>
      <Stats>
        <Stat label="Pairs scored" value={n} />
        <Stat label="Added latency" value={`${n * perPair} ms`} tone={n * perPair > 300 ? 'warn' : undefined} />
        <Stat label="Positions changed" value={moved} />
        <Stat label="Promoted into top K" value={promoted.length} tone={promoted.length ? 'good' : undefined} />
      </Stats>
      <Explain>
        Stage 1 compares one query vector with precomputed document vectors: cheap enough to run over millions of chunks, but it
        never sees the query's words next to the document's. Stage 2 scores each (query, document) pair together, which is far
        more accurate and far too slow for the whole corpus, so it only sees the top {n}.{' '}
        {promoted.length > 0
          ? `Here it promoted ${promoted.map((p) => p.doc.id).join(', ')} into the top ${k}: a document stage 1 had ranked too low.`
          : `Here the top ${k} did not change membership; the reranker still fixed the order within it.`}{' '}
        A reranker can only reorder what stage 1 retrieved: if the right chunk is not among the N candidates, no reranker can
        recover it. The stage 2 scorer in this demo is a simple stand-in for a cross-encoder model.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 12. Maximal Marginal Relevance                                       */
/* ------------------------------------------------------------------ */

export function Mmr() {
  const [query, setQuery] = useState('annual leave carry over and expenses')
  const [lambda, setLambda] = useState(0.7)
  const [k, setK] = useState(3)
  const rel = useMemo(() => denseScores(query).map((d) => d.score), [query])
  const sim = useMemo(() => DOC_VECTORS.map((a) => DOC_VECTORS.map((b) => cosine(a, b))), [])
  const steps = mmr(rel, sim, lambda, k)
  const plain = ranked(CORPUS.map((doc, i) => ({ doc, score: rel[i], i }))).slice(0, k)
  const maxPair = (idx: number[]) => {
    let m = 0
    idx.forEach((a, x) => idx.forEach((b, y) => x < y && (m = Math.max(m, sim[a][b]))))
    return m
  }
  const topics = (idx: number[]) => new Set(idx.map((i) => DOC_VECTORS[i].indexOf(Math.max(...DOC_VECTORS[i])))).size

  return (
    <Frame title="Maximal Marginal Relevance">
      <TextInput label="Query" value={query} onChange={setQuery} presets={['annual leave carry over and expenses', 'unused holiday', 'security rules for remote work']} />
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Slider label="λ (1 = relevance only, 0 = diversity only)" value={lambda} min={0} max={1} step={0.05} onChange={setLambda} format={(v) => v.toFixed(2)} />
        <Slider label="K" value={k} min={1} max={6} onChange={setK} />
      </div>
      <Formula>MMR(d) = λ × sim(d, query) − (1 − λ) × max sim(d, already selected)</Formula>
      <div className="grid gap-5 md:grid-cols-2">
        <RankList
          title="Plain top-K by relevance"
          tone="muted"
          max={1}
          rows={plain.map((p) => ({ id: p.doc.id, label: p.doc.title, score: p.score }))}
        />
        <RankList
          title="MMR selection, in order chosen"
          max={1}
          rows={steps.map((s) => ({
            id: CORPUS[s.picked].id,
            label: CORPUS[s.picked].title,
            score: Math.max(0, s.mmr),
            highlight: !plain.some((p) => p.i === s.picked),
            note: `${lambda.toFixed(2)}×${s.relevance.toFixed(2)} − ${(1 - lambda).toFixed(2)}×${s.redundancy.toFixed(2)} = ${s.mmr.toFixed(3)}`,
          }))}
        />
      </div>
      <Stats>
        <Stat label="Topics covered · plain" value={topics(plain.map((p) => p.i))} />
        <Stat label="Topics covered · MMR" value={topics(steps.map((s) => s.picked))} tone={topics(steps.map((s) => s.picked)) > topics(plain.map((p) => p.i)) ? 'good' : undefined} />
        <Stat label="Most similar pair · plain" value={k > 1 ? maxPair(plain.map((p) => p.i)).toFixed(2) : 'n/a'} tone={k > 1 && maxPair(plain.map((p) => p.i)) > 0.95 ? 'warn' : undefined} />
        <Stat label="Most similar pair · MMR" value={k > 1 ? maxPair(steps.map((s) => s.picked)).toFixed(2) : 'n/a'} />
      </Stats>
      <Explain>
        D1 and D8 say almost the same thing about leave carry-over. Plain top-K happily returns both, spending two context slots
        on one fact. MMR picks the most relevant document first, then penalises each remaining candidate by its similarity to
        what is already chosen.{' '}
        {lambda >= 0.95
          ? 'At λ near 1 the penalty vanishes and MMR is identical to plain top-K.'
          : lambda <= 0.2
            ? 'At λ this low, diversity dominates: later picks may be barely relevant to the query at all.'
            : 'Highlighted rows are documents plain top-K would not have returned.'}
      </Explain>
    </Frame>
  )
}
