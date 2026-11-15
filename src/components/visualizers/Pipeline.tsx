import { useState } from 'react'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CONCEPTS, dcg, denseScores, embed, ndcg, precisionAtK, ranked, recallAtK, reciprocalRank } from '../../lib/ragmath'
import { Explain, Frame, Formula, Segmented, Slider, Stat, Stats, TextInput, Toggle } from './kit'

/* ------------------------------------------------------------------ */
/* 13. RAG pipeline execution                                           */
/* ------------------------------------------------------------------ */

const STAGES = ['Question', 'Embed query', 'Search index', 'Select top-K', 'Build prompt', 'Generate'] as const

export function RagPipeline() {
  const [query, setQuery] = useState('How many leave days can I carry over?')
  const [k, setK] = useState(2)
  const [step, setStep] = useState(0)
  const qv = embed(query)
  const results = ranked(denseScores(query))
  const top = results.slice(0, k)
  const grounded = top[0]?.score >= 0.35
  const prompt = `System: Answer only from the context. Cite sources as [id]. If the answer is not in the context, say you don't know.

Context:
${top.map((t) => `[${t.doc.id}] ${t.doc.text}`).join('\n')}

Question: ${query}`

  return (
    <Frame title="one question through the RAG pipeline">
      <TextInput
        label="Question"
        value={query}
        onChange={(v) => {
          setQuery(v)
          setStep(0)
        }}
        presets={['How many leave days can I carry over?', 'Do I need a receipt for a £90 taxi?', 'When is my laptop replaced?', 'What is the capital of France?']}
      />
      <Slider label="K (chunks retrieved)" value={k} min={1} max={4} onChange={setK} />

      <ol className="flex flex-wrap gap-1.5" aria-label="Pipeline stages">
        {STAGES.map((s, i) => (
          <li key={s}>
            <button
              onClick={() => setStep(i)}
              aria-current={i === step ? 'step' : undefined}
              className={`rounded-full border px-2.5 py-1 text-[12.5px] ${
                i === step ? 'border-accent bg-accent font-medium text-accent-fg' : i < step ? 'border-accent text-accent' : 'border-line text-muted'
              }`}
            >
              {i + 1}. {s}
            </button>
          </li>
        ))}
      </ol>

      <div className="min-h-44 rounded-lg border border-line bg-bg p-4 text-[14px]">
        {step === 0 && (
          <p>
            The user asks: <strong>“{query}”</strong>. Nothing has been computed yet. Everything the system knows about Meridian
            is in the index, which was built ahead of time (the offline path).
          </p>
        )}
        {step === 1 && (
          <div>
            <p className="mb-3">The same embedding model used at indexing time turns the question into a vector:</p>
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
        )}
        {step === 2 && (
          <div>
            <p className="mb-2">The query vector is compared with every chunk vector (cosine similarity):</p>
            <ul className="space-y-1 font-mono text-[12.5px]">
              {results.map((r) => (
                <li key={r.doc.id} className="flex gap-3">
                  <span className="w-12">{r.score.toFixed(3)}</span>
                  <span className="text-muted">{r.doc.id}</span>
                  <span className="truncate font-sans">{r.doc.title}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {step === 3 && (
          <div>
            <p className="mb-2">The {k} highest-scoring chunk{k === 1 ? ' is' : 's are'} kept; the rest are discarded:</p>
            <ul className="space-y-1.5">
              {top.map((t) => (
                <li key={t.doc.id} className="rounded-md border border-accent bg-accent-soft px-3 py-1.5">
                  <span className="font-mono text-[11.5px] text-muted">{t.doc.id}</span> {t.doc.text}
                </li>
              ))}
            </ul>
          </div>
        )}
        {step === 4 && <pre className="thin-scroll overflow-x-auto whitespace-pre-wrap font-mono text-[12.5px] leading-relaxed">{prompt}</pre>}
        {step === 5 && (
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-faint">Simulated answer</p>
            <p className="rounded-md border border-line bg-surface px-3 py-2">
              {grounded ? (
                <>
                  {top[0].doc.text} <span className="font-mono text-accent">[{top[0].doc.id}]</span>
                </>
              ) : (
                <>I don't know. The provided context does not contain the answer to that question.</>
              )}
            </p>
            <p className="mt-2 text-xs text-muted">
              This visualiser has no language model: it quotes the top chunk when its score clears 0.35 and refuses otherwise. A
              real system sends the prompt from step 5 to an LLM, which writes the answer in its own words.
            </p>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button onClick={() => setStep(0)} disabled={step === 0} aria-label="Restart" className="rounded-md border border-line p-1.5 text-muted hover:text-fg disabled:opacity-40">
          <RotateCcw size={15} />
        </button>
        <button onClick={() => setStep(step - 1)} disabled={step === 0} aria-label="Previous stage" className="rounded-md border border-line p-1.5 text-muted hover:text-fg disabled:opacity-40">
          <ChevronLeft size={15} />
        </button>
        <button onClick={() => setStep(step + 1)} disabled={step === STAGES.length - 1} className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-fg disabled:opacity-40">
          Next stage <ChevronRight size={15} />
        </button>
        <span className="ml-2 text-xs text-faint">
          {step < 4 ? 'Retrieval' : 'Generation'} · stage {step + 1} of {STAGES.length}
        </span>
      </div>
      <Explain>
        {grounded
          ? `Retrieval found a strong match (${top[0].doc.id}, similarity ${top[0].score.toFixed(2)}), so the model has the fact it needs in front of it and can cite it.`
          : `The best match scores only ${(top[0]?.score ?? 0).toFixed(2)}: the handbook does not cover this question. A well-built system refuses here instead of letting the model answer from memory.`}{' '}
        Steps 2 to 4 are retrieval and need no language model; only step 6 does.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 14. Context window management                                        */
/* ------------------------------------------------------------------ */

const WINDOWS = [
  { value: '8000', label: '8k' },
  { value: '32000', label: '32k' },
  { value: '128000', label: '128k' },
  { value: '200000', label: '200k' },
] as const

export function ContextWindow() {
  const [win, setWin] = useState<(typeof WINDOWS)[number]['value']>('8000')
  const [system, setSystem] = useState(400)
  const [history, setHistory] = useState(1200)
  const [k, setK] = useState(8)
  const [chunk, setChunk] = useState(500)
  const [question, setQuestion] = useState(60)
  const [reserve, setReserve] = useState(1000)
  const W = Number(win)
  const chunks = k * chunk
  const parts = [
    { name: 'System prompt', v: system, cls: 'bg-muted' },
    { name: 'Chat history', v: history, cls: 'bg-warn' },
    { name: `Retrieved chunks (${k} × ${chunk})`, v: chunks, cls: 'bg-accent' },
    { name: 'Question', v: question, cls: 'bg-info' },
    { name: 'Reserved for the answer', v: reserve, cls: 'bg-good' },
  ]
  const used = parts.reduce((n, p) => n + p.v, 0)
  const over = used - W
  const room = W - system - history - question - reserve
  const fits = Math.max(0, Math.floor(room / chunk))
  const scale = Math.max(W, used)

  return (
    <Frame title="context window budget">
      <Segmented label="Model context window (tokens)" options={WINDOWS} value={win} onChange={setWin} />
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Slider label="System prompt" value={system} min={0} max={3000} step={50} onChange={setSystem} />
        <Slider label="Chat history" value={history} min={0} max={20000} step={200} onChange={setHistory} />
        <Slider label="Chunks retrieved (K)" value={k} min={0} max={40} onChange={setK} />
        <Slider label="Tokens per chunk" value={chunk} min={100} max={2000} step={50} onChange={setChunk} />
        <Slider label="Question" value={question} min={10} max={500} step={10} onChange={setQuestion} />
        <Slider label="Reserved for the answer" value={reserve} min={100} max={8000} step={100} onChange={setReserve} />
      </div>
      <div>
        <div className="relative flex h-7 overflow-hidden rounded-md bg-sunken" role="img" aria-label="Stacked token budget">
          {parts.map((p) => (
            <div key={p.name} className={`${p.cls} h-full transition-all`} style={{ width: `${(p.v / scale) * 100}%` }} title={`${p.name}: ${p.v}`} />
          ))}
          {over > 0 && <div className="absolute inset-y-0 border-l-2 border-danger" style={{ left: `${(W / scale) * 100}%` }} />}
        </div>
        <ul className="mt-3 grid gap-x-6 gap-y-1 text-[13px] sm:grid-cols-2">
          {parts.map((p) => (
            <li key={p.name} className="flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-sm ${p.cls}`} />
              <span className="flex-1 text-muted">{p.name}</span>
              <span className="font-mono">{p.v.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      </div>
      <Stats>
        <Stat label="Used" value={`${used.toLocaleString()} / ${W.toLocaleString()}`} tone={over > 0 ? 'danger' : 'good'} />
        <Stat label={over > 0 ? 'Over by' : 'Free'} value={Math.abs(over).toLocaleString()} tone={over > 0 ? 'danger' : undefined} />
        <Stat label="Chunks that fit" value={fits} tone={fits < k ? 'warn' : 'good'} />
        <Stat label="Share spent on context" value={`${Math.round((chunks / Math.max(used, 1)) * 100)}%`} />
      </Stats>
      <Formula>chunks that fit = ⌊(window − system − history − question − answer reserve) / tokens per chunk⌋ = ⌊{room.toLocaleString()} / {chunk}⌋ = {fits}</Formula>
      <Explain>
        {over > 0 ? (
          <>
            The request is <strong>{over.toLocaleString()} tokens over</strong> the window and would be rejected or truncated.
            Only {fits} of the {k} chunks fit. Options: retrieve fewer or smaller chunks, summarise the chat history, or rerank and
            keep the best {fits}.
          </>
        ) : (
          <>
            Everything fits with {Math.abs(over).toLocaleString()} tokens to spare. Fitting is not the same as helping: every
            token is paid for on every request, and models attend less reliably to material buried in the middle of a long
            context. Send the chunks that answer the question, not the chunks that fit.
          </>
        )}{' '}
        The answer reserve is easy to forget: input and output share the same window.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 15. RAG evaluation metrics                                           */
/* ------------------------------------------------------------------ */

export function EvalMetrics() {
  // graded relevance of each retrieved result, in ranked order: 0 = not relevant, 1 = relevant, 2 = highly relevant
  const [rel, setRel] = useState([0, 2, 1, 0, 0, 1, 0, 0, 0, 0])
  const [k, setK] = useState(5)
  const [missed, setMissed] = useState(1)
  const totalRel = rel.filter((r) => r > 0).length + missed
  const cycle = (i: number) => setRel((r) => r.map((v, j) => (j === i ? (v + 1) % 3 : v)))
  const p = precisionAtK(rel, k)
  const r = recallAtK(rel, k, totalRel)
  const rr = reciprocalRank(rel)
  const n = ndcg(rel, k, Array(missed).fill(1))
  const first = rel.findIndex((x) => x > 0)

  return (
    <Frame title="retrieval evaluation metrics">
      <p className="text-sm text-muted">
        These are the ten results a retriever returned for one query, best first. Click a result to cycle its relevance label.
      </p>
      <ol className="grid gap-1.5 sm:grid-cols-2">
        {rel.map((v, i) => (
          <li key={i}>
            <button
              onClick={() => cycle(i)}
              className={`flex w-full items-center gap-3 rounded-md border px-3 py-1.5 text-left text-[13.5px] ${
                i < k ? 'border-line' : 'border-dashed border-line opacity-50'
              } ${v === 2 ? 'bg-good-soft' : v === 1 ? 'bg-accent-soft' : ''}`}
            >
              <span className="font-mono text-[11px] text-faint">rank {i + 1}</span>
              <span className="flex-1">{v === 2 ? 'Highly relevant' : v === 1 ? 'Relevant' : 'Not relevant'}</span>
              <span className="font-mono text-xs text-muted">gain {v}</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Slider label="K (cut-off)" value={k} min={1} max={10} onChange={setK} />
        <Slider label="Relevant documents never retrieved" value={missed} min={0} max={5} onChange={setMissed} />
      </div>
      <Stats>
        <Stat label={`Precision@${k}`} value={p.toFixed(3)} />
        <Stat label={`Recall@${k}`} value={r.toFixed(3)} tone={r === 1 ? 'good' : undefined} />
        <Stat label="Reciprocal rank" value={rr.toFixed(3)} />
        <Stat label={`NDCG@${k}`} value={n.toFixed(3)} tone={n > 0.9 ? 'good' : undefined} />
      </Stats>
      <div className="space-y-1.5">
        <Formula>
          Precision@{k} = relevant in top {k} / {k} = {rel.slice(0, k).filter((x) => x > 0).length} / {k}
        </Formula>
        <Formula>
          Recall@{k} = relevant in top {k} / all relevant = {rel.slice(0, k).filter((x) => x > 0).length} / {totalRel}
        </Formula>
        <Formula>RR = 1 / rank of first relevant = {first === -1 ? '0 (none found)' : `1 / ${first + 1}`} &nbsp;(MRR is this averaged over many queries)</Formula>
        <Formula>
          DCG@{k} = Σ (2^gain − 1) / log₂(rank + 1) = {dcg(rel, k).toFixed(3)} &nbsp;·&nbsp; NDCG = DCG / ideal DCG
        </Formula>
      </div>
      <Explain>
        Precision and recall only count <em>how many</em> relevant results are in the top {k}; they ignore order. Reciprocal rank
        cares only about the <em>first</em> relevant result{first > 0 ? ` (here at rank ${first + 1}, so RR is ${rr.toFixed(2)})` : ''}.
        NDCG is the only one of the four that uses both position and graded relevance: move a “highly relevant” label to rank 1
        and watch NDCG rise while precision and recall stay the same. Results the retriever never returned still count against
        recall and NDCG, which is why an evaluation set must record every relevant document, not just the retrieved ones.
      </Explain>
    </Frame>
  )
}

/* ------------------------------------------------------------------ */
/* 16. Latency and cost comparison                                      */
/* ------------------------------------------------------------------ */

const MODELS = {
  small: { label: 'Small model', inPrice: 0.1, outPrice: 0.5, ttft: 250, tps: 160 },
  medium: { label: 'Mid-size model', inPrice: 2, outPrice: 10, ttft: 450, tps: 90 },
  large: { label: 'Large model', inPrice: 4, outPrice: 20, ttft: 700, tps: 60 },
} as const
type ModelKey = keyof typeof MODELS

export function LatencyCost() {
  const [model, setModel] = useState<ModelKey>('medium')
  const [rewrite, setRewrite] = useState(false)
  const [hybrid, setHybrid] = useState(true)
  const [rerank, setRerank] = useState(true)
  const [k, setK] = useState(5)
  const [chunk, setChunk] = useState(400)
  const [out, setOut] = useState(300)
  const [cache, setCache] = useState(0)
  const [qpd, setQpd] = useState(10000)
  const m = MODELS[model]

  const small = MODELS.small
  const stages = [
    { stage: 'Query rewrite', ms: rewrite ? small.ttft + (40 / small.tps) * 1000 : 0, usd: rewrite ? (150 * small.inPrice + 40 * small.outPrice) / 1e6 : 0 },
    { stage: 'Embed query', ms: 20, usd: 0.000001 },
    { stage: 'Vector search', ms: 15, usd: 0 },
    { stage: 'BM25 search', ms: hybrid ? 10 : 0, usd: 0 },
    { stage: 'Rerank', ms: rerank ? 30 * 8 : 0, usd: rerank ? 0.0005 : 0 },
    {
      stage: 'Generate',
      ms: m.ttft + (out / m.tps) * 1000,
      usd: ((300 + k * chunk + 60) * m.inPrice + out * m.outPrice) / 1e6,
    },
  ]
  // hybrid search runs its two retrievers in parallel, so only the slower one adds latency
  const latency = stages.reduce((n, s) => n + s.ms, 0) - (hybrid ? 10 : 0)
  const cost = stages.reduce((n, s) => n + s.usd, 0)
  const miss = 1 - cache / 100
  const avgLatency = miss * latency + (cache / 100) * 25
  const avgCost = miss * cost
  const data = stages.filter((s) => s.ms > 0).map((s) => ({ stage: s.stage, 'Latency (ms)': Math.round(s.ms), 'Cost (USD per 1,000 queries)': Number((s.usd * 1000).toFixed(3)) }))
  const gen = stages[5]

  return (
    <Frame title="where latency and cost come from">
      <p className="rounded-lg border border-dashed border-line px-3 py-2 text-[13px] text-muted">
        Every number here is an <strong>illustrative assumption</strong>, not a measurement or a price quote: per-stage timings,
        token prices per million tokens and generation speeds are round figures chosen to show proportions. Measure your own
        system before deciding anything.
      </p>
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Segmented
          label="Generation model"
          options={(Object.keys(MODELS) as ModelKey[]).map((v) => ({ value: v, label: MODELS[v].label }))}
          value={model}
          onChange={setModel}
        />
        <Toggle label="LLM query rewrite" checked={rewrite} onChange={setRewrite} />
        <Toggle label="Hybrid (add BM25)" checked={hybrid} onChange={setHybrid} />
        <Toggle label="Rerank 30 candidates" checked={rerank} onChange={setRerank} />
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Slider label="Chunks in the prompt (K)" value={k} min={1} max={30} onChange={setK} />
        <Slider label="Tokens per chunk" value={chunk} min={100} max={1500} step={50} onChange={setChunk} />
        <Slider label="Answer length (tokens)" value={out} min={50} max={1500} step={50} onChange={setOut} />
        <Slider label="Cache hit rate" value={cache} min={0} max={90} step={5} onChange={setCache} format={(v) => `${v}%`} />
        <Slider label="Queries per day" value={qpd} min={100} max={200000} step={100} onChange={setQpd} format={(v) => v.toLocaleString()} />
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 0, bottom: 0, left: -8 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="stage" tickLine={false} interval={0} tick={{ fontSize: 11 }} />
            <YAxis yAxisId="ms" tickLine={false} axisLine={false} />
            <YAxis yAxisId="usd" orientation="right" tickLine={false} axisLine={false} />
            <Tooltip cursor={{ fill: 'var(--sunken)' }} contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar yAxisId="ms" dataKey="Latency (ms)" fill="var(--accent)" radius={[3, 3, 0, 0]} />
            <Bar yAxisId="usd" dataKey="Cost (USD per 1,000 queries)" fill="var(--info)" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Stats>
        <Stat label="Latency, cache miss" value={`${(latency / 1000).toFixed(2)} s`} tone={latency > 5000 ? 'warn' : undefined} />
        <Stat label="Average latency" value={`${(avgLatency / 1000).toFixed(2)} s`} />
        <Stat label="Cost per 1,000 queries" value={`$${(avgCost * 1000).toFixed(2)}`} />
        <Stat label="Cost per month" value={`$${Math.round(avgCost * qpd * 30).toLocaleString()}`} />
      </Stats>
      <Formula>
        generation latency ≈ time to first token + output tokens / tokens per second = {m.ttft} ms + {out} / {m.tps} s = {(gen.ms / 1000).toFixed(2)} s
      </Formula>
      <Explain>
        Generation is {Math.round((gen.ms / latency) * 100)}% of the latency and {Math.round((gen.usd / cost) * 100)}% of the cost.
        Retrieval, even with hybrid search{rerank ? ' and reranking' : ''}, is a small slice of both. So the big levers are the
        ones that touch the model: a shorter answer cuts latency, fewer or smaller chunks cut input cost, a smaller model cuts
        both, and a cache removes the model call entirely for repeated questions
        {cache > 0 ? ` (at ${cache}% hits the average cost falls by ${cache}%)` : ''}. Streaming does not shorten total latency
        but lets the user start reading after about {m.ttft} ms.
      </Explain>
    </Frame>
  )
}
