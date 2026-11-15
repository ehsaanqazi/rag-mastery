/**
 * Small, exact implementations of the retrieval maths used by the visualisers.
 * Everything runs on a tiny built-in corpus so results can be checked by hand.
 */

export interface Doc {
  id: string
  title: string
  text: string
}

/** The Meridian handbook mini-corpus used across the curriculum. */
export const CORPUS: Doc[] = [
  { id: 'D1', title: 'Annual leave', text: 'Annual leave is 25 days per year. Up to 5 unused days carry over and must be used by 31 March.' },
  { id: 'D2', title: 'Remote work', text: 'Staff may work remotely up to 3 days a week. The home office equipment budget is £400.' },
  { id: 'D3', title: 'Expenses', text: 'Expenses must be submitted within 30 days. Any expense over £75 needs a receipt.' },
  { id: 'D4', title: 'Passwords', text: 'Passwords must be rotated every 90 days. Use the password manager for all work accounts.' },
  { id: 'D5', title: 'VPN', text: 'The VPN is required on public Wi-Fi. Connect to the VPN before opening any internal system.' },
  { id: 'D6', title: 'Laptop refresh', text: 'Laptops are refreshed on a 3 year cycle. Request a replacement laptop through the IT portal.' },
  { id: 'D7', title: 'Support desk', text: 'The support desk is open 08:00 to 18:00 UK time. Raise urgent incidents by phone.' },
  { id: 'D8', title: 'Leave carry-over', text: 'Unused annual leave: at most 5 days carry over into the next year, to be taken before 31 March.' },
  { id: 'D9', title: 'Travel expenses', text: 'Travel expenses for client visits are reimbursed when submitted with receipts within 30 days.' },
]

const STOP = new Set(
  'a an and any are as at be before by for from how i in into is it its may me my of on or our per the this to up use what when which who with within you your do does can many much must need needs needed all there their they we us am was will should would could if than then so not no'.split(
    ' ',
  ),
)

/** Lower-case word tokens. Numbers and £ amounts are kept; punctuation is dropped. */
export function tokenize(text: string, keepStop = false): string[] {
  const tokens = text.toLowerCase().match(/[a-z0-9£]+(?:[:.-][a-z0-9]+)*/g) ?? []
  return keepStop ? tokens : tokens.filter((t) => !STOP.has(t))
}

/** Very light stemming so "expenses" matches "expense" and "laptops" matches "laptop". */
export function stem(t: string): string {
  if (t.length > 4 && t.endsWith('ies')) return t.slice(0, -3) + 'y'
  if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) return t.slice(0, -1)
  return t
}

export const terms = (text: string) => tokenize(text).map(stem)

export const round = (x: number, d = 3) => {
  const f = 10 ** d
  return Math.round(x * f) / f
}

/* ---------- vectors ---------- */

export const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
export const norm = (a: number[]) => Math.sqrt(dot(a, a))
export const cosine = (a: number[], b: number[]) => {
  const d = norm(a) * norm(b)
  return d === 0 ? 0 : dot(a, b) / d
}
export const euclid = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0))
export const normalize = (a: number[]) => {
  const n = norm(a)
  return n === 0 ? a : a.map((x) => x / n)
}

/* ---------- sparse retrieval ---------- */

export interface Index {
  docs: Doc[]
  tokens: string[][]
  /** Number of documents containing each term. */
  df: Map<string, number>
  avgLen: number
}

export function buildIndex(docs: Doc[] = CORPUS): Index {
  const tokens = docs.map((d) => terms(d.text))
  const df = new Map<string, number>()
  for (const toks of tokens) for (const t of new Set(toks)) df.set(t, (df.get(t) ?? 0) + 1)
  return { docs, tokens, df, avgLen: tokens.reduce((n, t) => n + t.length, 0) / docs.length }
}

export const INDEX = buildIndex()

export const tf = (term: string, toks: string[]) => toks.filter((t) => t === term).length

/** Smoothed IDF as used by scikit-learn: ln((1 + N) / (1 + df)) + 1. */
export const idfSmooth = (term: string, ix: Index = INDEX) =>
  Math.log((1 + ix.docs.length) / (1 + (ix.df.get(term) ?? 0))) + 1

/** BM25 IDF (Lucene form, never negative): ln(1 + (N - df + 0.5) / (df + 0.5)). */
export const idfBm25 = (term: string, ix: Index = INDEX) => {
  const df = ix.df.get(term) ?? 0
  return Math.log(1 + (ix.docs.length - df + 0.5) / (df + 0.5))
}

export interface TermScore {
  term: string
  tf: number
  idf: number
  score: number
}

export interface Scored {
  doc: Doc
  score: number
  parts: TermScore[]
}

/** Raw TF × smoothed IDF, summed over the query terms. */
export function tfidfScores(query: string, ix: Index = INDEX): Scored[] {
  const q = [...new Set(terms(query))]
  return ix.docs.map((doc, i) => {
    const parts = q.map((term) => {
      const f = tf(term, ix.tokens[i])
      const idf = idfSmooth(term, ix)
      return { term, tf: f, idf, score: f * idf }
    })
    return { doc, parts, score: parts.reduce((s, p) => s + p.score, 0) }
  })
}

/** Okapi BM25 with the usual k1 and b parameters. */
export function bm25Scores(query: string, k1 = 1.5, b = 0.75, ix: Index = INDEX): Scored[] {
  const q = [...new Set(terms(query))]
  return ix.docs.map((doc, i) => {
    const len = ix.tokens[i].length
    const parts = q.map((term) => {
      const f = tf(term, ix.tokens[i])
      const idf = idfBm25(term, ix)
      const score = f === 0 ? 0 : (idf * f * (k1 + 1)) / (f + k1 * (1 - b + (b * len) / ix.avgLen))
      return { term, tf: f, idf, score }
    })
    return { doc, parts, score: parts.reduce((s, p) => s + p.score, 0) }
  })
}

/* ---------- a toy "embedding model" ---------- */

/**
 * Six hand-built concept dimensions. A real embedding model learns hundreds of
 * dimensions from data; this toy maps words to concepts with a small lexicon so
 * that synonyms ("holiday", "vacation", "leave") share a direction. It exists to
 * make semantic matching visible, not to be accurate.
 */
export const CONCEPTS = ['leave', 'remote work', 'expenses', 'security', 'hardware', 'support'] as const

const LEXICON: Record<string, number[]> = {}
const add = (words: string, concept: number, weight = 1) => {
  for (const w of words.split(' ')) {
    const v = (LEXICON[w] ??= [0, 0, 0, 0, 0, 0])
    v[concept] += weight
  }
}
add('leave holiday holidays vacation pto annual unused carry absence break', 0)
add('day days year march', 0, 0.35)
add('remote remotely home wfh hybrid telework commute', 1)
add('office week work', 1, 0.4)
add('expense expenses receipt receipts reimburse reimbursed reimbursement claim claims cost costs spend spent travel money', 2)
add('submit submitted budget £75 £400 client', 2, 0.45)
add('password passwords vpn wifi wi-fi secure security login credentials rotate rotated account accounts encrypt', 3)
add('public internal network manager', 3, 0.5)
add('laptop laptops computer device devices hardware macbook replacement refresh refreshed equipment', 4)
add('portal cycle new old', 4, 0.4)
add('support help helpdesk desk incident incidents phone urgent contact hours open call', 5)
add('raise time uk', 5, 0.35)

/** Concept vector for a text: the sum of its words' concept weights, L2-normalised. */
export function embed(text: string): number[] {
  const v = [0, 0, 0, 0, 0, 0]
  for (const t of tokenize(text)) {
    const w = LEXICON[t] ?? LEXICON[stem(t)]
    if (w) for (let i = 0; i < 6; i++) v[i] += w[i]
  }
  return normalize(v)
}

export const DOC_VECTORS = CORPUS.map((d) => embed(d.text))

export function denseScores(query: string): { doc: Doc; score: number }[] {
  const q = embed(query)
  return CORPUS.map((doc, i) => ({ doc, score: cosine(q, DOC_VECTORS[i]) }))
}

/** Projects a concept vector to 2-D by giving each concept a direction 60° apart. */
export function project2d(v: number[]): [number, number] {
  let x = 0
  let y = 0
  v.forEach((w, i) => {
    const a = (i * Math.PI) / 3 - Math.PI / 2
    x += w * Math.cos(a)
    y += w * Math.sin(a)
  })
  return [x, y]
}

/* ---------- ranking helpers ---------- */

export const ranked = <T extends { score: number }>(xs: T[]) => [...xs].sort((a, b) => b.score - a.score)

/** Min-max scaling to [0, 1]; all-equal inputs map to 0. */
export function minmax(xs: number[]): number[] {
  const lo = Math.min(...xs)
  const hi = Math.max(...xs)
  return hi === lo ? xs.map(() => 0) : xs.map((x) => (x - lo) / (hi - lo))
}

/** Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d)), ranks starting at 1. */
export function rrf(lists: string[][], k = 60): { id: string; score: number; ranks: (number | null)[] }[] {
  const ids = [...new Set(lists.flat())]
  return ids
    .map((id) => {
      const ranks = lists.map((l) => {
        const i = l.indexOf(id)
        return i === -1 ? null : i + 1
      })
      return { id, ranks, score: ranks.reduce<number>((s, r) => s + (r === null ? 0 : 1 / (k + r)), 0) }
    })
    .sort((a, b) => b.score - a.score)
}

export interface MmrStep {
  picked: number
  relevance: number
  redundancy: number
  mmr: number
}

/**
 * Maximal Marginal Relevance. At each step picks the candidate maximising
 * λ·relevance − (1 − λ)·max similarity to anything already picked.
 */
export function mmr(relevance: number[], sim: number[][], lambda: number, k: number): MmrStep[] {
  const picked: MmrStep[] = []
  const left = new Set(relevance.map((_, i) => i))
  while (picked.length < k && left.size) {
    let best: MmrStep | null = null
    for (const i of left) {
      const redundancy = picked.length ? Math.max(...picked.map((p) => sim[i][p.picked])) : 0
      const score = lambda * relevance[i] - (1 - lambda) * redundancy
      if (!best || score > best.mmr) best = { picked: i, relevance: relevance[i], redundancy, mmr: score }
    }
    picked.push(best!)
    left.delete(best!.picked)
  }
  return picked
}

/* ---------- evaluation metrics ---------- */

export const precisionAtK = (rel: number[], k: number) => rel.slice(0, k).filter((r) => r > 0).length / k

export const recallAtK = (rel: number[], k: number, totalRelevant: number) =>
  totalRelevant ? rel.slice(0, k).filter((r) => r > 0).length / totalRelevant : 0

export const reciprocalRank = (rel: number[]) => {
  const i = rel.findIndex((r) => r > 0)
  return i === -1 ? 0 : 1 / (i + 1)
}

export const dcg = (rel: number[], k: number) =>
  rel.slice(0, k).reduce((s, r, i) => s + (2 ** r - 1) / Math.log2(i + 2), 0)

/** NDCG@K against the ideal ordering of the given gains (plus any relevant items never retrieved). */
export const ndcg = (rel: number[], k: number, missing: number[] = []) => {
  const ideal = dcg([...rel, ...missing].sort((a, b) => b - a), k)
  return ideal === 0 ? 0 : dcg(rel, k) / ideal
}

/** Deterministic pseudo-random numbers (mulberry32) so layouts never change between renders. */
export function rng(seed: number) {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
