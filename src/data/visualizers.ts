export interface VisualizerMeta {
  n: number
  slug: string
  title: string
  blurb: string
  group: 'Representation' | 'Chunking' | 'Sparse retrieval' | 'Ranking' | 'Pipeline' | 'Advanced'
  /** Lesson that teaches the idea, as moduleSlug/topicSlug. */
  lesson?: string
}

type Raw = [slug: string, title: string, group: VisualizerMeta['group'], blurb: string, lesson?: string]

const raw: Raw[] = [
  ['embeddings-map', 'How embeddings represent text', 'Representation', 'Place sentences on a 2-D semantic map and watch similar meanings land near each other.', 'embedding-fundamentals/what-are-embeddings'],
  ['cosine-similarity', 'Cosine similarity', 'Representation', 'Drag two vectors and see the angle, dot product, cosine and Euclidean distance change together.', 'embedding-mathematics/cosine-similarity'],
  ['vector-search', 'Vector search', 'Representation', 'Move a query point through a set of document vectors and see its nearest neighbours under each metric.', 'vector-database-fundamentals/similarity-search'],
  ['chunking', 'Chunk size and overlap', 'Chunking', 'Slide chunk size and overlap and see exactly where a document is cut and what is duplicated.', 'chunking-fundamentals/chunk-overlap'],
  ['semantic-chunking', 'Semantic chunking', 'Chunking', 'Split where the topic changes: adjust the similarity threshold and watch the boundaries move.', 'advanced-chunking-strategies/semantic-chunking'],
  ['tf-idf', 'TF-IDF', 'Sparse retrieval', 'Type a query and see each term’s frequency, rarity and contribution to every document’s score.', 'sparse-retrieval/tf-idf'],
  ['bm25', 'BM25 scoring', 'Sparse retrieval', 'Tune k1 and b and see term-frequency saturation and length normalisation reshape the ranking.', 'sparse-retrieval/bm25'],
  ['hybrid-retrieval', 'Hybrid retrieval', 'Ranking', 'Blend dense and sparse scores with a weight and see which documents each retriever rescues.', 'hybrid-search/weighted-score-fusion'],
  ['rrf', 'Reciprocal Rank Fusion', 'Ranking', 'Fuse two ranked lists by rank alone and see how the constant k changes the outcome.', 'hybrid-search/reciprocal-rank-fusion'],
  ['reranking', 'Reranking', 'Ranking', 'Compare a fast first-stage ranking with a slower, more accurate second look at the candidates.', 'reranking-fundamentals/why-reranking-is-needed'],
  ['top-k', 'Top-K retrieval', 'Ranking', 'Change K and watch recall rise, precision fall and the context fill with noise.', 'retrieval-fundamentals/top-k-retrieval'],
  ['mmr', 'Maximal Marginal Relevance', 'Ranking', 'Trade relevance against diversity with lambda and watch near-duplicates drop out.', 'advanced-retrieval-optimisation/mmr'],
  ['rag-pipeline', 'RAG pipeline execution', 'Pipeline', 'Step a question through embedding, retrieval, context construction and generation.', 'rag-architecture-fundamentals/traditional-rag-architecture'],
  ['context-window', 'Context window management', 'Pipeline', 'Budget tokens between instructions, retrieved chunks, history and the answer.', 'context-engineering/context-windows'],
  ['eval-metrics', 'RAG evaluation metrics', 'Pipeline', 'Mark results relevant or not and see Recall@K, Precision@K, MRR and NDCG recomputed.', 'retrieval-evaluation/ndcg'],
  ['latency-cost', 'Latency and cost comparison', 'Pipeline', 'Assemble a pipeline from stages and see where the milliseconds and the money go.', 'cost-engineering/cost-performance-trade-offs'],
  ['hnsw', 'HNSW indexing concepts', 'Advanced', 'Watch a greedy search descend the layers of a small-world graph towards the query.', 'vector-indexing-algorithms/hnsw'],
  ['graph-rag', 'Graph RAG traversal', 'Advanced', 'Pick a start entity and a hop limit and see which facts a graph traversal collects.', 'graph-rag/graph-traversal'],
  ['query-decomposition', 'Query decomposition', 'Advanced', 'See a compound question split into sub-questions, each retrieving its own evidence.', 'advanced-retrieval/query-decomposition'],
  ['agentic-rag', 'Agentic RAG workflows', 'Advanced', 'Step through an agent deciding whether to search, search again or answer.', 'agentic-rag/agentic-retrieval'],
]

export const visualizers: VisualizerMeta[] = raw.map(([slug, title, group, blurb, lesson], i) => ({
  n: i + 1,
  slug,
  title,
  group,
  blurb,
  lesson,
}))

export const VIZ_GROUPS: VisualizerMeta['group'][] = ['Representation', 'Chunking', 'Sparse retrieval', 'Ranking', 'Pipeline', 'Advanced']

export const findVisualizer = (slug: string | undefined) => visualizers.find((v) => v.slug === slug)
