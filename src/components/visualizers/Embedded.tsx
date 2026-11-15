import type { ComponentType } from 'react'
import { Link } from 'react-router'
import { Maximize2 } from 'lucide-react'
import { findVisualizer } from '../../data/visualizers'
import { CosineSimilarity, EmbeddingsMap, TopK, VectorSearch } from './Representation'
import { Bm25, Chunking, SemanticChunking, TfIdf } from './TextViz'
import { HybridRetrieval, Mmr, Reranking, Rrf } from './Ranking'
import { ContextWindow, EvalMetrics, LatencyCost, RagPipeline } from './Pipeline'
import { AgenticRag, GraphRag, Hnsw, QueryDecomposition } from './Advanced'

const REGISTRY: Record<string, ComponentType> = {
  'embeddings-map': EmbeddingsMap,
  'cosine-similarity': CosineSimilarity,
  'vector-search': VectorSearch,
  chunking: Chunking,
  'semantic-chunking': SemanticChunking,
  'tf-idf': TfIdf,
  bm25: Bm25,
  'hybrid-retrieval': HybridRetrieval,
  rrf: Rrf,
  reranking: Reranking,
  'top-k': TopK,
  mmr: Mmr,
  'rag-pipeline': RagPipeline,
  'context-window': ContextWindow,
  'eval-metrics': EvalMetrics,
  'latency-cost': LatencyCost,
  hnsw: Hnsw,
  'graph-rag': GraphRag,
  'query-decomposition': QueryDecomposition,
  'agentic-rag': AgenticRag,
}

/** The bare visualiser, for the playground page. */
export function Visualizer({ slug }: { slug: string }) {
  const Component = REGISTRY[slug]
  return Component ? <Component /> : <p className="text-sm text-danger">Unknown visualiser “{slug}”.</p>
}

/** A visualiser embedded in a lesson, with a link to its playground page. */
export default function EmbeddedVisualizer({ slug }: { slug: string }) {
  const meta = findVisualizer(slug)
  return (
    <div>
      <Visualizer slug={slug} />
      {meta && (
        <p className="!mt-2 text-right text-[13px]">
          <Link to={`/playground/${slug}`} className="inline-flex items-center gap-1.5 !no-underline">
            <Maximize2 size={12} /> Open “{meta.title}” in the playground
          </Link>
        </p>
      )}
    </div>
  )
}
