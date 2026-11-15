export type Level = 'Beginner' | 'Intermediate' | 'Advanced'

export interface Project {
  n: number
  slug: string
  title: string
  level: Level
  blurb: string
  stack: string[]
  hours: string
  /** Has a complete, runnable end-to-end implementation rather than a guided build. */
  complete?: boolean
}

type Raw = [slug: string, title: string, level: Level, blurb: string, stack: string[], hours: string, complete?: boolean]

const raw: Raw[] = [
  [
    'pdf-question-answering',
    'Basic PDF question-answering system',
    'Beginner',
    'Load a PDF, chunk it, embed it locally and answer questions with page citations.',
    ['pypdf', 'Sentence Transformers', 'NumPy', 'Claude API'],
    '4–6 hours',
    true,
  ],
  [
    'personal-notes-chatbot',
    'Personal notes chatbot',
    'Beginner',
    'Chat with a folder of Markdown notes, with a persistent index that updates when notes change.',
    ['ChromaDB', 'Sentence Transformers', 'Claude API'],
    '4–6 hours',
  ],
  [
    'document-search-engine',
    'Document search engine',
    'Beginner',
    'A keyword search engine built from scratch: tokeniser, inverted index and BM25 ranking.',
    ['Python', 'rank-bm25', 'FastAPI'],
    '5–7 hours',
  ],
  [
    'csv-question-answering',
    'CSV data question-answering assistant',
    'Beginner',
    'Answer questions about tabular data by combining row retrieval with Pandas computation.',
    ['Pandas', 'Sentence Transformers', 'Claude API'],
    '4–6 hours',
  ],
  [
    'knowledge-base-chatbot',
    'Simple knowledge base chatbot',
    'Beginner',
    'A support-style chatbot over FAQ articles that refuses when the answer is not in the knowledge base.',
    ['FAISS', 'Sentence Transformers', 'Claude API'],
    '5–7 hours',
  ],
  [
    'hybrid-search-rag',
    'Hybrid search RAG application',
    'Intermediate',
    'Dense and BM25 retrieval fused with Reciprocal Rank Fusion, measured against each retriever alone.',
    ['Qdrant', 'rank-bm25', 'Sentence Transformers', 'FastAPI'],
    '8–12 hours',
    true,
  ],
  [
    'multi-document-research-assistant',
    'Multi-document research assistant',
    'Intermediate',
    'Decompose a research question, retrieve across many documents and synthesise a cited report.',
    ['LlamaIndex', 'ChromaDB', 'Claude API'],
    '8–12 hours',
  ],
  [
    'reranking-powered-rag',
    'Reranking-powered RAG',
    'Intermediate',
    'Add a cross-encoder second stage and measure what it does to precision and latency.',
    ['Sentence Transformers', 'FAISS', 'BGE reranker'],
    '6–10 hours',
  ],
  [
    'website-knowledge-base-assistant',
    'Website knowledge base assistant',
    'Intermediate',
    'Crawl a documentation site, keep the index fresh incrementally and answer with links to the source pages.',
    ['httpx', 'BeautifulSoup', 'PostgreSQL + pgvector', 'LangChain'],
    '10–14 hours',
  ],
  [
    'rag-with-citations',
    'RAG application with citations',
    'Intermediate',
    'Sentence-level citations that are verified against the retrieved sources before the answer is shown.',
    ['Pydantic', 'Qdrant', 'Claude API', 'FastAPI'],
    '8–12 hours',
  ],
  [
    'graph-rag-application',
    'Graph RAG application',
    'Advanced',
    'Extract entities and relationships into a graph and answer multi-hop questions vectors alone miss.',
    ['NetworkX', 'Sentence Transformers', 'Claude API'],
    '14–20 hours',
  ],
  [
    'multimodal-rag-system',
    'Multimodal RAG system',
    'Advanced',
    'Retrieve over text, tables and page images from PDFs using a shared image-text embedding space.',
    ['PyMuPDF', 'CLIP', 'Qdrant', 'Claude API'],
    '14–20 hours',
  ],
  [
    'agentic-rag-research-assistant',
    'Agentic RAG research assistant',
    'Advanced',
    'An agent that plans, searches iteratively, judges its own context and stops when it has enough.',
    ['LangGraph', 'Qdrant', 'Claude API'],
    '16–24 hours',
  ],
  [
    'multi-tenant-enterprise-rag',
    'Multi-tenant enterprise RAG',
    'Advanced',
    'Tenant isolation, document-level permissions and audit logging enforced at retrieval time.',
    ['FastAPI', 'Qdrant', 'PostgreSQL', 'JWT'],
    '16–24 hours',
  ],
  [
    'production-rag-platform',
    'Production-ready RAG platform',
    'Advanced',
    'The whole system: queued ingestion, hybrid retrieval, reranking, streaming API, evaluation, tracing and Docker Compose.',
    ['FastAPI', 'Qdrant', 'Redis', 'RAGAS', 'Docker'],
    '30–40 hours',
    true,
  ],
]

export const projects: Project[] = raw.map(([slug, title, level, blurb, stack, hours, complete], i) => ({
  n: i + 1,
  slug,
  title,
  level,
  blurb,
  stack,
  hours,
  complete,
}))

export const LEVELS: Level[] = ['Beginner', 'Intermediate', 'Advanced']

export const findProject = (slug: string | undefined) => projects.find((p) => p.slug === slug)
