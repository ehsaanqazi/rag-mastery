export interface Topic {
  title: string
  slug: string
}

export interface Module {
  n: number
  title: string
  slug: string
  blurb: string
  topics: Topic[]
  phase: number
}

export interface Phase {
  n: number
  title: string
  weeks: string
  blurb: string
  modules: Module[]
}

export function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/['’]/g, '')
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

type RawModule = [title: string, blurb: string, topics: string[]]
type RawPhase = [title: string, weeks: string, blurb: string, modules: RawModule[]]

const raw: RawPhase[] = [
  [
    'Foundations of Retrieval-Augmented Generation',
    'Week 1–3',
    'What RAG is, why it exists, how language models work, and the shape of every RAG system.',
    [
      [
        'Introduction to RAG',
        'The problem RAG solves, how it compares with fine-tuning and long context, and where it is used.',
        [
          'What is RAG?',
          'Why RAG was introduced',
          'Problems with traditional LLM applications',
          'Knowledge limitations of LLMs',
          'Hallucinations and factual reliability',
          'Parametric versus non-parametric knowledge',
          'How retrieval improves generation',
          'RAG versus fine-tuning',
          'RAG versus long-context prompting',
          'Real-world RAG applications',
        ],
      ],
      [
        'Understanding Large Language Models',
        'Tokens, attention, context windows and sampling: the generator half of every RAG system.',
        [
          'How LLMs work',
          'Tokens and tokenisation',
          'Embeddings',
          'Transformer architecture',
          'Attention mechanisms',
          'Context windows',
          'Temperature and sampling',
          'System prompts and user prompts',
          'Model inference',
          'LLM limitations',
        ],
      ],
      [
        'RAG Architecture Fundamentals',
        'The indexing, retrieval and generation pipelines, and how a question travels through them.',
        [
          'Traditional RAG architecture',
          'Indexing pipeline',
          'Retrieval pipeline',
          'Generation pipeline',
          'Offline and online processing',
          'Query-document matching',
          'Context construction',
          'Response generation',
          'End-to-end RAG lifecycle',
        ],
      ],
    ],
  ],
  [
    'Python and Data Engineering Foundations',
    'Week 3–5',
    'The Python and data-handling skills every ingestion pipeline and retrieval service is built from.',
    [
      [
        'Python for RAG Engineering',
        'Typed, validated, asynchronous Python: the working language of RAG systems.',
        [
          'Python fundamentals',
          'Functions and classes',
          'OOP',
          'Type hints',
          'Dataclasses and Pydantic',
          'Async programming',
          'Exception handling',
          'Virtual environments',
          'Dependency management',
          'Working with APIs',
        ],
      ],
      [
        'Data Processing',
        'Reading, cleaning and de-duplicating text from the formats real documents arrive in.',
        [
          'Text processing',
          'Regular expressions',
          'JSON and CSV',
          'XML and HTML',
          'PDF extraction',
          'DOCX extraction',
          'Markdown parsing',
          'Web scraping fundamentals',
          'Data cleaning',
          'Encoding and Unicode',
          'Duplicate detection',
        ],
      ],
      [
        'Document Engineering',
        'Treating documents as data: loaders, metadata, identifiers, versions and validation.',
        [
          'Document loaders',
          'Document metadata',
          'Document normalization',
          'Document classification',
          'Document versioning',
          'Document identifiers',
          'Data validation',
          'Structured document ingestion',
        ],
      ],
    ],
  ],
  [
    'Document Ingestion and Processing',
    'Week 6–8',
    'Getting documents of every kind out of their source systems and into clean, structured text.',
    [
      [
        'Understanding Document Ingestion',
        'Batch, streaming, incremental and event-driven ways to keep an index in step with its sources.',
        [
          'What is ingestion?',
          'Batch ingestion',
          'Streaming ingestion',
          'Incremental ingestion',
          'Event-driven ingestion',
          'Ingestion architecture',
        ],
      ],
      [
        'Advanced Document Parsing',
        'PDF libraries, OCR and layout-aware parsers, and how to choose between them.',
        [
          'PyMuPDF',
          'pypdf',
          'Unstructured',
          'Apache Tika',
          'HTML parsing',
          'Tables and structured content',
          'OCR',
          'Scanned documents',
          'Layout-aware parsing',
          'Multimodal document extraction',
        ],
      ],
      [
        'Document Preprocessing',
        'Cleaning, normalising and segmenting extracted text so that chunks carry signal, not noise.',
        [
          'Cleaning',
          'Normalisation',
          'Header and footer removal',
          'Noise removal',
          'Metadata extraction',
          'Language detection',
          'Document segmentation',
          'Data quality validation',
        ],
      ],
      [
        'Multimodal Ingestion',
        'Images, tables, charts, audio and video: turning non-text content into something retrievable.',
        [
          'Images',
          'Tables',
          'Charts',
          'Diagrams',
          'Audio transcripts',
          'Video transcripts',
          'OCR pipelines',
          'Multimodal document representation',
        ],
      ],
    ],
  ],
  [
    'Text Chunking and Semantic Segmentation',
    'Week 9–10',
    'How to split documents so that each piece is findable and makes sense on its own.',
    [
      [
        'Chunking Fundamentals',
        'Why chunks exist, and the size, overlap and boundary choices behind every splitter.',
        [
          'Why chunking matters',
          'Chunk size',
          'Chunk overlap',
          'Token-based chunking',
          'Character-based chunking',
          'Sentence-based chunking',
          'Paragraph-based chunking',
        ],
      ],
      [
        'Advanced Chunking Strategies',
        'Recursive, semantic and structure-aware splitting for prose, Markdown, code and tables.',
        [
          'Recursive character splitting',
          'Semantic chunking',
          'Document-aware chunking',
          'Structure-aware chunking',
          'Heading-based chunking',
          'Markdown chunking',
          'Code chunking',
          'Table chunking',
          'Sliding window chunking',
          'Agentic chunking',
        ],
      ],
      [
        'Chunking Optimisation',
        'Measuring chunk quality and tuning size, overlap and parent-child structure with experiments.',
        [
          'Chunk size experiments',
          'Overlap optimisation',
          'Boundary preservation',
          'Context preservation',
          'Parent-child chunking',
          'Small-to-big retrieval',
          'Late chunking',
          'Chunk quality evaluation',
        ],
      ],
    ],
  ],
  [
    'Embeddings and Vector Representations',
    'Week 11–13',
    'How text becomes vectors, which models produce them, and the mathematics of comparing them.',
    [
      [
        'Embedding Fundamentals',
        'Vector spaces and semantic similarity, from word vectors to sentence and document embeddings.',
        [
          'What are embeddings?',
          'Vector spaces',
          'Semantic similarity',
          'Word embeddings',
          'Sentence embeddings',
          'Document embeddings',
          'Dense representations',
          'Dimensionality',
          'Embedding models',
        ],
      ],
      [
        'Embedding Models',
        'The hosted and open-source model families, and what distinguishes them in practice.',
        [
          'OpenAI embeddings',
          'Sentence Transformers',
          'BGE models',
          'E5 models',
          'Nomic embeddings',
          'Jina embeddings',
          'Cohere embeddings',
          'Multilingual embeddings',
          'Domain-specific embeddings',
          'Local embedding models',
        ],
      ],
      [
        'Embedding Mathematics',
        'Dot products, cosine similarity, distance, normalisation and the geometry of embedding space.',
        [
          'Dot product',
          'Cosine similarity',
          'Euclidean distance',
          'Vector normalisation',
          'Vector dimensions',
          'Similarity distributions',
          'Embedding space geometry',
          'Dimensionality reduction',
        ],
      ],
      [
        'Embedding Optimisation',
        'Choosing, evaluating, fine-tuning and compressing embedding models for your own data.',
        [
          'Model selection',
          'Embedding evaluation',
          'Domain adaptation',
          'Fine-tuning embeddings',
          'Contrastive learning',
          'Hard negative mining',
          'Matryoshka embeddings',
          'Embedding quantisation',
        ],
      ],
    ],
  ],
  [
    'Vector Databases and Indexing',
    'Week 14–16',
    'Storing millions of vectors and searching them in milliseconds.',
    [
      [
        'Vector Database Fundamentals',
        'What a vector database stores, how it searches, and the operations it must support.',
        [
          'What is a vector database?',
          'Vector storage',
          'Similarity search',
          'Indexing',
          'Metadata',
          'Filtering',
          'CRUD operations',
          'Persistence',
        ],
      ],
      [
        'Vector Database Ecosystem',
        'Libraries, dedicated databases and vector extensions to databases you already run.',
        [
          'FAISS',
          'ChromaDB',
          'Qdrant',
          'Weaviate',
          'Milvus',
          'Pinecone',
          'PostgreSQL with pgvector',
          'Elasticsearch',
          'OpenSearch',
          'Redis',
        ],
      ],
      [
        'Vector Indexing Algorithms',
        'Flat, IVF, HNSW and quantisation: how approximate search trades recall for speed.',
        [
          'Flat indexes',
          'IVF',
          'HNSW',
          'Product quantisation',
          'Scalar quantisation',
          'Approximate nearest neighbour search',
          'Exact nearest neighbour search',
          'Recall versus latency',
          'Index construction',
        ],
      ],
      [
        'Advanced Vector Database Engineering',
        'Multi-tenancy, payload indexes, replication, sharding and keeping an index healthy over time.',
        [
          'Collections',
          'Namespaces',
          'Payload indexes',
          'Metadata filtering',
          'Hybrid storage',
          'Multi-tenancy',
          'Replication',
          'Sharding',
          'Backup and recovery',
          'Index updates',
        ],
      ],
    ],
  ],
  [
    'Retrieval Systems and Search Engineering',
    'Week 17–20',
    'Dense, sparse and hybrid search, and the query techniques that make retrieval reliable.',
    [
      [
        'Retrieval Fundamentals',
        'Top-K search, recall and precision, latency, and retrieval as candidate generation.',
        [
          'What is retrieval?',
          'Query-document similarity',
          'Top-K retrieval',
          'Recall@K',
          'Precision@K',
          'Retrieval latency',
          'Candidate generation',
        ],
      ],
      [
        'Dense Retrieval',
        'Semantic search with embeddings: how it works, and where it quietly fails.',
        [
          'Dense vector search',
          'Semantic search',
          'Query embeddings',
          'Document embeddings',
          'Similarity ranking',
          'Dense retrieval limitations',
        ],
      ],
      [
        'Sparse Retrieval',
        'Inverted indexes, TF-IDF and BM25: keyword search, still essential.',
        [
          'Inverted indexes',
          'TF-IDF',
          'BM25',
          'Keyword matching',
          'Term frequency',
          'Inverse document frequency',
          'Sparse representations',
        ],
      ],
      [
        'Hybrid Search',
        'Combining dense and sparse results with rank fusion and weighted scores.',
        [
          'Dense versus sparse retrieval',
          'Reciprocal Rank Fusion',
          'Weighted score fusion',
          'Query expansion',
          'Search fusion',
          'Hybrid retrieval architecture',
          'Hybrid search optimisation',
        ],
      ],
      [
        'Advanced Retrieval',
        'Query rewriting, decomposition, HyDE, self-querying and contextual retrieval.',
        [
          'Metadata-aware retrieval',
          'Filtered retrieval',
          'Multi-query retrieval',
          'Query decomposition',
          'Query rewriting',
          'Step-back prompting',
          'HyDE',
          'Self-query retrieval',
          'Contextual retrieval',
          'Multi-vector retrieval',
        ],
      ],
    ],
  ],
  [
    'Reranking and Retrieval Optimisation',
    'Week 21–22',
    'A second, more careful look at the candidates: cross-encoders, late interaction and diversity.',
    [
      [
        'Reranking Fundamentals',
        'Why a first-stage retriever is not enough, and how bi-encoders and cross-encoders differ.',
        [
          'Why reranking is needed',
          'Bi-encoders',
          'Cross-encoders',
          'Retrieval versus reranking',
          'Candidate selection',
          'Ranking scores',
        ],
      ],
      [
        'Reranking Models',
        'Open-source and hosted rerankers, and ColBERT-style late interaction.',
        ['BGE rerankers', 'Cohere rerank', 'Jina rerankers', 'Cross-encoder models', 'ColBERT', 'Late interaction'],
      ],
      [
        'Advanced Retrieval Optimisation',
        'Multi-stage pipelines, MMR, deduplication, score calibration and thresholds.',
        [
          'Two-stage retrieval',
          'Multi-stage retrieval',
          'Diversity-aware ranking',
          'MMR',
          'Context precision',
          'Context recall',
          'Duplicate removal',
          'Score calibration',
          'Retrieval thresholding',
        ],
      ],
    ],
  ],
  [
    'RAG Generation and Context Engineering',
    'Week 23–24',
    'Turning retrieved chunks into a grounded, cited, well-formed answer.',
    [
      [
        'Prompt Engineering for RAG',
        'Prompts that keep the model grounded, make it cite, and let it say "I don\'t know".',
        [
          'System prompts',
          'Context injection',
          'Grounded generation',
          'Citation instructions',
          'Answer formatting',
          'Refusal behaviour',
          'Handling missing information',
        ],
      ],
      [
        'Context Engineering',
        'Selecting, ordering, compressing and de-duplicating what goes into the context window.',
        [
          'Context windows',
          'Context selection',
          'Context ordering',
          'Context compression',
          'Context deduplication',
          'Lost-in-the-middle problem',
          'Context relevance',
          'Context prioritisation',
        ],
      ],
      [
        'Generation Strategies',
        'Citations, multi-document synthesis, structured output, streaming and multi-turn conversations.',
        [
          'Basic RAG',
          'Context-aware generation',
          'Citation-based generation',
          'Multi-document synthesis',
          'Structured outputs',
          'JSON generation',
          'Streaming responses',
          'Multi-turn RAG',
        ],
      ],
    ],
  ],
  [
    'Advanced RAG Architectures',
    'Week 25–28',
    'Corrective, adaptive, graph, hierarchical, agentic and multimodal RAG.',
    [
      [
        'Advanced RAG Patterns',
        'From naive to modular RAG, and the self-correcting patterns built on top.',
        [
          'Naive RAG',
          'Advanced RAG',
          'Modular RAG',
          'Corrective RAG',
          'Self-RAG',
          'Adaptive RAG',
          'Speculative RAG',
          'Fusion-based RAG',
        ],
      ],
      [
        'Graph RAG',
        'Knowledge graphs for questions that need relationships, not just similar passages.',
        [
          'Knowledge graphs',
          'Entity extraction',
          'Relationship extraction',
          'Graph construction',
          'Graph traversal',
          'Graph retrieval',
          'Microsoft GraphRAG concepts',
          'Local and global search',
          'Hybrid graph-vector retrieval',
        ],
      ],
      [
        'Hierarchical RAG',
        'Retrieving at several levels of detail: summaries, documents, sections and chunks.',
        [
          'Parent-child retrieval',
          'Document-level retrieval',
          'Hierarchical indexes',
          'Summary indexes',
          'Recursive retrieval',
          'Multi-resolution retrieval',
        ],
      ],
      [
        'Agentic RAG',
        'Letting a model decide when, where and how many times to search.',
        [
          'Agentic retrieval',
          'Tool-based retrieval',
          'Planning and reasoning',
          'Query routing',
          'Dynamic retrieval',
          'Iterative search',
          'Retrieval agents',
          'Agent evaluation',
          'Agent failure handling',
        ],
      ],
      [
        'Multimodal RAG',
        'Retrieval over images, audio, video, tables and diagrams.',
        [
          'Image retrieval',
          'Visual embeddings',
          'Image-text matching',
          'Audio RAG',
          'Video RAG',
          'Table retrieval',
          'Diagram retrieval',
          'Multimodal vector databases',
          'Cross-modal search',
        ],
      ],
    ],
  ],
  [
    'RAG Frameworks and Ecosystem',
    'Week 29–30',
    'LangChain, LlamaIndex, Haystack and DSPy, and when to write it yourself instead.',
    [
      [
        'LangChain for RAG',
        'Loaders, splitters, retrievers and LCEL chains.',
        [
          'LangChain architecture',
          'Document loaders',
          'Text splitters',
          'Embeddings',
          'Vector stores',
          'Retrievers',
          'Chains',
          'LCEL',
          'Runnable interfaces',
          'Custom retrievers',
        ],
      ],
      [
        'LlamaIndex',
        'Nodes, indexes, query engines, routers and response synthesis.',
        [
          'Index architecture',
          'Document ingestion',
          'Node parsing',
          'Query engines',
          'Vector indexes',
          'Router engines',
          'Sub-question engines',
          'Response synthesis',
        ],
      ],
      [
        'Haystack and Other Frameworks',
        'Haystack pipelines, DSPy optimisation, and an honest comparison of the options.',
        [
          'Haystack architecture',
          'Pipelines',
          'Components',
          'Document stores',
          'Retrieval pipelines',
          'DSPy for optimisation',
          'Framework comparisons',
          'When to use custom implementations',
        ],
      ],
    ],
  ],
  [
    'RAG Evaluation and Testing',
    'Week 31–33',
    'Measuring retrieval and generation quality so that changes are improvements, not guesses.',
    [
      [
        'Retrieval Evaluation',
        'Recall, precision, MRR, NDCG and hit rate, computed by hand and in code.',
        ['Recall@K', 'Precision@K', 'MRR', 'NDCG', 'Hit rate', 'Ranking quality', 'Retrieval benchmarks'],
      ],
      [
        'Generation Evaluation',
        'Faithfulness, relevance, correctness and citation accuracy.',
        [
          'Faithfulness',
          'Answer relevance',
          'Context relevance',
          'Answer correctness',
          'Hallucination detection',
          'Citation accuracy',
        ],
      ],
      [
        'RAG Evaluation Frameworks',
        'RAGAS, DeepEval, TruLens, Phoenix and LangSmith, and building your own harness.',
        ['RAGAS', 'DeepEval', 'TruLens', 'Arize Phoenix', 'LangSmith', 'Custom evaluation pipelines'],
      ],
      [
        'Testing and Benchmarking',
        'Golden datasets, synthetic data, regression tests, A/B tests and LLM judges.',
        [
          'Golden datasets',
          'Synthetic test data',
          'Test set construction',
          'Regression testing',
          'A/B testing',
          'Human evaluation',
          'LLM-as-a-judge',
          'Benchmark design',
        ],
      ],
    ],
  ],
  [
    'RAG Observability and Performance Engineering',
    'Week 34–35',
    'Seeing what a running system does, and making it faster and cheaper.',
    [
      [
        'RAG Observability',
        'Logs, traces and metrics for every stage, from query to answer.',
        [
          'Logging',
          'Tracing',
          'Distributed tracing',
          'Retrieval monitoring',
          'Generation monitoring',
          'Token tracking',
          'Cost tracking',
          'Failure analysis',
        ],
      ],
      [
        'Performance Optimisation',
        'Caching, batching, parallelism, streaming and GPU acceleration.',
        [
          'Latency optimisation',
          'Caching',
          'Semantic caching',
          'Query caching',
          'Batch processing',
          'Parallel retrieval',
          'Async execution',
          'Streaming',
          'GPU acceleration',
        ],
      ],
      [
        'Cost Engineering',
        'Where the money goes, and the trade-offs that reduce it.',
        [
          'Embedding costs',
          'LLM inference costs',
          'Token budgets',
          'Model selection',
          'Quantisation',
          'Caching strategies',
          'Cost-performance trade-offs',
        ],
      ],
    ],
  ],
  [
    'Production RAG and MLOps',
    'Week 36–38',
    'APIs, containers, CI/CD, index versioning and architectures that stay up.',
    [
      [
        'Backend Engineering',
        'A FastAPI service around your pipeline: validation, auth, async and streaming.',
        [
          'FastAPI',
          'REST APIs',
          'Request validation',
          'Authentication',
          'Async endpoints',
          'Streaming APIs',
          'Error handling',
          'API documentation',
        ],
      ],
      [
        'Deployment',
        'Docker, Kubernetes and the major clouds, with managed or self-hosted vector stores.',
        [
          'Docker',
          'Docker Compose',
          'Kubernetes fundamentals',
          'Cloud deployment',
          'AWS',
          'Azure',
          'Google Cloud',
          'Managed vector databases',
          'Infrastructure configuration',
        ],
      ],
      [
        'CI/CD and MLOps',
        'Versioning models, data and indexes; reindexing and rolling back safely.',
        [
          'Git workflows',
          'Automated testing',
          'CI/CD pipelines',
          'Model versioning',
          'Data versioning',
          'Index versioning',
          'Reindexing',
          'Rollbacks',
          'Blue-green deployments',
        ],
      ],
      [
        'Production Architecture',
        'Scaling, multi-tenancy, queues, fault tolerance and disaster recovery.',
        [
          'Scalable RAG systems',
          'Multi-tenant RAG',
          'High availability',
          'Fault tolerance',
          'Distributed ingestion',
          'Queue-based pipelines',
          'Horizontal scaling',
          'Load balancing',
          'Disaster recovery',
        ],
      ],
    ],
  ],
  [
    'Security, Governance and Enterprise RAG',
    'Week 39–40',
    'Prompt injection, poisoned documents, access control, privacy and compliance.',
    [
      [
        'RAG Security Fundamentals',
        'How RAG systems are attacked through their documents, and how to defend them.',
        [
          'Prompt injection',
          'Indirect prompt injection',
          'Data poisoning',
          'Malicious documents',
          'Retrieval manipulation',
          'Sensitive data exposure',
          'Output validation',
        ],
      ],
      [
        'Access Control and Privacy',
        'Making sure each user can only retrieve what they are allowed to read.',
        [
          'Authentication',
          'Authorisation',
          'Role-based access control',
          'Document-level permissions',
          'Tenant isolation',
          'Data encryption',
          'PII detection',
          'Data retention',
        ],
      ],
      [
        'Enterprise RAG Governance',
        'Lineage, audit, compliance and deletion that actually deletes.',
        [
          'Data lineage',
          'Audit logs',
          'Compliance',
          'GDPR concepts',
          'Data governance',
          'Source attribution',
          'Version control',
          'Secure deletion',
          'Responsible AI',
        ],
      ],
    ],
  ],
  [
    'Advanced Research and Capstone Engineering',
    'Week 41–44',
    'Current research directions, system design practice and eight capstone builds.',
    [
      [
        'Advanced RAG Research',
        'Reading the research: what is established, what is promising and what is unproven.',
        [
          'Modern RAG research',
          'Retrieval-augmented pretraining',
          'Long-context versus RAG',
          'Fine-tuned retrievers',
          'Late interaction',
          'Advanced query planning',
          'Retrieval reasoning',
          'Emerging research directions',
        ],
      ],
      [
        'Enterprise RAG System Design',
        'From requirements to architecture, capacity, cost and trade-offs.',
        [
          'Requirements gathering',
          'Architecture selection',
          'Technology selection',
          'Capacity planning',
          'Scaling strategies',
          'Reliability engineering',
          'Cost estimation',
          'Technical trade-offs',
        ],
      ],
      [
        'Capstone Projects',
        'Eight systems to design and build, from a personal assistant to an enterprise platform.',
        [
          'Personal document assistant',
          'PDF question-answering system',
          'Enterprise knowledge base',
          'Hybrid search engine',
          'Multimodal document assistant',
          'Graph RAG knowledge assistant',
          'Agentic research assistant',
          'Production-grade enterprise RAG platform',
        ],
      ],
    ],
  ],
]

let moduleCounter = 0

export const phases: Phase[] = raw.map(([title, weeks, blurb, mods], i) => ({
  n: i + 1,
  title,
  weeks,
  blurb,
  modules: mods.map(([mTitle, mBlurb, topics]) => ({
    n: ++moduleCounter,
    title: mTitle,
    slug: slugify(mTitle),
    blurb: mBlurb,
    phase: i + 1,
    topics: topics.map((t) => ({ title: t, slug: slugify(t) })),
  })),
}))

export const modules: Module[] = phases.flatMap((p) => p.modules)

export const totalTopics = modules.reduce((n, m) => n + m.topics.length, 0)

/** Whole-curriculum estimate shown on the home page and roadmap. */
export const TOTAL_WEEKS = 44

export function findModule(slug: string | undefined): Module | undefined {
  return modules.find((m) => m.slug === slug)
}

export function phaseOf(m: Module): Phase {
  return phases[m.phase - 1]
}

export const pad = (n: number) => String(n).padStart(2, '0')
