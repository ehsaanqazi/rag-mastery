Similarity search is the operation a vector database exists to perform: given a query vector, return the $k$ stored vectors most similar to it. Everything else in the database, from the storage layout to the index, is there to make that one operation fast and correct.

This lesson treats similarity search as a database operation. You already know how to compute a cosine or a dot product from [Embedding mathematics](/learn/embedding-mathematics); here the questions are different. Which similarity measure should a collection be configured with, and what goes wrong if it does not match the model? What exactly does a search return, and what do its scores mean in different databases? How is the top $k$ selected efficiently from millions of scores? And what extra knobs (thresholds, pagination, batching) does a production search API expose? How the database avoids scanning everything is the next lesson, [Indexing](/learn/vector-database-fundamentals/indexing).

## Learning objectives

- Define k-nearest-neighbour search precisely, for both similarity and distance measures.
- Choose the collection metric (cosine, dot product or Euclidean) that matches an embedding model, and explain when the three give the same ranking.
- Interpret the scores returned by Qdrant, FAISS and pgvector, including which direction is "better".
- Implement exact top-k search with efficient top-k selection, and state its cost.
- Use score thresholds, pagination and batched queries correctly, knowing what each does and does not guarantee.

## Prerequisites

- [What is a vector database?](/learn/vector-database-fundamentals/what-is-a-vector-database): collections, points and the query path.
- [Vector storage](/learn/vector-database-fundamentals/vector-storage): vectors as a contiguous matrix.
- [Cosine similarity](/learn/embedding-mathematics/cosine-similarity), [Dot product](/learn/embedding-mathematics/dot-product) and [Euclidean distance](/learn/embedding-mathematics/euclidean-distance): the measures themselves.
- [Vector normalisation](/learn/embedding-mathematics/vector-normalisation): scaling vectors to length 1.

## Why this matters

Similarity search is the step in a RAG pipeline that decides which text the model will read. If it returns the wrong chunks, nothing downstream can recover, and it fails in ways that raise no errors.

The most common silent failures are configuration mistakes rather than algorithmic ones. A collection created with the dot-product metric holding vectors that were never normalised ranks chunks partly by their length, so long, rambling chunks crowd out short, precise ones. A team that switches from Qdrant to FAISS forgets that FAISS's L2 index returns *squared* distances where lower is better, and a "minimum relevance" check written for cosine scores silently admits everything. A threshold of 0.7 copied from a blog post filters out every result, because the model in use produces question-to-answer cosines around 0.4. Each of these ships, passes a smoke test, and degrades answers for weeks.

Understanding precisely what a search computes, and what its numbers mean, is how you catch these before users do.

## The intuition

Imagine every chunk as a pin on a map, placed so that chunks about similar things are close together. A question becomes one more pin. Similarity search asks: which $k$ existing pins are nearest to the new one?

"Nearest" needs a ruler, and there are three common rulers:

- **Angle from the centre.** Draw a line from the map's centre to each pin and compare directions. Two pins in the same direction are similar, however far out they sit. This is **cosine similarity**.
- **Straight-line distance.** Measure pin to pin with a tape. This is **Euclidean distance**.
- **Direction and reach combined.** How far a pin extends in the query's direction. A pin far out in roughly the right direction can beat a nearer one in exactly the right direction. This is the **dot product**.

If every pin sits on a circle of the same radius (every vector normalised to length 1), the three rulers agree completely about which pins are nearest. They disagree only when pins sit at different distances from the centre, and then which ruler you chose decides the answer.

## How it works

### The definition

Let the collection hold vectors $x_1, \dots, x_N$ and let $q$ be the query vector. Fix a scoring function. For a **similarity** $s(q, x)$, larger means more alike; for a **distance** $\delta(q, x)$, smaller means more alike.

The **k-nearest-neighbour (kNN)** result is the set of $k$ stored vectors with the highest similarity (or the lowest distance) to $q$, returned in order, best first, with their scores. If the collection holds fewer than $k$ vectors that qualify, fewer are returned.

Three properties are worth stating plainly because each is a frequent source of confusion:

1. **kNN always returns something.** If the collection has at least $k$ points, you get $k$ results, relevant or not. "Nothing relevant" must be detected separately, by a threshold or a downstream check.
2. **Only the order inside the result is guaranteed, and only for exact search.** Approximate search ([Indexing](/learn/vector-database-fundamentals/indexing)) may miss a true neighbour; the scores of what it does return are still computed exactly.
3. **Scores are on the model's scale.** A cosine of 0.44 can be the best possible match for one model and a poor one for another.

### Choosing the metric

You choose the scoring function when you create a collection, and it should be whatever the embedding model was trained with. Model documentation (the model card) states this. The common cases:

| Model was trained with | Configure | Normalise vectors? |
| --- | --- | --- |
| Cosine similarity (most sentence-embedding models, including `all-MiniLM-L6-v2`) | Cosine, or dot product on normalised vectors | Yes, if you use dot product |
| Dot product, with meaningful vector lengths | Dot product | No; the length carries information |
| Euclidean distance | Euclidean | As the model card says |

When vectors are normalised to length 1, cosine, dot product and Euclidean distance produce **identical rankings**; the mathematics section proves it. That is why the common advice "normalise and use dot product" is safe for cosine-trained models: it gives cosine results with the cheapest arithmetic. Several databases do this internally: Qdrant's cosine metric normalises vectors when they are stored and then uses a dot product.

When vectors are **not** normalised, the metrics disagree. A dot product rewards length: a vector twice as long scores twice as high in the same direction. If length reflects nothing meaningful (it often reflects chunk length or some quirk of the model), dot-product search over unnormalised vectors is biased towards whichever chunks happen to have long vectors. Searching for the largest dot product is called **maximum inner product search (MIPS)**; it is the right operation only when the model intends length to matter.

### What the scores mean, by product

Each product reports scores in its own convention. Mixing them up is a classic porting bug.

| System and setting | Returned value | Better is | Range for unit vectors |
| --- | --- | --- | --- |
| Qdrant, `Distance.COSINE` | cosine similarity | higher | $-1$ to $1$ |
| Qdrant, `Distance.DOT` | dot product | higher | $-1$ to $1$ |
| Qdrant, `Distance.EUCLID` | Euclidean distance | lower | $0$ to $2$ |
| FAISS `IndexFlatIP` | inner product | higher | $-1$ to $1$ |
| FAISS `IndexFlatL2` | **squared** Euclidean distance | lower | $0$ to $4$ |
| pgvector `<=>` | cosine **distance** ($1 - \cos$) | lower | $0$ to $2$ |
| pgvector `<#>` | **negative** inner product | lower | $-1$ to $1$ |
| pgvector `<->` | Euclidean distance | lower | $0$ to $2$ |

pgvector negates the inner product so that `ORDER BY embedding <#> query` (ascending, as SQL sorts by default) puts the best match first. Always check the convention before writing a threshold or combining scores.

### Exact search, step by step

Exact (or **brute-force**, or **flat**) search computes the score for every stored vector. With vectors stored as an $N \times d$ matrix $X$ and normalised:

1. **Prepare the query.** Embed it with the same model; normalise it if the collection is normalised.
2. **Score everything.** Compute $s = Xq$: one dot product per row, a single matrix-vector product.
3. **Mask** any points that are deleted or excluded by a filter (setting their score to $-\infty$).
4. **Select the top $k$** scores without sorting all $N$ of them.
5. **Order** those $k$ by score.
6. **Fetch payloads** for the $k$ identifiers and return them.

```mermaid
flowchart LR
    A["Query vector q"] --> B["Scores s = X q<br/>(N dot products)"]
    B --> C["Mask deleted /<br/>filtered points"]
    C --> D["Select top k<br/>(partial selection)"]
    D --> E["Sort the k"]
    E --> F["Fetch payloads"]
    F --> G["k results<br/>id, score, payload"]
```

Step 2 dominates: $N \times d$ multiply-adds. Step 4 is where a naive implementation wastes time.

### Selecting the top k

Sorting all $N$ scores costs on the order of $N \log N$ comparisons, but you only need the best $k$, with $k$ usually between 5 and 100. Two standard approaches do less work:

- **A bounded heap.** Keep a min-heap of the best $k$ scores seen so far. For each new score, compare it with the smallest in the heap (the heap's root); if larger, replace the root and restore the heap. Cost: on the order of $N \log k$, and for most scores only one comparison, since most are smaller than the root. This is how streaming and per-segment search usually work.
- **Partial selection (quickselect).** Rearrange the array so that the $k$ largest are in the first $k$ positions, in no particular order, in time proportional to $N$ on average. NumPy exposes this as `np.argpartition`. Then sort just those $k$.

Watch a bounded heap with $k = 2$ process six scores:

```trace
{"title": "Top-2 selection with a bounded min-heap", "code": ["heap = []  # holds (score, id), smallest at heap[0]", "for i, s in enumerate([0.19, 0.25, 0.15, 0.44, 0.06, 0.24]):", "    if len(heap) < 2: heappush(heap, (s, i))", "    elif s > heap[0][0]: heapreplace(heap, (s, i))", "result = sorted(heap, reverse=True)"], "steps": [{"line": 1, "vars": {"heap": "[]"}, "note": "Start with an empty heap of capacity k = 2."}, {"line": 3, "vars": {"i": "0", "s": "0.19", "heap": "[(0.19, 0)]"}, "note": "Heap not full: push."}, {"line": 3, "vars": {"i": "1", "s": "0.25", "heap": "[(0.19, 0), (0.25, 1)]"}, "note": "Heap not full: push. The smallest kept score, 0.19, sits at the root."}, {"line": 4, "vars": {"i": "2", "s": "0.15", "heap": "[(0.19, 0), (0.25, 1)]"}, "note": "0.15 is not above the root 0.19, so it is discarded after one comparison."}, {"line": 4, "vars": {"i": "3", "s": "0.44", "heap": "[(0.25, 1), (0.44, 3)]"}, "note": "0.44 beats the root: replace 0.19 and restore the heap. 0.25 is the new root."}, {"line": 4, "vars": {"i": "4", "s": "0.06", "heap": "[(0.25, 1), (0.44, 3)]"}, "note": "Discarded after one comparison."}, {"line": 4, "vars": {"i": "5", "s": "0.24", "heap": "[(0.25, 1), (0.44, 3)]"}, "note": "0.24 is just below the root 0.25: discarded."}, {"line": 5, "vars": {"result": "[(0.44, 3), (0.25, 1)]"}, "note": "Sort the k survivors, best first. Only the two best of six were ever kept."}]}
```

Search over segments uses the same idea one level up: each segment returns its own top $k$, and the database merges those short lists into a global top $k$.

### Thresholds, pagination and batches

Production search APIs expose a few options around plain kNN.

**Score thresholds.** A minimum similarity (or maximum distance) drops results that do not reach it, so the search may return fewer than $k$. Qdrant's `query_points` takes a `score_threshold`; in SQL you add a `WHERE` condition on the distance. A threshold is the simplest way to let a RAG system say "nothing relevant was found". It is also fragile: the right value depends on the model and the corpus, so it must be set from measurements on labelled queries, not copied. [Retrieval thresholding](/learn/advanced-retrieval-optimisation/retrieval-thresholding) treats it properly.

**Range search.** A pure threshold with no $k$: "every vector within distance $r$". FAISS calls this `range_search`. It is useful for de-duplication ("find all near-copies of this chunk") more than for answering questions, because the number of results is unbounded.

**Pagination.** `limit` and `offset` request results 11 to 20, for example. For exact search this is well defined. For approximate search the database must still find the first $\text{offset} + \text{limit}$ candidates internally, so deep pages cost as much as a large $k$ and can be less accurate. RAG rarely needs page 2; if you need many candidates, ask for a larger $k$ once.

**Batched queries.** Several query vectors in one request become one matrix-matrix product ($Q X^\top$), which uses the hardware far better than separate matrix-vector products. Evaluation runs, multi-query retrieval and offline jobs should batch. Qdrant offers `query_batch_points`; FAISS's `search` takes a matrix of queries natively.

> [!KEY]
> Configure the metric the model was trained with, normalise if you use dot product for a cosine model, and learn each product's score convention before writing any threshold. Most similarity-search bugs are configuration bugs.

## The mathematics

### Why normalised vectors make the three metrics agree

**Intuition.** Squared Euclidean distance can be expanded into lengths and a dot product. If every length is 1, the only part that changes from one stored vector to the next is the dot product, so ranking by distance and ranking by dot product are the same ranking reversed.

$$
\lVert q - x \rVert^2 = \lVert q \rVert^2 + \lVert x \rVert^2 - 2\, q \cdot x
$$

and when $\lVert q \rVert = \lVert x \rVert = 1$:

$$
\lVert q - x \rVert^2 = 2 - 2\, q \cdot x = 2 - 2 \cos\theta
$$

$q$ is the query vector, $x$ a stored vector, $\lVert \cdot \rVert$ the Euclidean length, $q \cdot x$ the dot product and $\theta$ the angle between them. For unit vectors the dot product equals the cosine, and the squared distance is a decreasing function of it. The largest cosine is always the smallest distance.

**By hand.** Two unit vectors in the plane: $q = (0.6, 0.8)$ and $x = (0.8, 0.6)$.

- Dot product (and cosine, since both have length 1): $0.6 \times 0.8 + 0.8 \times 0.6 = 0.48 + 0.48 = 0.96$.
- Squared distance directly: $(0.6 - 0.8)^2 + (0.8 - 0.6)^2 = 0.04 + 0.04 = 0.08$.
- From the identity: $2 - 2 \times 0.96 = 0.08$. They agree.

Now lengthen $x$ to $x' = 2x = (1.6, 1.2)$. The cosine is unchanged at 0.96, but the dot product doubles to $0.6 \times 1.6 + 0.8 \times 1.2 = 1.92$, and the squared distance becomes $(0.6 - 1.6)^2 + (0.8 - 1.2)^2 = 1.0 + 0.16 = 1.16$. The three measures now disagree: dot product prefers $x'$, Euclidean prefers $x$, cosine cannot tell them apart.

```python
import numpy as np

q = np.array([0.6, 0.8]); x = np.array([0.8, 0.6]); x2 = 2 * x
for name, v in [("x", x), ("2x", x2)]:
    cos = q @ v / (np.linalg.norm(q) * np.linalg.norm(v))
    print(f"{name:<3} cosine={cos:.2f}  dot={q @ v:.2f}  squared L2={np.sum((q - v) ** 2):.2f}")
print("2 - 2 * cos for unit x:", round(2 - 2 * 0.96, 2))
```

```output
x   cosine=0.96  dot=0.96  squared L2=0.08
2x  cosine=0.96  dot=1.92  squared L2=1.16
2 - 2 * cos for unit x: 0.08
```

### The cost of exact search and of selection

Scoring costs $N d$ multiply-adds. Selection costs about $N \log_2 N$ comparisons with a full sort, $N \log_2 k$ in the worst case with a heap, and about $N$ on average with partial selection. For $N = 10^6$ and $k = 10$: a sort is about $10^6 \times 20 = 2 \times 10^7$ comparisons, a heap at most about $10^6 \times 3.3 \approx 3.3 \times 10^6$, and partial selection about $10^6$. Compared with scoring at $d = 384$, which is $3.84 \times 10^8$ multiply-adds, selection is small whichever method you use, but a full sort can still add a noticeable fraction; with GPU scoring, where the multiply-adds are cheap, selection can become the bottleneck.

## In code

**The problem.** Run one question against ten Meridian chunks three ways: exact search in NumPy with partial selection; FAISS flat indexes with inner-product and Euclidean metrics, confirming that they agree and that the L2 index returns squared distances; and a dot-product search over unnormalised vectors, to watch length distort the ranking.

**Dependencies.** NumPy, FAISS (CPU build) and sentence-transformers with the local `all-MiniLM-L6-v2` model.

```bash
pip install numpy faiss-cpu sentence-transformers
```

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = [
    "Annual leave is 25 days per year.",
    "Up to 5 unused leave days carry over and must be used by 31 March.",
    "Staff may work remotely up to 3 days a week.",
    "The home-office equipment budget is £400; contractors are not eligible.",
    "Expenses must be submitted within 30 days; anything over £75 needs a receipt.",
    "Passwords must be rotated every 90 days.",
    "VPN is required on public Wi-Fi.",
    "Laptops are refreshed every 3 years.",
    "The support desk is open 08:00 to 18:00 UK time.",
    "Report a lost laptop to the support desk at once.",
]
query = "How long can I wait before claiming back a taxi fare?"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model.encode(chunks, normalize_embeddings=True)          # (10, 384) float32, unit length
q = model.encode([query], normalize_embeddings=True)         # (1, 384)
k = 3

# 1. Exact search by hand: one matrix-vector product, then top-k selection.
scores = (X @ q.T).ravel()
part = np.argpartition(-scores, k)[:k]                       # the k best, in no particular order
top = part[np.argsort(-scores[part])]                        # sort only those k
print("NumPy, cosine (higher is better)")
for i in top:
    print(f"  {scores[i]:.4f}  [{i}] {chunks[i][:55]}")

# 2. FAISS flat indexes: inner product and (squared) Euclidean distance.
ip = faiss.IndexFlatIP(X.shape[1]); ip.add(X)
l2 = faiss.IndexFlatL2(X.shape[1]); l2.add(X)
D_ip, I_ip = ip.search(q, k)
D_l2, I_l2 = l2.search(q, k)
print("\nFAISS IndexFlatIP ids:", I_ip[0], "scores:", np.round(D_ip[0], 4))
print("FAISS IndexFlatL2 ids:", I_l2[0], "distances:", np.round(D_l2[0], 4))
print("2 - 2 x cosine:        ", np.round(2 - 2 * D_ip[0], 4))

# 3. The same vectors, NOT normalised, scored by raw dot product.
rng = np.random.default_rng(3)
lengths = rng.uniform(0.5, 3.0, size=len(chunks)).astype(np.float32)
X_raw = X * lengths[:, None]                                 # same directions, random lengths
dot = (X_raw @ q.T).ravel()
print("\nUnnormalised vectors, raw dot product (higher is better)")
for i in np.argsort(-dot)[:k]:
    print(f"  dot={dot[i]:.4f}  length={lengths[i]:.2f}  cosine={scores[i]:.4f}  [{i}] {chunks[i][:40]}")
```

```output
NumPy, cosine (higher is better)
  0.4372  [4] Expenses must be submitted within 30 days; anything ove
  0.2474  [1] Up to 5 unused leave days carry over and must be used b
  0.1946  [0] Annual leave is 25 days per year.

FAISS IndexFlatIP ids: [4 1 0] scores: [0.4372 0.2474 0.1946]
FAISS IndexFlatL2 ids: [4 1 0] distances: [1.1256 1.5052 1.6108]
2 - 2 x cosine:         [1.1256 1.5052 1.6108]

Unnormalised vectors, raw dot product (higher is better)
  dot=0.3698  length=2.50  cosine=0.1477  [2] Staff may work remotely up to 3 days a w
  dot=0.3215  length=0.74  cosine=0.4372  [4] Expenses must be submitted within 30 day
  dot=0.2702  length=1.09  cosine=0.2474  [1] Up to 5 unused leave days carry over and
```

**The important lines.**

- `np.argpartition(-scores, k)[:k]` is partial selection: the three highest scores move to the front in arbitrary order, in linear time. The following `np.argsort` sorts only those three. Negating the scores turns "largest" into "smallest", which is what `argpartition` selects.
- `faiss.IndexFlatIP` and `faiss.IndexFlatL2` are FAISS's exact indexes: they store vectors as a flat matrix and scan all of them. "Flat" in FAISS means exactly the brute-force search described above.
- `ip.search(q, k)` takes a **matrix** of queries (here one row) and returns two arrays, distances `D` and identifiers `I`, each of shape (number of queries, $k$). Passing a 1-D vector instead of a 2-D array is a common error.
- `X * lengths[:, None]` keeps every direction and changes only the lengths, isolating the effect of length on dot-product search.

**What the output means.**

- The question never mentions expenses, receipts or 30 days, yet the expenses chunk ranks first with a clear margin (0.437 against 0.247). Notice that the second and third results are about leave, not expenses: kNN returns three results because $k = 3$, not because three are relevant.
- FAISS's inner-product index returns exactly the NumPy scores. Its L2 index returns the same three identifiers in the same order, and the "distances" equal $2 - 2\cos$ to four decimal places. They are **squared** Euclidean distances, as the identity predicts; the true distances would be their square roots.
- With unnormalised vectors and a raw dot product, the remote-work chunk (cosine 0.148, irrelevant) jumps to first place purely because its vector is 2.5 units long, while the correct chunk, at 0.74 units, drops to second. Nothing raises an error; the ranking is simply wrong.

**Errors people commonly hit.**

- `AssertionError` from FAISS on `add` or `search`: the array is float64 or not C-contiguous. FAISS requires float32; `np.ascontiguousarray(X, dtype=np.float32)` fixes both.
- Reading `D_l2` as a similarity. Smaller is better, and it is squared.
- Normalising stored vectors but not the query, or the reverse. With dot product, an unnormalised query scales every score by the same factor, which preserves the ranking but breaks any threshold.

**Extending it.** Add `score_threshold`-style behaviour: drop results with cosine below a value, and choose the value by embedding 20 questions you know the answers to and 20 you know are unanswerable from the corpus, then picking the threshold that best separates them. You will find the separation is imperfect, which is the honest starting point for [score calibration](/learn/advanced-retrieval-optimisation/score-calibration).

## A real-world example

When Meridian's Platform team moved a copy of AskMeridian's index into FAISS for an offline evaluation job, Lena ported the retrieval code from Qdrant. The production code included a relevance floor: if the best result scored below 0.25, AskMeridian replied that it could not find the answer instead of guessing.

She built the FAISS index with `IndexFlatL2`, because the tutorial she followed used it, and kept the `score < 0.25` check. The evaluation came back looking excellent: almost no "could not find the answer" replies, even for the deliberately unanswerable questions in the test set. That was the clue. The "score" was now a squared Euclidean distance, where lower is better and even good matches sit around 1.1 to 1.3 for this model. A best "score" below 0.25 now meant an almost perfect match, which never happens, so the check that was meant to reject weak matches never fired at all, and weak matches went straight to the generator.

The fix was two lines: switch to `IndexFlatIP` over normalised vectors, so the scores were cosines again, and add an assertion in the evaluation harness that the top score for a known-good question exceeded the top score for a known-bad one. The team added the score-convention table from this lesson to the repository's README, next to every place a threshold appears.

## Common mistakes

- **Configuring a metric the model was not trained for.** Using Euclidean distance on unnormalised vectors from a cosine-trained model, or dot product without normalising, gives a ranking that partly reflects vector length.
- **Assuming all databases report scores the same way.** Similarity versus distance, squared versus not, negated inner products: the same "0.25" means different things in Qdrant, FAISS and pgvector.
- **Treating $k$ results as $k$ relevant results.** kNN always fills the list; low-ranked results are often irrelevant.
- **Copying a threshold.** Score scales differ between models and corpora; a threshold must be set from your own labelled queries.
- **Fully sorting millions of scores** in a hand-written search loop. Use partial selection or a heap.
- **Sending queries one at a time in batch jobs.** Batching turns many matrix-vector products into one matrix-matrix product, which is much faster per query.

## Best practices

- Read the embedding model's card for its similarity measure and normalisation, record it next to the collection, and test it with a known query.
- For cosine-trained models, normalise every vector at write time and every query at search time.
- Write down the score convention (similarity or distance, squared or not, direction) for every system in use, and keep thresholds in configuration alongside it.
- Request a larger $k$ than the prompt needs when a reranker or filter follows, and let the later stage cut it down.
- Batch queries in evaluation and offline workloads.
- Keep an exact (flat) search path available for testing: it is the ground truth that approximate search is measured against.

## Performance considerations

**Exact search is memory-bound and linear.** Scoring reads the whole matrix once per query. One million 384-dimension float32 vectors is 1.5 GB per query; at an assumed 20 GB/s per core that is about 77 ms, and a 16-core machine running queries in parallel shares the same memory bus, so total throughput is bounded by total bandwidth, not core count.

**Batching amortises the read.** Scoring 32 queries in one matrix multiplication reads the stored matrix roughly once (in cache-sized blocks) instead of 32 times. This is why batch throughput can be far higher than single-query throughput on the same hardware.

**Selection is cheap but not free.** As computed above, partial selection costs about $N$ comparisons, small next to $Nd$ multiply-adds. On a GPU, where scoring is very fast, top-k selection is a known bottleneck and libraries implement specialised kernels for it.

**$k$ affects cost only mildly in exact search, more in approximate search.** For a flat scan, a larger $k$ barely changes the cost. For an index, a larger $k$ requires exploring more of it, so latency grows with $k$.

## Security considerations

Similarity search ranks everything it can see. It does not know who is asking, so access control must be applied as a filter inside the search, not by trimming results afterwards; [Filtering](/learn/vector-database-fundamentals/filtering) explains why trimming afterwards also breaks result counts. Scores can also leak information: an API that returns raw similarity scores for arbitrary queries lets a caller probe whether text similar to their query exists in the collection, even if the text itself is withheld. Return scores to end users only when there is a reason to.

## Summary

- Similarity search returns the $k$ stored vectors with the highest similarity (or lowest distance) to the query, best first, and always returns $k$ results if the collection has them.
- Configure the metric the embedding model was trained with. On normalised vectors, cosine, dot product and Euclidean distance give identical rankings because $\lVert q - x \rVert^2 = 2 - 2\,q \cdot x$.
- On unnormalised vectors the metrics disagree, and dot product rewards long vectors; use it unnormalised only when the model intends length to carry meaning.
- Products report scores differently (similarity versus distance, squared, negated); learn the convention before writing thresholds.
- Exact search costs $Nd$ multiply-adds plus near-linear top-k selection; batching queries and partial selection keep it efficient, and indexes avoid it at scale.

## Exercises

**1.** A unit-length query has cosine similarities 0.82, 0.55 and 0.10 with three unit-length stored vectors. What squared Euclidean distances would FAISS `IndexFlatL2` report, and what values would pgvector's `<=>` operator return?

<details><summary>Show solution</summary>

Squared distances are $2 - 2\cos$: $2 - 1.64 = 0.36$, $2 - 1.10 = 0.90$ and $2 - 0.20 = 1.80$. pgvector's cosine distance is $1 - \cos$: $0.18$, $0.45$ and $0.90$. All three lists rank the vectors in the same order; only the numbers and their direction differ.

</details>

**2.** Rewrite step 1 of the code example to process a batch of three queries at once, returning a (3, k) array of identifiers in best-first order, without a Python loop over queries.

<details><summary>Show solution</summary>

```python
Qs = model.encode([query, "Can I work from home?", "Who do I tell about a stolen laptop?"],
                  normalize_embeddings=True)                  # (3, 384)
S = Qs @ X.T                                                  # (3, 10): every query against every chunk
part = np.argpartition(-S, k, axis=1)[:, :k]                  # top k per row, unordered
order = np.argsort(-np.take_along_axis(S, part, axis=1), axis=1)
top = np.take_along_axis(part, order, axis=1)                 # (3, k), best first
print(top)
```

```output
[[4 1 0]
 [2 3 6]
 [9 7 4]]
```

`np.take_along_axis` gathers each row's own selected scores so they can be sorted per row. One matrix product replaces three matrix-vector products.

</details>

**3.** A colleague proposes storing unnormalised vectors from a cosine-trained model in a dot-product collection "because the model's vectors are already roughly unit length". How would you test whether this is safe, and what would you check in the output?

<details><summary>Show solution</summary>

Measure the lengths: compute `np.linalg.norm(X, axis=1)` over a sample of real chunk embeddings and look at the minimum, maximum and spread. If all lengths are within a tiny tolerance of 1 (as with models that normalise internally), dot product and cosine rank almost identically. Then compare rankings directly: for a set of test queries, compute the top 10 by dot product on the raw vectors and by cosine, and measure the overlap. Anything less than near-complete overlap means length is influencing results, and the vectors should be normalised. Normalising costs nothing at query time, so the safe answer is simply to normalise.

</details>

**4.** Design decision. AskMeridian must reply "I could not find that in the handbook" when nothing relevant exists. Propose how to implement this on top of kNN, how to choose any numbers involved, and one failure mode of your design.

<details><summary>Show solution</summary>

Retrieve the top $k$ as usual, then apply a **minimum-score threshold** on the best result (and optionally drop individual results below it). Choose the threshold from data: collect labelled answerable and unanswerable questions, record the top-1 score for each, and pick the value that best separates the two distributions given the relative cost of each error (a wrong answer is usually worse than a false "I don't know"). Re-derive it whenever the embedding model changes.

Failure mode: the two score distributions overlap, so some answerable questions phrased unusually fall below the threshold, and some unanswerable questions that are close in topic to a real chunk ("what is the leave allowance for contractors?") pass it. A reranker score or an instruction to the generator to refuse when the context does not contain the answer is a useful second line of defence.

</details>

## Check your understanding

```quiz
[
  {"q": "All vectors in a collection and the query are normalised to length 1. Which statement is true?", "options": ["Cosine, dot product and Euclidean distance give the same ranking", "Euclidean distance gives a different ranking from cosine", "Dot product ranks longer chunks higher", "Only cosine similarity can be used"], "answer": 0, "why": "For unit vectors the squared distance equals 2 - 2 times the dot product, which equals 2 - 2 cos, so all three order the vectors identically."},
  {"q": "FAISS IndexFlatL2 returns 0.36 for a pair of unit vectors. What is their cosine similarity?", "options": ["0.36", "0.64", "0.82", "0.18"], "answer": 2, "why": "IndexFlatL2 returns squared distance, so 0.36 = 2 - 2 cos and cos = 0.82."},
  {"q": "```python\nimport numpy as np\ns = np.array([0.2, 0.9, 0.4, 0.7])\nprint(sorted(np.argpartition(-s, 2)[:2].tolist()))\n```\nWhat does this print?", "options": ["[0, 2]", "[2, 3]", "[1, 2]", "[1, 3]"], "answer": 3, "why": "The two largest scores are 0.9 (index 1) and 0.7 (index 3); argpartition puts their indices first in arbitrary order, and sorted() makes the output deterministic."},
  {"q": "Why can dot-product search over unnormalised vectors from a cosine-trained model return worse results?", "options": ["Dot products cannot be computed on unnormalised vectors", "Longer vectors score higher in the same direction, so vector length, which carries no intended meaning, influences the ranking", "Dot product ignores direction", "The database rejects unnormalised vectors"], "answer": 1, "why": "The dot product is length times length times cosine, so a long, less relevant vector can outscore a short, more relevant one."},
  {"q": "A RAG system must never place irrelevant chunks in the prompt. Which approach directly addresses the fact that kNN always returns k results?", "options": ["Increase k", "Switch from cosine to Euclidean distance", "Apply a score threshold chosen from labelled answerable and unanswerable queries", "Use batched queries"], "answer": 2, "why": "kNN fills the list regardless of relevance; a calibrated threshold lets the system return fewer results or none."}
]
```

## Further reading

- FAISS wiki, "Getting started" and "MetricType and distances": flat indexes, inner product versus L2, and why L2 results are squared. [github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki)
- Qdrant documentation, "Search" and "Collections": distance metrics, `query_points`, score thresholds and batch search. [qdrant.tech/documentation](https://qdrant.tech/documentation/)
- pgvector README: distance operators `<->`, `<#>` and `<=>` and how to order by them. [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- Sentence-Transformers documentation, "Semantic Search": which models expect cosine versus dot product. [sbert.net](https://www.sbert.net/)
- Manning, Raghavan and Schütze (2008), *Introduction to Information Retrieval*, chapter 7, "Computing scores in a complete search system": efficient top-k selection with heaps.
