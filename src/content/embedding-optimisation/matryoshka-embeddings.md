An embedding's dimensionality is usually fixed when you choose the model: 384, 768 or 1,536 numbers per chunk, every one of them stored in memory and touched by every search. **Matryoshka embeddings**, named after Russian nesting dolls, are trained so that the first few numbers of each vector already form a good, smaller embedding, the first few more a better one, and so on up to the full length. You can then cut vectors to whatever size your memory and latency budget allows, choose that size after training, and even use a short prefix for a fast first search and the full vector to re-rank. This lesson explains why ordinary embeddings cannot be shortened safely, how Matryoshka Representation Learning fixes that, how to train and use such models, and how to design a search that exploits them.

## Learning objectives

- Explain why truncating an ordinary embedding degrades retrieval, and what Matryoshka training changes.
- Write the Matryoshka loss as a weighted sum of losses over nested prefixes.
- Truncate and renormalise embeddings correctly, and show why skipping renormalisation changes scores.
- Fine-tune a model with `MatryoshkaLoss` and compare retrieval quality across dimensions with a standard fine-tune.
- Design a two-stage funnel search with short vectors for candidates and full vectors for re-ranking, and estimate its savings.

## Prerequisites

- [Vector dimensions](/learn/embedding-mathematics/vector-dimensions) and [Dimensionality](/learn/embedding-fundamentals/dimensionality): what dimensions are and what they cost.
- [Vector normalisation](/learn/embedding-mathematics/vector-normalisation): unit vectors and cosine similarity.
- [Contrastive learning](/learn/embedding-optimisation/contrastive-learning) and [Fine-tuning embeddings](/learn/embedding-optimisation/fine-tuning-embeddings): the loss and the training loop that Matryoshka training wraps.

## Why this matters

Vector memory is usually the largest running cost of a dense index, and it scales linearly with dimensionality. As [Model selection](/learn/embedding-optimisation/model-selection) computed, 200,000 chunks at 3,072 dimensions need about 2.5 GB in float32; at 256 dimensions, about 0.2 GB. Search time per comparison scales the same way.

Without Matryoshka training, your options for shrinking vectors are poor. You can pick a smaller model, which may be worse at everything. You can apply [dimensionality reduction](/learn/embedding-mathematics/dimensionality-reduction) such as PCA after the fact, which needs fitting, adds a transformation step to every query and is lossy in ways that depend on your data. Or you can simply cut vectors short, which, for a model not trained for it, throws away information spread across all dimensions. Matryoshka training makes the third option, the cheapest one, actually work. Several current hosted and open models support it, so understanding it lets you trade quality for cost with a configuration change instead of a model migration.

## The intuition

Imagine describing a colleague so that someone can find them in a crowded office. You could give a description of exactly fifty words. But if you are told you might be interrupted at any point, you would order it: most useful first ("tall, red jacket, by the window"), then refinements ("glasses, holding a blue mug"), then fine detail. Whoever hears only the first five words can already narrow the crowd; whoever hears all fifty can pick the one person out.

An ordinary embedding is the fifty-word description with the words in an arbitrary order: stop after ten and you have a random tenth of the information. A Matryoshka embedding is the ordered description. Training is what imposes the order: the model is graded not only on its full description, but also on how well its first five, ten and twenty words work alone.

## How it works

### Why plain truncation fails

A standard embedding model is trained with a loss computed on the full vector. Nothing in that loss cares *where* in the vector information lives, so the model spreads it across all dimensions. Keeping the first 64 of 384 dimensions keeps a fixed sixth of the coordinates, which is roughly like projecting the vectors onto an arbitrary 64-dimensional subspace. Some structure survives (nearby texts still tend to be nearby), but fine distinctions, such as which of two error codes a question is about, are lost first. The code below shows a base model's held-out MRR falling from 0.84 at 384 dimensions to 0.57 at 32.

### Matryoshka Representation Learning

**Matryoshka Representation Learning (MRL)** was introduced by Kusupati and colleagues (2022). The idea is simple: choose a set of nested sizes $m_1 < m_2 < \dots < m_K = d$, for example 32, 64, 128, 256, 768. In each training step, compute the usual loss not once but $K$ times, once on each prefix of the vectors, and add the results (optionally weighted). The gradient from the 32-dimensional loss pushes the most important information into the first 32 coordinates; the gradient from the 64-dimensional loss pushes the next most useful information into coordinates 33 to 64; and so on.

The architecture does not change and inference cost does not change: the model still outputs $d$ numbers. Only the training objective differs, so the same model serves every size. Because the sizes are nested, a 64-dimensional embedding is literally the first 64 numbers of the 768-dimensional one; you do not run the model twice.

In Sentence Transformers, `MatryoshkaLoss` wraps any loss, typically `MultipleNegativesRankingLoss`, with a list of `matryoshka_dims` and optional `matryoshka_weights`. There is also a `Matryoshka2dLoss`, which additionally trains the model so that its early layers produce usable embeddings, trading depth as well as width; treat that as a more specialised tool.

### Which models support it

A model supports truncation only if it was trained for it, and its model card or API documentation will say so. Examples include Nomic's `nomic-embed-text-v1.5`, which is documented as a Matryoshka model, and OpenAI's `text-embedding-3` models, whose API accepts a `dimensions` parameter that returns shortened embeddings. Many open models are not trained this way; truncating them works poorly, as the base model in the code below shows. When in doubt, measure: evaluate retrieval at each candidate size with your evaluation set before relying on it. See the [Embedding Models](/learn/embedding-models) module for the families.

### Using truncated vectors correctly

To use a $m$-dimensional prefix:

1. Take the first $m$ numbers of each vector.
2. **Renormalise** each truncated vector to unit length.
3. Compare with cosine similarity (equivalently, dot products of the renormalised vectors).

Step 2 is the one people miss. A unit-length vector cut short is no longer unit length, and different vectors lose different amounts of length. Dot products of unnormalised prefixes are not cosines and can reorder results. In Sentence Transformers, `encode(..., truncate_dim=m, normalize_embeddings=True)` truncates and then renormalises; `SentenceTransformer(name, truncate_dim=m)` sets a default truncation for the model.

Queries and documents must use the same $m$, and that $m$ becomes part of the [embedding contract](/learn/embedding-fundamentals/what-are-embeddings) recorded with the index.

### Funnel search

Because every size is a prefix of the full vector, you can store full vectors once and search in stages, a pattern sometimes called **funnel** or **adaptive** retrieval:

1. Keep a short prefix (say 64 dimensions) of every vector in the in-memory index.
2. For a query, search the short vectors for the top $K$ candidates (say 200), with a high $K$ to protect recall.
3. Fetch the full vectors of those $K$ candidates (from disk or a secondary store) and re-score them exactly.
4. Return the top $k$ (say 10) by full-vector score.

The first stage is cheap because the vectors are short; the second is cheap because $K$ is small. Whether it preserves quality depends on the recall of the first stage at $K$, which you measure exactly as in [Embedding evaluation](/learn/embedding-optimisation/embedding-evaluation): the fraction of the full-vector top 10 that appears in the short-vector top $K$. Funnel search composes naturally with a cross-encoder [reranker](/learn/reranking-fundamentals/why-reranking-is-needed) as a third stage, and with [quantisation](/learn/embedding-optimisation/embedding-quantisation) of the short vectors.

```mermaid
flowchart LR
    Q["Query"] --> E["Embed once:<br/>full d-dim vector"]
    E --> S["Take first m dims,<br/>renormalise"]
    S --> I["Search short-vector index<br/>(in RAM)"]
    I --> K["Top K candidates"]
    K --> F["Fetch full vectors,<br/>re-score with all d dims"]
    E --> F
    F --> R["Top k results"]
```

### Choosing a size

There is no universally right size. The procedure:

1. Evaluate retrieval at each supported size on your frozen evaluation set, per slice.
2. Plot quality against size; it usually rises steeply at first and then flattens.
3. Pick the smallest size whose quality is within your tolerance of the full size (judged with the paired interval, not the raw difference), or design a funnel whose final stage restores full quality.
4. Re-check thresholds: score distributions differ by size.

## The mathematics

### The Matryoshka loss

**Intuition.** Grade the model on several nested prefixes at once, so that each prefix must be useful on its own.

$$
\mathcal{L}_{\text{MRL}} = \sum_{k=1}^{K} w_k \, \mathcal{L}\!\left(\frac{q_{1:m_k}}{\lVert q_{1:m_k} \rVert}, \frac{p_{1:m_k}}{\lVert p_{1:m_k} \rVert}\right)
$$

$q$ and $p$ are the full $d$-dimensional embeddings of a query and its positive passage (in practice whole batches); $q_{1:m}$ is the vector of the first $m$ coordinates; dividing by its length renormalises it; $\mathcal{L}$ is the underlying loss, for example InfoNCE with in-batch negatives; $m_1 < \dots < m_K$ are the nested sizes and $w_k$ their weights (often all 1). The total is minimised only when every prefix retrieves well.

### Renormalisation

**Intuition.** Cutting a vector short shortens it by an amount that differs from vector to vector, so the dot product of two prefixes mixes direction with leftover length.

$$
\cos_m(q, d) = \frac{q_{1:m} \cdot d_{1:m}}{\lVert q_{1:m} \rVert \, \lVert d_{1:m} \rVert}
$$

**By hand.** $q = (0.5, 0.5, 0.5, 0.5)$ and $d = (0.7, 0.1, -0.1, 0.7)$, both unit length.

- Full cosine: $0.35 + 0.05 - 0.05 + 0.35 = 0.70$.
- First two coordinates: $q_{1:2} = (0.5, 0.5)$, length $\sqrt{0.5} \approx 0.707$; $d_{1:2} = (0.7, 0.1)$, length $\sqrt{0.5} \approx 0.707$.
- Raw dot product of the prefixes: $0.35 + 0.05 = 0.40$.
- Cosine of the prefixes: $0.40 / (0.707 \times 0.707) = 0.40 / 0.5 = 0.80$.

The raw prefix dot product (0.40) is not the cosine (0.80); with many vectors whose prefix lengths differ, using it would rank by length as well as direction.

### Funnel cost

With $N$ vectors, full dimension $d$, prefix $m$ and $K$ candidates re-scored, the multiply-adds per query are $N d$ for a full exact search and $N m + K d$ for a funnel. For $N = 1{,}000{,}000$, $d = 768$, $m = 64$, $K = 200$: $768{,}000{,}000$ against $64{,}000{,}000 + 153{,}600 \approx 64{,}153{,}600$, a twelve-fold reduction; in-memory vectors shrink from 3.07 GB to 0.26 GB in float32. Approximate indexes such as [HNSW](/learn/vector-indexing-algorithms/hnsw) change the first term but not the proportionality to dimension.

```python
import numpy as np

q = np.array([0.5, 0.5, 0.5, 0.5])
d = np.array([0.7, 0.1, -0.1, 0.7])
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"full cosine:            {cos(q, d):.3f}")
print(f"first 2 dims, raw dot:  {q[:2] @ d[:2]:.3f}")
print(f"first 2 dims, cosine:   {cos(q[:2], d[:2]):.3f}")

N, d_full, m, K = 1_000_000, 768, 64, 200
full = N * d_full
funnel = N * m + K * d_full
print(f"\nmultiply-adds per query: full {full:,}  funnel {funnel:,}  ratio {full / funnel:.1f}x")
print(f"RAM for the search vectors (float32): full {N * d_full * 4 / 1e9:.2f} GB, {m}-d {N * m * 4 / 1e9:.2f} GB")
```

```output
full cosine:            0.700
first 2 dims, raw dot:  0.400
first 2 dims, cosine:   0.800

multiply-adds per query: full 768,000,000  funnel 64,153,600  ratio 12.0x
RAM for the search vectors (float32): full 3.07 GB, 64-d 0.26 GB
```

## In code

**The problem.** Compare three versions of all-MiniLM-L6-v2 on AskMeridian's held-out shorthand queries at 384, 128, 64, 32 and 16 dimensions: the base model, a standard fine-tune with multiple negatives ranking loss, and a Matryoshka fine-tune with the same data and settings.

**Dependencies.** `sentence-transformers` (with PyTorch) and NumPy. The data are the `CORPUS`, `TRAIN` and `TEST` lists from the code in [Fine-tuning embeddings](/learn/embedding-optimisation/fine-tuning-embeddings), saved as `askmeridian_data.py`. The whole script runs on a CPU in under a minute.

```bash
pip install sentence-transformers numpy
```

```python
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from askmeridian_data import CORPUS, TRAIN, TEST   # the lists from "Fine-tuning embeddings"

ids = list(CORPUS)
DIMS = [384, 128, 64, 32, 16]

def mrr_at(model, dim):
    """Held-out MRR when every vector is cut to its first `dim` numbers and renormalised."""
    D = model.encode(list(CORPUS.values()), truncate_dim=dim, normalize_embeddings=True)
    Q = model.encode([q for q, _ in TEST], truncate_dim=dim, normalize_embeddings=True)
    order = np.argsort(-(Q @ D.T), axis=1)
    return np.mean([1 / (list(r).index(ids.index(d)) + 1) for r, (_, d) in zip(order, TEST)])

def build(kind):
    random.seed(0); torch.manual_seed(0)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    if kind == "base":
        return model
    loss_fn = MultipleNegativesRankingLoss(model)
    if kind == "matryoshka":
        loss_fn = MatryoshkaLoss(model, loss_fn, matryoshka_dims=DIMS)   # same loss at every prefix
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(4):
        for k in random.sample(range(3), 3):
            batch = TRAIN[k::3]                  # one query per passage: no duplicate positives
            features = [model.tokenize([q for q, _ in batch]),
                        model.tokenize([CORPUS[d] for _, d in batch])]
            loss = loss_fn(features, labels=None)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return model

print("held-out MRR by number of dimensions kept")
print(f"{'model':<11}" + "".join(f"{'d=' + str(d):>8}" for d in DIMS))
for kind in ("base", "standard", "matryoshka"):
    model = build(kind)
    print(f"{kind:<11}" + "".join(f"{mrr_at(model, d):>8.2f}" for d in DIMS))

v = model.encode(["Annual leave is 25 days a year."], truncate_dim=64)
print(f"\nnorm of a 64-d truncated vector: {np.linalg.norm(v):.3f} (before renormalising)")
```

```output
held-out MRR by number of dimensions kept
model         d=384   d=128    d=64    d=32    d=16
base           0.84    0.80    0.65    0.57    0.58
standard       1.00    0.94    0.88    0.71    0.72
matryoshka     0.89    0.94    0.92    0.88    0.76

norm of a 64-d truncated vector: 0.418 (before renormalising)
```

**The important lines.**

- `MatryoshkaLoss(model, loss_fn, matryoshka_dims=DIMS)` wraps the ordinary loss. Internally it computes the embeddings once, then evaluates the wrapped loss on the truncated, renormalised prefixes for each size in `DIMS`, and sums the results. Weights default to 1 for every size.
- `model.encode(..., truncate_dim=dim, normalize_embeddings=True)` takes the first `dim` numbers and renormalises. The last line shows why the second argument matters: a 64-dimensional prefix of a unit vector had length 0.418.
- `build` gives all three models the same starting point, the same data, the same seed and the same number of steps, so the only difference between "standard" and "matryoshka" is the loss.
- `batch = TRAIN[k::3]` produces batches with one query per passage, as in [Contrastive learning](/learn/embedding-optimisation/contrastive-learning), so there are no false in-batch negatives.

**What the output means.**

- **The base model degrades quickly when truncated**: 0.84 at full size, 0.65 at 64, 0.57 at 32. It was never trained to put the important information first.
- **The standard fine-tune is best at full size (1.00) but still degrades** at 32 dimensions (0.71). Fine-tuning improved the full vector without ordering its information.
- **The Matryoshka fine-tune holds up**: 0.92 at 64 and 0.88 at 32 dimensions, far better than either other model at those sizes. A 32-dimensional Matryoshka vector here does better than the base model's full 384-dimensional one, with a twelfth of the storage.
- **It paid at full size**: 0.89 against the standard fine-tune's 1.00. Spreading the training signal across five sizes left less capacity for the full vector in this tiny run. Published results with large training sets report that the cost at full size is usually small, but you should expect *some* trade-off and measure it.
- **The usual caveats apply**: 16 held-out queries, 18 passages, one seed. Differences of 0.05 are about one query. The pattern across sizes is the lesson, not any single number.

**Common errors.** Forgetting to renormalise after truncation; truncating queries and documents to different sizes; assuming a model supports truncation because its vectors are long; and evaluating truncation on the training queries.

**Extending it.** Add `matryoshka_weights` that favour small sizes (for example `[1, 1, 1, 2, 2]`) and see how the trade-off moves; implement a funnel search (exercise 3); and combine 64-dimensional prefixes with int8 quantisation from the [next lesson](/learn/embedding-optimisation/embedding-quantisation).

## A real-world example

The Platform team expects AskMeridian's index to grow from 200,000 to several million chunks as product documentation and support tickets are added. On the existing hardware, the full 768-dimensional vectors of their fine-tuned model will not fit in RAM at that scale.

Rather than buy bigger machines or switch to a smaller model, they retrain their fine-tuned model with `MatryoshkaLoss` over the sizes 768, 512, 256, 128 and 64, keeping data and settings the same. On the frozen evaluation set they compare it with the current model at each size. At 768 dimensions the two are indistinguishable within the interval. At 256 dimensions the Matryoshka model is within the interval of its own full size on every slice; at 128 it loses on the runbook slice, where error codes need fine distinctions.

They deploy a funnel: 128-dimensional prefixes in the in-memory Qdrant index, a candidate list of 200, and re-scoring with full 768-dimensional vectors stored alongside. They measure first-stage recall: the fraction of the full-vector top 10 found in the 128-dimensional top 200 is close to complete across slices, and end-to-end metrics match the full-vector baseline. RAM for the in-memory vectors falls to a sixth. The decision record notes the sizes evaluated, the chosen prefix and $K$, and that any future model must be Matryoshka-trained to keep this design.

## Common mistakes

- **Truncating a model that was not trained for it.** Quality drops sharply and unevenly across query types.
- **Not renormalising after truncation.** Dot products then mix direction with leftover length and reorder results.
- **Mixing sizes between queries and documents**, or between old and new vectors in one index.
- **Choosing the size from a public benchmark.** The quality curve depends on your data and slices; measure it.
- **A funnel with too small a candidate list.** Anything missed by the first stage cannot be recovered by re-scoring.
- **Reusing similarity thresholds across sizes.** Score distributions change with dimensionality.

## Best practices

- Confirm from the model card that a model is Matryoshka-trained before truncating, and verify with your evaluation.
- Evaluate retrieval at every candidate size, per slice, against the full size with a paired interval.
- Always renormalise truncated vectors and record the size in the embedding contract.
- When fine-tuning a model you may later want to shrink, wrap the loss in `MatryoshkaLoss` from the start; the cost is a little training time.
- Use funnel search when RAM is the constraint but full-size quality is required, and measure first-stage recall at $K$.
- Combine truncation with quantisation only after measuring each separately.

## Performance considerations

Truncation reduces memory and per-comparison compute in direct proportion: 768 to 128 dimensions is a six-fold reduction in both, before any index overhead. For flat search, latency falls roughly proportionally; for graph indexes such as HNSW, distance computations also shrink, though graph traversal and memory access patterns add costs that do not scale perfectly.

Embedding cost is unchanged. The model still computes the full vector; truncation happens afterwards. Matryoshka models do not make query embedding faster.

Training cost rises modestly: the transformer runs once per batch, and only the loss computation repeats for each size, which is small compared with the encoder.

Funnel search adds a second stage: fetching $K$ full vectors and $K \times d$ multiply-adds. If full vectors live on SSD, fetching 200 small records per query is fast but adds I/O latency; if they live in RAM alongside the short ones, the memory saving disappears. Some vector databases support storing multiple named vectors per point or re-scoring with original vectors; check your database's documentation.

## Summary

- Ordinary embeddings spread information across all dimensions, so truncating them loses quality quickly.
- Matryoshka Representation Learning trains with the loss applied to several nested prefixes, so each prefix is a usable embedding and one model serves many sizes.
- Truncate by taking the first $m$ numbers and renormalising; record $m$ in the embedding contract and use it for both queries and documents.
- In the AskMeridian experiment, a Matryoshka fine-tune kept 32-dimensional retrieval far closer to full quality than a standard fine-tune, at some cost at full size.
- Funnel search uses short prefixes to find candidates and full vectors to re-rank, cutting memory and compute roughly in proportion to the prefix size.

## Exercises

**1.** A model has 1,024 dimensions. Your index holds 5 million chunks in float32. How much RAM do the vectors need at 1,024, 256 and 64 dimensions?

<details><summary>Show solution</summary>

$5{,}000{,}000 \times 1{,}024 \times 4 = 20.48$ GB; at 256, 5.12 GB; at 64, 1.28 GB. Each quartering of dimensionality quarters memory.

</details>

**2.** Two unit vectors $a = (0.6, 0.0, 0.8)$ and $b = (0.0, 0.6, 0.8)$. Compute their full cosine, the raw dot product of their 2-dimensional prefixes, and the cosine of the prefixes. What does the result say about truncation?

<details><summary>Show solution</summary>

Full cosine: $0 + 0 + 0.64 = 0.64$. Prefixes $(0.6, 0.0)$ and $(0.0, 0.6)$: raw dot product 0, cosine 0. The similarity between these vectors lives entirely in the third coordinate, so truncating to two dimensions makes them look unrelated. A Matryoshka-trained model is trained so that the information needed to judge similarity appears in the early coordinates.

</details>

**3.** Extend the code example: after building the Matryoshka model, implement a funnel that searches with 32-dimensional prefixes for the top 5 candidates and re-scores them with full 384-dimensional vectors. Compute held-out MRR and the fraction of queries whose full-vector top 1 is among the 32-dimensional top 5.

<details><summary>Show solution</summary>

```python
D_full = model.encode(list(CORPUS.values()), normalize_embeddings=True)
Q_full = model.encode([q for q, _ in TEST], normalize_embeddings=True)
D_32 = model.encode(list(CORPUS.values()), truncate_dim=32, normalize_embeddings=True)
Q_32 = model.encode([q for q, _ in TEST], truncate_dim=32, normalize_embeddings=True)

rr, covered = [], 0
for i, (_, gold) in enumerate(TEST):
    shortlist = np.argsort(-(Q_32[i] @ D_32.T))[:5]                 # stage 1: 32-d
    rescored = shortlist[np.argsort(-(Q_full[i] @ D_full[shortlist].T))]   # stage 2: 384-d
    covered += int(np.argmax(Q_full[i] @ D_full.T) in shortlist)
    ranked = list(rescored) + [j for j in np.argsort(-(Q_full[i] @ D_full.T)) if j not in shortlist]
    rr.append(1 / (ranked.index(ids.index(gold)) + 1))
print(f"funnel MRR {np.mean(rr):.2f}, full-vector top-1 in shortlist for {covered}/{len(TEST)} queries")
```

```output
funnel MRR 0.87, full-vector top-1 in shortlist for 15/16 queries
```

The funnel's MRR (0.87) sits just below the model's own full-size MRR (0.89) and 32-dimensional MRR (0.88): on this tiny set the stages differ by about one query. For one query the full-vector top result was not in the 32-dimensional shortlist, so re-scoring could not recover it. That is the funnel's failure mode in miniature. With only 18 passages a shortlist of 5 is more than a quarter of the corpus; on a real corpus, measure coverage at several values of $K$ and pick the smallest $K$ that keeps it near complete.

</details>

**4.** Design decision. A hosted embedding API offers a model at 3,072 dimensions with a parameter to request fewer. You must choose a dimension for a 10-million-chunk index with a fixed RAM budget of 16 GB for vectors. Outline how you choose, and what you would do if no size that fits meets your quality bar.

<details><summary>Show solution</summary>

Budget arithmetic first: 16 GB / (10 million × 4 bytes) = 400 dimensions in float32, so candidates are 256 or 384 if offered (and higher with int8). Evaluate retrieval on your frozen set at 256, 512, 1,024 and 3,072 dimensions (the full size as the ceiling), per slice, with paired intervals.

If 256 meets the bar, use it. If not: quantise to int8 (four times smaller, so 1,024 dimensions fit in about 10 GB) and re-evaluate; or use a funnel with short prefixes in RAM and full vectors on disk for re-scoring; or add a cross-encoder reranker to recover precision from a smaller first stage. Record the choice and its evidence, since changing the dimension later means re-embedding or at least re-indexing.

</details>

## Check your understanding

```quiz
[
  {
    "q": "Why does truncating an ordinary (non-Matryoshka) embedding hurt retrieval?",
    "options": [
      "The model spreads information across all dimensions, so a prefix keeps only an arbitrary part of it",
      "Truncated vectors cannot be normalised",
      "Vector databases reject truncated vectors",
      "The first dimensions are always zero"
    ],
    "answer": 0,
    "why": "A loss computed only on the full vector gives the model no reason to place important information early."
  },
  {
    "q": "What does MatryoshkaLoss add to an ordinary loss such as MultipleNegativesRankingLoss?",
    "options": [
      "A second encoder for short vectors",
      "A penalty on vector length",
      "The same loss computed on several nested prefixes of the embeddings, summed",
      "Random dropping of dimensions at inference"
    ],
    "answer": 2,
    "why": "Each prefix is graded on its own, which pushes the most useful information into the earliest coordinates."
  },
  {
    "q": "What does this code print?\n```python\nimport numpy as np\nv = np.array([0.6, 0.8, 0.0])\nt = v[:1]\nprint(round(float(np.linalg.norm(t)), 2))\n```",
    "options": [
      "1.0",
      "0.8",
      "0.0",
      "0.6"
    ],
    "answer": 3,
    "why": "The 1-dimensional prefix is [0.6], whose length is 0.6. A truncated unit vector is no longer unit length, which is why renormalising is required."
  },
  {
    "q": "In a funnel search, 64-dimensional prefixes find 200 candidates which are re-scored with full vectors. What limits the funnel's final quality?",
    "options": [
      "The speed of the second stage",
      "Whether the relevant chunks are among the 200 candidates found by the first stage",
      "The number of dimensions in the query only",
      "Nothing; re-scoring always recovers full quality"
    ],
    "answer": 1,
    "why": "Re-scoring can only reorder the shortlist. Anything the first stage missed is lost, so measure first-stage recall at K."
  },
  {
    "q": "Does a Matryoshka model embed queries faster at 64 dimensions than at 768?",
    "options": [
      "No: the model computes the full vector and truncation happens afterwards; the savings are in storage and search",
      "Yes, ten times faster",
      "Yes, because fewer layers run",
      "Only on a GPU"
    ],
    "answer": 0,
    "why": "Matryoshka training changes the objective, not the architecture. Embedding cost is the same; memory and comparison cost shrink."
  }
]
```

## Further reading

- Kusupati et al. (2022), "Matryoshka Representation Learning", NeurIPS 2022, [arXiv:2205.13147](https://arxiv.org/abs/2205.13147).
- Sentence Transformers documentation, [sbert.net](https://www.sbert.net/), the Matryoshka embeddings training example and the `MatryoshkaLoss` reference.
- Nomic AI, model card for `nomic-embed-text-v1.5` on the Hugging Face Hub, describing its Matryoshka dimensions.
- OpenAI API documentation, embeddings guide, on the `dimensions` parameter of the `text-embedding-3` models.
