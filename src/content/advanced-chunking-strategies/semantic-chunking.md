The splitters met so far decide where to cut by looking at characters: blank lines, full stops, a size limit. None of them notices that the text has stopped talking about annual leave and started talking about the VPN. **Semantic chunking** places boundaries where the *meaning* changes. It embeds each sentence, measures how far apart consecutive sentences are in embedding space, and cuts where that distance jumps.

The idea is appealing and widely implemented (LangChain, LlamaIndex and others ship a version), but it is also widely oversold. This lesson builds a semantic chunker from scratch with a real embedding model, works through the threshold maths by hand, and then looks honestly at where the method helps, where it is noisy, and what it costs.

## Learning objectives

- Describe the semantic chunking pipeline: sentence splitting, optional buffering, embedding, adjacent-distance calculation and breakpoint selection.
- Compute a cosine distance and a percentile breakpoint threshold by hand and in NumPy.
- Implement a semantic chunker with `sentence-transformers` and interpret its distance profile.
- Explain how the percentile threshold controls average chunk length, and why semantic chunks need a separate size cap.
- Judge, for a given corpus, whether semantic chunking is likely to beat a well-configured recursive splitter, and how to find out.

## Prerequisites

- [Recursive character splitting](/learn/advanced-chunking-strategies/recursive-character-splitting): the baseline this lesson is compared against.
- [Sentence-based chunking](/learn/chunking-fundamentals/sentence-based-chunking): how text is split into sentences, the unit semantic chunking works on.
- [Embeddings](/learn/understanding-large-language-models/embeddings): text as vectors whose directions encode meaning.
- [Query-document matching](/learn/rag-architecture-fundamentals/query-document-matching): cosine similarity and why it is used.

## Why this matters

A chunk's embedding is, roughly, a summary of everything in it. If a chunk contains two unrelated topics, its vector sits somewhere between them and matches queries about either topic only moderately well. A recursive splitter creates such mixed chunks whenever a size limit lands in the middle of a section, or whenever two short sections fit into one chunk together.

The problem is worst for text with weak or missing formatting: meeting transcripts, chat exports, OCR output without reliable paragraph breaks, long emails, or web pages whose structure was lost in extraction. In that text, character separators carry little information about topic, and a chunker that can sense a change of subject has something real to offer. For well-formatted documents with headings, the document's own structure is usually a better signal, and the [structure-aware](/learn/advanced-chunking-strategies/structure-aware-chunking) and [heading-based](/learn/advanced-chunking-strategies/heading-based-chunking) lessons show how to use it.

Semantic chunking also has real costs: an embedding call per sentence at ingestion time, a threshold that needs tuning per corpus, and chunk sizes that are not controlled. Knowing the mechanism precisely is what lets you decide whether those costs buy anything for *your* documents.

## The intuition

Read a handbook aloud to a friend and ask them to raise a hand whenever "the subject changes". They will not raise it after every sentence: "Every employee receives 25 days of leave" and "Up to 5 unused days carry over" are obviously the same subject. They will raise it when you move from carry-over rules to the VPN.

Semantic chunking automates the hand-raising. Each sentence becomes a point in embedding space. Consecutive sentences on the same subject are close together; a change of subject is a jump. If you plot the size of each jump along the document, you get a jagged line with a few tall spikes. Cut at the spikes.

The hard question is "how tall is a spike?". There is no universal answer; jump sizes depend on the embedding model, the writing style and the length of the sentences. So instead of a fixed number, most implementations use a *relative* rule: cut at the jumps that are unusually large *for this document*, for example the largest 5 percent.

## How it works

### The pipeline, step by step

1. **Split into sentences.** Use a sentence splitter appropriate to the language (a regular expression for clean English prose; a trained segmenter for messier text). Sentences are the atoms; semantic chunking never cuts inside one.
2. **Optionally buffer.** Replace each sentence with a window containing it and its `b` neighbours on each side. This smooths the noise of single short sentences (more on this below).
3. **Embed.** Embed every sentence (or window) with an embedding model, normalising the vectors to unit length.
4. **Measure adjacent distances.** For each pair of neighbours $i$ and $i+1$, compute the cosine distance $d_i = 1 - \cos(\mathbf{e}_i, \mathbf{e}_{i+1})$. A document of $n$ sentences gives $n - 1$ distances, one per gap.
5. **Choose a threshold.** Compute a threshold from the distribution of the distances: a percentile, a number of standard deviations above the mean, or an interquartile-range rule.
6. **Cut.** Every gap whose distance exceeds the threshold becomes a chunk boundary. Consecutive sentences between boundaries are joined into a chunk.
7. **Enforce size.** Split any chunk that is too long (with a recursive splitter or a sentence-count cap), and optionally merge chunks that are too short.

```mermaid
flowchart LR
    A["Document"] --> B["Sentences"]
    B --> C["Buffer<br/>(optional)"]
    C --> D["Embed each<br/>sentence or window"]
    D --> E["Distance between<br/>neighbours"]
    E --> F["Threshold from<br/>the distribution"]
    F --> G["Cut at gaps<br/>above threshold"]
    G --> H["Size cap and<br/>minimum merge"]
    H --> I["Chunks"]
```

### Breakpoint rules

Implementations differ mainly in step 5. LlamaIndex's `SemanticSplitterNodeParser` uses a percentile (`breakpoint_percentile_threshold`, default 95) with a buffer (`buffer_size`, default 1). LangChain's `SemanticChunker`, in the `langchain-experimental` package, offers several `breakpoint_threshold_type` options, including `"percentile"`, `"standard_deviation"`, `"interquartile"` and `"gradient"`. All of them are relative to the document being chunked:

- **Percentile.** Cut at gaps above the $p$-th percentile of this document's distances. Predictable chunk *count*, regardless of how strongly the topics actually differ.
- **Standard deviation.** Cut at gaps more than $k$ standard deviations above the mean. Adapts to how spread out the distances are, but a document with no real topic changes still gets cuts if one gap happens to be an outlier.
- **Interquartile.** Cut at gaps above the third quartile plus a multiple of the interquartile range, the classic outlier rule; less affected by a few extreme values than the standard deviation.

Because every rule is relative, *every document gets cut*, even one that is entirely about a single subject. That is a design property you must account for, for example by not applying the chunker to documents shorter than a few hundred tokens.

### Why buffering exists, and what it costs

Single sentences make noisy embeddings. "Anything over £75 needs a receipt" contains no word about expenses; its embedding may sit some distance from "Expenses must be submitted within 30 days", producing a spurious spike in the middle of a topic. Buffering embeds each sentence together with its neighbours, so each window shares most of its text with the next one and the distance curve becomes smoother.

The price is blurring. With a buffer of 1, the window around the last sentence of topic A already contains the first sentence of topic B. The spike at the true boundary is flattened and spread over several gaps, and on short sections it can disappear entirely. A buffer should therefore be small compared with the typical topic length; you will see it misfire on three-sentence topics in the code below.

### Size is not controlled

Nothing in steps 1 to 6 refers to length. A 40-sentence stretch about one subject becomes one 40-sentence chunk; a one-sentence aside becomes a one-sentence chunk. Every production semantic chunker therefore needs step 7. The most common pattern is to treat semantic boundaries as *preferred* cut points and run a recursive splitter inside any semantic chunk that exceeds the size limit.

### Variants

Several variants exist beyond the adjacent-pair method. Some compare each new sentence with the *running mean* of the current chunk rather than with the previous sentence, so that a slowly drifting text eventually triggers a cut. Some cluster all sentence embeddings and draw boundaries where cluster membership changes. Others ask an LLM to place the boundaries, which is the subject of [agentic chunking](/learn/advanced-chunking-strategies/agentic-chunking). The adjacent-pair percentile method in this lesson is the one implemented in the mainstream frameworks and the one you should understand first.

### Does it work?

Evidence on whether semantic chunking improves retrieval is mixed, and you should treat claims in either direction with caution. Qu, Tu and Bao (2024) compared semantic and fixed-size chunking across several retrieval and question-answering tasks and reported that the benefits of semantic chunking were inconsistent and often did not justify the extra computation; their results are one study, on particular datasets and models, not a final verdict. Practitioners report similar mixed results. The sound engineering position is that semantic chunking is a hypothesis to test on your own labelled queries against a recursive baseline, not a default upgrade.

```playground
semantic-chunking
```

## The mathematics

### Cosine distance between neighbours

**Intuition.** Two sentences about the same subject point in similar directions in embedding space; the cosine of the angle between them is close to 1. Distance is one minus that.

$$
d_i = 1 - \frac{\mathbf{e}_i \cdot \mathbf{e}_{i+1}}{\lVert \mathbf{e}_i \rVert \, \lVert \mathbf{e}_{i+1} \rVert}
$$

- $\mathbf{e}_i$ is the embedding of sentence (or window) $i$, a vector of, here, 384 numbers.
- $\mathbf{e}_i \cdot \mathbf{e}_{i+1}$ is the dot product, the sum of element-wise products.
- $\lVert \cdot \rVert$ is the vector's length. If the vectors are normalised to unit length, the denominator is 1 and $d_i = 1 - \mathbf{e}_i \cdot \mathbf{e}_{i+1}$.
- $d_i$ ranges from 0 (same direction) to 2 (opposite); for sentence embeddings it typically falls between about 0.2 and 1.0.

**Worked example.** Take two toy unit vectors $\mathbf{e}_1 = (0.6, 0.8)$ and $\mathbf{e}_2 = (0.8, 0.6)$. The dot product is $0.6 \times 0.8 + 0.8 \times 0.6 = 0.96$, so $d_1 = 1 - 0.96 = 0.04$: nearly the same direction. With $\mathbf{e}_3 = (1, 0)$, $\mathbf{e}_1 \cdot \mathbf{e}_3 = 0.6$ and the distance is $0.4$.

### The percentile threshold

**Intuition.** Sort all the gaps; the $p$-th percentile is the value below which $p$ percent of them fall. Cutting above it keeps the largest $(100 - p)$ percent of gaps as boundaries.

NumPy's default (linear interpolation) computes the percentile of $m$ sorted values $x_{(0)} \le \dots \le x_{(m-1)}$ as

$$
P_p = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)\,\bigl(x_{(\lfloor h \rfloor + 1)} - x_{(\lfloor h \rfloor)}\bigr), \qquad h = \frac{p}{100}(m - 1)
$$

- $m$ is the number of gaps ($n - 1$ for $n$ sentences).
- $h$ is the fractional position in the sorted list; $\lfloor h \rfloor$ is its integer part.

**Worked example.** The code below produces eight gaps whose sorted values are 0.348, 0.382, 0.401, 0.531, 0.674, 0.766, 0.901, 0.936. For $p = 75$: $h = 0.75 \times 7 = 5.25$, so $P_{75} = 0.766 + 0.25 \times (0.901 - 0.766) = 0.766 + 0.034 = 0.800$. Two gaps (0.901 and 0.936) exceed it, so there are two boundaries and three chunks.

### What the percentile does to chunk length

Because the rule always keeps the top $(100 - p)$ percent of gaps, the number of boundaries is about $(1 - p/100)(n - 1)$, and the mean chunk length is about

$$
\bar{L} \approx \frac{n}{(1 - p/100)(n - 1) + 1} \approx \frac{1}{1 - p/100} \text{ sentences for large } n.
$$

With the common default $p = 95$, chunks average about $1/0.05 = 20$ sentences, *whatever the document is about*. With $p = 75$, about 4. This is the most useful fact about the percentile rule: it is really a chunk-length knob dressed up as a topic detector. If 20 sentences is longer than your target chunk, you must lower $p$ or rely on the size cap.

```python
import numpy as np

gaps = np.array([0.348, 0.401, 0.936, 0.531, 0.382, 0.901, 0.766, 0.674])
t = np.percentile(gaps, 75)
print(f"75th percentile = {t:.3f}; boundaries at gaps {np.where(gaps > t)[0] + 1}")
for p in (75, 90, 95):
    print(f"p={p}: mean chunk length for large n is about {1 / (1 - p / 100):.0f} sentences")
```

```output
75th percentile = 0.800; boundaries at gaps [3 6]
p=75: mean chunk length for large n is about 4 sentences
p=90: mean chunk length for large n is about 10 sentences
p=95: mean chunk length for large n is about 20 sentences
```

## In code

**The problem.** Chunk a nine-sentence passage of the Meridian handbook that covers three subjects (leave, the VPN, expenses) with no paragraph breaks, as text often arrives from a transcript or a badly extracted PDF. Inspect the distance profile, compare three threshold rules, and see what buffering does.

**Dependencies.** `sentence-transformers` (which installs PyTorch) and NumPy. The model `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, about 90 MB) downloads on first use. No API key.

```bash
pip install sentence-transformers numpy
```

```python
import re
import numpy as np
from sentence_transformers import SentenceTransformer

TEXT = (
    "Every employee receives 25 days of annual leave per year. "
    "Up to 5 unused leave days carry over into the next year. "
    "Carried-over days must be used by 31 March or they are lost. "
    "The VPN is required whenever you connect from public Wi-Fi. "
    "If the VPN client shows error E-4021, update it from the IT portal. "
    "The support desk can reset a VPN certificate between 08:00 and 18:00. "
    "Expenses must be submitted within 30 days. "
    "Anything over £75 needs a receipt. "
    "Late expense claims are returned by the finance department."
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentences = re.split(r"(?<=[.!?])\s+", TEXT.strip())      # 1. sentences


def gaps(sents, buffer=0):
    """Cosine distance between each sentence (plus `buffer` neighbours) and the next."""
    texts = [" ".join(sents[max(0, i - buffer): i + buffer + 1]) for i in range(len(sents))]
    emb = model.encode(texts, normalize_embeddings=True)   # 2. unit-length embeddings
    return 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)        # 3. adjacent distances


def split_at(sents, d, threshold, max_sentences=None):
    """4-5. Cut wherever the gap exceeds the threshold (or the chunk is full)."""
    chunks, current = [], [sents[0]]
    for i, gap in enumerate(d):
        full = max_sentences is not None and len(current) >= max_sentences
        if gap > threshold or full:
            chunks.append(current)
            current = []
        current.append(sents[i + 1])
    chunks.append(current)
    return chunks


d = gaps(sentences)
print("gap  distance  between")
for i, x in enumerate(d):
    print(f"{i + 1:>3}  {x:8.3f}  {sentences[i][:22]!r} -> {sentences[i + 1][:22]!r}")

rules = {
    "60th percentile": np.percentile(d, 60),
    "75th percentile": np.percentile(d, 75),
    "mean + 1 std":   d.mean() + d.std(),
}
for name, t in rules.items():
    chunks = split_at(sentences, d, t)
    print(f"\n{name}: threshold {t:.3f} -> chunk sizes {[len(c) for c in chunks]}")
    for c in chunks:
        print("   |", c[0][:60])

print("\nwith buffer=1:", " ".join(f"{x:.2f}" for x in gaps(sentences, buffer=1)))
```

The embedding model runs locally on the CPU; the distances below are from the sandbox run, and yours may differ in the third decimal place on other hardware or library versions.

```output
gap  distance  between
  1     0.348  'Every employee receive' -> 'Up to 5 unused leave d'
  2     0.401  'Up to 5 unused leave d' -> 'Carried-over days must'
  3     0.936  'Carried-over days must' -> 'The VPN is required wh'
  4     0.531  'The VPN is required wh' -> 'If the VPN client show'
  5     0.382  'If the VPN client show' -> 'The support desk can r'
  6     0.901  'The support desk can r' -> 'Expenses must be submi'
  7     0.766  'Expenses must be submi' -> 'Anything over £75 need'
  8     0.674  'Anything over £75 need' -> 'Late expense claims ar'

60th percentile: threshold 0.692 -> chunk sizes [3, 3, 1, 2]
   | Every employee receives 25 days of annual leave per year.
   | The VPN is required whenever you connect from public Wi-Fi.
   | Expenses must be submitted within 30 days.
   | Anything over £75 needs a receipt.

75th percentile: threshold 0.800 -> chunk sizes [3, 3, 3]
   | Every employee receives 25 days of annual leave per year.
   | The VPN is required whenever you connect from public Wi-Fi.
   | Expenses must be submitted within 30 days.

mean + 1 std: threshold 0.838 -> chunk sizes [3, 3, 3]
   | Every employee receives 25 days of annual leave per year.
   | The VPN is required whenever you connect from public Wi-Fi.
   | Expenses must be submitted within 30 days.

with buffer=1: 0.03 0.35 0.28 0.16 0.14 0.18 0.59 0.20
```

**The important lines.**

- `re.split(r"(?<=[.!?])\s+", ...)` splits after sentence-ending punctuation. It is adequate for this clean text and would mis-split "e.g. this" or "08:00 a.m. on"; real pipelines use a proper sentence segmenter.
- `model.encode(texts, normalize_embeddings=True)` returns unit vectors, so `np.sum(emb[:-1] * emb[1:], axis=1)` is the cosine of every adjacent pair in one vectorised line.
- `gap > threshold` is a strict comparison, as in LlamaIndex's implementation; a gap exactly equal to the threshold is not a boundary.
- `max_sentences` is the size cap from step 7, in its simplest form. It is unused in this run; exercise 3 uses it.

**What the output means.**

- **The profile is readable.** Gaps 3 and 6, the true topic changes, are the two largest distances (0.936 and 0.901). Within-topic gaps sit between 0.35 and 0.77.
- **But the margin is thin.** Gap 7, *inside* the expenses topic, is 0.766. "Anything over £75 needs a receipt" never mentions expenses, so its embedding is far from the previous sentence. The 60th percentile rule cuts there and splits the expenses topic in two, leaving "Anything over £75 needs a receipt" in a chunk that no longer says what the £75 refers to. That is exactly the kind of fragment chunking is meant to avoid.
- **The rule chose the count.** At the 75th percentile, two of eight gaps must be boundaries, and here the two largest happen to be the right ones. On a document with four topics the same rule would still produce about the same proportion of cuts.
- **Buffering hurt here.** With a buffer of 1 the profile flattens and its peak moves to gap 7 (0.59), while the true boundaries drop to 0.28 and 0.18. With three-sentence topics, every window near a boundary already contains sentences from both sides. Buffering helps on long, noisy text where topics run for many sentences; on short sections it destroys the signal.

**Errors people commonly hit.**

- Forgetting `normalize_embeddings=True` and then treating the dot product as a cosine. Unnormalised vectors give distances on an arbitrary scale.
- Feeding very long "sentences" (a table flattened into one line, a 300-word legal sentence) to a model with a short input limit. Anything past the limit is truncated before embedding, so the distance reflects only the beginning.
- Embedding one sentence per request against a hosted API. Batch the calls: `encode` accepts a list and batches internally, and hosted embedding APIs accept many inputs per request.
- Running the chunker on tiny documents. With five sentences there are four gaps, and any percentile rule will still cut.

**Extending it.** Replace the adjacent-pair comparison with a comparison against the running mean embedding of the current chunk. Or wrap `split_at` so that any chunk above a token limit is passed to a recursive splitter. LlamaIndex packages the method as `SemanticSplitterNodeParser(buffer_size=..., breakpoint_percentile_threshold=..., embed_model=...)` and LangChain as `SemanticChunker` in `langchain_experimental.text_splitter`; neither was run here, but both implement the same pipeline, so the output above is what to expect from them in kind.

## A real-world example

Meridian's Customer Success team asked the Platform team to make recorded customer calls about Meridian Reports searchable. The transcripts arrived as long runs of speaker turns with no headings and inconsistent paragraphing. A recursive splitter at 400 tokens produced chunks that routinely began in the middle of one issue (a billing question) and ended in the middle of another (an SSO configuration problem), and retrieval for "SAML login failing" returned chunks dominated by billing talk.

The team tried semantic chunking with LlamaIndex's defaults (buffer 1, 95th percentile). The first result was disappointing in an instructive way: chunks were enormous, often thousands of tokens, because the 95th percentile rule produces only one cut per twenty-odd sentences, and transcripts have many short sentences ("Right." "OK, got it."). Those huge chunks exceeded the embedding model's input limit and were silently truncated.

They made three changes. First, they merged consecutive turns from the same speaker into one unit before embedding, so the "sentences" were meaningful utterances rather than one-word acknowledgements. Second, they lowered the percentile to 80, after plotting the distance profile on a dozen transcripts and comparing the cuts with where a human reader would place topic changes. Third, they added a 350-token cap enforced by a recursive splitter inside any oversized semantic chunk. They then evaluated on 120 labelled queries against the recursive baseline and found that the semantic version retrieved the right call segment more often on multi-issue calls and made no difference on single-issue calls. That result justified keeping semantic chunking for transcripts only; the handbook and runbooks stayed on structure-based chunking, where headings already mark topic changes for free.

## Common mistakes

- **Treating the percentile as a topic detector.** It fixes the proportion of gaps that become boundaries, so a single-topic document is cut anyway and a many-topic document is under-cut.
- **Shipping without a size cap.** Semantic chunks can exceed the embedding model's input limit and be silently truncated; tiny chunks lose context.
- **Using a large buffer on text with short sections.** Windows straddle boundaries and the spikes you are looking for are smoothed away.
- **Tuning on one document.** A threshold that looks perfect on one handbook page may be wrong for the next. Tune on a sample and evaluate on held-out queries.
- **Assuming it beats recursive splitting.** Published evidence is mixed; without your own comparison you may be paying embedding costs for no gain.
- **Using semantic chunking on well-structured documents.** Headings and sections are a cheaper and more reliable boundary signal than embedding distances.

## Best practices

- Use semantic chunking where structure is weak (transcripts, chats, OCR text, long emails) and structural chunking where it is strong.
- Plot the distance profile for a handful of documents before choosing a threshold, and compare cuts with where a human would cut.
- Choose $p$ from the chunk length you want ($\bar{L} \approx 1/(1 - p/100)$ sentences), then check boundary quality, rather than the other way round.
- Always enforce a maximum size, measured in the embedding model's tokens, and merge fragments below a minimum size.
- Normalise "sentences" first: merge very short units, split very long ones, and remove boilerplate that would create spurious spikes.
- Evaluate against a recursive baseline on labelled queries, per document type, before switching.

## Performance considerations

Semantic chunking adds an embedding pass at ingestion: one embedding per sentence (or per window), in addition to the embedding of the final chunks. Suppose a corpus of 50,000 documents averaging 200 sentences: that is 10 million sentence embeddings. With a small local model on a CPU encoding, as an assumption, a few hundred short sentences per second, that is roughly $10^7 / 300 \approx 33{,}000$ seconds, about nine hours on one core, against perhaps a tenth as many chunk embeddings for the final index. A GPU or several worker processes bring this down by an order of magnitude or more. With a hosted embedding API the cost is per token: sentence embeddings cover the whole corpus once, so they roughly double the embedding bill compared with embedding only the chunks, and buffering multiplies it again by about $2b + 1$, since each sentence appears in that many windows.

Two cheaper options are worth considering. Use a small, fast model for boundary detection and a stronger one for the final chunk embeddings; boundaries need only coarse topic similarity. And re-chunk incrementally: only documents whose content hash changed need their sentences re-embedded.

At query time semantic chunking costs nothing extra: the index contains ordinary chunks.

## Security considerations

Semantic chunking feeds every sentence of every document to an embedding model. If that model is a hosted API, the full text of your corpus leaves your environment, including any sentences a later filter would have removed. Apply redaction and access-control filtering *before* chunking, check the provider's data-retention terms, and prefer a local model for boundary detection on sensitive corpora. Distances are also manipulable: a document author can insert sentences that force or suppress boundaries, for example to place an instruction-like sentence in the same chunk as a trusted policy statement. Treat ingested text as untrusted, as you would for any chunker.

## Summary

- Semantic chunking embeds sentences, measures the cosine distance between neighbours, and cuts where the distance exceeds a threshold computed from the document's own distances.
- The percentile rule fixes the proportion of gaps that become boundaries; mean chunk length is about $1/(1 - p/100)$ sentences, so $p = 95$ means about 20.
- Buffering smooths noise but blurs boundaries; keep it small relative to topic length.
- Semantic chunks have no size control, so a maximum (and usually a minimum) must be enforced separately.
- The method costs an extra embedding pass and its benefit is uneven in published comparisons; use it where structure is weak, and prove it against a recursive baseline.

## Exercises

**1.** Ten gaps have sorted distances 0.20, 0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.80, 0.85. Compute the 90th percentile with NumPy's linear interpolation, and say how many boundaries it creates.

<details><summary>Show solution</summary>

$m = 10$, so $h = 0.9 \times 9 = 8.1$. The values at sorted positions 8 and 9 are 0.80 and 0.85, so $P_{90} = 0.80 + 0.1 \times 0.05 = 0.805$. Only 0.85 exceeds it: one boundary, two chunks. Note that the gap of 0.80, clearly a second outlier, is not a boundary. A standard-deviation or interquartile rule would catch both.

</details>

**2.** A colleague says: "We switched from the 95th to the 90th percentile and retrieval got better, so our documents must have more topics than we thought." What is wrong with this reasoning?

<details><summary>Show solution</summary>

Lowering $p$ from 95 to 90 roughly halves the mean chunk length (from about 20 sentences to about 10) on *any* document, regardless of its topics. The improvement may come purely from smaller chunks, which a recursive splitter with a smaller size would also deliver. To attribute it to topic detection, compare against a recursive splitter producing chunks of the same average length.

</details>

**3.** Modify the code example so that chunks never exceed two sentences, using `max_sentences`, with the 75th percentile threshold. Which chunks change, and is the result better or worse for the query "Do I need a receipt for a £90 taxi?"

<details><summary>Show solution</summary>

Call `split_at(sentences, d, rules["75th percentile"], max_sentences=2)`. Each three-sentence topic now ends in a forced cut after its second sentence, so the sizes become `[2, 1, 2, 1, 2, 1]`. The third expenses sentence ("Late expense claims...") becomes its own chunk, and "Anything over £75 needs a receipt" now shares a chunk with "Expenses must be submitted within 30 days", which is good for the receipt query: the chunk says what the £75 threshold applies to. But the leave and VPN topics each lose their third sentence to a fragment. A size cap that cuts blindly reintroduces the fragmentation semantic chunking was meant to avoid; a better cap splits an oversized chunk at its *largest internal gap*, not at a fixed count.

</details>

**4.** Design decision. AskMeridian is about to ingest three new sources: the IT runbooks (Markdown with headings), a Slack export from the support channel, and scanned policy letters processed by OCR. For each, say whether you would use semantic chunking, and what you would measure to confirm the choice.

<details><summary>Show solution</summary>

- **Runbooks: no.** Headings and numbered steps already mark the boundaries; use heading-based chunking with a recursive fallback. Embedding distances would add cost and noise.
- **Slack export: possibly**, after preprocessing. Group messages into threads first (thread structure is a strong boundary signal); within long threads, semantic chunking over messages merged per speaker turn can separate sub-topics. Use a small buffer and a size cap.
- **OCR letters: possibly.** Paragraph breaks may be unreliable, but letters are short and usually about one subject, so a relative threshold would cut needlessly. A simpler choice is one chunk per letter, or recursive splitting for long letters.

In each case, build a labelled query set for that source and compare Recall@K for the candidate chunker against a recursive baseline with a similar average chunk length.

</details>

## Check your understanding

```quiz
[
  {
    "q": "In adjacent-pair semantic chunking, what is compared to decide whether to cut between sentence i and sentence i+1?",
    "options": [
      "The length of each sentence in tokens",
      "The cosine distance between the embeddings of sentence i and sentence i+1 (or their windows)",
      "The similarity of each sentence to the user's query",
      "The number of shared words between the two sentences"
    ],
    "answer": 1,
    "why": "Boundaries are placed where the distance between neighbouring embeddings is unusually large, signalling a change of subject."
  },
  {
    "q": "With breakpoint_percentile_threshold=95, roughly how long are chunks on average for a long document?",
    "options": [
      "About 5 sentences",
      "It depends entirely on how many topics the document contains",
      "About 95 sentences",
      "About 20 sentences"
    ],
    "answer": 3,
    "why": "The rule keeps the top 5 percent of gaps as boundaries, so there is about one cut per 20 gaps, whatever the content."
  },
  {
    "q": "Why can a buffer of 1 make boundaries harder to detect in a document whose topics are three sentences long?",
    "options": [
      "Windows near a boundary contain sentences from both topics, flattening the distance spike",
      "Buffering removes sentences from the document",
      "The embedding model rejects inputs containing more than one sentence",
      "Buffering changes the distance metric from cosine to Euclidean"
    ],
    "answer": 0,
    "why": "Adjacent windows share most of their text, and near a short topic's edges they mix both topics, so the jump in distance is spread out and reduced."
  },
  {
    "q": "Which of these is the strongest reason to add a size cap to a semantic chunker?",
    "options": [
      "Semantic chunks are always too small",
      "The percentile rule requires chunks of equal size",
      "Nothing in the boundary rule limits length, so long single-topic stretches can exceed the embedding model's input limit and be truncated",
      "Vector databases reject chunks with more than ten sentences"
    ],
    "answer": 2,
    "why": "Boundaries depend only on distances, so a long passage on one subject becomes one chunk of arbitrary length."
  },
  {
    "q": "Your corpus is well-formatted Markdown documentation with clear headings. What is the most defensible chunking choice?",
    "options": [
      "Semantic chunking with a 95th percentile threshold, because it detects topics",
      "Heading-based or structure-aware chunking with a recursive fallback, evaluated against alternatives on labelled queries",
      "Fixed 100-character chunks with no overlap",
      "One chunk per document, regardless of length"
    ],
    "answer": 1,
    "why": "Headings already mark topic boundaries reliably and for free; semantic chunking adds cost and noise there."
  }
]
```

## Further reading

- Qu, R., Tu, R. and Bao, F. (2024). "Is Semantic Chunking Worth the Computational Cost?" [arXiv:2410.13070](https://arxiv.org/abs/2410.13070).
- LlamaIndex documentation, "Node parsers", `SemanticSplitterNodeParser`: [docs.llamaindex.ai](https://docs.llamaindex.ai/).
- LangChain documentation, "How to split text based on semantic similarity" (`SemanticChunker` in `langchain-experimental`).
- Reimers, N. and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". [arXiv:1908.10084](https://arxiv.org/abs/1908.10084). The basis of the `sentence-transformers` models used here; documentation at [sbert.net](https://www.sbert.net/).
- Hearst, M. A. (1997). "TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages". *Computational Linguistics* 23(1). The classic lexical-cohesion ancestor of embedding-based topic segmentation.
- Kamradt, G. "5 Levels of Text Splitting", the tutorial notebook that popularised the buffered, percentile-based method.
