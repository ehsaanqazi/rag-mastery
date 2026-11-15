The [Reranking Fundamentals](/learn/reranking-fundamentals) module explained why a second, slower scoring stage helps and how a [cross-encoder](/learn/reranking-fundamentals/cross-encoders) reads a query and a passage together. This module turns to the models you can actually deploy, and it starts with the most widely used open-weight family: the **BGE rerankers** from the Beijing Academy of Artificial Intelligence (BAAI). They are free to download, run on your own hardware, and come in sizes from a few hundred million parameters to several billion.

This lesson covers what the family contains and how to choose a member, how a BGE reranker turns a pair of texts into a score, what that score does and does not mean, and how to run one in a real pipeline. The central practical lesson is about the score: you will run `bge-reranker-base` on Meridian's handbook and watch it give the correct chunk a score of 0.04 and a wrong chunk a score of 0.99, and you will see from the training objective exactly why that is allowed to happen.

## Learning objectives

- Describe the members of the BGE reranker family (encoder-based and LLM-based) and choose one for a given latency, language and hardware budget.
- Run a BGE reranker with `sentence-transformers` or `FlagEmbedding` over first-stage candidates and return a reordered top k.
- Explain the group-wise cross-entropy loss used to train BGE rerankers and show why it leaves the absolute score level unanchored.
- Distinguish a raw logit from a sigmoid-normalised score and explain why neither is a calibrated probability of relevance.
- Estimate reranking latency from the number of candidates and their token lengths.

## Prerequisites

- [Cross-encoders](/learn/reranking-fundamentals/cross-encoders): how a model that reads query and passage together produces one relevance score.
- [Retrieval versus reranking](/learn/reranking-fundamentals/retrieval-versus-reranking): why the reranker sees only a short candidate list.
- [BGE models](/learn/embedding-models/bge-models): the BGE embedding family and BGE-M3, which the rerankers are paired with.
- [Ranking scores](/learn/reranking-fundamentals/ranking-scores): why scores from different stages live on different scales.

## Why this matters

A first-stage retriever ranks by a single similarity between two vectors computed independently. It is fast enough to search millions of chunks, but it cannot look at how a specific question relates to a specific sentence. A cross-encoder can, and in most published retrieval evaluations adding one on top of a dense or BM25 first stage improves ranking quality noticeably. The question for an engineering team is not whether to rerank but **with what**.

Three constraints usually decide it: whether data may leave your infrastructure (which rules out hosted APIs such as [Cohere rerank](/learn/reranking-models/cohere-rerank) for some workloads), the latency budget, and the licence (some strong open rerankers are non-commercial). BGE rerankers are a common default because they satisfy all three for most teams: the main models carry permissive licences (check each model card before shipping), run on a CPU for small candidate sets, and include a multilingual option.

What goes wrong is rarely the ranking itself but everything around it: a threshold set on a score that was never meant to be thresholded, a `max_length` that silently cuts off the sentence containing the answer, an LLM-based reranker chosen for quality and then found to triple the latency budget. This lesson is about those decisions.

## The intuition

Imagine two examiners marking essays. Examiner A is told: "for each essay, give a mark out of 100 for how well it answers the question". Examiner B is told: "here are five essays for the same question; put the best one first". Examiner B never has to commit to an absolute level. One day B might rate the five essays 90, 85, 80, 75, 70; another day 30, 25, 20, 15, 10. Both satisfy the instruction, because only the order was ever checked.

BGE rerankers were trained like examiner B. During training each query came with one relevant passage and a group of irrelevant ones, and the model was rewarded for making the relevant one score highest **within that group**. So its scores are excellent for ordering the candidates of one query and much less reliable for saying, on their own, "this passage is relevant". A score of 0.04 can be the best answer to one question, and 0.99 can be a confident mismatch for another. Keep this picture in mind; the code will show both.

## How it works

### The family

BAAI publishes the rerankers alongside the BGE embedding models in its open-source FlagEmbedding project. They fall into two architectural groups. The table reflects the model cards at the time of writing; new members appear regularly, so check the [BAAI page on Hugging Face](https://huggingface.co/BAAI) before choosing.

| Model | Backbone | Approximate size | Notes |
| --- | --- | --- | --- |
| `bge-reranker-base` | XLM-RoBERTa (encoder) | 278 M parameters | Chinese and English; the 2023 original |
| `bge-reranker-large` | XLM-RoBERTa (encoder) | about 560 M | Same data, larger model |
| `bge-reranker-v2-m3` | Initialised from BGE-M3 (encoder) | about 568 M | Multilingual; the usual default today |
| `bge-reranker-v2-gemma` | Gemma 2B (decoder LLM) | about 2.5 B | Multilingual; higher quality, much slower |
| `bge-reranker-v2-minicpm-layerwise` | MiniCPM 2B (decoder LLM) | about 2.7 B | Lets you read the score from an earlier layer to trade quality for speed |

The parameter count of `bge-reranker-base` was checked by loading it (the code below prints 278 M); the others are as stated on their model cards and are approximate.

**Encoder-based rerankers** are classic cross-encoders. The query and the passage are joined into one sequence, the transformer runs over it with every token attending to every other, and a small classification head on the first token's output produces one number.

**LLM-based rerankers** use a decoder language model. The query and passage are placed in a prompt that asks, in effect, "does this passage answer this query? Answer Yes or No", and the score is the model's logit for the token "Yes" at the position where it would answer. The model never generates text; one forward pass yields the score. These models know more about the world and handle harder, more multilingual queries, but they are roughly five to ten times larger than the encoder rerankers, and the layerwise variant exists precisely because that cost is hard to bear: it was trained so that intermediate layers also produce usable scores, letting you stop the forward pass early.

### The scoring mechanism, step by step

For an encoder-based BGE reranker:

1. **Tokenise the pair.** The tokeniser builds `<s> query </s></s> passage </s>` (XLM-RoBERTa's equivalents of `[CLS]` and `[SEP]`). If the total exceeds `max_length`, the passage is truncated from the end.
2. **Encode jointly.** Every layer lets each passage token attend to each query token and vice versa. This is the step a bi-encoder cannot perform and the source of the reranker's accuracy.
3. **Read one vector.** The final hidden state of the first token summarises the pair.
4. **Project to a logit.** A linear layer maps that vector to one real number, the **relevance logit**. It is unbounded; in practice BGE logits for the base model fall roughly between −11 and +8.
5. **Optionally normalise.** A sigmoid, $\sigma(x) = 1/(1+e^{-x})$, maps the logit into (0, 1). Because the sigmoid is strictly increasing, this never changes the order.
6. **Sort and cut.** Sort candidates by score and keep the top k for the generator.

```mermaid
flowchart LR
    Q["Query"] --> T["Tokenise pair<br/>&lt;s&gt; query &lt;/s&gt;&lt;/s&gt; passage &lt;/s&gt;"]
    P["Candidate passage"] --> T
    T --> E["XLM-RoBERTa layers<br/>(full cross-attention)"]
    E --> H["First-token vector"]
    H --> L["Linear head<br/>one logit"]
    L --> S["Optional sigmoid<br/>(0, 1)"]
    S --> R["Sort candidates,<br/>keep top k"]
```

Steps 1 to 4 run once per candidate. With 50 candidates that is 50 forward passes per query, batched together on the hardware. This is the cost that a first-stage search avoids by encoding documents in advance, and it is why the reranker only ever sees a short list.

### How the models were trained

BGE rerankers are fine-tuned on large collections of (query, relevant passage) pairs, together with **hard negatives**: passages that a retriever ranked highly but that are not relevant. For each training query the model scores a group consisting of one positive and several negatives, and the loss is the cross-entropy of a softmax over the group (the maths is below). The FlagEmbedding fine-tuning scripts expose the group size as a setting, and the same recipe is what you use to [fine-tune a reranker on your own data](/learn/reranking-models/cross-encoder-models).

The consequence is the one the examiner analogy described. The loss depends only on the **differences** between scores within a group. Nothing in training ties a logit of +2 to "relevant" or −4 to "irrelevant" across different queries. The model usually learns a rough absolute scale anyway, because relevant pairs share features, but nothing forces it to, and on queries unlike its training data the scale drifts.

### Where BGE rerankers fit in the pipeline

BAAI's own guidance is to use the reranker on the top candidates returned by an embedding model, and that is the standard shape: retrieve a generous candidate list (often 20 to 100) with a fast first stage, rerank, keep a handful. How many candidates to fetch is the subject of [candidate selection](/learn/reranking-fundamentals/candidate-selection). Nothing ties a BGE reranker to a BGE embedder; it reads raw text, so it works after BM25, after any embedding model, or after a [hybrid](/learn/hybrid-search/hybrid-retrieval-architecture) first stage.

### Choosing a member

- **English or Chinese only, CPU, tight latency:** `bge-reranker-base`. It is the smallest and fastest.
- **Several languages, or a single GPU:** `bge-reranker-v2-m3`. It inherits BGE-M3's multilingual training and accepts longer inputs than the 512-token originals (check the model card for the exact limit you can rely on).
- **Quality matters more than latency, GPU available:** an LLM-based member, ideally measured against `v2-m3` on your own queries first, because a 2 to 3 billion parameter model per candidate is a large cost to justify by intuition.

Whatever you choose, measure it on your own labelled queries with [NDCG](/learn/retrieval-evaluation/ndcg) or [MRR](/learn/retrieval-evaluation/mrr); leaderboard positions rarely transfer cleanly to handbooks and runbooks.

## The mathematics

### Group-wise cross-entropy

**Intuition.** Turn the scores in a training group into probabilities with a softmax, then penalise the model by how little probability it puts on the positive passage. A model that makes the positive stand far above the negatives gets a small loss.

$$
\mathcal{L} = -\log \frac{e^{s_0}}{\sum_{j=0}^{n} e^{s_j}}
$$

- $s_0$ is the reranker's logit for the query paired with its relevant passage.
- $s_1, \dots, s_n$ are the logits for the same query paired with $n$ hard negatives.
- The fraction is the softmax probability the model assigns to the positive; $\mathcal{L}$ is its negative logarithm.

**Worked example.** A group of three with logits $s_0 = 2.0$, $s_1 = 0.5$, $s_2 = -1.0$.

- $e^{2.0} = 7.389$, $e^{0.5} = 1.649$, $e^{-1.0} = 0.368$; the sum is $9.406$.
- The positive's probability is $7.389 / 9.406 = 0.786$.
- The loss is $-\ln 0.786 = 0.241$.

Now subtract 6 from every logit: $-4.0, -5.5, -7.0$. Every exponential is multiplied by $e^{-6}$, which cancels between numerator and denominator, so the probability and the loss are unchanged. In symbols, $\frac{e^{s_0 + c}}{\sum_j e^{s_j + c}} = \frac{e^{s_0}}{\sum_j e^{s_j}}$ for any constant $c$. **Training cannot see the absolute level of a query's scores.** That single line is the reason reranker scores make poor thresholds.

### The sigmoid normalisation

$\sigma(x) = 1/(1 + e^{-x})$ maps any real number into (0, 1): $\sigma(2.0) = 1/(1 + 0.135) = 0.881$ and $\sigma(-4.0) = 1/(1 + 54.6) = 0.018$. It looks like a probability and is often called one, but it is only a squashed logit. A probability of relevance would require that, among all pairs scored 0.8, about 80 per cent are relevant; nothing in the training above makes that true. Mapping scores to real probabilities is a separate step called [score calibration](/learn/advanced-retrieval-optimisation/score-calibration).

The same arithmetic in Python:

```python
import numpy as np

def group_loss(scores):
    """Cross-entropy for one query: the positive's score is scores[0]."""
    scores = np.asarray(scores, dtype=float)
    log_probs = scores - np.log(np.exp(scores).sum())
    return -log_probs[0], np.exp(log_probs)

loss, probs = group_loss([2.0, 0.5, -1.0])
print("probabilities:", probs.round(4), " loss:", round(loss, 4))

shifted_loss, _ = group_loss([2.0 - 6, 0.5 - 6, -1.0 - 6])   # same gaps, lower level
print("loss after shifting every score by -6:", round(shifted_loss, 4))

sigmoid = lambda x: 1 / (1 + np.exp(-x))
print("sigmoid of 2.0 and of -4.0:", sigmoid(2.0).round(4), sigmoid(-4.0).round(4))
```

```output
probabilities: [0.7856 0.1753 0.0391]  loss: 0.2413
loss after shifting every score by -6: 0.2413
sigmoid of 2.0 and of -4.0: 0.8808 0.018
```

## In code

**The problem.** AskMeridian's first stage is a small embedding model. It often puts the right chunk in its top five but not always first, and the team wants to know what `bge-reranker-base` adds, and what its scores look like, before relying on them.

**Dependencies.** `sentence-transformers` (version 4 or later, for the `activation_fn` argument) and NumPy. The first run downloads `all-MiniLM-L6-v2` (about 90 MB) and `bge-reranker-base` (about 1.1 GB). No API key.

```bash
pip install "sentence-transformers>=4" numpy
```

The code runs a first-stage search over eight handbook chunks, reranks the top five with `bge-reranker-base`, and prints for each candidate its first-stage rank, its cosine similarity, the reranker's raw logit and the sigmoid of that logit.

```python
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

chunks = [
    "Annual leave is 25 days per year. Up to 5 unused days carry over and must be used by 31 March.",
    "Staff may work remotely up to 3 days a week.",
    "The home-office equipment budget is £400. Contractors are not eligible for the equipment budget.",
    "Expenses must be submitted within 30 days of purchase.",
    "Any expense over £75 needs a receipt; smaller amounts can be claimed without one.",
    "Passwords must be rotated every 90 days. VPN is required on public Wi-Fi.",
    "Laptops are refreshed every 3 years. Report a lost laptop to the support desk at once.",
    "The support desk is open 08:00 to 18:00 UK time.",
]
queries = [
    "Do I need a receipt for a £60 taxi?",
    "Can a contractor buy a monitor with the home-office budget?",
    "What is the parental leave policy?",
]

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

doc_vecs = embedder.encode(chunks, normalize_embeddings=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

for query in queries:
    q_vec = embedder.encode(query, normalize_embeddings=True)
    cosine = doc_vecs @ q_vec
    candidates = np.argsort(-cosine)[:5]                  # first stage: top 5
    pairs = [(query, chunks[i]) for i in candidates]
    logits = reranker.predict(pairs, activation_fn=torch.nn.Identity())  # raw logits
    order = np.argsort(-logits)
    print(f"\nQuery: {query}")
    print(f"  {'1st':>3} {'cos':>6} {'logit':>7} {'sigmoid':>8}  chunk")
    for new_rank, j in enumerate(order, start=1):
        i = candidates[j]
        old_rank = int(np.where(candidates == i)[0][0]) + 1
        print(f"  {old_rank:>3} {cosine[i]:>6.3f} {logits[j]:>7.2f} {sigmoid(logits[j]):>8.3f}  {chunks[i][:55]}")
```

```output

Query: Do I need a receipt for a £60 taxi?
  1st    cos   logit  sigmoid  chunk
    1  0.740   -3.17    0.040  Any expense over £75 needs a receipt; smaller amounts c
    3  0.276   -7.37    0.001  Expenses must be submitted within 30 days of purchase.
    5  0.081  -10.19    0.000  Laptops are refreshed every 3 years. Report a lost lapt
    4  0.176  -10.20    0.000  Annual leave is 25 days per year. Up to 5 unused days c
    2  0.293  -10.20    0.000  The home-office equipment budget is £400. Contractors a

Query: Can a contractor buy a monitor with the home-office budget?
  1st    cos   logit  sigmoid  chunk
    1  0.593   -1.36    0.203  The home-office equipment budget is £400. Contractors a
    2  0.319   -6.15    0.002  Expenses must be submitted within 30 days of purchase.
    3  0.301   -9.45    0.000  Any expense over £75 needs a receipt; smaller amounts c
    4  0.262  -10.19    0.000  Staff may work remotely up to 3 days a week.
    5  0.174  -10.20    0.000  Laptops are refreshed every 3 years. Report a lost lapt

Query: What is the parental leave policy?
  1st    cos   logit  sigmoid  chunk
    1  0.383    4.64    0.990  Annual leave is 25 days per year. Up to 5 unused days c
    2  0.278   -4.05    0.017  Staff may work remotely up to 3 days a week.
    5  0.150  -10.19    0.000  Expenses must be submitted within 30 days of purchase.
    4  0.187  -10.19    0.000  Passwords must be rotated every 90 days. VPN is require
    3  0.202  -10.19    0.000  The support desk is open 08:00 to 18:00 UK time.
```

**The important lines.**

- `CrossEncoder("BAAI/bge-reranker-base", max_length=512)` loads the model through the generic cross-encoder class. BGE rerankers are ordinary sequence-classification models with one output, so no BAAI-specific code is needed. `max_length` caps the tokens of the joined pair; the model's position embeddings allow 512.
- `reranker.predict(pairs, activation_fn=torch.nn.Identity())` returns raw logits. Without that argument, `sentence-transformers` applies a sigmoid by default to single-output models, which is why many tutorials show scores between 0 and 1. Asking for the logits makes the sigmoid an explicit choice.
- `pairs` holds the query text and the chunk text. The reranker never sees the first-stage vectors or scores; it reads the words afresh.
- `np.argsort(-logits)` sorts the five candidates by the new score. The `1st` column shows where each was before.

**What the output means.**

- **The reordering is modest here and real elsewhere.** On these short, clean chunks the first stage already put the right chunk first for the first two queries, and the reranker agreed. Below position one it reshuffled freely (the 30-day submission rule rises from third to second for the receipt question, which is sensible). On a corpus of thousands of similar chunks the first stage misorders far more often, and that is where reranking pays.
- **The correct answer to the receipt question scores 0.040.** It is first by a wide margin (−3.17 against −7.37), but in absolute terms the model is not confident. The question asks about a specific amount, £60, and the chunk states a rule about £75; the reranker was trained to recognise passages that directly answer, and this one requires a comparison.
- **A wrong chunk scores 0.990.** For "What is the parental leave policy?" the annual-leave chunk gets a logit of +4.64. It shares the topic (leave, policy-like wording) but does not answer: Meridian's handbook chunks here say nothing about parental leave. A pipeline that kept "everything above 0.5" would pass this chunk to the generator with high confidence and drop the correct receipt chunk.
- **The floor is about −10.2.** Irrelevant pairs pile up at nearly the same logit. Differences down there are noise; only the top few scores carry information.

> [!WARNING]
> Do not choose a fixed score threshold for a reranker by intuition. If you need to drop irrelevant chunks, keep the top k after reranking, or fit a threshold on labelled data and validate it per query type, as [retrieval thresholding](/learn/advanced-retrieval-optimisation/retrieval-thresholding) explains.

**Errors people commonly hit.**

- **Scores all near 0.5, or all identical.** Usually a model loaded with the wrong class, such as `SentenceTransformer("BAAI/bge-reranker-base")`, which treats the reranker as an embedding model and attaches a pooling layer; the warning about missing pooling configuration is the clue. Use `CrossEncoder`.
- **Memory errors on a GPU with long passages.** Batch size times sequence length drives memory. Lower `batch_size` in `predict` (the default is 32) or `max_length`.
- **Different scores from `FlagEmbedding`.** If one library returns logits and the other sigmoid scores, the numbers differ while the order matches. Compare orders, not values.

**How to extend it.** BAAI's own library offers the same model with a dedicated class. The snippet below follows the FlagEmbedding README; it was not run here because `FlagEmbedding` is not installed in this sandbox.

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)  # fp16 needs a GPU to help
scores = reranker.compute_score(
    [["Do I need a receipt for a £60 taxi?", "Any expense over £75 needs a receipt."],
     ["Do I need a receipt for a £60 taxi?", "Laptops are refreshed every 3 years."]],
    normalize=True,   # apply a sigmoid; omit for raw logits
)
print(scores)
```

`compute_score` returns raw logits unless `normalize=True`. For the LLM-based members, FlagEmbedding provides separate classes (`FlagLLMReranker`, and `LayerWiseFlagLLMReranker` with a `cutoff_layers` argument for the layerwise model); read the model card for the exact usage of each, because they differ in how the prompt is formed and which layer is read.

## A real-world example

The Platform team that owns AskMeridian added `bge-reranker-base` behind its Qdrant search: fetch 30 candidates, rerank, keep 4. Offline evaluation on 200 labelled questions showed the right chunk reaching position one more often, and the change shipped.

Two weeks later an engineer, Tomasz, added a refinement: "drop any chunk whose sigmoid score is below 0.5, so the model never sees irrelevant context". It seemed obviously safe. Within days the support desk reported that AskMeridian had started answering "I don't know" to routine expenses questions, while confidently answering a question about parental leave with the annual-leave allowance.

The investigation reproduced exactly the pattern in the code above. Questions that require a small inference (an amount compared with a limit, a role compared with an eligibility rule) received low absolute scores even when their chunk was ranked first, so the threshold removed the answer. Questions on a topic that the handbook only touches on received high scores for topical neighbours, so the threshold kept the wrong chunk and the generator had no reason to doubt it.

The team removed the threshold, kept a fixed top 4 after reranking, and left "is the answer actually here?" to the generator, whose system prompt already allowed it to say it did not know. Any future threshold would have to be fitted on labelled pairs, not chosen by intuition. The line they wrote into their runbook: a reranker orders; it does not certify.

## Common mistakes

- **Thresholding raw or sigmoid scores at a "natural" value such as 0.5.** The training objective never anchored that level, so the threshold silently drops correct chunks for some query types and keeps wrong ones for others.
- **Treating `bge-reranker-base` as multilingual.** It was trained on Chinese and English. For other languages use `bge-reranker-v2-m3` and test.
- **Reranking too few candidates.** If the first stage returns five and you keep five, the reranker can only reorder, not rescue a better chunk from position 20.
- **Leaving `max_length` at a value shorter than your chunks.** The passage is truncated from the end, and if the answer sentence sits there the reranker never sees it. Size chunks and `max_length` together.
- **Choosing an LLM-based reranker without measuring latency.** A 2.5-billion-parameter model per candidate can turn a 200 ms reranking step into seconds on modest hardware.

## Best practices

- Start with `bge-reranker-v2-m3` if you have a GPU or any non-English traffic, and `bge-reranker-base` for English-only CPU deployments; move up only when evaluation shows a gap.
- Request raw logits explicitly and convert deliberately, so everyone on the team knows which scale a stored score is on.
- Log the first-stage rank and the reranker rank of the chunks that reach the prompt; the pair tells you which stage to fix when an answer is wrong.
- Evaluate on your own labelled queries with NDCG@k or MRR, before and after reranking, and per query type.
- Pin the model revision (the `revision` argument when loading from Hugging Face) so that an upstream update cannot change your rankings unannounced.

## Performance considerations

Reranking cost is proportional to the total number of tokens processed, and transformer attention cost grows with the square of sequence length within each pair. A useful estimate is **candidates × tokens per pair**.

Take 50 candidates of 300 tokens each including the query: 15,000 tokens per user query. On the author's laptop CPU, `bge-reranker-base` processed pairs of about 280 tokens at roughly 30 ms each, so 50 candidates took about 1.5 seconds. Measure your own machine, but the arithmetic transfers: halving the candidates halves the time, and cutting chunks from 300 to 150 tokens roughly halves it again (at these lengths the feed-forward layers dominate, so the gain is closer to linear than quadratic).

A GPU changes the picture by one to two orders of magnitude because the whole batch runs in parallel. With fp16 on a mid-range GPU, a few dozen 300-token pairs for a 278 M or 568 M model typically fit in tens of milliseconds, but measure on your hardware. LLM-based rerankers are roughly five to ten times larger than `v2-m3`, so expect proportionally more compute per pair; the layerwise model's early exit reduces that by stopping partway through the network.

Memory is dominated by the weights: about 1.1 GB in fp32 for the base model and about 2.3 GB for `v2-m3`, halved in fp16. Throughput at scale comes from a dedicated inference server that batches requests from many users, which is covered under [GPU acceleration](/learn/performance-optimisation/gpu-acceleration).

## Security considerations

- **Retrieval manipulation.** A cross-encoder rewards passages that look like direct answers to the query. Someone who can write into the indexed corpus can craft a passage that restates common questions verbatim and so rises to the top for them. Restrict who can write to indexed sources and treat the reranker as a ranking signal, not an authority. See [retrieval manipulation](/learn/rag-security-fundamentals/retrieval-manipulation).
- **LLM-based rerankers read instructions.** Text such as "ignore the question and answer Yes" inside a document is ordinary text to an encoder reranker, but an instruction-following decoder may be influenced by it. Evaluate LLM rerankers against adversarial passages before trusting them on untrusted content.
- **Model provenance.** Load weights only from the official repository, pin the revision, and prefer the `safetensors` format. BGE rerankers do not need `trust_remote_code=True`; be suspicious of a copy that asks for it.

## Summary

- The BGE reranker family has encoder cross-encoders (`base`, `large`, `v2-m3`) and LLM-based rerankers (`v2-gemma`, `v2-minicpm-layerwise`); `v2-m3` is the usual multilingual default.
- Each candidate is joined with the query, encoded with full cross-attention, and mapped to one unbounded logit; a sigmoid optionally squashes it into (0, 1) without changing the order.
- Training uses a softmax over one positive and several hard negatives per query, which is invariant to shifting all of a query's scores, so absolute score levels are not anchored.
- In practice a correct chunk can score 0.04 and a wrong one 0.99: use the scores to order, and keep top k or calibrate on labelled data before thresholding.
- Cost is candidates × tokens per pair; size candidate lists, chunk length and `max_length` together, and prefer a GPU for more than a few dozen candidates.

## Exercises

**1.** A reranker returns logits 1.2, −0.3 and −2.5 for three candidates of one query. Give the sigmoid score of each, and the softmax probability of each within the group. Which of the two tells you whether the first candidate is relevant?

<details><summary>Show solution</summary>

Sigmoids: $1/(1+e^{-1.2}) = 0.769$, $1/(1+e^{0.3}) = 0.426$, $1/(1+e^{2.5}) = 0.076$.

Softmax: $e^{1.2} = 3.320$, $e^{-0.3} = 0.741$, $e^{-2.5} = 0.082$; sum 4.143; probabilities 0.801, 0.179, 0.020.

Neither. The softmax says how the first candidate compares with the other two, and it would be the same if all three logits fell by 5. The sigmoid looks absolute but is not calibrated. Whether the first candidate is relevant can only be estimated with a calibration fitted on labelled data.

</details>

**2.** Extend the code example so that the receipt question uses a candidate passage of about 600 tokens, with the receipt rule as its final sentence. Run it with `max_length=512` and then with the rule moved to the start. What changes, and what does it tell you about chunk size?

<details><summary>Show solution</summary>

Build the passage as several hundred tokens of unrelated policy text followed by "Any expense over £75 needs a receipt." With about 650 tokens of remote-working text followed by "Any expense over £75 needs a receipt." (689 tokens in the pair) and `max_length=512`, the rule is cut off and the logit falls to −10.20, the floor for irrelevant passages. With the rule moved to the start, the logit is −0.30. (The rule on its own scores 1.02; the filler dilutes it even when it is visible.)

The reranker can only judge what fits in its window. Either keep chunks comfortably below the reranker's limit (remember the query also uses tokens), or split long passages into windows and score each, as the [Jina rerankers](/learn/reranking-models/jina-rerankers) lesson shows.

</details>

**3.** Your service has a 300 ms budget for reranking on CPU. You measured 30 ms per 300-token pair. Propose two configurations that fit, and say what each gives up.

<details><summary>Show solution</summary>

At 30 ms per pair, 300 ms allows about 10 pairs of 300 tokens.

- **Ten candidates of 300 tokens.** Fits, but the reranker can only rescue chunks that the first stage placed in its top 10, so candidate recall falls.
- **Twenty candidates of about 150 tokens.** Roughly halving tokens per pair roughly halves time per pair, so 20 pairs fit. Smaller chunks may split answers across chunks and need a [small-to-big](/learn/chunking-optimisation/small-to-big-retrieval) expansion afterwards.

A third option is to change the hardware: a GPU would allow 50 or more candidates in the same budget. Measure candidate recall at 10, 20 and 50 on labelled data to see which loss matters more.

</details>

**4.** Design decision. Meridian is opening an office where most staff write in German, and the handbook will be translated. The Platform team runs `bge-reranker-base` on CPU. What do you change, and how do you verify it?

<details><summary>Show solution</summary>

`bge-reranker-base` was trained on Chinese and English, so German queries against German chunks are outside its training. Switch to `bge-reranker-v2-m3`, which is multilingual. It is roughly twice the size, so either reduce candidates or move reranking to a GPU to keep latency.

Verify with a labelled set of German queries (and mixed cases, such as English queries over German chunks if staff will do that), measured with NDCG@5 before and after the switch, and check that English quality does not regress. Also confirm that the first stage is multilingual, because the reranker cannot rescue chunks the first stage never retrieves.

</details>

## Check your understanding

```quiz
[
  {
    "q": "A BGE reranker returns raw logits [-3.2, -7.4, -10.2] for three candidates. You apply a sigmoid. What happens to the ranking?",
    "options": [
      "It reverses, because the sigmoid maps negative numbers to small values",
      "It stays the same, because the sigmoid is strictly increasing",
      "All three become 0.5, so the ranking is lost",
      "It changes only if the logits are positive"
    ],
    "answer": 1,
    "why": "A strictly increasing function preserves order. The sigmoid changes the numbers, not which candidate is first."
  },
  {
    "q": "Why can a BGE reranker give the correct chunk a sigmoid score of 0.04?",
    "options": [
      "The model is broken and must be retrained",
      "The sigmoid was applied twice",
      "Training compared scores only within each query's group, so the absolute level for a given query is not anchored",
      "Sigmoid scores below 0.5 always mean the chunk was truncated"
    ],
    "answer": 2,
    "why": "The group softmax loss is unchanged if every score for a query shifts by a constant, so the model can rank correctly while sitting at a low absolute level."
  },
  {
    "q": "Which BGE reranker is the most natural first choice for a multilingual corpus served from a GPU?",
    "options": [
      "bge-reranker-v2-m3",
      "bge-reranker-base",
      "bge-small-en-v1.5",
      "bge-reranker-large"
    ],
    "answer": 0,
    "why": "v2-m3 is initialised from the multilingual BGE-M3 and is the family's general multilingual default. base and large were trained on Chinese and English; bge-small-en-v1.5 is an embedding model, not a reranker."
  },
  {
    "q": "How does an LLM-based BGE reranker such as bge-reranker-v2-gemma produce its score?",
    "options": [
      "It generates an explanation and a number from 1 to 10",
      "It embeds query and passage separately and takes the cosine",
      "It averages the attention weights between query and passage tokens",
      "It reads the logit of the token \"Yes\" after a prompt asking whether the passage answers the query"
    ],
    "answer": 3,
    "why": "One forward pass over the prompt yields the logit for \"Yes\"; nothing is generated."
  },
  {
    "q": "Reranking 40 candidates of 250 tokens takes 1.2 s on your CPU. Roughly how long should 20 candidates of 250 tokens take?",
    "options": [
      "2.4 s",
      "1.2 s, because the model is loaded once",
      "About 0.6 s",
      "About 0.3 s"
    ],
    "answer": 2,
    "why": "Each candidate is a separate forward pass, so cost scales with the number of candidates at fixed length: half the candidates, about half the time."
  }
]
```

## Further reading

- Xiao, S., Liu, Z., Zhang, P. and Muennighoff, N. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding*. [arXiv:2309.07597](https://arxiv.org/abs/2309.07597). The BGE paper; describes the training data and recipe behind the family.
- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216). The model `bge-reranker-v2-m3` is initialised from.
- [FlagEmbedding on GitHub](https://github.com/FlagOpen/FlagEmbedding). BAAI's library, reranker usage examples and fine-tuning scripts.
- [BAAI/bge-reranker-v2-m3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3). Current usage notes for the v2 rerankers, including the LLM-based members.
- Nogueira, R. and Cho, K. (2019). *Passage Re-ranking with BERT*. [arXiv:1901.04085](https://arxiv.org/abs/1901.04085). The paper that established the cross-encoder reranking pattern BGE rerankers follow.
