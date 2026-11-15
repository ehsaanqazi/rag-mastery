Sentence Transformers is the open-source Python library through which most teams run open embedding models. The name refers to two things that are worth keeping apart: a 2019 *method* for turning a BERT-style model into a sentence encoder, and the *library* that grew from it, which today loads thousands of models from the Hugging Face Hub, including the BGE, E5 and Nomic families covered later in this module, behind one `encode` call.

This lesson opens that call up. You will see why the original method was needed, what a Sentence Transformers model actually consists of, what happens to your text between `encode` and the vector that comes back, and which settings (sequence length, pooling, normalisation, prompts, batch size) silently change your retrieval results. By the end you will reproduce the library's output by hand, to within floating-point rounding, and you will be able to read any model's configuration and know how it must be used.

## Learning objectives

- Explain why a plain BERT model is a poor sentence encoder and how the Sentence-BERT bi-encoder fixes the cost problem.
- Name the modules of a Sentence Transformers model (Transformer, Pooling, optional Dense, Normalize) and say what each does to the data.
- Compute masked mean pooling by hand and reproduce `model.encode` with the `transformers` library.
- Configure `encode` correctly: normalisation, batch size, prompts, `max_seq_length` and `truncate_dim`.
- Diagnose silent truncation of long chunks and fix it in the chunking stage.

## Prerequisites

- [OpenAI embeddings](/learn/embedding-models/openai-embeddings): the hosted alternative, and the idea that every vector belongs to one model's space.
- [Sentence embeddings](/learn/embedding-fundamentals/sentence-embeddings): one vector per passage.
- [Transformer architecture](/learn/understanding-large-language-models/transformer-architecture): how a transformer produces one contextual vector per token.
- [Cosine similarity](/learn/embedding-mathematics/cosine-similarity): the score most of these models are trained for.

## Why this matters

When you embed text with an open model, you are no longer calling a service whose behaviour someone else has fixed. You are running the model yourself, and several decisions the service would have made for you are now yours: which pooling the model expects, whether to normalise, how long an input may be, what prefix a query needs. Each wrong choice produces vectors that look perfectly normal and retrieve worse, with no error message.

The most common case is truncation. Every model has a maximum number of tokens it reads; anything beyond is dropped without warning. A chunker that produces 600-token chunks for a model that reads 256 tokens builds an index in which more than half of every chunk is invisible to search. Nothing fails. The answers that live in the second half of a chunk simply never come back.

Sentence Transformers is also the common interface to the rest of this module. BGE, E5 and Nomic models are all distributed as Sentence Transformers models; they differ in pooling, prefixes and length limits, and those differences live in configuration files this library reads. Understanding the library is what lets you switch between model families safely.

## The intuition

Imagine a library with 10,000 index cards and a reader who must find, for each card, the most similar other card.

One way is to put every possible pair of cards on a desk and have an expert read both together and score them. The expert is very accurate because they see both texts at once, but there are almost 50 million pairs. This is what a **cross-encoder** does: one model reads the two texts jointly.

The other way is to give each card a **map reference** once, by reading it alone, and then compare map references, which is just arithmetic. That needs 10,000 readings instead of 50 million. The references are a little less precise than the expert's joint reading, because no card was read with the other in view, but they can be computed in advance and searched in milliseconds. This is a **bi-encoder**, and producing good map references is exactly what Sentence-BERT trained BERT to do.

The library is the machinery that turns a card into its map reference: read the words (the transformer), summarise the reading into one position (pooling), and put it on a common scale (normalisation).

## How it works

### From BERT to Sentence-BERT

A BERT-style transformer outputs one vector per token. To get one vector per sentence, early practitioners took either the output for the special `[CLS]` token or the average of all token vectors. Neither was trained for similarity, and Reimers and Gurevych (2019) showed that such vectors performed poorly on semantic similarity tasks, in some cases worse than averaging older GloVe word vectors.

BERT can score sentence similarity well when given both sentences together, as a cross-encoder, but that does not scale. Their paper's example: finding the most similar pair among 10,000 sentences needs about 50 million BERT computations, roughly 65 hours; with Sentence-BERT it takes about 5 seconds.

Sentence-BERT (SBERT) fine-tunes the transformer in a **siamese** arrangement: the same network, with the same weights, encodes each sentence separately, a pooling layer turns token vectors into one sentence vector, and the training loss compares the two sentence vectors. The original models were trained on natural-language-inference pairs and then on semantic-similarity data. Modern Sentence Transformers models are trained with **contrastive** objectives on very large sets of related pairs (questions and answers, titles and bodies); `all-MiniLM-L6-v2`, used throughout this lesson, was fine-tuned on about a billion sentence pairs according to its model card. Contrastive training itself is covered in [Contrastive learning](/learn/embedding-optimisation/contrastive-learning).

### Anatomy of a model

A `SentenceTransformer` is an ordered list of modules. Printing a model shows them:

1. **Transformer.** Tokenises the text, truncates it to `max_seq_length` tokens, and runs the transformer, producing one vector per token (for MiniLM, 384 numbers per token).
2. **Pooling.** Collapses the token vectors into one vector. The mode is fixed by how the model was trained: `mean` (average of token vectors, ignoring padding), `cls` (the first token's vector), `max`, or `lasttoken` (used by decoder-based models that read left to right).
3. **Dense** (optional). A linear layer that projects to a different dimension. Some models use it; MiniLM does not.
4. **Normalize** (optional). Divides by the L2 norm so that every vector has length 1. If present, dot product and cosine give the same scores.

These modules, and a few settings, are stored next to the weights in a model repository:

- `modules.json` lists the modules in order.
- `sentence_bert_config.json` holds `max_seq_length`.
- `1_Pooling/config.json` holds the pooling mode.
- `config_sentence_transformers.json` holds the default similarity function and any **prompts**: named prefixes such as `"query": "query: "` that the model expects before certain inputs.

Reading these four files tells you how a model must be used. In particular, pooling mismatches cannot occur if you load the model through this library, because the library applies the pooling the model was published with; they occur when someone reimplements the pipeline by hand and guesses.

```mermaid
flowchart LR
    T["Text"] --> P["Prompt prefix<br/>(if configured)"]
    P --> K["Tokenise and truncate<br/>to max_seq_length"]
    K --> X["Transformer:<br/>one vector per token"]
    X --> O["Pooling:<br/>mean / cls / lasttoken"]
    O --> D["Dense projection<br/>(optional)"]
    D --> N["Normalize<br/>(optional)"]
    N --> V["Sentence embedding"]
```

### What `encode` does

Calling `model.encode(texts)` runs these steps:

1. **Prepend a prompt** if you pass `prompt_name` or `prompt`, or call `encode_query` / `encode_document`, which use the model's `query` and `document` prompts when it defines them (these two methods were added in version 5).
2. **Sort the inputs by length** so that each batch holds texts of similar length, which minimises padding; the results are put back in your original order.
3. **Tokenise each batch** of `batch_size` texts (default 32), padding to the longest text in the batch and truncating at `max_seq_length`.
4. **Run the modules** in order on the batch.
5. **Post-process**: optionally normalise (`normalize_embeddings=True`), optionally keep only the first `truncate_dim` dimensions (meaningful only for Matryoshka-trained models), optionally quantise (`precision="int8"` or `"binary"`), and convert to a NumPy array or a PyTorch tensor.

`model.similarity(a, b)` then scores every row of `a` against every row of `b` with the model's configured function (`cosine` unless the model says otherwise; `dot`, `euclidean` and `manhattan` are the alternatives). Using it rather than hand-written cosine code means you automatically use the function the model was trained for.

### Truncation: the silent limit

`max_seq_length` is the number of **tokens** (word pieces, not words) the model reads. For `all-MiniLM-L6-v2` it is 256, and its model card notes that it was trained with shorter inputs, 128 tokens. The library truncates without warning. The practical rule is that the chunk size in your [chunking pipeline](/learn/chunking-fundamentals/chunk-size) must be measured in the embedding model's own tokens and kept under the limit, preferably well under it, since quality often degrades before the hard limit is reached for models trained on short texts.

You can raise `model.max_seq_length` up to the transformer's position limit (512 for BERT-style models), but a model trained on short texts does not necessarily produce good vectors for long ones. If you need long chunks, choose a model trained for them; [Nomic embeddings](/learn/embedding-models/nomic-embeddings) is one such family.

### The wider library

A few more parts are worth knowing about, each owned by a later lesson:

- **`CrossEncoder`** loads rerankers that score a query and a passage jointly, the "expert at the desk" from the intuition. See [Cross-encoders](/learn/reranking-fundamentals/cross-encoders).
- **Training.** `SentenceTransformerTrainer` and a catalogue of losses fine-tune any model on your own pairs. See [Fine-tuning embeddings](/learn/embedding-optimisation/fine-tuning-embeddings).
- **Sparse encoders.** Version 5 added a `SparseEncoder` class for learned sparse models in the SPLADE family; see [Sparse representations](/learn/sparse-retrieval/sparse-representations).
- **Faster inference.** `SentenceTransformer(name, backend="onnx")` or `backend="openvino"` runs the transformer through those runtimes, which can speed up CPU inference. [Local embedding models](/learn/embedding-models/local-embedding-models) covers serving and hardware.

## The mathematics

### Masked mean pooling

**Intuition.** A sentence vector should be the average of its real tokens' vectors. Padding tokens, added only so that a batch forms a rectangle, must not take part, or a short sentence's vector would depend on how long its batch-mates happened to be.

$$
\mathbf{s} = \frac{\sum_{i=1}^{L} m_i \, \mathbf{h}_i}{\sum_{i=1}^{L} m_i},
\qquad
\hat{\mathbf{s}} = \frac{\mathbf{s}}{\lVert \mathbf{s} \rVert}
$$

$L$ is the padded sequence length, $\mathbf{h}_i$ is the transformer's output vector for position $i$, $m_i$ is the attention mask (1 for a real token, 0 for padding), $\mathbf{s}$ is the pooled sentence vector and $\hat{\mathbf{s}}$ its unit-length version.

**By hand.** Take 2-dimensional token vectors $\mathbf{h}_1 = (1, 2)$, $\mathbf{h}_2 = (3, 0)$, $\mathbf{h}_3 = (2, 4)$ and a padding position $\mathbf{h}_4 = (9, -9)$ with mask $(1, 1, 1, 0)$.

- Masked sum: $(1 + 3 + 2,\; 2 + 0 + 4) = (6, 6)$. Divide by 3: $\mathbf{s} = (2, 2)$.
- Length: $\sqrt{4 + 4} = 2.8284$, so $\hat{\mathbf{s}} = (0.7071, 0.7071)$.
- A naive mean over all four positions gives $(15/4,\; -3/4) = (3.75, -0.75)$: a different direction, caused entirely by padding.

### Why bi-encoders scale

Comparing every pair among $n$ texts with a cross-encoder needs one model pass per pair, $n(n-1)/2$ passes. A bi-encoder needs $n$ passes, after which each comparison is a dot product. For $n = 10{,}000$: $10{,}000 \times 9{,}999 / 2 = 49{,}995{,}000$ passes against $10{,}000$. In retrieval the asymmetry is even sharper, because document vectors are computed once at indexing time and only the query is encoded at run time.

```python
import numpy as np

H = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0], [9.0, -9.0]])  # last row is padding
m = np.array([1, 1, 1, 0])

masked = (H * m[:, None]).sum(axis=0) / m.sum()
naive = H.mean(axis=0)
print("masked mean:", masked, "-> unit:", np.round(masked / np.linalg.norm(masked), 4))
print("naive mean :", naive)

n = 10_000
print("cross-encoder passes for all pairs:", n * (n - 1) // 2)
print("bi-encoder passes:", n)
```

```output
masked mean: [2. 2.] -> unit: [0.7071 0.7071]
naive mean : [ 3.75 -0.75]
cross-encoder passes for all pairs: 49995000
bi-encoder passes: 10000
```

## In code

**The problem.** The AskMeridian team wants to run embeddings locally, and wants to understand exactly what the library does before trusting it. The script below loads `all-MiniLM-L6-v2`, embeds a few handbook sentences, reproduces the library's vectors by hand with the `transformers` library, runs three searches, and demonstrates silent truncation.

**Dependencies.** The first run downloads the model (about 90 MB) from the Hugging Face Hub and caches it.

```bash
pip install sentence-transformers numpy
```

**The source.**

```python
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

NAME = "sentence-transformers/all-MiniLM-L6-v2"
torch.manual_seed(0)

model = SentenceTransformer(NAME, device="cpu")
print(model)
print("max_seq_length:", model.max_seq_length, "| similarity:", model.similarity_fn_name)

docs = [
    "Annual leave is 25 days per year.",
    "Up to 5 unused leave days carry over and must be used by 31 March.",
    "Staff may work remotely up to 3 days a week.",
    "The home-office equipment budget is £400.",
    "Passwords must be rotated every 90 days.",
    "VPN is required on public Wi-Fi.",
]

# 1. The library's pipeline
lib = model.encode(docs, batch_size=32, normalize_embeddings=True)

# 2. The same pipeline by hand: tokenise -> transformer -> mean pool -> normalise
tok = AutoTokenizer.from_pretrained(NAME)
bert = AutoModel.from_pretrained(NAME).eval()
batch = tok(docs, padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    tokens = bert(**batch).last_hidden_state          # (6, seq_len, 384)
mask = batch["attention_mask"].unsqueeze(-1).float()  # 1 for real tokens, 0 for padding
pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1)
manual = torch.nn.functional.normalize(pooled, dim=1).numpy()
print("token tensor:", tuple(tokens.shape), f"| max abs difference: {np.abs(lib - manual).max():.1e}")

# 3. Search
queries = ["How much holiday do I get?", "Can I work from home?", "Do I need a VPN in a café?"]
q = model.encode(queries, normalize_embeddings=True)
scores = model.similarity(q, lib)                      # cosine, a (3, 6) tensor
for query, row in zip(queries, scores):
    best = int(row.argmax())
    print(f"{query:<28} -> {float(row[best]):.3f}  {docs[best]}")

# 4. Silent truncation
filler = "This section describes general policy background. " * 60
long_doc = filler + "Laptops are refreshed every 3 years."
print("tokens in long_doc:", len(tok(long_doc)["input_ids"]), "| model keeps:", model.max_seq_length)
cands = model.encode([long_doc, "Laptops are refreshed every 3 years."], normalize_embeddings=True)
lq = model.encode(["How often are laptops replaced?"], normalize_embeddings=True)
print("score vs long doc:", round(float(model.similarity(lq, cands)[0, 0]), 3),
      "| score vs short doc:", round(float(model.similarity(lq, cands)[0, 1]), 3))
```

The output below comes from sentence-transformers 6.3 on a CPU; the module printout's exact wording varies between library versions.

```output
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
max_seq_length: 256 | similarity: cosine
token tensor: (6, 18, 384) | max abs difference: 1.5e-08
How much holiday do I get?   -> 0.357  Annual leave is 25 days per year.
Can I work from home?        -> 0.381  Staff may work remotely up to 3 days a week.
Do I need a VPN in a café?   -> 0.680  VPN is required on public Wi-Fi.
tokens in long_doc: 432 | model keeps: 256
score vs long doc: 0.045 | score vs short doc: 0.731
```

**The important lines.**

- `print(model)` is the first thing to run on any unfamiliar model. Here it shows three modules: a BERT transformer, **mean** pooling over 384-dimensional token vectors, and a Normalize module. A BGE model printed the same way shows `cls` pooling instead; you will see it in [BGE models](/learn/embedding-models/bge-models).
- `tok(docs, padding=True, truncation=True, max_length=256)` pads the batch to its longest sentence (18 tokens here, including the special `[CLS]` and `[SEP]` tokens) and would cut anything beyond 256.
- `(tokens * mask).sum(dim=1) / mask.sum(dim=1)` is the masked-mean formula from the mathematics section, applied to a whole batch at once.
- `model.similarity(q, lib)` uses the model's configured function, cosine, and returns a queries-by-documents matrix.

**What the output means.**

- The hand-built vectors match the library's to about $10^{-8}$, which is floating-point rounding. There is no hidden step: tokenise, transform, pool, normalise.
- All three queries find the right sentence, including "holiday" to "annual leave" and "work from home" to "work remotely", which share no content words. The absolute scores (0.36 to 0.68) are not probabilities and not comparable with another model's; only the ranking matters.
- The long document holds the answer to the laptop question at its end, after 432 tokens. The model read only the first 256, all filler, so the query scored 0.045 against it and 0.731 against the same sentence on its own. In a real index that chunk would never be retrieved for the laptop question.

**Common errors.**

- Forgetting `normalize_embeddings=True` with a model that has no Normalize module, then storing vectors in a database configured for dot product: longer vectors win regardless of meaning. Check `print(model)`, or normalise anyway.
- Mismatched models between indexing and querying, for example after someone updates the model name in one service only. Store the model name with the index.
- Running out of memory on a GPU with long inputs: memory grows with `batch_size` times sequence length; lower the batch size.

**Extending it.** Swap `NAME` for another model and rerun: the printout tells you its pooling, and `model.prompts` tells you whether it expects query and document prefixes. Then measure `max_seq_length` against your real chunk-length distribution, using the model's own tokeniser.

## A real-world example

Meridian's Platform team moved AskMeridian's IT-runbook index from a hosted model to `all-MiniLM-L6-v2` running inside the ingestion workers, to keep runbook content on Meridian's own machines. Offline tests on the employee handbook looked good. Two weeks after launch, the support desk reported that AskMeridian could not answer questions about Atlas error codes that were plainly in the runbooks.

The cause was the chunker. It had been configured for the previous hosted model, whose input limit was in the thousands of tokens, and produced chunks of about 800 tokens. Each Atlas runbook page began with a long standard preamble (scope, owners, escalation contacts) and listed error codes such as E-4021 near the end. MiniLM read the first 256 tokens of each chunk, which were almost all preamble, so every runbook chunk embedded as "a generic runbook" and the error-code sections were invisible to dense search.

The team fixed it at the source rather than in the model: chunk size was set in MiniLM tokens with a ceiling of 200, the preamble was split into its own chunk, and the ingestion pipeline gained a check that logs any chunk exceeding the model's `max_seq_length`. They also added runbook questions to their evaluation set, which until then had held only handbook questions, so that the next change of model or chunker would be tested against the content that had broken.

## Common mistakes

- **Chunk sizes set in words or characters, or in another model's tokens.** The model counts its own word pieces; the excess is dropped silently. Measure with the model's tokeniser.
- **Reimplementing the pipeline with the wrong pooling.** Using CLS pooling on a mean-pooled model, or the reverse, gives vectors that are subtly wrong. Load through the library, or read `1_Pooling/config.json`.
- **Ignoring prompts.** Models that define `query` and `document` prompts expect them. Use `encode_query` and `encode_document`, which apply them when present.
- **Assuming normalisation.** Not every model ends in a Normalize module. Pass `normalize_embeddings=True` when your index uses dot product.
- **Using `truncate_dim` on a model not trained for it.** Truncating an ordinary model's vectors discards information in no particular order; quality drops much faster than with a Matryoshka-trained model.
- **Treating score values as portable.** A threshold of 0.5 means something different for every model. Calibrate per model.

## Best practices

- Print the model and read `model.max_seq_length`, `model.prompts` and `model.similarity_fn_name` before writing any indexing code.
- Use `encode_query` for queries and `encode_document` for chunks, even for models without prompts, so that switching to a model with prompts needs no code change.
- Count chunk lengths with the embedding model's tokeniser and fail or warn in ingestion when a chunk exceeds the limit.
- Pin the model by name and revision (the `revision` argument accepts a commit hash) so that an upstream update cannot change your vectors under you.
- Keep a small evaluation set covering every document type in the index, and rerun it whenever the model, chunker or settings change.

## Performance considerations

**Compute.** Transformer cost grows with model size and, because of attention, faster than linearly with sequence length. MiniLM-L6 has 6 layers and about 22 million parameters; a "base" model typically has 12 layers and around 110 million; a "large" one 24 layers and around 335 million. As a rough rule, encoding cost scales with parameter count, so a large model is something like 15 times the work of MiniLM per token.

**Padding.** The library sorts inputs by length before batching. If you batch yourself, sort too: a batch with one 256-token text and thirty-one 20-token texts pads everything to 256, and $31 \times 236 / (32 \times 256) \approx 0.89$: almost 90% of the positions computed are padding.

**Batch size.** Larger batches use a GPU more efficiently up to the memory limit; on CPU the gain levels off quickly. Tune it by measuring throughput at a few values.

**Memory.** At 384 dimensions in `float32`, a vector is 1,536 bytes; a million chunks is about 1.5 GB before index overhead. `precision="int8"` or `"binary"` in `encode` shrinks this; see [Embedding quantisation](/learn/embedding-optimisation/embedding-quantisation).

## Security considerations

- **Remote code.** Some models on the Hub need `trust_remote_code=True`, which downloads and runs Python code from the model repository on your machine. Read that code, pin the `revision`, and prefer models that do not need it.
- **Supply chain.** Load weights in the `safetensors` format where available, which cannot execute code when loaded, unlike older pickle-based formats. Mirror approved models into an internal registry rather than downloading from the public Hub in production.
- **Data locality.** Running models locally keeps document text on your machines, which is often the reason for choosing open models; make sure logging and caching do not undo that by writing chunk text somewhere less protected.

## Summary

- Sentence-BERT made BERT a practical sentence encoder by training it as a siamese bi-encoder, replacing a quadratic number of joint model passes with one pass per text.
- A Sentence Transformers model is a list of modules: Transformer, Pooling, optional Dense and optional Normalize, configured by files in the model repository.
- `encode` adds prompts, sorts by length, tokenises and truncates, runs the modules, then optionally normalises, truncates dimensions and quantises. You can reproduce it exactly with `transformers` and masked mean pooling.
- `max_seq_length` truncates silently; chunk sizes must be measured in the model's own tokens.
- The library is the common front door for BGE, E5, Nomic and many other open models; their differences are in configuration you can read.

## Exercises

1. Masked mean pooling: a batch has token vectors $(2, 0)$, $(0, 2)$, $(4, 4)$ and two padding positions $(10, 0)$ and $(10, 0)$. Compute the masked mean, its unit vector, and the naive mean over all five positions.

<details><summary>Show solution</summary>

Masked sum: $(2 + 0 + 4,\; 0 + 2 + 4) = (6, 6)$; divided by 3: $(2, 2)$; unit vector $(0.7071, 0.7071)$. Naive mean: $(26/5,\; 6/5) = (5.2, 1.2)$, which points mostly along the first axis purely because of padding.

</details>

2. Write a function `check_chunks(model, chunks)` that returns the indices of chunks longer than the model's `max_seq_length` in the model's own tokens. Run it on the `docs` list plus `long_doc` from the code above.

<details><summary>Show solution</summary>

```python
def check_chunks(model, chunks):
    limit = model.max_seq_length
    lengths = [len(model.tokenizer(c)["input_ids"]) for c in chunks]
    return [i for i, n in enumerate(lengths) if n > limit]

print(check_chunks(model, docs + [long_doc]))   # [6]
```

The count includes the special tokens the tokeniser adds, which also occupy positions in the model's window. Run this in ingestion and either re-split or log any chunk it reports.

</details>

3. A colleague loads `all-MiniLM-L6-v2` and calls `model.encode(chunks, truncate_dim=128)` to save memory. Explain what will happen to retrieval quality and what you would recommend instead.

<details><summary>Show solution</summary>

`truncate_dim` keeps the first 128 of 384 dimensions. MiniLM was not trained with a Matryoshka objective, so its information is not concentrated in the leading dimensions, and keeping a third of them discards information indiscriminately; quality is likely to drop noticeably. (Normalising again after truncation is also needed for cosine scores to be comparable.) Recommend either a model trained for truncation, such as Nomic's v1.5 or the OpenAI v3 models, or keep 384 dimensions and reduce memory with int8 or binary quantisation, measuring recall on an evaluation set in each case.

</details>

4. Design decision: AskMeridian's handbook chunks average 180 tokens, but 5% of IT-runbook chunks exceed 400 tokens because they contain long command listings. Propose two options and the trade-offs of each.

<details><summary>Show solution</summary>

Option one: keep MiniLM and split the long runbook chunks (for example by command block, with the section heading repeated in each piece). It keeps the fast model and the existing index, but small pieces may lose context, which [Parent-child chunking](/learn/chunking-optimisation/parent-child-chunking) can restore at generation time. Option two: switch to a model with a longer window, such as one with 512 or 8,192 tokens. It handles long chunks natively but costs more compute per token, requires re-embedding everything, and long single vectors can blur several topics together. Measure both on runbook questions before choosing.

</details>

## Check your understanding

```quiz
[
  {
    "q": "Why did Sentence-BERT train BERT as a bi-encoder rather than use it as a cross-encoder for similarity search?",
    "options": ["Cross-encoders cannot score sentence pairs", "A bi-encoder encodes each text once, so comparing n texts needs n passes instead of a pass per pair", "Bi-encoders are always more accurate", "BERT cannot be fine-tuned"],
    "answer": 1,
    "why": "Joint scoring needs a model pass per pair, which grows quadratically; independent encoding plus vector arithmetic does not."
  },
  {
    "q": "A model's printout shows `Pooling({'pooling_mode': 'cls', ...})` and no Normalize module. Which is true?",
    "options": ["Its vectors may not have unit length, so normalise them if your index uses dot product", "It averages all token vectors", "It cannot be used for retrieval", "Its vectors are always unit length"],
    "answer": 0,
    "why": "CLS pooling takes the first token's vector; without a Normalize module, lengths vary unless you pass `normalize_embeddings=True`."
  },
  {
    "q": "A chunk is 600 tokens long and the model's `max_seq_length` is 256. What happens when it is encoded?",
    "options": ["An exception is raised", "The library splits it into three vectors", "The model averages over all 600 tokens", "Only the first 256 tokens are read; the rest is ignored without warning"],
    "answer": 3,
    "why": "Truncation is silent; content beyond the limit has no influence on the vector."
  },
  {
    "q": "Why must padding positions be excluded from mean pooling?",
    "options": ["They are too slow to process", "They contain the answer", "Otherwise a text's vector would depend on the lengths of the other texts in its batch", "The tokeniser cannot produce them"],
    "answer": 2,
    "why": "Padding is added to fill the batch; including it shifts the average by an amount that depends on batch composition."
  },
  {
    "q": "What do `encode_query` and `encode_document` do for a model whose configuration defines `query` and `document` prompts?",
    "options": ["They prepend the corresponding prompt text before encoding", "They use two different neural networks", "They change the pooling mode", "They normalise only the queries"],
    "answer": 0,
    "why": "They select the named prompt from the model's configuration and prepend it; for models without prompts they behave like `encode`."
  }
]
```

## Further reading

- Reimers, N. and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP-IJCNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
- [Sentence Transformers documentation](https://www.sbert.net/): usage, pretrained models, prompts, training and efficiency guides.
- Wang, W. and colleagues (2020). "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers." NeurIPS 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957).
- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
