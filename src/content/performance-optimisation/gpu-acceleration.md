Every model in a RAG system is a stack of matrix multiplications: the embedding model that turns chunks into vectors, the cross-encoder that reranks candidates, the language model that writes the answer, and even brute-force vector search, which is one large matrix product. A **graphics processing unit (GPU)** is hardware built to do enormous numbers of such multiplications in parallel. Used well, it turns an overnight re-embedding job into a coffee break and a two-second rerank into a few hundred milliseconds. Used badly, it is an expensive accelerator that sits idle while single requests trickle through it.

This lesson explains what a GPU is good at and why, where it pays in a RAG pipeline and where it does not, and the levers that decide how much of its capacity you actually use: batching, padding, numeric precision and memory. You will benchmark a real embedding model and a real reranker on CPU and GPU, and do the arithmetic for embedding a corpus and for serving a language model.

## Learning objectives

- Explain why GPUs accelerate transformer inference and vector search, and why they need large batches of parallel work to do so.
- Identify which RAG stages benefit from a GPU (bulk embedding, reranking, self-hosted generation, large-scale vector search) and which usually do not (single-query embedding, small indexes).
- Measure CPU and GPU throughput and latency correctly, including warm-up and asynchronous execution.
- Estimate embedding time from model size and corpus size, and GPU memory for serving a language model, including the KV cache.
- Choose numeric precision and a serving approach (in-process, inference server, hosted API) for a given workload, and verify that reduced precision does not change rankings.

## Prerequisites

- [Model inference](/learn/understanding-large-language-models/model-inference): prefill, decoding, the KV cache and batching.
- [Async execution](/learn/performance-optimisation/async-execution): micro-batching concurrent requests.
- [Batch processing](/learn/performance-optimisation/batch-processing): building large batches for offline work.
- [Cross-encoders](/learn/reranking-fundamentals/cross-encoders): why reranking is expensive per candidate.
- [Flat indexes](/learn/vector-indexing-algorithms/flat-indexes): brute-force search as a matrix product.

## Why this matters

Three moments in a RAG system's life make compute the bottleneck.

- **Re-embedding the corpus.** Changing the embedding model, the chunking or the [embedding dimensions](/learn/embedding-optimisation/matryoshka-embeddings) means embedding everything again. For a million chunks with a large model on CPU, that can take most of a day, which turns every experiment into an overnight job and every model upgrade into a project.
- **Reranking on every query.** A cross-encoder reads the query and each candidate together, so reranking 100 candidates is 100 transformer passes per question. On CPU this can cost seconds, which is outside any interactive latency budget.
- **Self-hosting the language model.** Data-residency rules, cost at high volume or the need for a fine-tuned model can push a team to run its own generator. Without a GPU this is impractical for anything but small models, and with one, memory rather than speed is usually the limit.

But GPUs are rented by the hour whether busy or idle, and they do not help everything: query embedding sent one query at a time can cost far more for no measurable gain. The skill is choosing which workloads to send and keeping the GPU busy.

## The intuition

A CPU is a small team of expert chefs. Each can cook anything, switches tasks quickly and finishes a single complicated dish fast. A GPU is a vast canteen kitchen with hundreds of cooks who can each only follow simple instructions, all doing the same step at the same moment: everyone chops, then everyone stirs.

For a single omelette, the expert wins: the canteen needs time to hand out instructions and fetch ingredients, and most of its cooks stand idle. For two thousand identical omelettes, the canteen wins by a wide margin. The ingredients also have to be carried from the stores to the canteen and the dishes carried back; if each delivery is one egg, the walking dominates.

That is the whole story of GPU acceleration in miniature: enormous parallel throughput, a fixed cost per batch of work, data transfer between the CPU's memory and the GPU's, and a strong need for large, uniform batches.

## How it works

### Why GPUs suit transformers

A transformer layer is dominated by matrix multiplications: projecting token vectors into queries, keys and values, mixing them with attention, and passing them through feed-forward layers. A matrix multiplication of an $m \times k$ matrix by a $k \times n$ one is $m \times n$ independent dot products, which is exactly the kind of uniform, independent work a GPU's thousands of arithmetic units can do simultaneously. GPUs also have far higher **memory bandwidth** (bytes per second between their memory and their arithmetic units) than CPUs, which matters because many operations are limited by how fast the weights and activations can be read rather than by arithmetic.

Two consequences follow:

1. **Parallel work must be plentiful.** One short query is a small matrix (a dozen tokens by a few hundred dimensions). It cannot keep thousands of units busy, so the GPU is mostly idle and the fixed costs (launching each of many small GPU programs, called **kernels**, and moving data) dominate. A batch of 128 passages of 200 tokens is a 25,600-row matrix, enough to fill the hardware.
2. **Execution is asynchronous.** When Python launches GPU work, the call returns before the work finishes; the CPU queues kernels and moves on. Timing code must wait for the GPU to finish (`torch.cuda.synchronize()` on NVIDIA, `torch.mps.synchronize()` on Apple silicon), or it measures only the launch. Copying results back to the CPU, as `encode` does when it returns NumPy arrays, also forces a wait.

### Where a GPU pays in RAG

| Stage | Workload shape | GPU benefit |
| --- | --- | --- |
| Bulk embedding at ingestion | Millions of chunks, batchable | Large: the classic GPU job |
| Query embedding | One short text per request | Small unless micro-batched at high traffic |
| Cross-encoder reranking | Tens to hundreds of pairs per query, batchable within the query | Large: often the difference between seconds and milliseconds |
| Self-hosted generation | Long sequential decoding, many concurrent users | Essential; memory is the main constraint |
| Brute-force or IVF vector search | Large indexes, batches of queries | Large at scale; little for small indexes or single queries |
| OCR and layout models at ingestion | Images, batchable | Large |

Query embedding is the instructive exception. A small model such as MiniLM embeds one query in a few milliseconds on a modern CPU; the GPU cannot make that much faster, because the work is too small to fill it. GPUs help query embedding only when a server [micro-batches](/learn/performance-optimisation/async-execution) many concurrent queries together.

### Batching, padding and sorting

Transformers process a batch as one rectangular tensor, so every sequence in a batch is **padded** to the length of the longest. A batch with one 512-token passage and 127 passages of 50 tokens costs as much as 128 passages of 512 tokens. Two practices limit the waste:

- **Sort by length before batching**, so each batch contains similar lengths. Sentence Transformers' `encode` does this internally and restores the original order afterwards; if you write your own batching loop, do it yourself.
- **Cap sequence length** at what the model was trained for (`max_seq_length`); longer inputs are truncated anyway, and attention cost grows with the square of length.

Batch size has a sweet spot. Too small and the GPU idles between kernel launches; too large and memory runs out (an out-of-memory error), or throughput plateaus while each batch takes longer, which hurts latency in an online service. Measure throughput at a few sizes and pick the knee of the curve, as the [async execution](/learn/performance-optimisation/async-execution) lesson did for micro-batches.

### Precision

Models are trained and stored in 32-bit floating point (fp32) or, increasingly, 16-bit formats. Running inference in 16-bit (**fp16**, or **bf16**, which keeps fp32's range with less precision) halves memory and on most GPUs speeds up matrix multiplication substantially, because GPUs have dedicated hardware for low-precision maths. For embeddings and reranking, the small numeric changes rarely alter rankings, but "rarely" is not "never": verify on your own data by comparing rankings at full and reduced precision, as the main example does. Going further, 8-bit and 4-bit **quantisation** of model weights (covered in [Quantisation](/learn/cost-engineering/quantisation)) is standard for self-hosted language models, where it lets a larger model fit in GPU memory. Quantising stored *vectors* is a separate technique, covered in [Embedding quantisation](/learn/embedding-optimisation/embedding-quantisation).

### Memory: the real constraint for generation

Generating text has two phases, as [Model inference](/learn/understanding-large-language-models/model-inference) explained. **Prefill** reads the whole prompt in parallel and is limited by arithmetic. **Decoding** produces one token at a time, and for each token the GPU must read every weight of the model from memory, so it is limited by memory bandwidth; a single sequence leaves the arithmetic units mostly idle. Batching many users' sequences together shares each read of the weights across them, which is why serving systems batch aggressively.

Batching needs memory. Besides the weights, every sequence needs a **KV cache**: the stored keys and values for every token so far, in every layer. RAG prompts are long (several thousand tokens of retrieved context), so the KV cache per request is large, and it, not arithmetic, usually caps how many requests a GPU can serve at once. Inference servers such as **vLLM** manage this memory carefully: its PagedAttention technique (Kwon and colleagues, 2023) stores the cache in fixed-size blocks, like an operating system's virtual memory pages, so memory is not wasted on fragmentation, and its continuous batching adds and removes sequences at every decoding step.

### Serving models on GPUs

How a model reaches the GPU matters as much as the GPU itself:

- **In-process.** The web process loads the model onto the GPU and calls it directly. Simple, but each process holds its own copy of the weights in GPU memory, requests are batched only within one process, and a GPU fault takes the web process down.
- **An inference server.** A dedicated service owns the GPU and batches requests from every client: Hugging Face **Text Embeddings Inference** for embedding and reranking models, **vLLM** or similar for generation, **NVIDIA Triton** for general models. The web tier makes network calls, as [Async execution](/learn/performance-optimisation/async-execution) recommends, and the GPU stays busy because it sees everyone's traffic. This is established practice in production.
- **A hosted API.** The provider owns the GPUs. For generation with a model such as Claude, or for hosted embedding and rerank APIs, the GPU question becomes a question of rate limits, latency and price. Many teams use a hosted generator and self-host only embedding and reranking, which are small enough to run cheaply.

### Vector search on GPUs

Brute-force search computes a query-by-corpus matrix product, which GPUs do very well, and IVF and graph indexes have GPU implementations too. FAISS has GPU versions of its flat, IVF and product-quantisation indexes, and NVIDIA's cuVS library provides GPU-native indexes including a graph index called CAGRA; some vector databases, such as Milvus, offer GPU index types. GPU search pays when the index is large, when queries arrive in batches (offline evaluation, deduplication, building nearest-neighbour graphs) or when index construction is the bottleneck. For a typical RAG service with a few million vectors and one query at a time, a well-tuned CPU [HNSW](/learn/vector-indexing-algorithms/hnsw) index already answers in milliseconds, and the GPU adds cost and memory limits (the index must fit in GPU memory) for little latency gain.

The FAISS calls to move an index to an NVIDIA GPU are shown below. They need the `faiss-gpu` build and a CUDA device and were not executed for this lesson.

```python
import faiss
import numpy as np

d = 384
xb = np.random.default_rng(0).standard_normal((1_000_000, d), dtype=np.float32)
cpu_index = faiss.IndexFlatIP(d)
cpu_index.add(xb)

res = faiss.StandardGpuResources()                      # GPU memory and streams for FAISS
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)   # copy the index to GPU 0
scores, ids = gpu_index.search(xb[:1000], 10)           # a batch of 1,000 queries
```

## The mathematics

### How long to embed a corpus

**Intuition.** A transformer forward pass costs roughly two floating-point operations (one multiply, one add) per model parameter per token. Multiply by the number of tokens in the corpus and divide by the hardware's effective speed.

$$
T \approx \frac{2 \, P \, N_{\text{chunks}} \, L}{F_{\text{eff}}}
$$

$P$ is the number of parameters, $N_{\text{chunks}}$ the number of chunks, $L$ the tokens per chunk and $F_{\text{eff}}$ the effective throughput in floating-point operations per second (FLOP/s), which is the hardware's peak multiplied by the fraction you actually achieve. The approximation ignores attention's extra cost for long sequences and padding waste, so treat it as a lower bound.

**By hand.** MiniLM-L6 has about 22.7 million parameters. A million chunks of 256 tokens is $2.56 \times 10^8$ tokens, so the work is $2 \times 2.27 \times 10^7 \times 2.56 \times 10^8 \approx 1.16 \times 10^{16}$ FLOPs. At an assumed effective 0.5 TFLOP/s ($5 \times 10^{11}$) on a CPU, that is about 23,000 s, roughly 6.5 hours. At an assumed effective 20 TFLOP/s on a GPU, about 580 s, under 10 minutes. A model ten times larger takes ten times longer on either.

### GPU memory for serving a language model

**Intuition.** Memory holds the weights once, plus a KV cache for every token of every sequence in flight.

$$
M \approx P \, b_w + n_{\text{seq}} \, T_{\text{seq}} \, (2 \, n_{\text{layers}} \, n_{\text{kv}} \, d_{\text{head}} \, b_{kv})
$$

$b_w$ is bytes per weight (2 for 16-bit), $n_{\text{seq}}$ the number of concurrent sequences, $T_{\text{seq}}$ tokens per sequence, $n_{\text{layers}}$ the number of layers, $n_{\text{kv}}$ the number of key-value heads (fewer than attention heads in models that use grouped-query attention), $d_{\text{head}}$ the dimension per head and $b_{kv}$ bytes per cached value. The factor 2 counts keys and values. Activations and framework overhead come on top.

**By hand.** Take a hypothetical 8-billion-parameter model with 32 layers, 8 KV heads of dimension 128, in 16-bit. Weights: $8 \times 10^9 \times 2 = 16$ GB. KV cache per token: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, 128 KiB. A 4,000-token RAG request needs about 0.52 GB. On a 24 GB GPU, $24 - 16 = 8$ GB remain, enough for about 15 such requests at once, before overheads. Halving the retrieved context roughly doubles the concurrency; 4-bit weights (4 GB) free another 12 GB.

```python
# 1. Compute to embed a corpus: about 2 FLOPs per parameter per token.
params = 22.7e6                 # all-MiniLM-L6-v2, roughly
chunks, tokens = 1_000_000, 256
flops = 2 * params * chunks * tokens
for name, tflops in [("CPU, assumed 0.5 TFLOP/s effective", 0.5), ("GPU, assumed 20 TFLOP/s effective", 20)]:
    print(f"{name}: {flops / (tflops * 1e12) / 60:6.1f} minutes for {flops:.2e} FLOPs")

# 2. Memory to serve a hypothetical 8B-parameter model with grouped-query attention.
weights_gb = 8e9 * 2 / 1e9      # 16-bit weights
layers, kv_heads, head_dim, bytes_ = 32, 8, 128, 2
kv_per_token = 2 * layers * kv_heads * head_dim * bytes_      # keys and values
prompt = 4_000                  # a typical RAG prompt plus answer, in tokens
per_seq_gb = kv_per_token * prompt / 1e9
gpu_gb = 24
print(f"weights {weights_gb:.0f} GB; KV cache {kv_per_token / 1024:.0f} KiB per token, "
      f"{per_seq_gb:.2f} GB per {prompt}-token sequence")
print(f"sequences that fit beside the weights on a {gpu_gb} GB GPU: {int((gpu_gb - weights_gb) / per_seq_gb)}")
```

```output
CPU, assumed 0.5 TFLOP/s effective:  387.4 minutes for 1.16e+16 FLOPs
GPU, assumed 20 TFLOP/s effective:    9.7 minutes for 1.16e+16 FLOPs
weights 16 GB; KV cache 128 KiB per token, 0.52 GB per 4000-token sequence
sequences that fit beside the weights on a 24 GB GPU: 15
```

The throughput figures are assumptions for illustration; real effective rates depend heavily on the hardware, the batch size and the software stack, so measure.

## In code

**The problem.** The Platform team is deciding whether AskMeridian's ingestion and reranking should move to GPUs. Before renting hardware they benchmark what they can on a development laptop with an Apple-silicon GPU, which PyTorch exposes as the `mps` device; the same code uses an NVIDIA GPU (`cuda`) if one is present. They measure three things: bulk embedding throughput at several batch sizes, single-query embedding latency, and reranking 100 candidates with a cross-encoder at full and half precision.

```bash
pip install sentence-transformers torch numpy
```

```python
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

torch.manual_seed(0)
rng = np.random.default_rng(0)
words = ("annual leave carry over remote work equipment budget expenses receipt password "
         "rotation VPN public Wi-Fi laptop refresh support desk hours handbook policy").split()
passages = [" ".join(rng.choice(words, size=rng.integers(40, 120))) for _ in range(2048)]
query = "How many days of annual leave carry over?"

def bench(fn, repeats=3):
    fn()                                         # warm-up: load kernels, allocate memory
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()              # GPU work is asynchronous: wait for it
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs))

devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
                  + (["cuda"] if torch.cuda.is_available() else [])

print("Embedding 2,048 passages with all-MiniLM-L6-v2 (passages/s)")
for device in devices:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    row = []
    for bs in (1, 16, 128):
        n = 256 if bs == 1 else len(passages)    # batch size 1 is slow; time fewer
        t = bench(lambda: model.encode(passages[:n], batch_size=bs))
        row.append(f"batch {bs:3d}: {n / t:6.0f}")
    print(f"  {device:4s}", " | ".join(row))

print("Single query embedding latency (ms)")
for device in devices:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    print(f"  {device:4s} {1000 * bench(lambda: model.encode([query]), repeats=20):5.1f}")

print("Reranking 100 candidates with bge-reranker-base (ms per query)")
pairs = [(query, p) for p in passages[:100]]
for device in devices:
    ce = CrossEncoder("BAAI/bge-reranker-base", device=device)
    print(f"  {device:4s} fp32 {1000 * bench(lambda: ce.predict(pairs, batch_size=32)):6.0f}")
    if device != "cpu":
        full = ce.predict(pairs, batch_size=32)
        ce.model.half()                          # 16-bit weights and activations
        print(f"  {device:4s} fp16 {1000 * bench(lambda: ce.predict(pairs, batch_size=32)):6.0f}")
        half = ce.predict(pairs, batch_size=32)
        same = np.array_equal(np.argsort(-full)[:10], np.argsort(-half)[:10])
        print(f"       max score change {np.abs(full - half).max():.5f}; same top 10: {same}")
```

```output
Embedding 2,048 passages with all-MiniLM-L6-v2 (passages/s)
  cpu  batch   1:     63 | batch  16:     74 | batch 128:    122
  mps  batch   1:    124 | batch  16:    513 | batch 128:    571
Single query embedding latency (ms)
  cpu    8.9
  mps    5.8
Reranking 100 candidates with bge-reranker-base (ms per query)
  cpu  fp32   6341
  mps  fp32   1391
  mps  fp16    378
       max score change 0.00025; same top 10: True
```

These figures come from one run on a laptop whose CPU was also busy with other work. An earlier run on a quieter machine gave CPU embedding rates of about 240 to 350 passages per second, GPU rates of about 290 to 740, and a single-query latency of about 5 ms on both devices. That variability is itself a lesson: benchmark on the hardware and under the load you will deploy with, and repeat runs.

**The important lines.**

- `device=device` places the weights on the chosen device; `encode` and `predict` move inputs there.
- The warm-up call in `bench` absorbs one-off costs: compiling or loading GPU kernels and allocating memory. 
- `torch.mps.synchronize()` waits for queued GPU work. Here `encode` and `predict` return NumPy arrays, which already forces a wait, but the habit prevents wrong timings in code that keeps tensors on the GPU.
- `batch_size=bs` controls how many texts go through the model at once; `encode` also sorts texts by length to reduce padding.
- `ce.model.half()` converts the cross-encoder's underlying PyTorch model to fp16. The ranking check compares the top 10 at both precisions.

**What the output means.**

- **Bulk embedding** on the GPU went from 124 passages per second at batch size 1 to 571 at 128: the GPU needs batches to be useful. At batch size 1, the GPU was only about twice as fast as this busy CPU, and on the quieter run barely faster at all.
- **Single-query latency** was a few milliseconds on either device. A lone short query is too little work to benefit, which is why query embedding rarely justifies a GPU on its own.
- **Reranking** is where the GPU earns its place: 100 query-passage pairs of up to 120 words took over 6 s on the CPU, 1.4 s on the GPU in fp32 and under 0.4 s in fp16. Half precision changed scores by at most 0.00025 and left the top 10 identical on this data.

A data-centre GPU would widen the gaps; the shape (big wins for batched work, little for single small requests) is what transfers.

**Common errors.** `RuntimeError: ... out of memory` means the batch, the sequence length or the model is too large for GPU memory; reduce the batch size first. An error saying tensors are on different devices means some input or model part was not moved; let library calls handle placement, or move everything explicitly. If a GPU appears slower than the CPU, check for missing warm-up, batch size 1, or timing without synchronisation.

**Extending it.** Add a 512-token passage to every batch of short ones and measure the throughput drop caused by padding; then sort your own batches by length and compare. Run the reranker comparison on real AskMeridian questions and chunks to confirm that fp16 leaves rankings unchanged on your data, not just on synthetic text.

## A real-world example

The Platform team planned to switch AskMeridian from MiniLM to a larger embedding model and, at the same time, to add a cross-encoder reranker. Two problems surfaced in planning. Re-embedding the handbook, runbooks and product documentation with the larger model on the existing CPU workers was estimated, using the FLOP arithmetic above and a short measured run, at most of a working day, and the team expected to iterate on chunking several times. And reranking 50 candidates on CPU took well over a second, against a retrieval budget of 300 ms.

A first proposal was to put a GPU in every API pod and run both models in-process. Load testing showed the GPUs idle most of the time: each pod saw only a few queries per second, sent one at a time, and each pod held its own copy of both models in GPU memory. The team moved instead to a single small GPU pool running an inference server for the embedding model and the reranker, which batched requests from all pods together; the API pods became pure network clients. Re-embedding jobs ran on the same pool overnight, at large batch sizes, when interactive traffic was low.

Before switching the reranker to fp16, the team compared rankings at both precisions on 500 real questions from their [golden dataset](/learn/testing-and-benchmarking/golden-datasets): the top 5 matched in all but a handful of cases, all involving near-tied scores, and answer-quality metrics did not change. Generation stayed on a hosted API: the volume did not justify the cost and operational burden of serving a language model themselves.

## Common mistakes

- **Sending single requests to a GPU.** Most of its capacity sits idle while fixed overheads dominate. Batch, or micro-batch concurrent requests.
- **Timing without warm-up or synchronisation.** The first call includes one-off setup, and asynchronous launches return before the work is done, so measured times are wrong in both directions.
- **One model copy per process.** Each web worker loading the model onto the same GPU multiplies memory use and prevents cross-request batching.
- **Mixed-length batches.** Padding to the longest sequence wastes most of the computation.
- **Assuming reduced precision is harmless.** Usually it is, but verify rankings on your own data before switching.
- **Forgetting the KV cache.** Sizing a GPU for the model's weights alone, then discovering it can serve only a few long RAG prompts at once.
- **Putting a small index on a GPU.** A few million vectors in a good CPU index already answer in milliseconds; the GPU adds cost and a memory ceiling.

## Best practices

- Use GPUs first where work is batchable and heavy: bulk embedding, reranking and self-hosted generation.
- Serve models through an inference server that batches across clients, and keep web processes as network clients.
- Benchmark with warm-up, synchronisation, realistic text lengths and several batch sizes; pick the knee of the throughput curve.
- Sort inputs by length and cap sequence length to limit padding.
- Run inference in fp16 or bf16 where supported, after checking rankings against full precision.
- Size GPU memory for weights plus KV cache at your real prompt lengths and concurrency.
- Schedule bulk jobs (re-embedding, index builds) for when interactive traffic is low, or on separate capacity.

## Performance considerations

The economics come down to utilisation. A GPU rented by the hour costs the same whether it processes one request per second or a thousand, so the cost per embedding or per rerank is the hourly price divided by the work done in that hour. Batching and sharing one server across all clients are therefore cost optimisations as much as speed optimisations. For bursty workloads, a GPU that is idle most of the day may cost more than CPU workers that scale with demand, or than a hosted API priced per token; compare cost per unit of work at your real traffic, not peak throughput.

Data movement is the hidden cost. Moving inputs from CPU memory to the GPU and results back takes time over the PCIe bus (on discrete GPUs), so pipelines that bounce data back and forth between CPU-side Python code and the GPU for each step lose much of the benefit. Keep consecutive GPU steps on the GPU and transfer in large blocks.

A GPU cuts one request's latency only when that request contains substantial parallel work, like reranking 100 candidates; it raises throughput whenever work can be batched.

## Security considerations

Shared GPU inference servers process many tenants' text in the same batches; batching itself does not mix results between requests, but logs, caches and debugging dumps on the server can, so apply the same access and retention rules to the inference tier as to the rest of the system. Model weights downloaded from public hubs are code-adjacent artefacts: prefer the `safetensors` format over pickle-based checkpoints, which can execute code when loaded, and pin model revisions. Finally, GPU capacity is a denial-of-service target: very long inputs inflate attention cost quadratically, so cap input length and batch size at the server.

## Summary

- GPUs execute the matrix multiplications at the heart of transformers and vector search in massive parallel, but need large batches to be used well.
- In RAG, GPUs pay most for bulk embedding, cross-encoder reranking, self-hosted generation and large-scale batched vector search; single-query embedding and small indexes rarely justify them.
- Measure correctly: warm up, synchronise, use realistic lengths and several batch sizes.
- Precision (fp16, bf16, quantisation) and padding control how much of the GPU's capacity you use; verify rankings when lowering precision.
- For generation, GPU memory, dominated by weights and the KV cache of long RAG prompts, limits concurrency; inference servers with continuous batching and paged KV caches make the most of it.

## Exercises

1. **Re-embedding estimate.** A corpus has 3 million chunks of 300 tokens. The new embedding model has 335 million parameters. Estimate the time on a GPU with an assumed effective 40 TFLOP/s, and the number of GPUs needed to finish within one hour.

<details><summary>Show solution</summary>

Work: $2 \times 3.35 \times 10^8 \times 3 \times 10^6 \times 300 = 6.03 \times 10^{17}$ FLOPs. At $4 \times 10^{13}$ FLOP/s: about 15,000 s, a little over 4 hours. Within one hour needs at least 5 GPUs, assuming the work splits evenly across them (it does: chunks are independent) and ignoring data loading. In practice add a margin for padding, tokenisation and writing vectors to the database.

</details>

2. **Concurrency on a GPU.** Using the memory formula and the hypothetical 8-billion-parameter model from the lesson, how many concurrent 6,000-token requests fit on an 80 GB GPU with 16-bit weights? How many with 4-bit weights, keeping the KV cache in 16-bit?

<details><summary>Show solution</summary>

KV cache per request: $131{,}072 \times 6{,}000 \approx 0.79$ GB. With 16-bit weights: $(80 - 16)/0.79 \approx 81$ requests. With 4-bit weights (4 GB): $(80 - 4)/0.79 \approx 96$ requests. Overheads for activations and the serving framework reduce both figures somewhat. On a large GPU, the KV cache dominates, so shorter prompts (fewer, better-chosen chunks) raise concurrency more than weight quantisation does.

</details>

3. **Padding experiment.** Modify the main benchmark so that each batch of 128 passages contains one 500-word passage, the rest being 40 to 120 words. Disable length sorting by encoding each batch of 128 in a separate `encode` call. Compare throughput with the original and explain the result.

<details><summary>Show solution</summary>

Build the batches explicitly: for each block of 127 short passages add one long one, and call `model.encode(batch, batch_size=128)` per block, so the library cannot regroup passages across blocks. On the laptop GPU, throughput fell from about 720 passages per second (short passages only) to about 280, because each batch is padded to the longest sequence (truncated at the model's maximum length of 256 word-pieces for MiniLM), so every short passage costs as much as a long one. Letting `encode` see all passages at once restored most of it (about 640 per second), because its internal length sorting puts the long passages together. The general rule: batch by similar length.

</details>

4. **Design.** AskMeridian's traffic is 5 queries per second at peak and near zero at night. Each query reranks 50 candidates. Re-embedding runs about once a month. Propose a deployment for embedding and reranking, and justify whether GPUs should run continuously.

<details><summary>Show solution</summary>

At 5 queries per second, reranking needs 250 pairs per second, well within one small GPU behind an inference server that batches each query's 50 pairs. A single GPU instance serving both the query-embedding and reranking models, shared by all API pods, is enough; a second instance (or a CPU fallback that reranks fewer candidates) protects availability. Because traffic is near zero at night, scale the serving pool to its minimum out of hours if the platform allows. Monthly re-embedding should run as a batch job on separately provisioned GPUs that are released when it finishes, rather than on the interactive pool, so it does not affect latency. Generation stays on a hosted API unless volume or data rules change.

</details>

## Check your understanding

```quiz
[
  {"q": "Why does embedding one short query on a GPU often take about as long as on a modern CPU?", "options": ["The work is too small to fill the GPU, so fixed costs such as kernel launches and data transfer dominate", "GPUs cannot run transformers", "GPUs only support batch size 1", "Embedding models are memory-bound on CPUs only"], "answer": 0, "why": "A GPU's advantage is massive parallelism. One short sequence offers little parallel work, so overheads dominate. Batching many inputs is what unlocks the speed-up."},
  {"q": "A benchmark measures a GPU call at 0.2 ms, far faster than expected. What is the most likely explanation?", "options": ["The model is cached", "fp16 was used", "The timing stopped before the asynchronous GPU work finished", "The batch was too large"], "answer": 2, "why": "GPU launches return immediately. Without a synchronisation call (or a copy of results back to the CPU), the timer measures only the launch."},
  {"q": "For a self-hosted language model answering long RAG prompts, what usually limits how many requests a GPU serves at once?", "options": ["The number of arithmetic units", "GPU memory for the KV cache of all sequences in flight", "The tokeniser speed", "The PCIe bus"], "answer": 1, "why": "Each sequence's KV cache grows with its length; with prompts of thousands of tokens, the cache for concurrent requests fills memory long before arithmetic is saturated."},
  {"q": "Which RAG workload benefits most from a GPU?", "options": ["Embedding one query per request at low traffic", "Searching a 100,000-vector HNSW index one query at a time", "Building the prompt string", "Reranking 100 candidates per query with a cross-encoder"], "answer": 3, "why": "Reranking is heavy, batchable transformer work within each request, so a GPU cuts its latency substantially. The other workloads are too small or not GPU-shaped."},
  {"q": "A team switches its reranker to fp16. What should it check first?", "options": ["That the model file is smaller on disk", "That the GPU temperature is lower", "That rankings on its own evaluation data match fp32 closely", "Nothing; fp16 never changes results"], "answer": 2, "why": "Lower precision usually leaves rankings unchanged but can reorder near-tied candidates. Verifying on real data is cheap and catches the exceptions."}
]
```

## Further reading

- [PyTorch documentation](https://pytorch.org/docs/stable/index.html): devices, CUDA and MPS backends, `synchronize` and mixed precision.
- [Sentence Transformers documentation](https://www.sbert.net/): `encode` batching, devices and cross-encoders.
- Woosuk Kwon and colleagues (2023), "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. The vLLM paper.
- [Hugging Face Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/index): a GPU inference server for embedding and reranking models.
- Jeff Johnson, Matthijs Douze and Hervé Jégou (2019), "Billion-scale similarity search with GPUs", *IEEE Transactions on Big Data*. The paper behind FAISS's GPU indexes.
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki): GPU indexes and their limitations.
