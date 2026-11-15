Some questions are hard to retrieve for because they are too specific. "I'm in a hotel in Lisbon on Thursday and the Wi-Fi has no password; anything to do before I open the finance dashboard?" is a perfectly clear question to a person, but almost every word in it (hotel, Lisbon, Thursday, password, finance dashboard) points a retriever away from the one sentence that answers it: *VPN is required on public Wi-Fi.* **Step-back prompting** fixes this by asking a language model to step back from the particulars and pose the more general question behind them ("What are the security rules for working on public Wi-Fi?"), then retrieving for that general question as well as the original.

This lesson explains where the idea comes from, how to turn it into a retrieval stage, how it differs from the other query transformations in this module, and when it is worth one extra model call. You will run it on a Meridian corpus full of distractors that share the question's surface details, and see exactly which retrieval failure it repairs.

## Learning objectives

- Explain what a step-back question is and why abstraction can retrieve documents that a detailed query misses.
- Distinguish step-back prompting from query rewriting, decomposition, multi-query retrieval and HyDE by what each changes about the query.
- Build a step-back retrieval stage that retrieves for both questions and merges the results by rank.
- Write a few-shot prompt that produces useful step-back questions, and an answer prompt that applies general rules to specific facts.
- Decide, for a given workload, whether step-back prompting should run on every query, on some, or not at all.

## Prerequisites

- [Query rewriting](/learn/advanced-retrieval/query-rewriting): using a model to turn the user's words into a better search query.
- [Query decomposition](/learn/advanced-retrieval/query-decomposition): splitting a compound question into sub-questions.
- [Semantic search](/learn/dense-retrieval/semantic-search): asymmetric search between short questions and the passages that answer them.
- [Reciprocal Rank Fusion](/learn/hybrid-search/reciprocal-rank-fusion): merging ranked lists whose scores are not comparable.

## Why this matters

Users rarely ask the question a policy document answers. The handbook says "Expenses must be submitted within 30 days." Nobody types "What is the expense submission deadline?" They type their situation: "I bought a £90 monitor arm on 2 May and it is now 10 June; can I still claim it?" The question is wrapped in particulars (an amount, a product, two dates) that are essential for the *answer* and mostly irrelevant for *finding* the rule.

To an embedding model, those particulars are not irrelevant. A query embedding is a summary of everything in the query, so each detail pulls the vector towards documents that mention similar things. If the corpus contains a page about the office guest Wi-Fi password, a query that mentions a Wi-Fi password will drift towards it. In a ten-page toy corpus the right page usually wins anyway. In a real corpus of tens of thousands of chunks, there is nearly always some chunk that shares the surface details better than the rule does, and the rule falls out of the top $k$.

The failure is silent and plausible: the retriever returns the guest Wi-Fi page, the generator explains how to find the guest password, and the employee connects to an open network without the VPN. Retrieval matched the *scene* instead of the *rule*.

## The intuition

Ask an experienced HR adviser the monitor-arm question and watch what they do. They do not search their memory for "monitor arm" or "2 May". They think: *this is an expenses question, and also a home-office equipment question.* They recall the two rules (submit within 30 days, receipts over £75; a £400 equipment budget), and only then return to the particulars: 2 May to 10 June is 39 days, which is past the deadline.

That move has two halves. First, **abstraction**: recognise which general principle governs the situation. Second, **application**: reason from the principle back to the specific facts. The adviser is good at their job because they do the first half before looking anything up.

Step-back prompting gives a retrieval system the first half. A model reads the detailed question and writes the general question behind it; the retriever looks up the rule with the general question; the generator then receives both the rule and the original particulars, and does the second half.

## How it works

### Where the idea comes from

Zheng and colleagues at Google DeepMind introduced step-back prompting in *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models* (2023). Their setting was reasoning: before answering a physics question about a gas, the model first answers a **step-back question** such as "What physics principles are involved?", then reasons from the recalled principle to the specific answer. The abstraction step was taught with a few worked examples in the prompt (few-shot prompting).

For the knowledge-intensive tasks, they combined the technique with retrieval: passages were retrieved for both the original question and the step-back question, and both sets were given to the model. That variant is the one that matters for RAG engineering, and it is what this lesson builds. Treat the paper as evidence that the idea works on its benchmarks; whether it helps on your corpus is something you measure, as with every query transformation.

### What a step-back question is

A step-back question asks about the **concept, rule or category** that governs the original question. It is deliberately less specific: it drops names, dates, amounts and circumstances, and keeps the subject.

| Original question | Step-back question |
| --- | --- |
| I have 7 days of holiday left at Christmas; how many can I use in April? | What are the rules for carrying over unused annual leave? |
| My MacBook was issued in June 2023; can I get a new one this year? | How often are company laptops replaced? |
| I'm in a Lisbon hotel with open Wi-Fi; anything to do before I open the finance dashboard? | What are the security rules for working on public Wi-Fi? |
| Atlas returned E-4021 after the 2.3 upgrade; what now? | How are Atlas client errors after an upgrade resolved? |

A step-back question must not answer (a guessed "every 4 years" would steer retrieval to the superseded 2023 handbook), and it should stop one level up: "What are Meridian's HR policies?" retrieves everything and nothing.

### How it differs from its neighbours

This module contains several query transformations, and they are easily confused. They differ in *what* they change about the query:

| Technique | What it produces | Relation to the original |
| --- | --- | --- |
| [Query rewriting](/learn/advanced-retrieval/query-rewriting) | One cleaner query | Same meaning, better words |
| [Multi-query retrieval](/learn/advanced-retrieval/multi-query-retrieval) | Several paraphrases | Same meaning, varied words |
| [Query decomposition](/learn/advanced-retrieval/query-decomposition) | Several sub-questions | Parts of the original, each narrower |
| **Step-back prompting** | One more general question | The principle *behind* the original, broader |
| [HyDE](/learn/advanced-retrieval/hyde) | A hypothetical answer passage | What the answer might look like |

Decomposition and step-back move in opposite directions. Decomposition makes the query *narrower and more numerous*; step-back makes it *broader*. They combine naturally: a compound question can be decomposed, and each part stepped back.

### The pipeline

```mermaid
flowchart TD
    Q["User question<br/>(detailed, situational)"] --> SB["LLM: write the step-back question<br/>(few-shot prompt)"]
    Q --> R1["Retrieve top-k<br/>for the original"]
    SB --> R2["Retrieve top-k<br/>for the step-back question"]
    R1 --> M["Merge by rank,<br/>remove duplicates"]
    R2 --> M
    M --> G["Generate: apply the general rules<br/>to the specific facts"]
    Q --> G
    G --> A["Answer with citations"]
```

1. **Generate the step-back question.** One short model call with a few-shot prompt: instructions plus four to eight examples of detailed questions and their step-back versions, drawn from your own domain.
2. **Retrieve for the original question.** This keeps whatever the specific details legitimately match: a named product page, an error code, a person's team.
3. **Retrieve for the step-back question.** This finds the governing rule even when the details pull elsewhere.
4. **Merge.** Combine the two ranked lists by position, not by score (scores for two different queries are not on a shared scale), and remove duplicates. Interleaving and [Reciprocal Rank Fusion](/learn/hybrid-search/reciprocal-rank-fusion) both work; the next section explains the choice.
5. **Generate the answer.** Give the model the original question (not the step-back one) and the merged context, and instruct it to identify the governing rule first and then apply it to the user's particulars.

Steps 1 and 2 are independent, so run them concurrently: the original-query retrieval does not wait for the model.

### Why you keep the original query

Do not retrieve only with the step-back question: specifics are sometimes exactly what retrieval needs. "Atlas returned E-4021" must reach the page for E-4021, and a step-back question about "Atlas client errors" retrieves the whole error catalogue indifferently. The paper retrieved for both questions, and that is the robust default: the step-back query adds recall for the rule; the original query preserves precision for the particulars.

### Merging the two lists

Interleave the lists (first of each, then second of each, skipping duplicates) or use [Reciprocal Rank Fusion](/learn/hybrid-search/reciprocal-rank-fusion), which also lifts documents that *both* queries found. With only two lists the difference is small; use RRF if your pipeline already has it.

### The step-back prompt

The quality of the whole stage rests on the abstraction prompt. Four things make it work:

- **Few-shot examples from your domain.** The paper relied on few-shot examples, and in practice they do most of the work: they show the model how far to step back. Write five or six pairs from real (anonymised) user questions.
- **An explicit "do not answer" instruction.** Without it, some step-back questions arrive with a guessed answer attached.
- **A single line of output.** You are going to embed it, so you want the question and nothing else.
- **A pass-through for already-general questions.** "How many days of annual leave do I get?" is already at the level of the rule. Allow the model to return the question unchanged, rather than forcing an artificial abstraction.

### When step-back helps and when it does not

It helps when:

- Questions are **situational**: users describe their circumstances rather than naming the topic (HR, IT support, insurance, legal and compliance assistants).
- Documents are **normative**: they state rules, principles or procedures in general terms.
- The corpus contains **surface-similar distractors**: pages that share the situational vocabulary but not the rule.

It does little or harm when:

- Questions are already general, or are **lookups of specific identifiers** (an error code, a ticket, a person's name). The step-back query adds noise; the original query does the work.
- The answer *is* a specific fact attached to a specific entity ("Which plan includes SAML single sign-on for Meridian Reports?"). Stepping back to "What features do Meridian products have?" broadens to the point of uselessness.
- Latency budgets are tight and the corpus is small enough that the original query already finds the rule.

Because of this, many systems run step-back **conditionally**: a lightweight [query router](/learn/agentic-rag/query-routing) or a simple rule (question longer than some number of words, contains dates or amounts, no identifier pattern) decides which queries get the extra stage.

## In code

**The problem.** Retrieve for five Meridian questions with and without a step-back question, on a corpus that contains deliberate surface-level distractors (a guest Wi-Fi page, a hotel-booking page, a Christmas closure notice, a battery safety notice), and count how often the answering document reaches the top two.

**Dependencies.** NumPy and sentence-transformers. The embedding model downloads once (about 90 MB).

```bash
pip install numpy sentence-transformers
```

To keep the run reproducible and free, the step-back questions are written into the script: they are what a model following the prompt above would plausibly write. The real model call comes after, in its own block. Everything else, embedding, search and merging, is real.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS = {
    "leave":     "Annual leave is 25 days per year. Up to 5 unused days carry over and must be used by 31 March.",
    "booking":   "Book annual leave in the HR portal; your line manager approves each request.",
    "remote":    "Staff may work remotely up to 3 days a week.",
    "equipment": "The home-office equipment budget is £400. Contractors are not eligible for it.",
    "expenses":  "Expenses must be submitted within 30 days. Anything over £75 needs a receipt.",
    "vpn":       "VPN is required on public Wi-Fi.",
    "password":  "Passwords must be rotated every 90 days.",
    "laptop":    "The laptop refresh cycle is 3 years.",
    "lost":      "Report a lost laptop to the support desk at once.",
    "hours":     "The support desk is open 08:00 to 18:00 UK time.",
    "guestwifi": "The office guest Wi-Fi password changes every Monday and is shown at reception.",
    "travel":    "Hotels for business travel are booked through the travel portal.",
    "finance":   "Access to the finance dashboard is requested from the finance department.",
    "closure":   "The office is closed between Christmas and New Year.",
    "macbook":   "New starters choose a MacBook or a Windows laptop when they join.",
    "battery":   "If a laptop battery swells, stop using it and contact the support desk.",
    "claims":    "Claims for mileage are paid at the rate set by the finance department.",
}
ids = list(DOCS)

# (query, step-back question, ids of the documents that answer it).
# The step-back questions stand in for what the LLM would write.
CASES = [
    ("I joined in September and still have 7 days of holiday left at Christmas. "
     "How many can I use in April?",
     "What are the rules for carrying over unused annual leave?", {"leave"}),
    ("My MacBook was issued in June 2023 and the battery is dying. Can I get a new one this year?",
     "How often are company laptops replaced?", {"laptop"}),
    ("I'm in a hotel in Lisbon on Thursday and the Wi-Fi has no password. "
     "Anything to do before I open the finance dashboard?",
     "What are the security rules for working on public Wi-Fi?", {"vpn"}),
    ("I bought a £90 monitor arm for my flat on 2 May and it is now 10 June. Can I still claim it?",
     "What are the rules for claiming expenses and home-office equipment?", {"expenses", "equipment"}),
    ("How many days of annual leave do I get?",
     "What is the annual leave policy?", {"leave"}),
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = model.encode([DOCS[i] for i in ids], normalize_embeddings=True)

def search(text, k=2):
    q = model.encode([text], normalize_embeddings=True)[0]
    scores = D @ q                       # unit vectors: dot product = cosine
    top = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in top]

def interleave(*ranked_lists):
    """Merge ranked lists by position (1st of each, then 2nd of each...), no duplicates.
    Scores from different queries are not comparable, so we never sort by them."""
    out = []
    for rank_group in zip(*ranked_lists):
        for doc_id, _ in rank_group:
            if doc_id not in out:
                out.append(doc_id)
    return out

def show(hits, gold):
    return ", ".join(f"{d}{'*' if d in gold else ''} {s:.2f}" for d, s in hits)

found = {"original": 0, "step-back": 0, "combined": 0}
total = 0
for query, step_back, gold in CASES:
    orig, back = search(query), search(step_back)
    combined = interleave(orig, back)
    print(f"Q: {query[:72]}...")
    print(f"   original : {show(orig, gold)}")
    print(f"   step-back: {show(back, gold)}")
    print(f"   combined : {', '.join(d + ('*' if d in gold else '') for d in combined)}")
    found["original"] += len(gold & {d for d, _ in orig})
    found["step-back"] += len(gold & {d for d, _ in back})
    found["combined"] += len(gold & set(combined))
    total += len(gold)
print("answering documents retrieved:", {k: f"{v}/{total}" for k, v in found.items()})
```

```output
Q: I joined in September and still have 7 days of holiday left at Christmas...
   original : leave* 0.48, closure 0.35
   step-back: leave* 0.78, booking 0.54
   combined : leave*, closure, booking
Q: My MacBook was issued in June 2023 and the battery is dying. Can I get a...
   original : laptop* 0.47, battery 0.47
   step-back: laptop* 0.61, lost 0.45
   combined : laptop*, battery, lost
Q: I'm in a hotel in Lisbon on Thursday and the Wi-Fi has no password. Anyt...
   original : guestwifi 0.53, finance 0.41
   step-back: vpn* 0.57, guestwifi 0.41
   combined : guestwifi, vpn*, finance
Q: I bought a £90 monitor arm for my flat on 2 May and it is now 10 June. C...
   original : expenses* 0.51, equipment* 0.32
   step-back: equipment* 0.53, expenses* 0.53
   combined : expenses*, equipment*
Q: How many days of annual leave do I get?...
   original : leave* 0.80, booking 0.57
   step-back: leave* 0.64, booking 0.60
   combined : leave*, booking
answering documents retrieved: {'original': '5/6', 'step-back': '6/6', 'combined': '6/6'}
```

**Reading the output.** Look at the cases one by one, because the totals hide the interesting part.

- **Lisbon hotel.** The original query fails outright. "Wi-Fi", "password" and "finance dashboard" pull the vector to the guest Wi-Fi page (0.53) and the finance access page (0.41); the VPN rule is not in the top two. The step-back question goes straight to it (0.57). This is the failure the technique exists for.
- **MacBook.** The original query finds the refresh-cycle page, but only just: it ties with the battery safety notice at 0.47 (the third decimal decides). One more battery-related chunk in a real corpus and the rule drops out. The step-back query finds it with a clear margin (0.61 against 0.45). **Margins matter**: a query that wins narrowly on a toy corpus loses on a large one.
- **Holiday carry-over.** Both succeed, but the step-back cosine is far higher (0.78 against 0.48), because the abstract question and the policy share their subject and nothing else competes.
- **Monitor arm.** The original already finds both relevant pages, the equipment one weakly (0.32). The step-back question, which names both topics, scores them equally.
- **Already general.** For "How many days of annual leave do I get?" the step-back adds nothing and its score is lower than the original's (0.64 against 0.80). The combined list still contains the answer, so the cost of stepping back needlessly was latency, not quality.

The combined list keeps the distractors the original query found (closure, guest Wi-Fi). That is deliberate: in other questions those specific pages would be the relevant ones, and the generator, told to identify the governing rule, can ignore them. If you need to cut context, rerank the merged list with a [cross-encoder](/learn/reranking-fundamentals/cross-encoders) against the *original* question.

**Common errors.** If the step-back list is identical to the original list, the model is paraphrasing rather than abstracting; strengthen the few-shot examples. If you sort the merged list by score, you will see the step-back results float to the top regardless of relevance, because abstract questions score higher against short policy sentences; merge by rank.

### Generating the step-back question with Claude

In production the step-back question comes from a model call. This block was not executed (it needs an API key); it uses the Anthropic SDK in the standard way.

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

STEP_BACK_SYSTEM = """You help a search system for Meridian's internal documents.
Given an employee's question, write the more general question about the policy,
rule or procedure that governs it. Drop names, dates, amounts, places and personal
circumstances. Do not answer the question. Do not step back further than the
relevant policy area. If the question is already general, repeat it unchanged.
Reply with the step-back question only, on one line."""

FEW_SHOT = """Examples:
Question: I started in March and have 3 days left in December. Can I keep them?
Step-back: What are the rules for carrying over unused annual leave?

Question: I'm a contractor; can I expense a £120 desk lamp for home?
Step-back: Who is eligible for the home-office equipment budget, and what does it cover?

Question: Atlas showed E-4021 after I upgraded. What do I do?
Step-back: Atlas showed E-4021 after I upgraded. What do I do?

Question: How often must I change my password?
Step-back: How often must I change my password?"""

def step_back(question: str) -> str:
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=100,
        system=STEP_BACK_SYSTEM,
        messages=[{"role": "user", "content": f"{FEW_SHOT}\n\nQuestion: {question}\nStep-back:"}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text.splitlines()[0] if text else question   # fall back to the original
```

Two details in the examples deserve attention. The error-code example is returned *unchanged*: it teaches the model that identifier lookups should not be abstracted. The password example is already general and is also returned unchanged. Without these, a model will step back on everything. In production you would choose a smaller, faster model for this call than for the final answer (the task is short and formulaic), and fall back to the original question on any error or timeout so that the stage can only add recall, never block a request.

**Extending it.** Run the original retrieval and the model call concurrently (the SDK has an `AsyncAnthropic` client), swap `interleave` for RRF, and measure on your own labelled situational questions.

## A real-world example

The Platform team at Meridian ran AskMeridian's monthly review of thumbs-down answers and found a recurring pattern in IT-security questions. Employees described situations ("I'm at a conference and the hall Wi-Fi is open", "working from my parents' house on their router", "café near the client office"), and the answers explained how to get on the network rather than what to do once there. In each case the retrieved chunks were about networks: guest Wi-Fi, the office wireless setup, the visitor network at reception. The one-sentence rule that VPN is required on public Wi-Fi was ranked somewhere below the top five.

A context header on the rule chunk helped some phrasings but not others, because the query embedding was still dominated by the scene. The team added step-back prompting only for queries their router tagged as situational (personal context, places or dates, no identifier pattern), retrieving for both questions in parallel and merging with the RRF code they already had for hybrid search. On their labelled set of past situational questions the rule chunk reached the top five far more often, and answers changed shape: "VPN is required on public Wi-Fi, and a conference hall network counts as public, so connect before opening internal tools."

One regression appeared: questions such as "since Tuesday, Relay shows E-4021" were routed to step-back because they contained a date, and the stepped-back query retrieved the generic troubleshooting page. Because both queries were retrieved, the right page was still in the context, only lower. The team added an identifier pattern to the router and added such questions to the few-shot examples as pass-throughs.

## Common mistakes

- **Retrieving only with the step-back question.** Precision on identifiers, product names and specific facts collapses. Always retrieve for the original too.
- **Stepping back too far.** "What are Meridian's policies?" retrieves a random sample of the handbook. Few-shot examples should show one level of abstraction: the policy area, not the whole domain.
- **Letting the model answer inside the step-back question.** A query such as "Laptops are replaced every 4 years; what is the policy?" embeds a guessed (and here superseded) fact and steers retrieval to the wrong version. Instruct it not to answer and check outputs.
- **Merging by raw score.** Similarities for two different queries are not comparable; abstract queries often score systematically higher against short rule sentences, so a score merge silently ranks the step-back results first.
- **Passing the step-back question to the generator instead of the original.** The model then answers the general question and ignores the user's dates and amounts, producing a policy summary instead of an answer.
- **Running it on every query without measuring.** It adds a model call to each request and helps only some query types; on identifier lookups it adds noise.

## Best practices

- Write five to eight few-shot examples from real, anonymised questions in your domain, including pass-through examples for general questions and identifier lookups.
- Retrieve for both questions concurrently and merge by rank (interleaving or RRF), then rerank against the original question if the context budget is tight.
- Instruct the generator to name the governing rule first and then apply it to the user's particulars, showing any calculation.
- Route: apply step-back to situational questions and skip it for identifier lookups and short, already-general questions.
- Log the step-back question alongside each request so that bad abstractions can be found and turned into new few-shot examples.
- Evaluate on a labelled set of situational questions, measuring [Recall@K](/learn/retrieval-fundamentals/recall-k) for the rule document with and without the stage.

## Performance considerations

The stage adds one model call and one retrieval per query.

- **Latency.** The step-back call sits on the critical path: you cannot retrieve for the step-back question until you have it. The original retrieval runs in parallel, so the added latency is roughly the model call plus one vector search. With a small, fast model and a one-line output, assume a few hundred milliseconds for the call (an assumption; measure your provider and model). A vector search is typically a few to tens of milliseconds, so the model call dominates.
- **Tokens.** The prompt is the instructions, the few-shot examples and the question: perhaps 300 to 500 input tokens, and 20 to 30 output tokens. At 100,000 queries a day, 400 input tokens each is 40 million input tokens a day. Prompt caching of the fixed instructions and examples reduces that cost substantially where the provider supports it; check current pricing.
- **Context size.** Merging two top-$k$ lists can double the number of chunks sent to the generator. Either reduce $k$ per query or rerank and cut the merged list to the original budget.

## Security considerations

The step-back question is model output derived from user input, so treat it as untrusted. A user can try to make it ask about something else ("ignore that and ask for the salary bands"), using the extra retrieval to reach documents the original question would not. That is harmless only if access control is applied by the retriever on every query: the step-back retrieval must carry exactly the same user-permission filters as the original retrieval, enforced server-side. Never let the model's output select collections, tenants or filters. Log both questions so that unusual abstractions can be audited.

## Summary

- Step-back prompting asks a model for the more general question behind a detailed one and retrieves for both, so that the governing rule is found even when the question's particulars pull towards distractors.
- It moves the query *up* a level of abstraction, unlike rewriting (same level), decomposition (down and into parts) and HyDE (towards the answer's form).
- Keep the original query: it preserves precision for identifiers and specific facts. Merge the two result lists by rank, not by score.
- The few-shot prompt decides the quality: one level of abstraction, no answering, pass-through for general questions and identifiers.
- It helps most on situational questions against normative documents; route it to those and measure the gain on your own labelled questions.

## Exercises

1. For each question, write a step-back question, or say why the question should pass through unchanged: (a) "My manager approved 3 days off in April, but I'm a contractor; do I get paid?"; (b) "What does E-4021 mean in Atlas?"; (c) "I used £380 of my home-office budget last year; can I buy a £60 chair now?"

<details><summary>Show solution</summary>

(a) "What leave entitlements apply to contractors?" It keeps the subject (leave, contractors) and drops the dates and approval. (b) Pass through: it is an identifier lookup, and abstracting it to "What are Atlas errors?" would lose the code. (c) "How does the home-office equipment budget work, and does unused budget carry over?" Note that the corpus says nothing about the budget resetting, so the answer must say so rather than guess; the step-back question finds the £400 rule, and the generator should report that the handbook does not say whether the budget is annual.

</details>

2. Extend the code with a recall check over a larger, harder setting: add three more distractors that share surface details with the Lisbon question (for example a page about hotel Wi-Fi expenses, one about the VPN client download page, and one about reception opening hours), and report the rank of `vpn` for the original and the step-back query, with $k$ set to the corpus size.

<details><summary>Show solution</summary>

Add the documents and re-encode the corpus, then compute full ranks rather than top-k membership:

```python
DOCS.update({
    "hotelwifi": "Hotel Wi-Fi charges on business trips can be claimed as expenses.",
    "vpnclient": "Download the VPN client from the IT portal.",
    "reception": "Reception is open from 08:00 to 18:00 on weekdays.",
})
ids = list(DOCS)
D = model.encode([DOCS[i] for i in ids], normalize_embeddings=True)

def rank_of(text, target):
    q = model.encode([text], normalize_embeddings=True)[0]
    order = [ids[i] for i in np.argsort(-(D @ q))]
    return order.index(target) + 1

for query, step_back, gold in CASES:
    for g in gold:
        print(g, rank_of(query, g), rank_of(step_back, g))
```

```output
leave 1 1
laptop 1 1
vpn 6 1
expenses 1 2
equipment 2 1
leave 1 1
```

The VPN rule falls to rank 6 for the original Lisbon question once more scene-sharing pages exist, while the step-back question still ranks it first. The other questions are barely affected, which is the pattern to expect: step-back matters exactly where the particulars have competitors.

</details>

3. Your team proposes replacing the original query with the step-back question to save one retrieval. Give two concrete Meridian questions for which this would make answers worse, and explain why.

<details><summary>Show solution</summary>

"What does Atlas error E-4021 mean?" and "Does Meridian Reports support SAML single sign-on on the Enterprise plan?" Both depend on a specific identifier or entity named in the question. A step-back question ("How are Atlas errors handled?", "What features do Meridian products have?") retrieves broad pages that may not contain the specific code or plan detail, and the precise page drops out because nothing in the query points at it any more. A retrieval costs milliseconds; losing precision on identifier questions costs correctness.

</details>

## Check your understanding

```quiz
[
  {"q": "What does a step-back question ask about?", "options": ["A paraphrase of the original question with the same specificity", "The general concept, rule or category that governs the original question", "One narrow sub-part of the original question", "A hypothetical answer to the original question"], "answer": 1, "why": "Step-back prompting moves up a level of abstraction: it drops the particulars and asks about the governing principle."},
  {"q": "Why should a step-back pipeline still retrieve with the original question?", "options": ["Because the step-back question cannot be embedded", "Because vector databases require at least two queries", "Because specific details such as error codes and product names often identify the right page, and the step-back question discards them", "Because the original query is always more accurate"], "answer": 2, "why": "The step-back query adds recall for the rule; the original preserves precision for identifiers and specific facts."},
  {"q": "The original query returns cosines 0.53 and 0.41; the step-back query returns 0.78 and 0.54. How should the two lists be merged?", "options": ["By rank (interleaving or RRF), because scores from different queries are not comparable", "By sorting all four results by cosine", "By keeping only the list with the higher top score", "By averaging the two lists' scores per position"], "answer": 0, "why": "Each cosine is relative to its own query vector. Abstract queries often score higher against short rule sentences, so a score merge would be biased."},
  {"q": "Which question is the poorest candidate for step-back prompting?", "options": ["I'm in a café with open Wi-Fi; can I check my email?", "I bought a chair on 1 May and it's 20 June; can I still claim?", "I have 6 days of leave left in December; what happens to them?", "What does Atlas error E-4021 mean?"], "answer": 3, "why": "It is an identifier lookup. Abstracting it to Atlas errors in general discards the code that identifies the right page."},
  {"q": "Which question should the generator receive in a step-back pipeline?", "options": ["Only the step-back question", "The original question, with context retrieved for both questions", "Only the step-back question's retrieved context, without any question", "Both questions, with an instruction to answer the step-back one"], "answer": 1, "why": "The answer must apply the general rule to the user's particulars, which are in the original question."}
]
```

## Further reading

- Zheng, H. S., Mishra, S., Chen, X., Cheng, H.-T., Chi, E. H., Le, Q. V. and Zhou, D. (2023). *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*. [arXiv:2310.06117](https://arxiv.org/abs/2310.06117). The original paper, including the retrieval-augmented variant.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). The few-shot prompting that the step-back prompt depends on.
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903). The main baseline step-back prompting is compared against.
- Anthropic. *Prompt engineering* and *Prompt caching* guides in the Claude developer documentation: writing instructions and examples, and caching the fixed parts of a prompt.
