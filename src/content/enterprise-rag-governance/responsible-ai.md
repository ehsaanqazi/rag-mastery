A RAG system can pass every technical test (retrieval recall is high, answers are faithful, latency is low) and still treat some of its users worse than others, mislead people about what it is, or quietly become the decision-maker in a process that needed a human. None of these failures shows up in an aggregate accuracy score. They show up when someone asks who the system works for, who it fails, and who is accountable when it does.

**Responsible AI** is the set of practices for answering those questions before launch and keeping them answered afterwards: understanding the risks of a use case, measuring performance for different groups of users, being open about what the system is and is not, keeping people in control where the stakes demand it, and documenting decisions so they can be reviewed. This lesson translates the broad principles into engineering work for an enterprise RAG system. It closes the module: [audit logs](/learn/enterprise-rag-governance/audit-logs), [compliance](/learn/enterprise-rag-governance/compliance) and [data governance](/learn/enterprise-rag-governance/data-governance) supply the evidence, and this lesson decides what it should show.

## Learning objectives

- Describe the main responsible-AI risk areas for a RAG system (unequal performance, over-reliance, opacity, privacy, harmful output) and the engineering control for each.
- Perform a disaggregated evaluation of retrieval and refusal behaviour by user group, and explain why a pooled metric hides a gap.
- Explain why a single global confidence threshold can place the burden of refusals unevenly, and how identity-aware retrieval filters change the picture.
- Classify a use case by risk tier and choose proportionate controls, including human oversight.
- Name the components of a system card and the main published frameworks (NIST AI RMF, the EU AI Act's risk tiers) without treating this lesson as legal advice.

## Prerequisites

- [Source attribution](/learn/enterprise-rag-governance/source-attribution): showing users where an answer came from.
- [Compliance](/learn/enterprise-rag-governance/compliance): regulatory obligations that apply to AI systems and data.
- [Retrieval evaluation](/learn/retrieval-evaluation): measuring whether retrieval finds what it should.
- [Document-level permissions](/learn/access-control-and-privacy/document-level-permissions): filtering retrieval by who is asking.

## Why this matters

Enterprise RAG is deployed where decisions about people are made: HR questions, benefits, customer eligibility, compliance guidance. The technology that saves an engineer ten minutes on an error code can, with a different corpus, shape whether an employee believes they are entitled to leave. Three properties make RAG prone to quiet harm.

**The corpus has a point of view.** If the handbook covers permanent employees in detail and contractors in one paragraph, the system serves employees well and contractors badly, with no bug anywhere.

**Fluency reads as authority.** A well-formed paragraph with a citation looks reliable. Research on human-automation interaction has long documented *automation bias*, the tendency to accept a machine's output over contrary evidence, and fluent text strengthens it.

**Errors are unevenly distributed.** Retrievers, embedding models and thresholds behave differently on terse, informal or non-native phrasing, thinly covered topics, and groups the documents barely mention. A system right 92 per cent of the time overall can be right 98 per cent for one group and 70 per cent for another.

Without deliberate practice the outcome is a slow loss: one group stops using the tool, a wrong answer on eligibility goes unchallenged, a regulator asks for documentation that was never written.

## The intuition

Imagine Meridian's HR helpdesk staffed by one very fast, well-read clerk. If the clerk's files covered full-time staff thoroughly and contractors in a single page, contractors would get vague or wrong answers, politely delivered. If the clerk never said "I don't know", people would stop checking. And if the clerk quietly started deciding who got approved for what, a role meant to inform would become a role that decides.

A responsible organisation checks how the answers vary by who asked, makes the clerk say when it is unsure, labels the desk plainly ("automated assistant; confirm anything important with HR"), and keeps a human signature on decisions that matter. None of this needs a less capable clerk; it needs a system designed for the cases where capability is not enough.

## How it works

### Risk areas and their controls

| Risk area | What it looks like in RAG | Engineering control |
| --- | --- | --- |
| Unequal performance | Contractors, non-native speakers or thinly covered topics get worse answers | Disaggregated evaluation; coverage audits of the corpus; identity-aware retrieval |
| Over-reliance | Users act on answers without checking | Citations, visible uncertainty, abstention, friction on high-stakes answers |
| Opacity | Users do not know they are talking to an AI or what it knows | Disclosure; a plain statement of scope and limits; system card |
| Privacy | Personal data surfaces in answers or logs | Access control, PII detection, retention (see [PII detection](/learn/access-control-and-privacy/pii-detection)) |
| Harmful or inappropriate output | Advice outside scope, discriminatory phrasing, leaked content | Scope limits, [output validation](/learn/rag-security-fundamentals/output-validation), red-teaming |
| Loss of human control | The system decides instead of informing | Human review on defined decision types; no autonomous action on people |
| Weak accountability | No owner, no record, no way to contest | Named owner, audit logs, an appeals path |

### Risk tiering

Not every use case deserves the same scrutiny. Tiering by consequence lets you spend effort where harm is possible.

- **Low consequence**: answers that save time and are easily checked, such as error-code lookups, "where is the form" and meeting-room policy. Wrong answers cause inconvenience. Light controls: citations, feedback button, logging.
- **Medium consequence**: answers people may act on that affect their entitlements or obligations: leave, expenses, benefits. Wrong answers can cause real loss. Add disaggregated evaluation, number and entity verification of citations (see [source attribution](/learn/enterprise-rag-governance/source-attribution)), abstention on low confidence, and a prominent route to a human.
- **High consequence**: use that informs or makes decisions about individuals: hiring, promotion, discipline, credit, health, legal outcomes. Here the system should support a human who decides, never decide itself; require documented evaluation, human review of every outcome, a record of the basis, and legal and ethics sign-off before launch.

This mirrors the risk-based structure of the EU AI Act, which ranges from prohibited practices through high-risk systems (including certain uses in employment) to systems with transparency obligations and minimal-obligation systems. Definitions and dates are in the regulation and evolving guidance, so treat this as orientation and take legal advice on whether a deployment is regulated. The [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) organises the work into *Govern* (policies and accountability), *Map* (context and risks), *Measure* (assess and track) and *Manage* (prioritise and act). It is voluntary, and useful as a vocabulary.

### Where unequal performance comes from

In a RAG system, uneven results arise at identifiable points, and each points to a different fix.

1. **Corpus coverage.** Documents about a group, a region or a language are missing or thin. Fix by auditing coverage and commissioning content, not by tuning the retriever.
2. **Audience mismatch.** The right document exists but applies to a different group, and the retriever, which sees only text similarity, returns the wrong audience's rule. This is the failure demonstrated in the code below. Fix with metadata (`applies_to`) and identity-aware filtering.
3. **Query-style mismatch.** The embedding model or lexical index handles formal text better than slang, abbreviations, typos or another dialect, so some users' queries retrieve worse. Fix by evaluating on queries that represent real users, not on developer-written ones.
4. **Threshold effects.** A single refusal threshold on a similarity score refuses more often for groups whose documents are shorter or terser, because their best matches score lower. The burden of "I don't know" falls unevenly.
5. **Generation effects.** The model may phrase advice differently, or hedge more, depending on cues in the question. Detect with paired prompts that differ only in a group cue, and compare outputs.

Embedding models may also carry social biases from their training data, documented in the research literature, so evaluate retrieval on slices of real queries instead of assuming neutrality.

### Disaggregated evaluation

**Disaggregated evaluation** means computing every metric you care about separately for each group (user type, region, language, topic) instead of only in total. The principle is simple and the discipline is rare: define the groups, make sure the evaluation set contains enough queries from each, and report the per-group numbers together with the pooled one. Track at least answer correctness, retrieval to the right audience, the rate of *refusing an answerable question* and the rate of *answering an unanswerable one*; the last two are the two sides of abstention and trade against each other.

### Abstention and honest uncertainty

A system that always answers will sometimes answer from irrelevant context. **Abstention** (declining to answer when evidence is thin) turns many confident errors into "I could not find this; here is who to ask". Two cautions follow from the code later.

- A similarity score is **not a probability** of correctness (see [query-document matching](/learn/rag-architecture-fundamentals/query-document-matching)). Its scale differs between models, and its distribution differs between topics and groups. A threshold must be chosen by measuring, per slice, not by feel.
- Abstention is itself a service outcome that can be distributed unfairly. If one group's questions are refused three times as often, the system is unreliable for that group even though it never gave them a wrong answer. Report the refusal rate per group.

Good refusals say where to go next: "I could not find a policy for contractors on this; HR People Operations can confirm" beats a bare "I don't know".

### Human oversight that is real

"A human is in the loop" is only a control if the human can disagree. Oversight fails when the reviewer sees only the conclusion, has seconds per case, is measured on throughput, or stops looking because the system is usually right. It needs the following.

- The reviewer sees the **evidence**, not only the answer.
- The reviewer has **time and authority** to overrule, and overrides are not punished.
- Reviewers are **audited**, for example by seeding known-wrong answers and measuring whether they are caught.
- The system **never takes the action itself** for decisions about people; it prepares material for the person who decides.

### Transparency and contestability

People should know when they are interacting with an automated system, what it is for, and what it cannot do. In practice this means a visible notice in the interface, a short statement of scope ("answers questions about the UK employee handbook and IT runbooks; does not give legal or medical advice"), citations that link to the source, and a clear path to a human and to report a wrong answer. **Contestability** means a person can challenge an outcome and have it examined, which requires the audit trail described in [audit logs](/learn/enterprise-rag-governance/audit-logs): you can only review a decision whose basis you recorded.

### Documentation: the system card

Two artefacts are standard in the research literature: **model cards** (Mitchell et al., 2019), which describe a model's intended use, evaluation across groups and limitations, and **datasheets for datasets** (Gebru et al., 2018, revised 2021), which describe how a dataset was assembled and what it should and should not be used for. For a RAG system the useful equivalent is a short **system card** covering:

- purpose, owner and users, and what is explicitly out of scope;
- the corpus: sources, owners, coverage, known gaps, update frequency;
- components and their versions (the [release bundle](/learn/enterprise-rag-governance/version-control));
- evaluation results, **disaggregated**, with sample sizes and known weaknesses;
- risk tier, the controls applied, and the residual risks accepted, with the name of the person who accepted them;
- oversight: who reviews what, and the incident and escalation path;
- review date: when this card will next be reconsidered.



### Monitoring and incidents

Responsible behaviour decays: documents change, new groups arrive, models are updated. Monitor refusal and complaint rates per group, sample answers for review, run a fixed set of fairness-focused probes on every release (see [regression testing](/learn/testing-and-benchmarking/regression-testing)), and have an incident process for harm that treats a discovered disparity like any other defect: reproduce, fix, add a regression test, record.

## The mathematics

A per-group rate computed from a few queries is noisy, and the temptation is to read a difference into noise. A **confidence interval** states the range of true rates consistent with what you observed. The Wilson interval for a proportion is a standard choice because it behaves well for small samples and for rates near 0 or 1.

For $k$ events out of $n$ trials, with observed rate $\hat{p} = k/n$ and a normal quantile $z$ ($z = 1.96$ for roughly 95 per cent coverage), the interval is

$$
\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

**Intuition.** More queries give a narrower interval. If two groups' intervals do not overlap, the difference is unlikely to be luck; if they overlap heavily, you cannot tell yet.

**Worked example.** Suppose group A has 2 refusals of an answerable question in 40 tries (5 per cent) and group B has 8 in 40 (20 per cent). Plugging into the formula gives about 0.01 to 0.17 for A and 0.10 to 0.35 for B. The intervals overlap, so with 40 queries per group you cannot conclude the groups differ. With ten times the data at the same rates (20 of 400 and 80 of 400) the intervals are about 0.03 to 0.08 and 0.16 to 0.24: clearly apart.

```python
from math import sqrt

def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half

for name, k, n in [("group A", 2, 40), ("group B", 8, 40), ("group A (large)", 20, 400), ("group B (large)", 80, 400)]:
    lo, hi = wilson(k, n)
    print(f"{name:16s} {k}/{n} = {k/n:.2f}  95% interval [{lo:.2f}, {hi:.2f}]")
```

```output
group A          2/40 = 0.05  95% interval [0.01, 0.17]
group B          8/40 = 0.20  95% interval [0.10, 0.35]
group A (large)  20/400 = 0.05  95% interval [0.03, 0.08]
group B (large)  80/400 = 0.20  95% interval [0.16, 0.24]
```

Before claiming a disparity, or its absence, check that you have enough queries per group. The code below uses eight per group, so it illustrates mechanisms rather than measuring rates.

## In code

### A disaggregated retrieval audit with an abstention gate

**The problem.** Meridian's corpus describes some rules for employees only, some for contractors only and some for both. We want to measure, per user group, how often the top-1 retrieved chunk applies to the user's group, how often the system answers a question the corpus cannot answer, and how often a similarity threshold makes it refuse a question it could have answered. Then we compare retrieval that ignores identity with retrieval that filters by group.

**Dependencies.** `sentence-transformers` and NumPy. The first run downloads `all-MiniLM-L6-v2`.

```bash
pip install sentence-transformers numpy
```

**The source.** Each document carries an `applies_to` set. The queries are written without any group cue ("How much can I spend on a home office desk?"), as real users write them. For each query we record whether the corpus can answer it for that group.

```python
"""Disaggregated evaluation of a retriever, plus an abstention gate."""
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS = [  # (text, who the rule applies to)
    ("Employees may claim up to £400 for home-office equipment.", {"employee"}),
    ("Contractors are not eligible for the home-office equipment budget.", {"contractor"}),
    ("Annual leave for employees is 25 days; up to 5 unused days carry over.", {"employee"}),
    ("Expenses must be submitted within 30 days; anything over £75 needs a receipt.", {"employee", "contractor"}),
    ("Employees receive a laptop refresh every 3 years.", {"employee"}),
    ("Contractors use laptops supplied by their agency or their own approved device.", {"contractor"}),
    ("Password rotation is every 90 days and VPN is required on public Wi-Fi.", {"employee", "contractor"}),
]
QUERIES = {  # (query, answerable from the corpus for this group?)
    "employee": [("How much can I spend on a home office desk?", True), ("How many days of holiday do I get?", True),
                 ("When do I get a new laptop?", True), ("What is the parental leave policy?", False)],
    "contractor": [("How much can I spend on a home office desk?", True), ("How many days of holiday do I get?", False),
                   ("When do I get a new laptop?", True), ("What is the parental leave policy?", False)],
}
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = model.encode([t for t, _ in DOCS], normalize_embeddings=True)


def top1(query, group=None):
    """Best chunk for the query; if group is given, only chunks that apply to it are eligible."""
    s = D @ model.encode(query, normalize_embeddings=True)
    if group:
        s = np.where([group in who for _, who in DOCS], s, -1.0)
    i = int(np.argmax(s))
    return i, float(s[i])


def evaluate(threshold, use_group):
    rows = {}
    for group, qs in QUERIES.items():
        bad_audience = answered_unanswerable = refused_answerable = 0
        for q, answerable in qs:
            i, score = top1(q, group if use_group else None)
            if score < threshold:
                refused_answerable += answerable
                continue
            bad_audience += group not in DOCS[i][1]
            answered_unanswerable += not answerable
        rows[group] = (bad_audience, answered_unanswerable, refused_answerable)
    return rows


if __name__ == "__main__":
    for use_group in (False, True):
        for th in (0.0, 0.5):
            print(f"identity filter={use_group!s:5s} threshold={th}")
            for g, (bad, unans, ref) in evaluate(th, use_group).items():
                print(f"  {g:10s} wrong-audience={bad} answered-unanswerable={unans} refused-answerable={ref}")
    print("\nscores, contractor with filter:")
    for q, ans in QUERIES["contractor"]:
        i, sc = top1(q, "contractor")
        print(f"  {sc:.2f} answerable={ans!s:5s} {q}")
```

```output
identity filter=False threshold=0.0
  employee   wrong-audience=0 answered-unanswerable=1 refused-answerable=0
  contractor wrong-audience=4 answered-unanswerable=2 refused-answerable=0
identity filter=False threshold=0.5
  employee   wrong-audience=0 answered-unanswerable=0 refused-answerable=1
  contractor wrong-audience=2 answered-unanswerable=0 refused-answerable=0
identity filter=True  threshold=0.0
  employee   wrong-audience=0 answered-unanswerable=1 refused-answerable=0
  contractor wrong-audience=0 answered-unanswerable=2 refused-answerable=0
identity filter=True  threshold=0.5
  employee   wrong-audience=0 answered-unanswerable=0 refused-answerable=1
  contractor wrong-audience=0 answered-unanswerable=0 refused-answerable=2

scores, contractor with filter:
  0.34 answerable=True  How much can I spend on a home office desk?
  0.28 answerable=False How many days of holiday do I get?
  0.44 answerable=True  When do I get a new laptop?
  0.23 answerable=False What is the parental leave policy?
```

**What the output means.** Read it group by group, and notice what a pooled number would have hidden.

- **Without identity (filter=False, threshold 0).** Every one of the four contractor queries is answered from an employee-only chunk: the contractor asking about a home-office desk is told the £400 employee rule. Pooled, 4 of 8 answers are wrong-audience (50 per cent); the employee figure is 0 and the contractor figure 100 per cent.
- **A threshold of 0.5 does not fix it.** It cuts the contractor wrong-audience answers from 4 to 2, but only because it refuses the lower-scoring ones; the two that pass are still wrong-audience. A confidence threshold measures similarity, not whether the chunk applies to the asker.
- **With the identity filter**, wrong-audience answers fall to zero for both groups at either threshold, because ineligible chunks cannot be retrieved. The contractor desk query now finds the contractor-ineligibility rule (the correct answer: not eligible).
- **The threshold's cost lands on contractors.** At 0.5 with the filter, the employee loses 1 answerable question and the contractor loses 2 (their correct matches score 0.34 and 0.44, below 0.5), so contractors are refused more often. The contractor chunks are short and terse and match queries less strongly. The unanswerable queries score 0.28 and 0.23 and are rightly refused, but a single global threshold also refuses the answerable ones. A per-group or per-topic threshold, or a better-matched embedding model, is the fix to evaluate.
- **Even at threshold 0 unanswerable questions get answers** (1 for employees, 2 for contractors): the retriever always returns something, which is what abstention is for.

Eight queries per group demonstrate mechanisms, not rates. Scores may differ slightly across library versions; the pattern should not.

**Errors people commonly hit.** Building the audit set from developer-written queries, which sound like the documents; labelling "answerable" by whether *any* chunk matches instead of one that applies to the group; reading one run on a tiny set as a finding.

**How to extend it.** Add groups (region, language, seniority); add real queries from logs, consented and stripped of personal data; report Wilson intervals per cell; sweep thresholds per group (see the [eval metrics visualiser](/playground/eval-metrics)); add paired-prompt tests for the generator.

## A real-world example

The Platform team at Meridian is asked to extend AskMeridian to the 60 contractors who work alongside employees. The evaluation set that gave 91 per cent answer accuracy was written by the team and the HR office, who are all employees. Nobody built a contractor slice. A product manager, Imani, insists on one and gets an HR partner to write 30 contractor questions.

The slice result is poor. Many contractor queries retrieve employee-only chunks, which is the failure above. Worse, the answers read well and cite the handbook, so a contractor would have no reason to doubt them. The team makes four changes. They add `applies_to` metadata at ingestion and a filter from the user's identity (the contractor flag comes from the identity provider, not from anything typed). They commission a short contractor section of the handbook, since half the contractor questions had no answer in the corpus at all. They set a separate refusal threshold per group, chosen from per-group evaluation, and make the refusal message name the right contact. And they write a one-page system card stating the contractor slice's size (30 questions, small), its results with intervals, and that the slice will be re-run each release and grown from real, consented queries.

They also decide, with HR, that questions of the form "Is X entitled to Y?" about a named individual are routed to a person. The assistant informs; people keep deciding.

## Common mistakes

- **Reporting only pooled metrics.** A good average hides a failing group. Always slice.
- **Evaluating on developer-written queries.** They share vocabulary and assumptions with the documents. Use real or realistic users' phrasing.
- **Treating similarity as applicability.** A high score means textual closeness, not "this rule applies to you". Use metadata and identity.
- **One global threshold for everyone.** It can shift refusals onto groups with thinly covered or terse documents. Tune and report per slice.
- **Reading noise as signal.** A gap on 40 queries per group may be chance. Report intervals.
- **Paper oversight.** A human who sees only the conclusion is not oversight.
- **No disclosure or exit.** Users who do not know it is an AI, or cannot reach a person, cannot protect themselves.
- **Documenting once.** A system card never revisited describes a system that no longer exists.

## Best practices

- Tier each use case by consequence and apply controls in proportion; record the tier and the owner.
- Build evaluation sets with slices for each user group that matters, and report per-slice metrics with sample sizes and intervals.
- Audit corpus coverage by audience, language and topic, and treat gaps as content tasks with owners.
- Put audience metadata on documents and enforce eligibility by identity from the identity provider, not from user text.
- Let the system abstain, tune thresholds per slice, and make refusals say where to go next.
- Never let the system act on a person; keep a human who can see the evidence and overrule.
- Disclose that the assistant is automated, state its scope, link sources, and offer a route to a human.
- Keep a one-page system card current.
- Run fairness probes in the regression suite and monitor per-group refusal and complaint rates.

## Performance considerations

Responsible-AI controls cost compute, annotation time and latency, and the costs are modest compared with the harm they prevent, but they need budgeting.

**Evaluation.** Disaggregation multiplies the evaluation set. At roughly 400 queries per group for tight intervals and 5 groups, that is 2,000 labelled queries; at 2 minutes of expert time each (an assumption) about 67 hours. Retrieval runs take seconds; generation-based evaluation costs tokens (see [cost engineering](/learn/cost-engineering)).

**Filters.** An identity-aware metadata filter adds a predicate to each search. In most vector databases a payload filter on a low-cardinality field such as audience costs little, but very selective filters can interact with approximate indexes and reduce recall, so measure recall under the filter, as in [document-level permissions](/learn/access-control-and-privacy/document-level-permissions).

**Abstention.** A threshold check is free; a verifier on every answer adds latency (measure yours), so apply it by risk tier.

**Human review.** The most expensive control. Sampling 2 per cent of 5,000 medium-consequence answers a week at 3 minutes each is 100 reviews and 5 hours: affordable and informative. Reviewing every high-consequence case is a cost of using the system for such decisions.

## Security considerations

- **Fairness data is sensitive data.** Group labels (contractor status, region, language, above all protected characteristics) are personal data. Collect the minimum, restrict access, and check the legal basis before building group slices; take advice before storing protected characteristics to measure bias.
- **Identity must come from the system, not the prompt.** A user who types "I am an employee" must not change which documents they can retrieve. Take eligibility from the authenticated identity (see [authorisation](/learn/access-control-and-privacy/authorisation)).
- **Transparency must not leak.** "Contractors cannot see the restructuring plan" confirms a plan exists. Word refusals neutrally.
- **Review samples hold real queries.** Apply retention and access controls and strip personal data from evaluation sets (see [secure deletion](/learn/enterprise-rag-governance/secure-deletion)).
- **Confident tone is not evidence.** [Prompt injection](/learn/rag-security-fundamentals/prompt-injection) can produce authoritative-looking answers; do not let tone or citations replace review on high-consequence outputs.

## Summary

- Responsible AI for RAG means finding and managing who the system serves poorly, how users may over-trust it, how transparent it is, and who is accountable, in proportion to the consequence of each use case.
- Pooled metrics hide group gaps; disaggregated evaluation with enough queries per group (and intervals) is the basic tool.
- Retrieval similarity is not applicability: audience metadata and identity-aware filters prevent an employee-only rule being served to a contractor, and a global refusal threshold can put the cost of refusals on the groups whose documents are thinnest.
- Human oversight is real only if the reviewer sees the evidence, has time and authority to overrule, and is audited; for decisions about people the system informs and never decides.
- A short, current system card (purpose, corpus, versions, disaggregated results, accepted risks, review date), plus frameworks such as the NIST AI RMF and the EU AI Act's risk tiers as vocabulary, turns good intentions into reviewable evidence.

## Exercises

**1. Concept.** Why might a RAG system with 92 per cent overall accuracy be unacceptable, and which two practices would reveal it?

<details>
<summary>Show solution</summary>

The average can combine a very high score for a majority group with a poor one for a minority (98 against 70 per cent). Disaggregated evaluation with enough queries per group, and per-group refusal rates with confidence intervals, reveal it; so does a coverage audit by audience.

</details>

**2. Calculation.** A group has 6 refusals of answerable questions out of 30. Compute the observed rate, and say whether you could tell this group apart from another group with 3 of 30, using the idea of overlapping intervals (an approximate answer is fine).

<details>
<summary>Show solution</summary>

The observed rate is 6/30 = 0.20; the other group's is 3/30 = 0.10. A Wilson 95 per cent interval for 6 of 30 is roughly 0.10 to 0.37, and for 3 of 30 roughly 0.03 to 0.26. They overlap substantially, so with 30 queries per group the data does not distinguish the groups. You would need several times more queries (or a larger observed gap) before concluding that the groups differ.

</details>

**3. Code.** Using the lesson's code as a base, write a function that, for each group, finds the threshold from `[0.2, 0.3, 0.4, 0.5]` that minimises `answered_unanswerable + refused_answerable` for that group (with the identity filter on). What does it suggest about using one threshold?

<details>
<summary>Show solution</summary>

```python
def best_threshold(group, grid=(0.2, 0.3, 0.4, 0.5)):
    best = None
    for th in grid:
        bad, unans, ref = evaluate(th, True)[group]
        cost = unans + ref
        if best is None or cost < best[1]:
            best = (th, cost)
    return best

for g in QUERIES:
    print(g, best_threshold(g))
```

Running it gives `employee (0.2, 1)` and `contractor (0.3, 0)`. Contractors reach zero errors at 0.3, which refuses both unanswerable queries (scores 0.28 and 0.23) and keeps both answerable ones (0.34 and 0.44). Employees cannot get below one error on this grid: their unanswerable parental-leave query scores 0.403 and their answerable holiday query 0.425, so only a threshold near 0.41, which is not on the grid, separates them. The groups want different thresholds, so one global value is a compromise that costs one group more. With eight queries per group these choices are unstable; confirm on a larger set before adopting per-group thresholds.

</details>

**4. Design.** AskMeridian is asked to help managers decide which employees to put on a performance-improvement plan by summarising each employee's documents. Classify the use case, decide whether you would build it, and if so list the controls.

<details>
<summary>Show solution</summary>

High consequence: it informs a decision about an individual's employment, a regulated, high-risk area in many jurisdictions. Do not build it without legal, HR and ethics sign-off. If it proceeds, the system never recommends an outcome; it presents verbatim cited evidence; disaggregated evaluation is documented before launch; the manager and an HR reviewer both see the evidence and record their own reasoning; the employee is told automated tools were used and can contest the outcome; use is logged and sampled; protected characteristics and unrelated personal data are out of scope. The system retrieves and presents; an accountable human decides.

</details>

## Check your understanding

```quiz
[
  {
    "q": "In the lesson's audit, why did a similarity threshold of 0.5 not remove wrong-audience answers for contractors when retrieval ignored identity?",
    "options": [
      "Thresholds only work with BM25",
      "A similarity score measures textual closeness, not whether the chunk applies to the asker, so some wrong-audience chunks still scored above the threshold",
      "The threshold was applied to the wrong model",
      "Contractor queries have no embeddings"
    ],
    "answer": 1,
    "why": "Two contractor queries still passed the threshold and were answered from employee-only chunks; audience needs metadata and an identity filter."
  },
  {
    "q": "With the identity filter on and a global threshold of 0.5, which group bore more refusals of answerable questions, and why?",
    "options": [
      "Employees, because they have more documents",
      "Contractors, because their short, terse chunks scored lower (0.34 and 0.44) against their queries",
      "Neither; both lost the same number",
      "Employees, because of the filter"
    ],
    "answer": 1,
    "why": "Contractors lost 2 answerable questions against 1 for employees, an uneven burden produced by a single threshold."
  },
  {
    "q": "Group A has 5 per cent refusals and group B 20 per cent, each from 40 queries. What can you conclude?",
    "options": [
      "B is definitely worse",
      "The 95 per cent intervals overlap, so you cannot yet tell the groups apart and need more queries",
      "A is definitely worse",
      "Nothing can ever be concluded from rates"
    ],
    "answer": 1,
    "why": "The intervals are about 0.01 to 0.17 and 0.10 to 0.35, which overlap; at 400 queries each they do not."
  },
  {
    "q": "Which description best fits meaningful human oversight of a RAG-assisted decision about an individual?",
    "options": [
      "A human clicks approve on the system's conclusion",
      "The reviewer sees the cited evidence, has time and authority to overrule, and is audited for actually reviewing",
      "The system acts automatically and a human is notified afterwards",
      "A human reviews only answers that the system flags as wrong"
    ],
    "answer": 1,
    "why": "Oversight is a control only if the reviewer can disagree, which needs evidence, time, authority and checks that review is real."
  },
  {
    "q": "What is the main purpose of a system card for a RAG deployment?",
    "options": [
      "To advertise the product",
      "To record purpose, corpus, versions, disaggregated evaluation, risks accepted and review date in a form others can review",
      "To store users' personal data",
      "To replace audit logs"
    ],
    "answer": 1,
    "why": "A system card makes decisions and their evidence reviewable by people who were not there, and is only useful if kept current."
  }
]
```

## Further reading

- NIST (2023), *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, at [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework), for the Govern, Map, Measure and Manage functions.
- Regulation (EU) 2024/1689 (the EU Artificial Intelligence Act), available on EUR-Lex, for the risk-based categories and obligations; check the current text and guidance for application dates and definitions.
- Mitchell, M. et al. (2019), "Model Cards for Model Reporting", Proceedings of the Conference on Fairness, Accountability, and Transparency.
- Gebru, T. et al. (2021), "Datasheets for Datasets", Communications of the ACM, on documenting how datasets are built and should be used.
- Parasuraman, R. and Manzey, D. (2010), "Complacency and Bias in Human Use of Automation: An Attentional Integration", Human Factors, on automation bias.
