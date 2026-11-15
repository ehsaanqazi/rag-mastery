Some questions cannot be answered by one search. "Does the product that exports PDFs support single sign-on, and how long does Relay keep failed events?" needs two independent lookups and one that depends on the first. Three earlier lessons cover the basics of this: [Query decomposition](/learn/advanced-retrieval/query-decomposition) splits a question into sub-questions and chains answers, [Planning and reasoning](/learn/agentic-rag/planning-and-reasoning) compares planning strategies for agents, and [Iterative search](/learn/agentic-rag/iterative-search) covers loops that search, read and search again.

This lesson takes the next step, which is what the research has been doing since 2022: treating a plan as a **program** that can be validated, scheduled, costed and evaluated, much as a database treats a query. You will see why a plan is better as a graph than as a chain, how the main published planners differ, and how to measure a plan rather than only its final answer. The code builds a plan executor with validation, concurrent scheduling, failure propagation and plan arithmetic, all of which run without an API key.

## Learning objectives

- Distinguish interleaved, plan-first and graph-structured planning, and place Self-Ask, IRCoT, ReAct, ReWOO and LLMCompiler among them.
- Represent a plan as a validated directed acyclic graph and schedule its steps level by level, with concurrency.
- Compute a plan's total work, critical-path latency, speed-up and end-to-end reliability, and say what each ignores.
- Order retrieval operations by cost and selectivity using the optimiser rule, and verify it numerically.
- Design plan-level evaluation that exposes planner errors invisible in final-answer accuracy.

## Prerequisites

- [Query decomposition](/learn/advanced-retrieval/query-decomposition): sub-questions, plans as data and error compounding along a chain.
- [Planning and reasoning](/learn/agentic-rag/planning-and-reasoning): the three planning strategies and replanning.
- [Iterative search](/learn/agentic-rag/iterative-search): the search-read-search loop.
- [Adaptive RAG](/learn/advanced-rag-patterns/adaptive-rag): choosing a strategy by question complexity.
- [Modern RAG research](/learn/advanced-rag-research/modern-rag-research): the evidence ladder and how to read results.

## Why this matters

A planner's output is the most consequential text in a multi-step system, because everything downstream trusts it. A wrong step silently poisons later steps; an unnecessary dependency serialises work that could run in parallel; a plan referring to an answer that no step produces fails at run time, at the user's expense. These are not model-quality problems in the usual sense. They are *program* problems, and software engineering already knows how to handle programs: check them before running, schedule them by their dependencies, cost them, and test them.

The research also matters because choices about planning change latency and cost by large factors. A system that decides each step after seeing the last pays a model call per step, serially. A system that plans everything first pays one planning call but cannot adapt. A system that plans a graph and executes independent branches concurrently can pay for the width of the question, not its length. Knowing the trade lets you choose, and lets you tell which published claims apply to your workload.

## The intuition

Think of a kitchen preparing a meal. A beginner cooks by reading one instruction at a time and doing it: chop onions, then boil water, then cook pasta. Everything is sequential, but she can adjust if the onions are bad. A better cook writes the whole recipe up front, notices that the water can boil while the onions are chopped, and does both at once; the meal is ready sooner, but if the onions turn out bad, the recipe has to be rewritten. A head chef does a third thing: writes the recipe as a dependency chart (sauce needs onions; pasta needs boiling water; plating needs both), checks it makes sense before cooking begins, and assigns independent tasks to different people.

A plan graph is the head chef's chart. Its two virtues are visible in the metaphor: independent branches run together, and a malformed chart (a dish that needs itself) can be caught before anyone lights a stove.

## How it works

### Three planning shapes

**Interleaved (decide, observe, decide).** The model emits one step, sees the result, and chooses the next. **Self-Ask** (Press and colleagues, 2022) has the model write follow-up questions and answer each; **IRCoT** (Trivedi and colleagues, 2022) interleaves retrieval with chain-of-thought sentences, using each new thought as the next query; **ReAct** (Yao and colleagues, 2022) interleaves reasoning, tool calls and observations. They adapt naturally to surprises and handle questions whose later steps depend on what earlier steps found. Their cost is a model call per step, strictly in series.

**Plan first.** The model writes the whole plan before any retrieval. **Least-to-Most** prompting (Zhou and colleagues, 2022) and **Plan-and-Solve** (Wang and colleagues, 2023) decompose first and solve afterwards. **ReWOO** (Xu and colleagues, 2023) goes a step further: the plan contains placeholders for the results of earlier tools, so the planner is called once and the reasoning is "decoupled from observations"; the paper's motivation was the cost of re-sending the whole growing context at every step of an interleaved agent. The price is rigidity, since a surprise in step one cannot change step two without replanning.

**Graph-structured.** The plan is a directed acyclic graph (DAG): nodes are retrieval or tool steps, edges are data dependencies. **LLMCompiler** (Kim and colleagues, 2023) is the best-known example. A planner emits the graph, a scheduler launches every step whose dependencies are complete, and independent steps run in parallel; the authors report reduced latency and cost on multi-call benchmarks against sequential reasoning-and-acting agents. The graph shape is what makes parallelism visible to the scheduler.

| Shape | Model calls | Parallelism | Adapts to surprises | Typical failure |
| --- | --- | --- | --- | --- |
| Interleaved | One per step | None | Yes, every step | Drift, runaway loops |
| Plan first (chain) | One for the plan, plus per-step reads | Little | Only by replanning | A wrong early step corrupts the rest |
| Graph | One for the plan, plus per-step reads | Yes, across independent branches | By replanning, with checkpoints | Invalid graphs; over-decomposition |

A fourth idea sits above all three: **choose the shape per question**. Adaptive-RAG (Jeong and colleagues, 2024) trains a classifier to route a question to no retrieval, single-step retrieval or iterative multi-step retrieval according to its complexity ([Adaptive RAG](/learn/advanced-rag-patterns/adaptive-rag)). Most traffic in a real system is simple, so the cheapest adequate strategy should be the default.

### A plan as a program

If a plan is data, you can check it before it runs. The checks that matter, in order of cost:

1. **Schema.** Every step has an id, a question, a list of dependencies and an operation. A structured-output API makes this nearly free ([Planning and reasoning](/learn/agentic-rag/planning-and-reasoning) shows the pattern).
2. **Referential integrity.** Every dependency names a step that exists; every placeholder in a question names a step it depends on.
3. **Acyclicity.** A dependency cycle can never be scheduled.
4. **Necessity.** A step that no later step uses and that is not the answer is wasted work (a heuristic, since a step may feed the final answer directly).
5. **Budget.** Step count, estimated tokens and critical-path latency are within limits.

The first three are mechanical and catch real planner errors. Validation failures are fed back to the planner as a repair request, with a bounded number of attempts.

### Scheduling and failure

Given a valid DAG, **topological generations** group steps into levels: level 0 has no dependencies, level 1 depends only on level 0, and so on. Run each level's steps concurrently, then move on. A step whose dependency produced no answer is **skipped**, not run with a missing value, and the final answer is composed from what succeeded, flagged as partial. An answer of "none" is a legitimate outcome (the corpus may not contain the fact) and must stay distinct from an error.

### The optimiser analogy

A database separates a *logical* plan (what to compute) from *physical* operators (how). Retrieval planning has the same split. A step "find the product that exports PDFs" can run as BM25, dense search, a metadata lookup or a graph query; choosing among them is an optimisation problem with costs and selectivities. Two lessons transfer directly: **put cheap, selective operations first** (a metadata filter before a cross-encoder), as the maths shows, and **reuse results** (cache identical sub-queries across steps and across users).

### What the research evaluates, and its caveats

Multi-step planners are mostly evaluated on multi-hop question sets such as HotpotQA (Yang and colleagues, 2018), 2WikiMultihopQA (Ho and colleagues, 2020) and MuSiQue (Trivedi and colleagues, 2022). Two cautions matter. First, many "multi-hop" questions can be answered by a single retrieval, because of shortcuts in how they were built (Min and colleagues, 2019, argued this for HotpotQA); MuSiQue was constructed to require connected reasoning and to resist such shortcuts. A planner that scores well on a shortcut-prone set has shown less than it seems. Second, these are Wikipedia questions with clean entities; a corpus of internal policies has abbreviations, versions and gaps, so plan quality has to be measured on your questions.

### Evaluating the plan, not only the answer

Final-answer accuracy hides planner errors, because a lucky retrieval can rescue a bad plan and a good plan can fail at one step. Add plan-level measurements:

- **Validity rate**: the share of plans that pass validation on the first attempt.
- **Necessity and sufficiency**: against a human-written reference plan for a sample, which needed steps are missing and which steps are redundant.
- **Dependency correctness**: false dependencies (serialising independent work) and missing dependencies (using an answer before it exists).
- **Per-step success**: whether each step's retrieval contained its evidence, which locates the failing step.
- **Critical-path latency and total tokens**, per plan.

## The mathematics

### Work, critical path and speed-up

**Intuition.** *Work* is the time if every step ran one after another. The *critical path* is the longest chain of dependent steps; no schedule can finish faster than that, however many workers you have.

Let step $i$ take time $c_i$. Then

$$W = \sum_i c_i, \qquad L = \max_{\text{paths } \pi} \sum_{i \in \pi} c_i, \qquad \text{speed-up} \le \frac{W}{L},$$

where $W$ is the work, $L$ the critical-path length (the maximum over all dependency paths $\pi$ of the summed step times) and the speed-up is bounded by their ratio. Here $W/L$ is an upper bound, since scheduling overheads and limited concurrency reduce it.

**Worked example by hand.** Three steps, each costing 0.7 seconds (a 0.2-second search plus a 0.5-second model call). As a chain ($a \to b \to c$), $W = 2.1$ and $L = 2.1$: speed-up 1.0. As a DAG with only $a \to b$ and $c$ independent, $W = 2.1$ but $L = 1.4$ (the path $a \to b$), so the bound is $2.1/1.4 = 1.5$. The code below measures 2.1 seconds and 1.4 seconds.

### End-to-end reliability

If every step must be correct for the final answer to be correct, and step $i$ succeeds with probability $p_i$ independently of the others, then

$$P(\text{correct}) = \prod_i p_i .$$

With $p_i = 0.9$ for each of three steps, $0.9^3 = 0.729$. Parallelism improves latency but does **not** improve this; only better steps, verification or redundancy do. This is why over-decomposition hurts: each extra step multiplies in another factor below 1. A question that needs two steps should not be planned as five.

### Ordering filters by cost and selectivity

**Intuition.** If two checks must both pass, run the one that is cheap and rejects many first, so the expensive one runs on fewer items.

Let operation $i$ cost $c_i$ per item and pass a fraction $s_i$ of items. For two operations in sequence, the expected cost per input item is $c_1 + s_1 c_2$ if operation 1 runs first, or $c_2 + s_2 c_1$ if operation 2 does. Operation 1 should go first when $c_1 + s_1 c_2 < c_2 + s_2 c_1$, which rearranges to

$$\frac{c_1}{1 - s_1} < \frac{c_2}{1 - s_2}.$$

For more operations, sort by this ratio (valid when the passes are independent).

**Worked example by hand.** A metadata filter costs 1 unit per candidate and passes 50 per cent ($s = 0.5$); a rerank costs 10 and passes 10 per cent ($s = 0.1$). Filter first: $1 + 0.5 \times 10 = 6$. Rerank first: $10 + 0.1 \times 1 = 10.1$. The ratios are $1/0.5 = 2$ and $10/0.9 = 11.1$, so the filter goes first. The code reproduces these two numbers.

## In code

**The problem.** Build the machinery around a plan, with no model in the loop, so the mechanism is visible and testable: a typed plan, a validator that catches cycles, unknown dependencies and unbound placeholders, a scheduler that runs each dependency level concurrently, and failure propagation that skips steps whose inputs are missing. The question is the one from the introduction, over a seven-document Meridian corpus, with BM25 for retrieval.

**What is stand-in.** The planner is replaced by hand-written plans (a real planner is shown afterwards), and the "reader" is a small regular-expression extractor with simulated latencies (0.2 seconds per search and 0.5 seconds per model call, assumed figures). So the timings show scheduling, not model speed.

**Dependencies.** `networkx`, `pydantic`, `rank-bm25`. No download, no API key.

```bash
pip install networkx pydantic rank-bm25
```

**The source.**

```python
import asyncio, re, time
import networkx as nx
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

# --- 1. A tiny Meridian corpus and a retriever --------------------------------------------------------
DOCS = [
    "Meridian Relay retries failed deliveries for up to 24 hours and then moves the event to a dead-letter queue.",
    "Events in the Meridian Relay dead-letter queue are kept for 14 days.",
    "Meridian Relay accepts events of up to 256 KB.",
    "Meridian Reports supports PDF and CSV export. SAML single sign-on is available on the Enterprise plan.",
    "Meridian Sync keeps files synchronised across devices and supports offline editing.",
    "The VPN is required whenever you use public Wi-Fi.",
    "Expenses must be submitted within 30 days and anything over 75 pounds needs a receipt.",
]
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tok(d) for d in DOCS])
def retrieve(query: str):
    scores = bm25.get_scores(tok(query))
    return DOCS[int(scores.argmax())] if scores.max() > 0 else None     # None means "nothing matched"

# --- 2. A plan is data: steps, dependencies, placeholders ------------------------------------------------
class Step(BaseModel):
    id: str
    question: str                 # may contain {other_id} placeholders for earlier answers
    depends_on: list[str] = []
    extract: str                  # how to read the answer from the retrieved passage

class Plan(BaseModel):
    steps: list[Step]

def validate(plan: Plan) -> list[str]:
    problems, ids = [], {s.id for s in plan.steps}
    g = nx.DiGraph()
    for s in plan.steps:
        g.add_node(s.id)
        for d in s.depends_on:
            if d not in ids: problems.append(f"{s.id} depends on unknown step {d}")
            else: g.add_edge(d, s.id)
        for ph in re.findall(r"\{(\w+)\}", s.question):
            if ph not in s.depends_on: problems.append(f"{s.id} uses {{{ph}}} without depending on it")
    if not nx.is_directed_acyclic_graph(g): problems.append("the plan contains a cycle")
    return problems

# --- 3. Execution: run each dependency level concurrently ------------------------------------------------
RETRIEVE_S, READ_S = 0.2, 0.5     # simulated latencies of a search and of a model call (assumptions)
def read(kind: str, passage: str, question: str):
    if kind == "product": m = re.search(r"Meridian (Sync|Relay|Reports)", passage); return m and "Meridian " + m.group(1)
    if kind == "days":    m = re.search(r"(\d+) days", passage); return m and f"{m.group(1)} days"
    if kind == "saml":    return "yes, on the Enterprise plan" if "SAML" in passage else "no"
    return None

async def run_step(step, answers):
    question = step.question.format(**{d: answers[d] for d in step.depends_on})
    await asyncio.sleep(RETRIEVE_S); passage = retrieve(question)
    await asyncio.sleep(READ_S)
    return step.id, (read(step.extract, passage, question) if passage else None)

async def execute(plan: Plan, concurrent=True):
    g = nx.DiGraph([(d, s.id) for s in plan.steps for d in s.depends_on]); g.add_nodes_from(s.id for s in plan.steps)
    by_id, answers, t0 = {s.id: s for s in plan.steps}, {}, time.perf_counter()
    for level in nx.topological_generations(g):
        runnable = [by_id[i] for i in level if all(answers.get(d) for d in by_id[i].depends_on)]
        if concurrent: results = await asyncio.gather(*(run_step(s, answers) for s in runnable))
        else:          results = [await run_step(s, answers) for s in runnable]
        answers.update(dict(results))
    return answers, time.perf_counter() - t0

chain = Plan(steps=[                                              # the naive plan: one long chain
    Step(id="a", question="Which Meridian product offers PDF and CSV export?", extract="product"),
    Step(id="b", question="Does {a} support SAML single sign-on?", depends_on=["a"], extract="saml"),
    Step(id="c", question="How many days are events kept in the Relay dead-letter queue?", depends_on=["b"], extract="days")])
dag = Plan(steps=[                                                # the same steps, true dependencies only
    chain.steps[0], chain.steps[1],
    Step(id="c", question="How many days are events kept in the Relay dead-letter queue?", extract="days")])
bad = Plan(steps=[Step(id="a", question="Does {z} support SAML?", depends_on=["b"], extract="saml"),
                  Step(id="b", question="Which product has {a}?", depends_on=["a", "ghost"], extract="product")])
dead_end = Plan(steps=[Step(id="a", question="Which canteen serves lunch?", extract="product"),
                       Step(id="b", question="Does {a} support SAML single sign-on?", depends_on=["a"], extract="saml"),
                       dag.steps[2]])

for name, p in (("chain", chain), ("dag", dag), ("bad", bad)):
    print(f"validate({name}):", validate(p) or "ok")
for name, p, conc in (("chain, sequential", chain, False), ("dag, sequential", dag, False), ("dag, concurrent", dag, True),
                      ("dead end, concurrent", dead_end, True)):
    answers, secs = asyncio.run(execute(p, conc))
    print(f"{name:20s} {secs:4.1f} s  answers={answers}")

# --- 4. Plan arithmetic: work, critical path and filter ordering ----------------------------------------------------------------
def critical_path(plan, cost):
    g = nx.DiGraph([(d, s.id) for s in plan.steps for d in s.depends_on]); g.add_nodes_from(s.id for s in plan.steps)
    return len(nx.dag_longest_path(g)) * cost
step_cost = RETRIEVE_S + READ_S
for name, p in (("chain", chain), ("dag", dag)):
    print(f"{name}: work {len(p.steps) * step_cost:.1f} s, critical path {critical_path(p, step_cost):.1f} s")
order_cost = lambda first, second: first[0] + first[1] * second[0]    # (cost, pass rate)
metadata, rerank = (1.0, 0.5), (10.0, 0.1)
print("filter then rerank:", order_cost(metadata, rerank), "| rerank then filter:", order_cost(rerank, metadata))
```

**What the important lines do.** `Plan` and `Step` are Pydantic models, so a plan is typed data. `validate` builds a graph with `networkx`, reporting unknown dependencies, placeholders not listed in `depends_on` and cycles. `execute` groups steps with `nx.topological_generations` and runs each level either with `asyncio.gather` (concurrent) or one by one; a step runs only if all its dependencies have a non-empty answer, which implements failure propagation. `retrieve` returns `None` when BM25 finds no matching term, so "nothing matched" is distinct from a wrong answer. The final lines compute work, critical path, and the filter-ordering costs.

**Output.** (Times are wall-clock and vary slightly between runs.)

```output
validate(chain): ok
validate(dag): ok
validate(bad): ['a uses {z} without depending on it', 'b depends on unknown step ghost', 'the plan contains a cycle']
chain, sequential     2.1 s  answers={'a': 'Meridian Reports', 'b': 'yes, on the Enterprise plan', 'c': '14 days'}
dag, sequential       2.1 s  answers={'a': 'Meridian Reports', 'c': '14 days', 'b': 'yes, on the Enterprise plan'}
dag, concurrent       1.4 s  answers={'a': 'Meridian Reports', 'c': '14 days', 'b': 'yes, on the Enterprise plan'}
dead end, concurrent  0.7 s  answers={'a': None, 'c': '14 days'}
chain: work 2.1 s, critical path 2.1 s
dag: work 2.1 s, critical path 1.4 s
filter then rerank: 6.0 | rerank then filter: 10.1
```

**What the output means.** `validate(bad)` catches all three planted faults before anything runs. The chain and the DAG return the same three answers, but the DAG with concurrent execution finishes in 1.4 seconds instead of 2.1, because the independent step `c` runs alongside `a`; the DAG run sequentially takes 2.1 seconds, which shows the saving comes from scheduling, not from the plan alone. In the `dead end` plan, step `a` finds nothing (no document mentions a canteen), so step `b`, which needs it, is skipped: the answer contains `a` as `None` and `c` as `14 days`, with `b` absent. That is a correct partial result, and the caller can say so, rather than a confident answer built from nothing. The last lines match the hand calculations: the chain has work and critical path both 2.1, the DAG has a critical path of 1.4, and filtering before reranking costs 6.0 against 10.1.

**Common errors.** Running a dependent step with a missing value (use an explicit skip); treating a cyclic plan as "just slow" (validate first); using `asyncio.gather` over steps that mutate shared state; counting the planner call as free in latency estimates.

**The planner with Claude.** The block below was **not executed** (it needs an API key), and the plan it returns will differ between runs. It asks the model for a `Plan` using structured output, validates it with the function above, and feeds validation errors back for up to three attempts.

```python
import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from the environment

PLANNER_SYSTEM = (
    "Break the user's question into the smallest set of retrieval steps. Give each step a short id, a question, "
    "and the ids of earlier steps whose answers it needs; write those answers into the question as {id} "
    "placeholders. Steps that do not need each other must not depend on each other. "
    "extract must be one of: product, days, saml."
)

def make_plan(question: str, attempts: int = 3) -> Plan:
    feedback = ""
    for _ in range(attempts):
        response = client.messages.parse(
            model="claude-opus-5-5",
            max_tokens=1024,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": question + feedback}],
            output_format=Plan,
        )
        plan = response.parsed_output
        problems = validate(plan)                 # the validator from the main example
        if not problems:
            return plan
        feedback = "\n\nYour previous plan was invalid: " + "; ".join(problems) + ". Return a corrected plan."
    raise RuntimeError("the planner did not produce a valid plan")
```

**How to extend it.** Add a necessity check (steps with no consumers that are not the answer), a per-step budget, and a cache keyed on the normalised question. Replace the regular-expression reader with a model call whose output is validated. Add a replan step that, when a level fails, re-invokes the planner with the failed step's question and what was learned.

## A real-world example

AskMeridian's Platform team notices that compound IT questions are its slowest: "Which VPN client version do I need for error E-4021, and does it work on the company laptop model?" The agent loop decides one step at a time and takes four model turns in sequence. Latency traces show that the second and third turns do not use each other's output: one looks up the error code, the other the laptop's supported clients, and only the fourth combines them.

The team replaces the loop for this class of question with a planned graph and the three checks from this lesson. The planner emits two independent lookups and one combining step; the validator rejects the occasional plan that invents a dependency or leaves a placeholder unbound, and the planner retries with the error message. Independent lookups run concurrently, so the critical path is two levels rather than four. They keep the interleaved agent for open-ended investigations, where adapting to surprises is the point, and route between them with a simple complexity classifier, as Adaptive-RAG suggests. They also add plan-level logging (validity, steps, critical path, per-step success) and discover that most failures sit in one step: the retrieval of laptop-model compatibility, which turns out to need an exact match on the model name. The fix is a keyword channel for that step, not a better planner.

## Common mistakes

- **Over-decomposing.** Each extra step multiplies in another success probability below 1 and adds latency; plan the fewest steps that cover the question.
- **Serial plans for parallel questions.** A planner that writes a chain by habit leaves speed on the table; ask for dependencies explicitly and validate.
- **Running unvalidated plans.** Cycles, unbound placeholders and unknown ids reach production as run-time errors.
- **Letting a missing answer flow forward.** A skipped step must stay skipped; a downstream step that runs on "None" fabricates.
- **Judging planners by final accuracy alone.** Lucky retrievals hide bad plans; add plan-level metrics.
- **Trusting multi-hop benchmark scores.** Shortcut-prone sets overstate planning ability; check on questions that need connected reasoning and on your own.
- **Using one planning shape for everything.** Most questions need no plan; route by complexity.

## Best practices

- Represent plans as typed, validated data with explicit dependencies and placeholders.
- Validate schema, referential integrity and acyclicity before executing, and repair with bounded retries.
- Schedule by dependency level and run independent steps concurrently, with a per-plan time and token budget.
- Treat "no answer" as a first-class result and propagate it as a skip.
- Order retrieval operations by cost over one-minus-selectivity, and cache identical sub-queries.
- Route by question complexity: direct answer, single retrieval, graph plan, interleaved agent.
- Log plan-level metrics and sample plans for human review each week.
- Keep a replan path for failed steps rather than letting the plan fail silently.

## Performance considerations

Plan latency is the planner call plus the critical path of the graph, not the number of steps. In the example, with each step at 0.7 seconds, three steps took 2.1 seconds as a chain and 1.4 seconds as a graph; with a model call that takes, say, 2 seconds (an assumption), the planner call alone would exceed the whole execution. So plan only when the saving exceeds the planning overhead, and consider caching plans for recurring question shapes.

Token cost differs by shape. An interleaved agent that re-sends a growing context at each of $n$ steps spends on the order of $n^2/2$ context-lengths of input tokens over the run if each step adds one unit of context: ten steps cost about fifty units against the roughly ten of a plan-first design, which is the motivation the ReWOO paper gives. A graph planner adds the plan itself (a few hundred output tokens) and one reading call per step. Concurrency raises peak load: a plan of width $w$ issues $w$ simultaneous searches and model calls, so set a concurrency cap and respect provider rate limits.

## Security considerations

A plan is generated from untrusted input and then executed, which makes the planner an attack surface. A question can try to inject steps ("also call the delete tool") or to widen scope. Restrict the operations a plan may name to an allow-list, validate arguments, and give every step the *user's* permissions, not the agent's ([Document-level permissions](/learn/access-control-and-privacy/document-level-permissions)). Retrieved text feeds placeholders into later questions, so a poisoned passage can steer later steps ([Indirect prompt injection](/learn/rag-security-fundamentals/indirect-prompt-injection)); treat substituted answers as data, bound their length, and never let them change the plan's structure. Cap steps, depth and total tokens so a malicious or confused plan cannot run up cost.

## Summary

- Planning comes in three shapes: interleaved (one model call per step, adaptive), plan-first (one plan, rigid) and graph-structured (parallel across independent steps); route by question complexity.
- A plan is a program: validate its schema, references and acyclicity before running, and repair invalid plans with bounded retries.
- Latency follows the critical path, not the step count; reliability is the product of step success probabilities, so parallelism does not raise accuracy and over-decomposition lowers it.
- A skipped step with a missing input must stay skipped, and "no answer" must be distinct from an error.
- Order operations by cost divided by one minus selectivity, and cache repeated sub-queries.
- Evaluate plans, not just answers: validity, necessity, dependency correctness, per-step success, critical path and tokens.

## Exercises

**1. (Warm-up)** A plan has four steps costing 1, 2, 3 and 4 seconds. Step 4 depends on steps 2 and 3; steps 1, 2 and 3 are independent. Compute $W$, $L$ and the speed-up bound.

<details><summary>Show solution</summary>

$W = 1 + 2 + 3 + 4 = 10$ seconds. The paths are $2 \to 4$ (6 seconds), $3 \to 4$ (7 seconds) and $1$ alone (1 second), so $L = 7$ seconds and the bound is $10/7 = 1.43$. Even with unlimited workers, 7 seconds is the floor.

</details>

**2. (Applied)** A pipeline must apply a language filter (cost 0.1, passes 80 per cent), a permissions check (cost 0.5, passes 30 per cent) and a cross-encoder (cost 20, passes 20 per cent). Order them using the ratio rule and compute the expected cost per input item.

<details><summary>Show solution</summary>

Ratios $c/(1-s)$: language $0.1/0.2 = 0.5$; permissions $0.5/0.7 = 0.714$; cross-encoder $20/0.8 = 25$. Order: language, permissions, cross-encoder. Expected cost: $0.1 + 0.8 \times 0.5 + 0.8 \times 0.3 \times 20 = 0.1 + 0.4 + 4.8 = 5.3$. For comparison, putting the cross-encoder first costs $20 + 0.2 \times 0.1 + 0.2 \times 0.8 \times 0.5 = 20.1$.

</details>

**3. (Design)** Meridian wants an AskMeridian planner for questions that compare two policies ("how does the contractor equipment budget differ from the employee one?"). Sketch the plan graph, the validation checks and the plan-level metrics you would log.

<details><summary>Show solution</summary>

Graph: step A retrieves the employee home-office equipment budget, step B retrieves the contractor eligibility rule, both independent, and step C compares them using A and B. Checks: schema, both placeholders `{A}` and `{B}` listed in C's dependencies, no cycles, at most four steps. Metrics: validity rate on first attempt, whether A and B were planned as independent (false-dependency rate), per-step success (does each retrieval contain its evidence), critical path length (two levels), and total tokens. For a comparison question the plan should never have more than about three steps; a plan with six steps is a warning of over-decomposition.

</details>

**4. (Challenge)** Extend the executor with a *cache* keyed on the lower-cased, whitespace-normalised question so that two steps with the same question run once. Then explain why caching changes the critical-path calculation but not the reliability product.

<details><summary>Show solution</summary>

Keep a dictionary `cache` shared by `run_step`: compute `key = " ".join(question.lower().split())`; if `key` is in the cache, return its answer without sleeping; otherwise run the step and store the result. Because concurrent steps may race, store an `asyncio.Future` in the cache so a second step awaits the first instead of duplicating work. Caching shortens the time of the repeated step to nearly zero, so the critical path can shrink if the repeated step lay on it, and the work $W$ falls. The reliability product is unchanged in a useful sense: both steps share one outcome, so their errors are perfectly correlated; if the first retrieval was wrong, both consumers inherit the error. Independence, which the product formula assumes, no longer holds, and the true success probability is closer to $p$ than to $p^2$ for those two steps.

</details>

## Check your understanding

```quiz
[
  {
    "q": "What is the main advantage of a graph-structured plan over a chain when two sub-questions are independent?",
    "options": [
      "Plans no longer need validation",
      "The final answer becomes more accurate",
      "Fewer model tokens are needed for the final answer",
      "Independent steps can run concurrently, so latency follows the critical path"
    ],
    "answer": 3,
    "why": "A graph makes independence explicit so a scheduler can run steps in parallel. Reliability still depends on every step being right."
  },
  {
    "q": "Three steps each succeed with probability 0.9 and all must be right. What is the probability the plan is correct, and does running two steps in parallel change it?",
    "options": [
      "0.9, and yes",
      "0.729, and no",
      "0.729, and yes",
      "0.81, and no"
    ],
    "answer": 1,
    "why": "0.9 cubed is 0.729. Parallelism changes latency, not the probability that each step is correct."
  },
  {
    "q": "A metadata filter costs 1 and passes 50 percent; a reranker costs 10 and passes 10 percent. Which order has the lower expected cost per candidate?",
    "options": [
      "Reranker first, cost 10.1",
      "Both orders cost the same",
      "Filter first, cost 6.0",
      "It depends on the number of candidates"
    ],
    "answer": 2,
    "why": "Filter first costs 1 + 0.5 x 10 = 6; reranker first costs 10 + 0.1 x 1 = 10.1. The ratio c/(1-s) gives 2 versus 11.1."
  },
  {
    "q": "In the lesson's dead-end example, step a finds nothing and step b depends on a. What should the executor do with b?",
    "options": [
      "Skip b and return a partial result with a as None",
      "Run b with an empty string",
      "Retry a forever",
      "Replace b with a guess from the model's memory"
    ],
    "answer": 0,
    "why": "A step whose input is missing must be skipped; running it would fabricate an answer from nothing. The caller reports the answer as partial."
  },
  {
    "q": "Why can final-answer accuracy alone be a poor measure of a planner?",
    "options": [
      "Accuracy cannot be computed for multi-step questions",
      "Planners do not affect accuracy",
      "A lucky retrieval can rescue a bad plan and a good plan can fail at one step, so planner errors stay hidden",
      "Accuracy is only defined for single-hop questions"
    ],
    "answer": 2,
    "why": "Plan-level measures such as validity, dependency correctness and per-step success show where the planner or a step failed."
  }
]
```

## Further reading

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
- Trivedi, H. et al. (2022). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions* (IRCoT).
- Xu, B. et al. (2023). *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*.
- Kim, S. et al. (2023). *An LLM Compiler for Parallel Function Calling* (LLMCompiler).
- Jeong, S. et al. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*.
- Trivedi, H. et al. (2022). *MuSiQue: Multihop Questions via Single-hop Question Composition*.
