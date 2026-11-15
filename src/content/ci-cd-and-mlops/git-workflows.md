A RAG system is made of more kinds of thing than an ordinary web service: code, but also prompts, model choices, chunking settings, evaluation questions and the stored results of past evaluations. Every one of them can change an answer, and every change needs to be proposed, reviewed, recorded and, when it goes wrong, found and undone. **Git** is the tool that does this for text files, and a **Git workflow** is the set of team rules about how changes travel through it: where they are made, who must approve them, and how they reach the main line of development.

This lesson is about those rules for a RAG repository. It covers what belongs in the repository and what must stay out, how to choose a branching model, how pull requests and code owners route each change to the right reviewer, how a pre-commit hook stops the commonest mistakes before they are recorded, and how `git bisect` finds the commit that broke retrieval. It assumes you can already commit, branch and merge; the [Pro Git book](https://git-scm.com/book/en/v2) covers those basics. Testing and automation build on this foundation in [Automated testing](/learn/ci-cd-and-mlops/automated-testing) and [CI/CD pipelines](/learn/ci-cd-and-mlops/ci-cd-pipelines).

## Learning objectives

- Decide, for each artefact of a RAG system, whether it belongs in Git, in Git with a large-file extension, or outside Git behind a versioned pointer.
- Compare trunk-based development with long-lived branch models and justify a choice for a team that deploys continuously.
- Write CODEOWNERS rules that route prompt, model and golden-set changes to the people qualified to review them, and predict which rule wins for a given path.
- Build a pre-commit check that blocks secrets, oversized files, unversioned prompt edits and malformed golden-set items.
- Use `git bisect run` with a retrieval check to locate a regression, and calculate how many evaluations it needs.

## Prerequisites

- [Dependency management](/learn/python-for-rag-engineering/dependency-management): lock files, which this lesson assumes are committed.
- [Golden datasets](/learn/testing-and-benchmarking/golden-datasets): the evaluation set this lesson keeps in the repository.
- [Regression testing](/learn/testing-and-benchmarking/regression-testing): what "a change made retrieval worse" means and how it is detected.

## Why this matters

In a conventional application the code is the behaviour: read the diff and you know what changed. In a RAG system a one-word edit to a system prompt, a changed `top_k` in a YAML file or a new embedding model name can change thousands of answers without touching a line of Python. If those artefacts live outside version control (in a dashboard, a notebook, an environment variable someone set by hand), three things go wrong.

First, **nobody reviews them**: the prompt edit that removed "If the answer is not there, say you don't know" is never seen by anyone who would object. Second, **nothing can be reproduced**: you cannot say which prompt, settings and model produced last Tuesday's wrong answer. Third, **nothing can be undone precisely**.

The opposite mistake is just as common: putting everything in Git, including a 2 GB index file, the raw HR documents and an API key. Git keeps every version of every file forever and copies the whole history to every clone, so the binary slows every clone, the personal document is on every laptop, and the key stays in the history after a later commit deletes it. A Git workflow for RAG has to decide both what is versioned in Git and how a change reaches the main branch with the right people having looked at it.

## The intuition

Think of the repository as a ship's logbook. Every entry is dated, signed and permanent: you can only add a correcting entry, never tear a page out. That is what makes the log trustworthy, and it is why you do not paste the cargo into it. The log records "loaded 400 crates, manifest 7781, inspected by the harbour master"; the crates live in the hold.

In a RAG repository, prompts, settings and code are log entries: short, frequently changed, needing review. Corpora, model weights and indexes are cargo: large, sometimes sensitive, often rebuilt. The log records exactly which cargo was used. The workflow decides who may write in the log, who must countersign which entries, and how mistakes are found by reading back through it.

## How it works

### What goes where

| Artefact | Where it lives | Why |
| --- | --- | --- |
| Application and pipeline code | Git | Small text, reviewed line by line |
| Prompts (system prompt, templates) | Git, as files, not strings buried in code | Diffs show exactly which words changed |
| Settings: top-k, thresholds, chunk size, model identifiers | Git, in a config file | A model change becomes a reviewable one-line diff |
| Lock files, Dockerfiles, CI definitions, infrastructure code | Git | They define how the system is built |
| Golden set (JSON Lines) and evaluation baselines | Git, while they are small | Reviewed like code; versioned with the code that is scored on them |
| Model weights | Model hub or registry, pinned by revision | Large binaries; see [Model versioning](/learn/ci-cd-and-mlops/model-versioning) |
| Source documents and processed chunks | Object storage or a data-versioning tool | Large, sometimes personal, owned by other teams; see [Data versioning](/learn/ci-cd-and-mlops/data-versioning) |
| Vector indexes | The vector database, with snapshots | Derived data, rebuilt from the above; see [Index versioning](/learn/ci-cd-and-mlops/index-versioning) |
| Secrets | A secrets manager, injected as environment variables | Never in any repository, ever |

The rule of thumb: if a human writes it and a human should review it, it goes in Git. If a program produces it, or it is large, or it contains other people's data, Git holds a pointer (a hash, a revision, a manifest) and the thing itself lives in a system built for it.

**Git LFS** (Large File Storage) replaces large files with small pointer files and stores the content on a separate server. It fixes clone size for team-owned binaries such as test fixtures, but not governance: a confidential document in LFS is still in the history and still awkward to delete. Do not use it for the corpus.

### Branching models

A **branching model** is the team's agreement about which branches exist and how long they live.

**Trunk-based development** keeps one long-lived branch, `main` (the trunk). Each change is made on a short-lived branch, typically lasting hours or a day or two, merged through a pull request, and deleted. Unfinished features hide behind **feature flags**, configuration switches that keep new code paths off in production until they are ready. Releases are cut from `main`, often by tagging a commit.

**Long-lived branch models**, of which GitFlow (Vincent Driessen, 2010) is the best known, keep a `develop` branch alongside `main`, plus feature, release and hotfix branches with rules for merging between them. GitFlow was designed for software shipped in numbered versions to customers. Its author later added a note recommending simpler models for software that is delivered continuously, such as web services.

For a RAG service that deploys several times a week, trunk-based development is the usual choice, for a reason specific to RAG: **evaluation results only mean something against a known baseline.** A feature branch that lives for three weeks accumulates a prompt change, a chunker change and a dependency upgrade, then merges into a `main` that has itself moved. The evaluation compares the branch with a baseline that no longer exists, and if quality drops, nobody can say which of the changes caused it. Short-lived branches with one logical change each make every evaluation a clean comparison of two adjacent states.

```mermaid
flowchart LR
    M1["main @ a1"] --> B["branch: prompt-cite-sources<br/>(hours, one change)"]
    B --> PR["Pull request:<br/>CI tests + evaluation +<br/>code-owner review"]
    PR -->|"approved and green"| M2["main @ b7<br/>(squash merge)"]
    M2 --> T["tag v1.8.0<br/>= release candidate"]
    M1 --> M2
```

### Pull requests and merge strategies

A **pull request** (GitLab calls it a merge request) is a proposal to merge a branch, with a discussion, automated checks and approvals attached. Three settings on the main branch turn it into a control:

1. **Require a pull request**: nobody pushes directly to `main`.
2. **Require status checks**: the CI jobs (tests, evaluation gate) must pass before merging. [CI/CD pipelines](/learn/ci-cd-and-mlops/ci-cd-pipelines) builds those jobs.
3. **Require review from code owners**: the people responsible for the changed files must approve.

On GitHub these live in branch protection rules or the newer repository rulesets; other platforms have equivalents.

When the pull request merges, the platform offers three strategies. A **merge commit** keeps every commit from the branch and adds one commit joining the histories. **Squash merging** combines the branch into a single commit on `main`. **Rebase merging** replays each commit on top of `main` without a merge commit. For RAG repositories, squash merging has a useful property: one pull request becomes one commit on `main`, so each commit on the trunk corresponds to one reviewed, evaluated change. That makes the history easy to bisect (below) and easy to revert, because `git revert <commit>` undoes exactly one change. The cost is that the individual commits from the branch are lost from `main`'s history; the pull request page still has them.

> [!KEY]
> One pull request, one logical change. A model upgrade, a dependency upgrade and a prompt rewrite in one pull request produce one evaluation result that cannot be attributed to any of them. Split them, merge them one at a time, and each gets its own clean comparison against the baseline.

### Code owners

A **CODEOWNERS** file maps path patterns to the people or teams who must review changes to matching files. On GitHub it lives at `.github/CODEOWNERS`, the repository root or `docs/CODEOWNERS`; GitLab and Bitbucket offer similar features. When "require review from code owners" is on, a pull request that touches a file cannot merge until one of that file's owners approves.

The rule that trips people up: **when several patterns match a file, the last matching line in the file wins**, not the most specific one. Put broad patterns at the top and specific ones below them.

```text
# .github/CODEOWNERS  (order matters: the last match wins)
*                          @meridian/platform
/prompts/                  @meridian/platform @meridian/ai-quality
/config/models.yaml        @meridian/platform @meridian/ai-quality
/eval/golden/handbook*     @meridian/hr
/eval/golden/it*           @meridian/it-ops
```

This encodes a RAG-specific truth: the people qualified to review a change are not always engineers. The HR team owns the employee handbook, so it should approve changes to the handbook questions and expected answers in the golden set; an engineer cannot tell whether "5 days carry over" is the right expected answer. Note that the last two rules replace the Platform team rather than adding HR alongside it; list both teams on the line if both must approve.

### Pre-commit hooks

A **hook** is a script that Git runs at a particular moment. The `pre-commit` hook runs when you type `git commit`, before the commit is recorded; if it exits with a non-zero status, the commit is refused. The **pre-commit framework** (pre-commit.com) manages these hooks from a `.pre-commit-config.yaml` file so the whole team runs the same checks. Widely used hooks include `check-added-large-files` from the `pre-commit-hooks` repository (by default it rejects files over 500 KB) and secret scanners such as gitleaks and detect-secrets.

Local hooks are a convenience, not a control: `git commit --no-verify` skips them, so the same checks must also run in CI. Their value is speed: the mistake is caught in two seconds, before it is in any history.

### Finding the commit that broke retrieval

**`git bisect`** finds the commit that introduced a problem by binary search. You mark one commit as good (retrieval worked) and one as bad (it does not); Git checks out a commit halfway between them; you test it and mark it good or bad; Git halves the range again. With `git bisect run <command>` the testing is automated: Git runs the command at each step and reads its exit code, where 0 means good, 125 means "cannot test this commit, skip it", and any other code from 1 to 127 means bad.

For RAG this pairs naturally with a small retrieval check: a script that builds the pipeline at the checked-out commit, runs twenty critical queries and exits with 1 if any expected document is missing. The steps:

1. Write the check script outside the repository (or in a path that does not change between commits), so every commit is tested by the same check.
2. `git bisect start`, then `git bisect bad HEAD` and `git bisect good v1.7.0`.
3. `git bisect run python /tmp/check_retrieval.py`.
4. Read the first bad commit that Git reports, then `git bisect reset` to return to where you were.

Bisect has a RAG-specific trap. It checks out **code**, not data or models. If the index was rebuilt between the good and bad commits, or a hosted model changed behind an alias, every commit tests bad and bisect blames the oldest one. The check script must therefore pin everything Git does not version: build a fresh in-memory index from a fixed test corpus and use a pinned model revision, rather than querying the production index.

## The mathematics

### How many tests does bisect need?

**Intuition.** Each test halves the set of commits that could be the culprit, so the number of tests grows with the number of times you can halve the range, not with the size of the range.

$$
s = \lceil \log_2 n \rceil
$$

Here $n$ is the number of candidate commits between the known-good and the known-bad commit, $s$ is the number of tests needed in the worst case, and $\lceil \cdot \rceil$ rounds up to a whole number. The total time is $s \times t$, where $t$ is the time for one test.

**By hand.** AskMeridian's `main` has 200 commits between the last good release and today. $\log_2 200 \approx 7.64$, so $s = 8$. If the retrieval check takes 6 minutes (building a small index and running the queries), bisect finishes in about $8 \times 6 = 48$ minutes, unattended. Testing commits one by one from the oldest would take up to $200 \times 6 = 1{,}200$ minutes. For 1,000 commits, $s = 10$: five times the commits costs two more tests.

```python
import math

t = 6  # minutes per retrieval check (an assumption for this example)
for n in [50, 200, 1000]:
    s = math.ceil(math.log2(n))
    print(f"{n:>5} commits: {s:>2} tests, about {s * t} minutes")
```

```output
   50 commits:  6 tests, about 36 minutes
  200 commits:  8 tests, about 48 minutes
 1000 commits: 10 tests, about 60 minutes
```

This is another argument for small pull requests: bisect finds a commit, and if that commit is a one-change pull request, the culprit is the answer rather than a large diff to read.

## In code

**The problem.** Build the check that runs before every commit to AskMeridian's repository. It blocks the four mistakes that matter most in a RAG repository (a secret, a large binary, a prompt edited without a version bump, and a malformed golden-set item) and reports which teams must review the change under the CODEOWNERS rules.

**Dependencies.** Python's standard library only.

```bash
# nothing to install; Python 3.10 or later
python guard.py
```

To keep the example self-contained, the staged files are given as a dictionary of path to content, and a binary file as a `("<binary>", size)` pair. In a real hook you would build that dictionary from `git diff --cached --name-only` and `git show :<path>` (the staged version of a file); that wiring is shown after the output.

```python
import fnmatch
import json
import re

MAX_BYTES = 500 * 1024          # same default as pre-commit-hooks' check-added-large-files

SECRET_PATTERNS = {
    "an Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "an AWS access key ID": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "a hard-coded credential": re.compile(r"(?i)(api_key|password|secret)\s*=\s*['\"][^'\"]{8,}['\"]"),
}

# CODEOWNERS-style rules: the LAST matching pattern wins.
CODEOWNERS = [
    ("*",                      ["@meridian/platform"]),
    ("prompts/*",              ["@meridian/platform", "@meridian/ai-quality"]),
    ("eval/golden/handbook*",  ["@meridian/hr"]),
    ("eval/golden/it*",        ["@meridian/it-ops"]),
    ("config/models.yaml",     ["@meridian/platform", "@meridian/ai-quality"]),
]

def prompt_version(text):
    m = re.match(r"version:\s*(\d+)", text)
    return int(m.group(1)) if m else None

def check(staged, head):
    """staged: {path: content or ("<binary>", size)}; head: {path: content} at HEAD."""
    problems, reviewers = [], set()
    for path, content in staged.items():
        size = content[1] if isinstance(content, tuple) else len(content.encode())
        if size > MAX_BYTES:
            problems.append(f"{path}: {size / 1e6:.1f} MB is over the {MAX_BYTES // 1024} KB limit; "
                            "store it outside git and commit a manifest instead")
            continue
        if isinstance(content, tuple):
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{path}: looks like {name}; remove it and rotate the key")
        if path.startswith("prompts/") and path.endswith(".txt"):
            old, new = prompt_version(head.get(path, "version: 0")), prompt_version(content)
            if new is None or new <= old:
                problems.append(f"{path}: prompt text changed but version is still {old}; bump it")
        if path.endswith(".jsonl"):
            seen = set()
            for n, line in enumerate(content.splitlines(), 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"{path}:{n}: invalid JSON ({e.msg})")
                    continue
                missing = {"id", "question", "relevant_ids"} - item.keys()
                if missing:
                    problems.append(f"{path}:{n}: missing {sorted(missing)}")
                if item.get("id") in seen:
                    problems.append(f"{path}:{n}: duplicate id {item['id']}")
                seen.add(item.get("id"))
        owners = []
        for pattern, teams in CODEOWNERS:
            if fnmatch.fnmatch(path, pattern):
                owners = teams                  # later rules override earlier ones
        reviewers.update(owners)
    return problems, sorted(reviewers)

head = {
    "prompts/answer.txt": "version: 3\nAnswer only from the provided context.",
}
staged = {
    "src/askmeridian/retriever.py": "def retrieve(q, k=5):\n    ...\n",
    "src/askmeridian/settings.py": 'API_KEY = "sk-ant-api03-EXAMPLEONLYexampleonly1234"\n',
    "prompts/answer.txt": "version: 3\nAnswer only from the provided context. Cite sources.",
    "eval/golden/handbook.jsonl": (
        '{"id": "hb-001", "question": "How many days of annual leave?", "relevant_ids": ["hb-leave"]}\n'
        '{"id": "hb-002", "question": "When must carried-over days be used?"}\n'
        '{"id": "hb-001", "question": "Equipment budget?", "relevant_ids": ["hb-remote"]}\n'
    ),
    "data/index/askmeridian.faiss": ("<binary>", 48_000_000),
}

problems, reviewers = check(staged, head)
print(f"{len(problems)} problem(s):")
for p in problems:
    print("  -", p)
print("required reviewers:", ", ".join(reviewers))

# The same commit after fixing everything.
fixed = dict(staged)
del fixed["data/index/askmeridian.faiss"]
fixed["src/askmeridian/settings.py"] = 'import os\nAPI_KEY = os.environ["ANTHROPIC_API_KEY"]\n'
fixed["prompts/answer.txt"] = "version: 4\nAnswer only from the provided context. Cite sources."
fixed["eval/golden/handbook.jsonl"] = (
    '{"id": "hb-001", "question": "How many days of annual leave?", "relevant_ids": ["hb-leave"]}\n'
    '{"id": "hb-002", "question": "When must carried-over days be used?", "relevant_ids": ["hb-leave"]}\n'
    '{"id": "hb-003", "question": "Equipment budget?", "relevant_ids": ["hb-remote"]}\n'
)
problems, reviewers = check(fixed, head)
print(f"\nafter fixes: {len(problems)} problem(s); required reviewers:", ", ".join(reviewers))
```

```output
6 problem(s):
  - src/askmeridian/settings.py: looks like an Anthropic API key; remove it and rotate the key
  - src/askmeridian/settings.py: looks like a hard-coded credential; remove it and rotate the key
  - prompts/answer.txt: prompt text changed but version is still 3; bump it
  - eval/golden/handbook.jsonl:2: missing ['relevant_ids']
  - eval/golden/handbook.jsonl:3: duplicate id hb-001
  - data/index/askmeridian.faiss: 48.0 MB is over the 500 KB limit; store it outside git and commit a manifest instead
required reviewers: @meridian/ai-quality, @meridian/hr, @meridian/platform

after fixes: 0 problem(s); required reviewers: @meridian/ai-quality, @meridian/hr, @meridian/platform
```

**The important lines.**

- `MAX_BYTES` mirrors the default of the widely used large-file hook. An index file is the classic offender: it is produced by a program, so it belongs in the vector database or object storage, with its identity recorded in a manifest that *is* committed.
- `SECRET_PATTERNS` are deliberately simple. Real scanners such as gitleaks ship hundreds of rules and entropy checks; the point here is the placement of the check, before the commit exists. Note the advice in the message: *rotate the key*. Once a secret is in a commit that has been pushed, assume it is compromised.
- The prompt rule compares the `version:` header of the staged prompt with the one at `HEAD`. Giving each prompt an explicit version that is logged with every answer lets you say, months later, exactly which wording produced a given response. Requiring the bump at commit time is what keeps the version honest.
- The JSON Lines check validates every golden item. A malformed item that slips through either crashes the evaluation job or, worse, is silently skipped, quietly shrinking the test set.
- The CODEOWNERS loop assigns `owners = teams` on every match rather than appending, which is what implements "last match wins". `fnmatch` is a simplification: real CODEOWNERS patterns follow `.gitignore`-style rules (a leading `/` anchors to the root, a trailing `/` matches a directory), which you should check in your platform's documentation.

**What the output means.** The first commit has six problems and would be refused. After the fixes the hook passes, and the reviewers list tells the author before opening the pull request that HR must approve it, because it touches the handbook questions.

**Wiring it into Git.** This shell fragment, which was not executed here because it needs a real repository, shows how a hook gathers the staged files. Files are read from the index (`git show :path`), not from the working directory, because the working copy may contain further unstaged edits.

```bash
#!/bin/sh
# .git/hooks/pre-commit (or a "local" hook in .pre-commit-config.yaml)
git diff --cached --name-only --diff-filter=ACM | python scripts/guard_staged.py
```

**Common errors.** Checking the working copy instead of the staged content lets a fixed-but-unstaged file pass while the broken version is committed. Forgetting `--diff-filter=ACM` (added, copied, modified) makes the hook try to read deleted files. A slow hook gets bypassed with `--no-verify`; leave heavy work to CI.

**Extending it.** Exercise 3 adds a model-versus-index check; run the same script as the first CI job so `--no-verify` cannot skip it.

## A real-world example

The Platform team ran AskMeridian's repository with two long-lived branches: `main` for production and `develop` for integration. Prompts were stored as Python string constants inside the code, and the golden set lived in a shared spreadsheet that the HR team edited.

In one release cycle, `develop` received a chunk-size change, an upgrade of the sentence-transformers library and a rewritten system prompt. When `develop` was merged to `main` after three weeks, answer correctness on the evaluation dropped by several points on handbook questions. Nobody could say which change was responsible. Bisect was little help: many of the 41 merged commits did not build on their own, and the evaluation depended on the spreadsheet, which had also changed. The team eventually found the cause by hand: the prompt rewrite had moved the instruction to refuse when the context lacked an answer below the context block, and on long contexts the model was answering from general knowledge.

The team changed four things:

1. **Trunk-based development** with squash-merged pull requests, each holding one logical change, and a rule that model changes and dependency upgrades ship separately.
2. **Prompts moved to versioned files** under `prompts/`, with the version logged on every answer.
3. **The golden set moved into the repository** as JSON Lines, with a CODEOWNERS rule giving HR approval over handbook items. HR now edits it through pull requests, which is slower for them and is the price of reproducible evaluations.
4. **The pre-commit guard** from this lesson, run both locally and as the first CI job.

The next quality drop, two months later, was bisected in nine automated steps to a single pull request that had changed the default `top_k` from 5 to 3.

## Common mistakes

- **Committing generated artefacts** (indexes, embedding caches, evaluation outputs in the megabytes). Clones become slow, and because history is permanent, deleting the file later does not shrink the repository.
- **Committing a secret and "fixing" it with a second commit.** The key remains in the history and in every clone. Rotate it immediately; rewriting history is a clean-up, not a remedy.
- **Prompts as string literals scattered through the code.** Reviewers miss prompt changes buried in Python diffs, and nobody can log which prompt version produced an answer.
- **Long-lived feature branches for "experiments".** The branch drifts from `main`, its evaluation compares against a stale baseline, and merging brings in many changes at once.
- **Several unrelated changes in one pull request.** One evaluation result for several causes; a regression cannot be attributed, and a revert undoes good changes with the bad.
- **Ordering CODEOWNERS from specific to general.** Because the last match wins, a catch-all `*` at the bottom overrides every specific rule above it.
- **Bisecting against live data.** If the index or a hosted model changed during the range, the test result depends on when you run it, not on the commit, and bisect blames the wrong change.

## Best practices

- Keep every human-authored input that affects answers in Git, and commit pointers (a model revision, a corpus manifest hash, an index version) for everything generated, large or personal.
- Use short-lived branches and squash merges so that each commit on `main` is one reviewed, evaluated change.
- Protect `main`: pull requests only, required CI checks, required code-owner review.
- Give non-engineering owners (HR, IT operations) CODEOWNERS entries for the golden-set items and documents they are responsible for.
- Version prompts explicitly and log the prompt version, the settings version and the commit hash with every answer.
- Run fast checks in a pre-commit hook and run them again in CI, where they cannot be skipped.
- Tag releases (`v1.8.0`) so that rollbacks and bisect have named, known-good starting points.

## Performance considerations

Repository size matters more than it seems. Every CI clone downloads the history, so 3 GB of old index files is added to every pipeline run. Shallow clones (`git clone --depth 1`) help CI but break `git bisect`, so keep large files out rather than working around them.

Hook latency decides whether hooks are used at all. The guard above runs in well under a second; a hook that loads an embedding model costs seconds every commit and will be bypassed.

Bisect cost is logarithmic in commits but linear in test time: a 6-minute check bisects 1,000 commits in an hour, a 40-minute full evaluation in nearly seven. Keep a fast retrieval-only check for bisecting and smoke testing.

## Security considerations

- **Secrets.** Scan before commit and again in CI, and enable your Git platform's secret scanning where available. On a hit in pushed history, rotate first, then clean the history with a tool such as `git filter-repo`, and tell everyone with a clone to re-clone.
- **Personal data in golden sets.** Questions copied from real user logs can contain names, employee numbers and health details. Redact before committing, because a golden set in Git is readable by everyone with repository access and is kept forever. The [Golden datasets](/learn/testing-and-benchmarking/golden-datasets) lesson covers redaction.
- **Prompts are security controls.** The system prompt carries grounding and refusal instructions; a malicious or careless edit can weaken them. Code-owner review on `prompts/` treats them accordingly.

## Summary

- A RAG system's behaviour lives in prompts, settings and evaluation data as well as code; all human-authored inputs belong in Git, reviewed like code.
- Large, generated or personal artefacts (weights, corpora, indexes) stay out of Git; the repository holds precise pointers to them.
- Trunk-based development with short-lived, single-change pull requests keeps every evaluation a clean comparison against a known baseline.
- CODEOWNERS routes each change to qualified reviewers, including non-engineers; the last matching pattern wins.
- Pre-commit hooks catch secrets, large files, unversioned prompt edits and malformed golden items in seconds, and CI repeats them where they cannot be skipped.
- `git bisect run` finds a regression in about $\lceil \log_2 n \rceil$ tests, provided the test pins the data and models that Git does not version.

## Exercises

1. **Where does it live?** For each item, say whether it goes in Git, Git LFS, or outside Git with a pointer, and why: (a) a 30 KB sample PDF used by a parser unit test; (b) the 1.2 GB fine-tuned embedding model; (c) `prompts/answer.txt`; (d) last night's evaluation results for 400 golden items (2 MB of JSON); (e) a CSV of 50 real employee questions with names in them.

<details><summary>Show solution</summary>

(a) Git, or Git LFS if you have many such fixtures: small, team-owned, needed by tests. (b) Outside Git, in a model registry or hub, referenced by name and revision in `config/models.yaml`. (c) Git: short, human-authored, needs review. (d) Outside Git, in an artefact store or the evaluation tool, because it is generated and grows with every run; commit only the approved baseline that gates use, if it is small. (e) Not in Git at all until redacted; after redaction, as golden items in JSON Lines.

</details>

2. **CODEOWNERS order.** A colleague writes this file. Which team must approve a change to `eval/golden/handbook.jsonl`, and is that what was intended? Fix it.

```text
/eval/golden/handbook*   @meridian/hr
/eval/                   @meridian/ai-quality
*                        @meridian/platform
```

<details><summary>Show solution</summary>

All three patterns match, and the last match wins, so only `@meridian/platform` is required; HR and AI Quality are never asked. Reverse the order so the general rule comes first and the specific rule last:

```text
*                        @meridian/platform
/eval/                   @meridian/ai-quality
/eval/golden/handbook*   @meridian/hr @meridian/ai-quality
```

HR now owns the handbook items, with AI Quality listed on the same line if both are wanted.

</details>

3. **Extend the guard.** Add a check to `guard.py` that fails when `config/models.yaml` changes the `embedding_model:` value without changing `index_version:` in the same file. Test it on a staged file that changes only the model.

<details><summary>Show solution</summary>

```python
def field(text, name):
    m = re.search(rf"^{name}:\s*(\S+)", text, re.M)
    return m.group(1) if m else None

# inside check(), for each staged path:
if path == "config/models.yaml":
    old = head.get(path, "")
    if (field(content, "embedding_model") != field(old, "embedding_model")
            and field(content, "index_version") == field(old, "index_version")):
        problems.append(f"{path}: embedding model changed but index_version did not; "
                        "a new model needs a new index")
```

With `head["config/models.yaml"] = "embedding_model: all-MiniLM-L6-v2\nindex_version: 3\n"` and a staged version that changes only the model to `bge-small-en-v1.5`, the check reports one problem; bumping `index_version` to 4 clears it.

</details>

4. **Design a bisect.** AskMeridian started returning the 2023 handbook for carry-over questions some time in the last 300 commits. During that period the production index was rebuilt twice. Describe the script you would pass to `git bisect run`, what it pins, and how many runs to expect.

<details><summary>Show solution</summary>

The script builds a fresh in-memory index at the checked-out commit from a fixed test corpus (stored outside the repository) containing both the 2025 and 2023 handbook carry-over passages, using a pinned embedding model revision from a local cache with network access disabled. It runs the carry-over queries and exits 1 if any top-3 result is from the 2023 handbook, 0 otherwise, and 125 if the commit fails to build. It never touches the production index, because the two rebuilds would make the result depend on the date rather than the commit. With 300 commits, $\lceil \log_2 300 \rceil = 9$ runs, plus a few more if some commits are skipped.

</details>

## Check your understanding

```quiz
[
  {
    "q": "A CODEOWNERS file contains `/prompts/ @meridian/ai-quality` on line 2 and `* @meridian/platform` on line 9. Who must approve a change to `prompts/answer.txt`?",
    "options": ["@meridian/ai-quality only", "Both teams", "@meridian/platform only", "Nobody, because two rules conflict"],
    "answer": 2,
    "why": "When several patterns match, the last matching line wins. The catch-all on line 9 overrides the prompts rule, which is why general rules belong at the top."
  },
  {
    "q": "Why is trunk-based development especially suited to RAG systems that deploy often?",
    "options": ["Each short-lived, single-change branch is evaluated against a baseline that still exists, so quality changes can be attributed", "It removes the need for code review", "It lets large index files be stored on the trunk", "It makes evaluation unnecessary because changes are small"],
    "answer": 0,
    "why": "Evaluation results only mean something compared with a known baseline. Long-lived branches mix many changes and compare against a stale baseline."
  },
  {
    "q": "A developer committed an API key, then deleted it in the next commit, and pushed both. What is the first thing to do?",
    "options": ["Nothing, the key is no longer in the code", "Squash the two commits", "Rotate (revoke and replace) the key", "Add the file to .gitignore"],
    "answer": 2,
    "why": "The key is in the pushed history and every clone. Treat it as compromised and rotate it; cleaning the history comes afterwards."
  },
  {
    "q": "`git bisect run` is given a script that queries the production vector index. The index was rebuilt halfway through the commit range. What is the likely result?",
    "options": ["Bisect finds the right commit, only more slowly", "Bisect refuses to run", "Bisect reports the rebuild as a commit", "Every commit tests the same way, so bisect blames the wrong commit"],
    "answer": 3,
    "why": "Bisect varies only the code. A test that depends on unversioned live data gives the same answer for every commit, so the reported culprit is meaningless."
  },
  {
    "q": "About how many automated tests does bisect need to search 500 commits?",
    "options": ["9", "250", "500", "25"],
    "answer": 0,
    "why": "Each test halves the range: the base-2 logarithm of 500 is about 8.97, which rounds up to 9."
  }
]
```

## Further reading

- Scott Chacon and Ben Straub, *Pro Git*, 2nd edition, chapters on branching and Git tools: [git-scm.com/book](https://git-scm.com/book/en/v2)
- Git reference documentation for `git bisect`: [git-scm.com/docs/git-bisect](https://git-scm.com/docs/git-bisect)
- GitHub Docs, "About code owners": [docs.github.com](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- The pre-commit framework and its hook catalogue: [pre-commit.com](https://pre-commit.com/)
- Vincent Driessen (2010), "A successful Git branching model", the original GitFlow article, including its later note on continuously delivered software.
- Nicole Forsgren, Jez Humble and Gene Kim (2018), *Accelerate: The Science of Lean Software and DevOps*, for the research on trunk-based development and delivery performance.
