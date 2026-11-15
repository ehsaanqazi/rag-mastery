A RAG pipeline is a conveyor belt of records: a document with its metadata, the chunks cut from it, a search request, a ranked result, a model's answer with its citations. So far in this module those records have been dictionaries and hand-written classes. Dictionaries accept any key and any value, so a typo (`"secton"`) or a wrong type (`"page": "four"`) passes silently. Hand-written classes need a page of boilerplate (`__init__`, `__repr__`, `__eq__`) before they hold three fields.

This lesson covers the two standard tools for records. **Dataclasses**, in the standard library, write the boilerplate for you from type annotations, and can make records immutable. **Pydantic** goes further: it *validates* data against those annotations at run time, converting what it safely can and rejecting what it cannot with precise error messages, and it serialises records to and from JSON and JSON Schema. Knowing which to use where (dataclasses for trusted data inside your program, Pydantic at every boundary where data arrives from outside) is one of the most practical design skills in RAG engineering.

## Learning objectives

- Define dataclasses with defaults, `default_factory`, `frozen`, `slots` and `kw_only`, and explain what code the decorator generates.
- Build Pydantic models with field constraints, field and model validators and configuration such as `extra="forbid"`, and interpret their validation errors.
- Distinguish lax (coercing) from strict validation and choose between them for a given input source.
- Serialise and parse records with `model_dump`, `model_dump_json`, `model_validate_json` and `model_json_schema`, and load configuration with `pydantic-settings`.
- Decide, for each record in a RAG pipeline, whether it should be a dataclass or a Pydantic model, and justify the choice by trust and cost.

## Prerequisites

- [Type hints](/learn/python-for-rag-engineering/type-hints): annotations, `Literal`, `X | None`, and why annotations alone check nothing at run time.
- [OOP](/learn/python-for-rag-engineering/oop): `__eq__`, `__hash__` and why hashed fields must not change.
- [Indexing pipeline](/learn/rag-architecture-fundamentals/indexing-pipeline): the stage where most records are created.

## Why this matters

Every serious RAG incident review contains a sentence like "the data was not what we thought it was". Some examples of what loose records allow:

- An export tool starts writing `"page": "3"` instead of `"page": 3`. Sorting chunks by page now sorts "10" before "9", and citations point to the wrong page.
- A new field `tenant` is added upstream, but the ingestion code builds chunks with `dict(**row)`, so the field flows into the vector store's payload unnoticed, and the access-control filter that should have used it does not.
- A language model is asked for JSON with `answer` and `citations`, and occasionally returns `"citations": "see above"`. Code that iterates over the citations iterates over the characters of a string.
- A chunk's metadata dictionary is shared and mutated after it was indexed (the bug from [Python fundamentals](/learn/python-for-rag-engineering/python-fundamentals)).

Each is caught by a record type that states its fields, their types and their constraints, and either refuses to be built from bad data (Pydantic) or refuses to change once built (a frozen dataclass or model). Records are also the *contract* between pipeline stages, in the same way that an abstract retriever is the contract between components.

## The intuition

A dictionary is a cardboard box: you can put anything in it and label it however you like. A **dataclass** is a printed form with named boxes: it is clear what goes where, and a form missing a required box cannot be created, but nobody checks that you wrote a date in the date box. A **Pydantic model** is the same form handed to a careful clerk at the counter. The clerk reads every box, quietly fixes obvious formatting ("3" becomes the number 3, "2024-04-01" becomes a date), refuses the form if anything is wrong, and tells you exactly which boxes are wrong and why.

You do not need a clerk for forms your own team fills in and passes between desks inside the office. You do need one at the front door, where forms arrive from the public.

## How it works

### Dataclasses: records without boilerplate

The `@dataclass` decorator (PEP 557, in the standard library since Python 3.7) reads the class's annotated attributes and generates methods from them. Step by step, when Python executes the decorated class:

1. It collects the annotated class attributes in order (`chunk_id: str`, `page: int | None = None`, ...). These become the **fields**. Annotations are required; they are how the decorator finds the fields, though it does not check the values against them.
2. It generates `__init__` with one parameter per field, using the given defaults.
3. It generates `__repr__` showing every field, and `__eq__` comparing all fields in order.
4. Depending on the options, it also generates ordering methods (`order=True`), makes assignment raise an error (`frozen=True`), generates `__hash__` (automatic when `frozen=True` and `eq=True`), uses `__slots__` instead of a per-instance dict (`slots=True`, Python 3.10+), or makes every parameter keyword-only (`kw_only=True`, Python 3.10+).
5. If the class defines `__post_init__`, the generated `__init__` calls it after assigning the fields: the place for checks and derived values.

```python
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str | None = None
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError(f"{self.chunk_id}: empty text")
        if not self.chunk_id.startswith(self.doc_id + "#"):
            raise ValueError(f"{self.chunk_id}: does not belong to {self.doc_id}")


a = Chunk(chunk_id="hr-leave#0", doc_id="hr-leave", text="Annual leave is 25 days.", tags=("leave",))
b = Chunk(chunk_id="hr-leave#0", doc_id="hr-leave", text="Annual leave is 25 days.", tags=("leave",),
          embedding=(0.1, 0.2))
print(a)
print(a == b, len({a, b}))               # embedding excluded from equality and hashing

try:
    a.text = "Annual leave is 30 days."
except FrozenInstanceError as e:
    print("FrozenInstanceError:", e)

c = replace(a, section="Leave")          # a modified copy; a is untouched
print(c.section, a.section)
print(asdict(c))

for bad in (dict(chunk_id="hr-leave#1", doc_id="hr-leave", text="   "),
            dict(chunk_id="it-vpn#0", doc_id="hr-leave", text="VPN is required.")):
    try:
        Chunk(**bad)
    except ValueError as e:
        print("ValueError:", e)

d = Chunk(chunk_id="x#0", doc_id="x", text="ok", section=42)   # wrong type: not checked
print(type(d.section).__name__)
```

```output
Chunk(chunk_id='hr-leave#0', doc_id='hr-leave', text='Annual leave is 25 days.', section=None, tags=('leave',))
True 1
FrozenInstanceError: cannot assign to field 'text'
Leave None
{'chunk_id': 'hr-leave#0', 'doc_id': 'hr-leave', 'text': 'Annual leave is 25 days.', 'section': 'Leave', 'tags': ('leave',), 'embedding': ()}
ValueError: hr-leave#1: empty text
ValueError: it-vpn#0: does not belong to hr-leave
int
```

What each choice buys:

- **`frozen=True`** makes the record immutable: assignment raises `FrozenInstanceError`, so the shared-metadata bug becomes impossible, and the instance is hashable, so chunks can go in sets for de-duplication. To "change" a frozen record, `dataclasses.replace(obj, field=value)` returns a modified copy.
- **`slots=True`** stores fields in fixed slots rather than a per-instance `__dict__`, which saves memory when you hold hundreds of thousands of chunk objects and turns a misspelt attribute assignment into an error.
- **`kw_only=True`** forces `Chunk(chunk_id=..., text=...)`: with several string fields in a row, positional construction is an accident waiting to happen (exactly the argument for keyword-only parameters in [Functions and classes](/learn/python-for-rag-engineering/functions-and-classes)).
- **`field(repr=False, compare=False)`** tunes one field. An embedding of 384 numbers should not flood the `repr`, and two chunks with the same text and identifiers are the same chunk whether or not one has been embedded yet.
- **Tuples, not lists, inside a frozen record.** `frozen` stops reassigning `tags`, not appending to a list stored in it. Immutable field types make the immutability real.

Mutable defaults are guarded for you: writing `tags: list[str] = []` raises `ValueError: mutable default <class 'list'> for field tags is not allowed: use default_factory`. Use `field(default_factory=list)`, which calls `list()` afresh for each instance, the dataclass equivalent of the `None` sentinel.

The last line of output is the limitation that motivates the rest of this lesson: `section=42` was accepted. A dataclass trusts its caller. `asdict` also has a cost worth knowing: it recursively deep-copies every field, which is slow for large nested records.

### Pydantic: validation from annotations

A Pydantic model is a class that inherits from `pydantic.BaseModel`. It looks like a dataclass, but constructing one (or calling `Model.model_validate(data)` or `Model.model_validate_json(text)`) runs a **validator** generated from the annotations. Pydantic v2's validation core, `pydantic-core`, is written in Rust, which is why it is fast enough to run on every record.

What happens when a model is validated:

1. **Before validators** run, if any (`@model_validator(mode="before")` receives the raw input).
2. Each field is validated in turn: its value is checked against the annotation, and in the default **lax mode** converted when the conversion is unambiguous: the string `"3"` becomes the integer 3, `"2024-04-01"` becomes a `datetime.date`, `"yes"` becomes `True`. Values that cannot be converted safely (`"four"` for an `int`, `2.5` for an `int`) are errors.
3. **Constraints** declared with `Field(...)` are checked: `min_length`, `max_length`, `ge` and `le` (greater or less than or equal), `pattern` and others.
4. **Field validators** (`@field_validator("name")`) run: your own functions that can check or transform one field.
5. Errors from every field are collected, not just the first. If there are any, a single `ValidationError` is raised listing all of them, each with a location, a message and a machine-readable type.
6. Only if every field succeeded do **after model validators** (`@model_validator(mode="after")`) run, receiving the constructed instance, for rules that involve several fields at once.

```mermaid
flowchart TD
    I["Raw input<br/>(dict or JSON text)"] --> B["Before model validators"]
    B --> F["Per field: type check and<br/>lax conversion"]
    F --> C["Field constraints<br/>(min_length, ge, pattern)"]
    C --> V["Field validators"]
    V --> Q{"Any field errors?"}
    Q -- "Yes" --> E["ValidationError<br/>listing every error"]
    Q -- "No" --> A["After model validators<br/>(cross-field rules)"]
    A --> M["Validated model instance"]
    A -- "rule fails" --> E
```

**Lax versus strict.** Lax mode suits data from text formats such as CSV, query strings and environment variables, where everything arrives as a string. **Strict mode** (`ConfigDict(strict=True)` for a model, or `Field(strict=True)` for one field) accepts only the exact type, which suits trusted structured data where a string in a numeric field means something upstream is broken:

```python
from pydantic import BaseModel, ConfigDict, ValidationError

class Lax(BaseModel):
    k: int
    score: float
    rerank: bool

class Strict(Lax):
    model_config = ConfigDict(strict=True)

data = {"k": "5", "score": "0.5", "rerank": "yes"}
print(Lax.model_validate(data))
try:
    Strict.model_validate(data)
except ValidationError as e:
    print([(err["loc"][0], err["type"]) for err in e.errors()])
try:
    Lax.model_validate({"k": 2.5, "score": 1, "rerank": "maybe"})
except ValidationError as e:
    print([(err["loc"][0], err["msg"]) for err in e.errors()])
```

```output
k=5 score=0.5 rerank=True
[('k', 'int_type'), ('score', 'float_type'), ('rerank', 'bool_type')]
[('k', 'Input should be a valid integer, got a number with a fractional part'), ('rerank', 'Input should be a valid boolean, unable to interpret input')]
```

Even lax mode refuses to lose information: `2.5` is not silently truncated to 2. One subtlety: when you validate JSON *text* with `model_validate_json`, strict mode still accepts a JSON string for a `date` field, because JSON has no date type.

**Configuration** is set with `model_config = ConfigDict(...)`. The options that matter most for RAG records:

| Option | Effect | Why you want it |
| --- | --- | --- |
| `extra="forbid"` | Unknown fields are errors (default is `"ignore"`, which drops them) | Catches renamed or misspelt fields upstream, and stops clients setting fields you did not intend |
| `frozen=True` | Assignment raises; instances are hashable | Records cannot change after validation, so they stay valid |
| `str_strip_whitespace=True` | Strips leading and trailing whitespace from strings | Extracted text often carries stray spaces |
| `strict=True` | No type conversion | Trusted structured inputs |

**Reusable constrained types.** `Annotated[str, Field(pattern=r"^[a-z0-9-]+$", max_length=64)]` attaches constraints to a type, which you can name (`DocId = ...`) and reuse across many models, so the definition of "a valid document id" lives in one place.

### Serialisation and JSON Schema

A validated model converts back out cleanly:

- `model.model_dump()` gives a dict of Python objects; `model_dump(mode="json")` gives only JSON-compatible values (dates become strings); `model_dump_json()` gives JSON text directly. All accept `include=` and `exclude=`, for example to keep an embedding out of a log line.
- `Model.model_json_schema()` produces a **JSON Schema**, a standard, machine-readable description of the model's fields, types and constraints.

That last method connects Pydantic to language models. Asking a model for **structured output** means giving it a schema and getting back JSON that fits it. The Anthropic Python SDK accepts a Pydantic model directly and validates the response into an instance. The code below follows the SDK's documented API but was **not executed** for this lesson, because it needs an API key; the shape of the call is what matters here:

```python
import anthropic
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    quote: str = Field(max_length=300)


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    answerable: bool


client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

response = client.messages.parse(
    model="claude-opus-5-5",
    max_tokens=1024,
    system="Answer only from the provided context. If the answer is not there, say you don't know.",
    messages=[{"role": "user", "content": prompt}],
    output_format=GroundedAnswer,
)
result: GroundedAnswer = response.parsed_output
for c in result.citations:
    print(c.chunk_id, c.quote)
```

Even when a provider constrains the output to the schema, keep validating: the schema guarantees shape, not truth. A citation's `chunk_id` may still name a chunk that was not in the context, which is a check only your code can make. Structured generation is covered fully in [Structured outputs](/learn/generation-strategies/structured-outputs).

### Validating things that are not models

`pydantic.TypeAdapter(T)` validates any type without defining a model: `TypeAdapter(list[int]).validate_python(["1", "2"])` returns `[1, 2]`, and the [type hints lesson](/learn/python-for-rag-engineering/type-hints) used it to validate a `TypedDict`. It is handy for one-off checks on API responses and for collections of models.

### Settings from the environment

Configuration is data from outside too: environment variables, `.env` files, secrets mounted by a deployment platform. The separate `pydantic-settings` package provides `BaseSettings`, a model whose fields are filled from environment variables and validated like any other:

```python
import os

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASKMERIDIAN_")

    anthropic_api_key: SecretStr
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = Field(default=5, ge=1, le=50)


os.environ["ASKMERIDIAN_ANTHROPIC_API_KEY"] = "sk-ant-example-not-a-real-key"
os.environ["ASKMERIDIAN_TOP_K"] = "8"
settings = Settings()
print(settings)
print(settings.top_k + 1, settings.anthropic_api_key.get_secret_value()[:7] + "...")

os.environ["ASKMERIDIAN_TOP_K"] = "500"
try:
    Settings()
except ValidationError as e:
    print(e.errors()[0]["loc"], e.errors()[0]["msg"])
```

```output
anthropic_api_key=SecretStr('**********') qdrant_url='http://localhost:6333' embedding_model='sentence-transformers/all-MiniLM-L6-v2' top_k=8
9 sk-ant-...
('top_k',) Input should be less than or equal to 50
```

The environment variable `ASKMERIDIAN_TOP_K` arrived as the string `"8"` and became the integer 8. A value of 500 fails at start-up, with a clear message, instead of producing a mysteriously slow service. `SecretStr` hides the key in `repr`, `print` and logs; you must call `get_secret_value()` deliberately to read it. (The key here is a fake set by the script itself.) Real keys belong in the environment or a secrets manager, never in code; [Virtual environments](/learn/python-for-rag-engineering/virtual-environments) and the deployment modules cover where they live.

### Choosing between them

```mermaid
flowchart LR
    X["Outside world:<br/>files, exports, HTTP,<br/>env vars, LLM output"] -->|"Pydantic validates<br/>at the boundary"| P["Validated models"]
    P --> D["Inside the pipeline:<br/>frozen dataclasses or<br/>frozen models"]
    D -->|"model_dump_json /<br/>asdict"| O["Outside again:<br/>vector store payloads,<br/>API responses, logs"]
```

| Question | Dataclass | Pydantic model |
| --- | --- | --- |
| Does the data come from outside the process? | Not on its own | Yes: validate it |
| Do you need type conversion and constraint checks? | Hand-written in `__post_init__` | Built in |
| Do you need JSON in and out, or a JSON Schema? | Manual | Built in |
| Is it created millions of times from already-trusted data? | Cheapest | Measurable overhead |
| Extra dependency? | None (standard library) | `pydantic` |

A common, sound arrangement: Pydantic models at the edges (the ingestion loader, the HTTP API, configuration, model output) and either frozen Pydantic models or frozen dataclasses for the internal records that pass between pipeline stages. Many teams simply use Pydantic everywhere for consistency, and that is also defensible when volumes are modest. The mistake is the opposite arrangement: unvalidated dicts at the edges.

## In code

**The problem.** AskMeridian's ingestion job reads a JSON Lines export (one JSON object per line) of chunks from the documentation system. Validate every line before it reaches the index: normalise what can be normalised safely, reject what cannot, and report every problem on every bad line so that the export's owners can fix them in one pass.

**Dependencies.** Pydantic v2 (the example was run with Pydantic 2.14).

```bash
pip install "pydantic>=2"
```

```python
import json
from datetime import date
from typing import Annotated, Literal

from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      field_validator, model_validator)

DocId = Annotated[str, Field(pattern=r"^[a-z0-9-]+$", max_length=64)]


class ChunkRecord(BaseModel):
    """One line of the ingestion export, validated before it reaches the index."""
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    chunk_id: str
    doc_id: DocId
    text: str = Field(min_length=1)
    source: Literal["handbook", "runbook", "product-docs"]
    owner: str = "Platform"
    page: int | None = Field(default=None, ge=1)
    effective: date
    access_groups: list[str] = Field(default_factory=lambda: ["all-staff"])

    @field_validator("access_groups")
    @classmethod
    def lower_case_groups(cls, groups: list[str]) -> list[str]:
        return sorted({g.lower() for g in groups})

    @model_validator(mode="after")
    def chunk_belongs_to_doc(self):
        if not self.chunk_id.startswith(f"{self.doc_id}#"):
            raise ValueError(f"chunk_id {self.chunk_id!r} does not start with doc_id {self.doc_id!r}#")
        return self


export = """\
{"chunk_id": "hr-leave#0", "doc_id": "hr-leave", "text": "  Annual leave is 25 days.  ", "source": "handbook", "page": "3", "effective": "2024-04-01", "access_groups": ["All-Staff", "HR"]}
{"chunk_id": "it-vpn#0", "doc_id": "it-vpn", "text": "VPN is required on public Wi-Fi.", "source": "runbook", "effective": "2024-01-15"}
{"chunk_id": "hr-remote#0", "doc_id": "hr-remote", "text": "", "source": "handbook", "page": 0, "effective": "2024-04-01"}
{"chunk_id": "fin-exp#0", "doc_id": "Finance Expenses", "text": "Receipts over £75.", "source": "wiki", "effective": "1 April"}
{"chunk_id": "it-pw#0", "doc_id": "it-vpn", "text": "Rotate passwords every 90 days.", "source": "runbook", "effective": "2024-01-15", "tenant": "meridian"}
{"chunk_id": "it-pw#0", "doc_id": "it-vpn", "text": "Rotate passwords every 90 days.", "source": "runbook", "effective": "2024-01-15"}
"""

good, bad = [], []
for line_no, line in enumerate(export.splitlines(), start=1):
    try:
        good.append(ChunkRecord.model_validate_json(line))
    except ValidationError as err:
        bad.append(line_no)
        print(f"line {line_no}: {err.error_count()} error(s)")
        for e in err.errors():
            print(f"   {'.'.join(map(str, e['loc'])) or '(model)'}: {e['msg']}")

print(f"\nvalid {len(good)}, rejected {len(bad)} (lines {bad})")
first = good[0]
print(repr(first.text), first.page, type(first.page).__name__, first.effective, first.access_groups)
print(good[1].owner, good[1].access_groups, good[1].page)
print(first.model_dump_json(exclude={"owner"}))
try:
    first.page = 4
except ValidationError as e:
    print("frozen:", e.errors()[0]["msg"])
print(json.dumps(ChunkRecord.model_json_schema()["properties"]["source"]))
```

```output
line 3: 2 error(s)
   text: String should have at least 1 character
   page: Input should be greater than or equal to 1
line 4: 3 error(s)
   doc_id: String should match pattern '^[a-z0-9-]+$'
   source: Input should be 'handbook', 'runbook' or 'product-docs'
   effective: Input should be a valid date or datetime, input is too short
line 5: 1 error(s)
   tenant: Extra inputs are not permitted
line 6: 1 error(s)
   (model): Value error, chunk_id 'it-pw#0' does not start with doc_id 'it-vpn'#

valid 2, rejected 4 (lines [3, 4, 5, 6])
'Annual leave is 25 days.' 3 int 2024-04-01 ['all-staff', 'hr']
Platform ['all-staff'] None
{"chunk_id":"hr-leave#0","doc_id":"hr-leave","text":"Annual leave is 25 days.","source":"handbook","page":3,"effective":"2024-04-01","access_groups":["all-staff","hr"]}
frozen: Instance is frozen
{"enum": ["handbook", "runbook", "product-docs"], "title": "Source", "type": "string"}
```

**The important lines.**

- `model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)` sets three policies for every field at once: unknown keys are errors, records are immutable after validation, and strings are trimmed.
- `DocId = Annotated[str, Field(pattern=..., max_length=64)]` defines once what a valid document id looks like: lower-case letters, digits and hyphens. Any model can reuse it.
- `source: Literal[...]` limits the source to the three systems AskMeridian indexes. The error message lists the allowed values.
- `page: int | None = Field(default=None, ge=1)` makes page optional, but if present it must be a positive integer.
- `access_groups: list[str] = Field(default_factory=lambda: ["all-staff"])` gives each record its own fresh default list. (Because the model is frozen, the list cannot be reassigned; for full immutability, a `tuple[str, ...]` would be stricter still.)
- `@field_validator("access_groups")` normalises group names to lower case, removes duplicates and sorts them, so that permission checks downstream compare like with like. By convention it is written as a `classmethod` (Pydantic accepts it either way), and it returns the cleaned value.
- `@model_validator(mode="after")` checks a rule that involves two fields: the chunk id must start with its document id. It receives the built instance and must return it.
- `ChunkRecord.model_validate_json(line)` parses and validates in one step, which is faster than `json.loads` followed by `model_validate`, and reports JSON syntax errors as validation errors too.

**What the output means.**

- **Line 1 was repaired, not rejected.** Its text had stray spaces (stripped), its page was the string `"3"` (converted to the integer 3), its date was a string (converted to a `date`) and its groups were mixed case (normalised). Each repair is unambiguous and loses no information.
- **Line 2 shows defaults.** `owner`, `page` and `access_groups` were absent and took their defaults.
- **Lines 3 and 4 report every error at once**, not just the first: two on line 3, three on line 4. That is what lets the export's owners fix everything in one round trip.
- **Line 5 has a field nobody declared.** `tenant` might be harmless or might be an access-control field the pipeline is now ignoring. `extra="forbid"` makes someone decide. Note that line 5 *also* breaks the cross-field rule, but the after-validator never ran: model validators in `"after"` mode run only once every field is valid.
- **Line 6 fails only the cross-field rule**, reported with the location `(model)` because it belongs to no single field.
- The JSON dump shows the date serialised as `"2024-04-01"`, and the schema fragment shows how the `Literal` becomes a JSON Schema `enum`: the form a language model or an API client would receive.

**Errors people commonly hit.**

- `PydanticUserError: Decorators defined with incorrect fields` when a `@field_validator` names a field the model does not have, often after a rename. It is raised when the class is defined, which is the helpful place.
- An after model validator that forgets `return self`. Pydantic 2.14 emits a `UserWarning` ("A custom validator is returning a value other than `self`") rather than failing, so read your warnings.
- Using Pydantic v1 idioms (`.dict()`, `.json()`, `parse_obj`, `@validator`, `class Config:`) from older tutorials. In v2 they are deprecated or removed; the names are `model_dump`, `model_dump_json`, `model_validate`, `@field_validator` and `model_config`.
- Expecting `frozen=True` to freeze a list field's contents. It stops reassignment, not mutation of the list; use tuples.

**Extending it.** Write rejected lines, with their errors, to a quarantine file instead of printing them, and fail the ingestion run if the rejection rate exceeds a threshold; add a `content_hash` computed field; and convert valid records into the payload format of your vector store with `model_dump(mode="json")`. [Data validation](/learn/document-engineering/data-validation) builds this into a full document-validation stage.

## A real-world example

In its first months, AskMeridian's ingestion loader turned each line of the documentation export into a dict and passed it straight on. When the documentation team migrated to a new content system, the export changed in two small ways: `effective_date` was renamed to `effective`, and `page` numbers became strings. Neither change raised an error. The loader read `row.get("effective_date")`, got `None` for every chunk, and the freshness filter, which preferred the most recent policy, quietly stopped working. Questions about equipment budgets started to be answered from the superseded 2023 handbook (£250) as often as from the current one (£400), because nothing distinguished them any more.

Fatima, on the Platform team, replaced the dict with a Pydantic `ChunkRecord` model using `extra="forbid"`, a required `effective: date`, and the quarantine-and-threshold pattern above. The next export with an unexpected change failed loudly within seconds: 100% of lines were rejected with `effective: Field required` and `effective_date: Extra inputs are not permitted`, and the nightly job stopped before touching the index. The fix took five minutes because the error named both the missing field and the unexpected one. Inside the pipeline, the team kept frozen dataclasses for the chunks produced after chunking, which are created from already-validated records in large numbers, and used `model_json_schema()` to publish the export contract to the documentation team so that future changes were agreed in advance.

## Common mistakes

- **Validating nothing at the boundary.** Dicts from `json.loads` flow into the index with whatever shape the upstream system produced this week.
- **Leaving `extra` at its default.** Pydantic's default is to *ignore* unknown fields, which silently drops renamed fields and new access-control attributes.
- **Treating a dataclass annotation as a check.** `section: str` accepts 42. Dataclasses validate only what `__post_init__` checks.
- **Mutable defaults.** A dataclass refuses `= []` (use `default_factory`); in Pydantic, mutable defaults are copied per instance, but `default_factory` states the intent clearly.
- **Coercion where it hides problems.** Lax mode turning `"3"` into 3 is helpful for CSV; for an internal JSON API it can hide an upstream bug. Choose strict mode where types should already be right.
- **Assuming structured LLM output is correct because it parsed.** Schema-valid output can still cite chunks that were never retrieved; validate the content too.
- **Mixing Pydantic v1 and v2 APIs** from old examples, producing deprecation warnings or attribute errors.

## Best practices

- Validate every input that crosses a trust boundary (files, exports, HTTP requests, environment variables, model output) with a Pydantic model, and report every error, not just the first.
- Set `extra="forbid"` on input models unless you have a reason not to, and make records `frozen` once validated.
- Name reusable constraints with `Annotated` types (`DocId`, `ChunkText`) so that rules live in one place.
- Use `field_validator` for single-field normalisation, `model_validator(mode="after")` for cross-field rules, and keep both free of side effects (no network calls).
- Inside the pipeline, pass validated, immutable records: frozen dataclasses with `slots=True` for high-volume internal records, or frozen models if you prefer one tool.
- Load configuration with `pydantic-settings`, keep secrets in `SecretStr`, and fail at start-up on invalid settings.
- Publish `model_json_schema()` as the contract with upstream teams and with language models.

## Performance considerations

Validation is real work, so measure it rather than guess. The script below times object creation on the machine used to write this lesson; absolute numbers will differ on yours, the ratios less so.

```python
import timeit
from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(slots=True)
class ChunkDC:
    chunk_id: str
    text: str
    page: int | None = None


class ChunkPD(BaseModel):
    chunk_id: str
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)


row = {"chunk_id": "hr-leave#0", "text": "Annual leave is 25 days.", "page": 3}
raw = '{"chunk_id": "hr-leave#0", "text": "Annual leave is 25 days.", "page": 3}'
n = 200_000
for label, stmt in [("dict(**row)", lambda: dict(**row)),
                    ("dataclass", lambda: ChunkDC(**row)),
                    ("pydantic model_validate", lambda: ChunkPD.model_validate(row)),
                    ("pydantic model_validate_json", lambda: ChunkPD.model_validate_json(raw))]:
    t = min(timeit.repeat(stmt, number=n, repeat=3)) / n
    print(f"{label:<30}{t * 1e6:6.2f} µs per object")
```

```output
dict(**row)                     0.09 µs per object
dataclass                       0.11 µs per object
pydantic model_validate         0.39 µs per object
pydantic model_validate_json    0.42 µs per object
```

Pydantic costs a few times as much as a dataclass per object here, and `model_validate_json` does the JSON parsing as well for only slightly more. Put that in context with arithmetic. Validating 10 million chunks at about 0.4 µs each is $10^7 \times 0.4 \times 10^{-6} = 4$ seconds of CPU, while embedding the same 10 million chunks with a model that manages, say, 1,000 chunks per second (an assumption; it depends heavily on hardware and model) takes $10^4$ seconds, nearly three hours. Validation is a rounding error in ingestion. The overhead grows with model complexity (nested models, many validators), and it matters in genuinely hot loops, such as constructing a result object for each of thousands of candidates per query, which is where a slotted dataclass or plain tuples are the better choice.

Memory follows the same logic: `slots=True` dataclasses avoid a per-instance dict, a saving worth having when hundreds of thousands of chunk objects are alive at once.

## Security considerations

Validation is a security control as well as a correctness one, provided it is configured to be strict where it matters.

- **Forbid extra fields on anything a client sends.** With the default `extra="ignore"`, unknown fields are dropped (safe); with `extra="allow"`, they are kept and may flow into code that trusts them. The classic failure is **mass assignment**: a request body that sets `access_groups` or `tenant` because the server built its record from whatever the client sent.
- **Bound sizes.** `max_length` on query strings and text fields, and limits on list lengths, stop a single request from submitting a megabyte "question" that becomes a very expensive embedding and prompt.
- **Keep secrets out of logs.** `SecretStr` hides values in `repr` and in error messages; exclude sensitive fields from `model_dump` used for logging.
- **Validate model output before acting on it.** If generated JSON drives anything (a filter, a tool call, a database query), validate it against a model with tight constraints, and check identifiers against what you actually provided. See [Output validation](/learn/rag-security-fundamentals/output-validation).
- **Validation is not sanitisation.** A string that passes `min_length=1` can still contain a prompt injection or HTML. Validation guarantees shape; what is safe to do with the content is a separate question.

## Summary

- Dataclasses generate `__init__`, `__repr__` and `__eq__` from annotated fields; `frozen`, `slots`, `kw_only` and `default_factory` make them safe, compact records, but they do not validate types.
- Pydantic models validate data against the same annotations at run time: converting unambiguous values in lax mode, checking `Field` constraints, running field and model validators, and reporting every error in one `ValidationError`.
- `ConfigDict(extra="forbid", frozen=True)` is the right default for input records; strict mode suits inputs that should already have the right types.
- Pydantic serialises to dicts, JSON and JSON Schema, which is how records cross into vector stores, APIs and structured LLM output; `pydantic-settings` applies the same validation to configuration.
- Validate at boundaries with Pydantic, pass immutable records inside; validation is cheap next to embedding, but not free in per-query hot loops.

## Exercises

**1.** What happens when Python executes this class definition, and how do you fix it?

```python
from dataclasses import dataclass

@dataclass
class Doc:
    doc_id: str
    tags: list[str] = []
```

<details><summary>Show solution</summary>

The decorator raises an error as soon as the class is defined:

```output
ValueError: mutable default <class 'list'> for field tags is not allowed: use default_factory
```

The dataclass machinery knows that a shared list default would be the mutable-default bug in disguise, so it refuses. Fix it with a factory, called afresh for every instance:

```python
from dataclasses import dataclass, field

@dataclass
class Doc:
    doc_id: str
    tags: list[str] = field(default_factory=list)
```

If the record should be immutable, prefer `tags: tuple[str, ...] = ()` with `frozen=True`: an empty tuple is immutable, so it is allowed as a plain default.

</details>

**2.** Using the `Lax` and `Strict` models from the lesson, predict whether each input is accepted and with what values: (a) `Lax` with `{"k": "07", "score": 1, "rerank": 0}`; (b) `Strict` with `{"k": 7, "score": 1, "rerank": False}`; (c) `Lax` with `{"k": "seven", "score": 0.5, "rerank": True}`.

<details><summary>Show solution</summary>

(a) Accepted as `k=7 score=1.0 rerank=False`. Lax mode converts the numeric string `"07"` to 7, the integer 1 to the float 1.0, and the integer 0 to `False`.

(b) Accepted as `k=7 score=1.0 rerank=False`. Strict mode still accepts an `int` for a `float` field, because every integer is exactly representable as a float and no information is lost; it rejects strings for numbers and numbers for booleans.

(c) Rejected with one error on `k`: `"seven"` cannot be parsed as an integer. Pydantic does not interpret words as numbers.

Run them to check; predicting coercion rules is the habit that keeps you from relying on behaviour you have not verified.

</details>

**3.** Write a Pydantic model `SearchRequest` for AskMeridian's search API: `query` (whitespace-stripped, 1 to 500 characters), `k` (1 to 20, default 5), `mode` (`"dense"`, `"sparse"` or `"hybrid"`, default `"hybrid"`), and `sources` (a list of allowed source names, default empty). Unknown fields must be rejected. Validate one good and three bad JSON bodies and print the error locations and types.

<details><summary>Show solution</summary>

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    sources: list[Literal["handbook", "runbook", "product-docs"]] = Field(default_factory=list)


print(SearchRequest.model_validate_json('{"query": "  how much annual leave?  ", "k": "3"}'))

for body in ('{"query": "   "}',
             '{"query": "vpn", "k": 100, "mode": "semantic"}',
             '{"query": "vpn", "sources": ["runbook", "slack"], "debug": true}'):
    try:
        SearchRequest.model_validate_json(body)
    except ValidationError as e:
        print([(".".join(map(str, err["loc"])), err["type"]) for err in e.errors()])
```

```output
query='how much annual leave?' k=3 mode='hybrid' sources=[]
[('query', 'string_too_short')]
[('k', 'less_than_equal'), ('mode', 'literal_error')]
[('sources.1', 'literal_error'), ('debug', 'extra_forbidden')]
```

Note that a query of only spaces is rejected: whitespace is stripped *before* the length check. The location `sources.1` pinpoints the second list item. The machine-readable `type` values are what an API should return to clients, alongside the human message. [Request validation](/learn/backend-engineering/request-validation) shows FastAPI doing this automatically from the same model.

</details>

**4.** Design decision. AskMeridian's pipeline has these records: (a) a line of the documentation export; (b) a chunk produced by the chunker, of which a full re-index creates about 2 million; (c) the body of a `/search` HTTP request; (d) a candidate result inside the retriever, of which each query creates about 200; (e) the JSON answer returned by the language model; (f) the service's configuration. For each, choose a plain tuple, a frozen dataclass or a Pydantic model, and give one reason.

<details><summary>Show solution</summary>

A defensible set of choices:

- **(a) Export line: Pydantic model** with `extra="forbid"`. It comes from another team's system; it must be validated and every error reported.
- **(b) Chunk: frozen dataclass with `slots=True`** (or a frozen Pydantic model). It is built by your own code from already-validated records, so re-validation adds little; immutability prevents the shared-metadata bug, and slots save memory across 2 million objects. Validation cost would be affordable here too (about a second of CPU), so a frozen model is also reasonable if the team prefers one tool.
- **(c) Search request: Pydantic model.** Untrusted input from clients; needs size limits, allowed values and rejection of extra fields.
- **(d) Candidate result: a tuple `(chunk_id, score)` or a slotted dataclass.** Created hundreds of times per query on a latency-sensitive path from trusted internal data; validation would be pure overhead.
- **(e) LLM answer: Pydantic model**, used both as the schema sent to the model and to validate what comes back, plus a check that cited chunk ids were in the context.
- **(f) Configuration: `pydantic-settings` `BaseSettings`**, with `SecretStr` for keys, so invalid settings stop the service at start-up.

The principle: validate where trust changes, then pass cheap immutable records inside.

</details>

## Check your understanding

```quiz
[
  {
    "q": "What does this print?\n```python\nfrom dataclasses import dataclass\n\n@dataclass\nclass C:\n    page: int\n\nprint(C(page=\"4\").page * 2)\n```",
    "options": ["8", "44", "It raises TypeError", "It raises ValidationError"],
    "answer": 1,
    "why": "Dataclasses do not validate or convert types, so page holds the string \"4\", and \"4\" * 2 is \"44\"."
  },
  {
    "q": "A Pydantic model with default configuration receives an input containing a field it does not declare. What happens?",
    "options": ["The field is silently ignored and dropped", "ValidationError for an extra field", "The field is stored as an extra attribute", "A warning is printed and the field is kept"],
    "answer": 0,
    "why": "The default is extra=\"ignore\". Use extra=\"forbid\" to reject unknown fields, or \"allow\" to keep them."
  },
  {
    "q": "A model has field errors on two fields and an after model validator that checks a cross-field rule. What does the ValidationError contain?",
    "options": ["Only the first field error", "The two field errors and the model validator's error", "The two field errors; the after validator does not run", "Only the model validator's error"],
    "answer": 2,
    "why": "Pydantic collects all field errors, but after-mode model validators run only once every field has validated successfully."
  },
  {
    "q": "Which configuration is the best default for a model that validates request bodies from API clients?",
    "options": ["extra=\"allow\" so that new client fields are kept", "strict=True and nothing else", "No configuration; defaults are safest", "extra=\"forbid\", with max_length limits on free-text fields"],
    "answer": 3,
    "why": "Forbidding extra fields prevents mass assignment and surfaces client mistakes; length limits bound the cost of each request."
  },
  {
    "q": "Why make a chunk record frozen once it has been created?",
    "options": ["Frozen records validate faster", "It prevents later code from mutating shared records, and makes them hashable for sets and de-duplication", "Vector databases only accept frozen objects", "Frozen dataclasses check field types at run time"],
    "answer": 1,
    "why": "Immutability rules out the shared-mutation bugs common in pipelines, and frozen dataclasses and models get a __hash__ so they can be set members."
  }
]
```

## Further reading

- The `dataclasses` module documentation, [docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html).
- PEP 557, "Data Classes" (Eric V. Smith, 2017), [peps.python.org/pep-0557](https://peps.python.org/pep-0557/).
- Pydantic documentation, especially the Models, Fields, Validators, Strict Mode and JSON Schema concept pages, [docs.pydantic.dev](https://docs.pydantic.dev/).
- Pydantic Settings documentation, in the same site's "Settings Management" concept page, [docs.pydantic.dev](https://docs.pydantic.dev/).
- JSON Schema specification and guides, [json-schema.org](https://json-schema.org/).
- Anthropic documentation on structured outputs, [docs.anthropic.com](https://docs.anthropic.com/).
