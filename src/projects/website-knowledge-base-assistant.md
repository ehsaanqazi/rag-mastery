Documentation lives on websites, and websites change. In this project you build an assistant that crawls a documentation site politely, extracts the readable text from each page, stores chunk embeddings in a database, and on every later run re-embeds only the pages whose content actually changed. Answers come with links to the pages they were drawn from.

**What was run.** The crawler, extractor, chunker, indexer, file-backed store, command line and all 29 tests were run as shown, against a small site served on localhost with `python -m http.server`. PostgreSQL was not available, so `schema.sql` and `app/pg_store.py` were written against the documented pgvector and psycopg 3 APIs and **not executed**. The final Claude call was not executed either, for want of an API key.

## Overview

The pipeline has two halves.

**Sync** (run on a schedule): fetch `robots.txt`, crawl the site breadth-first within one host at a limited rate, turn each page into clean text, hash that text, and compare the hash with the one stored last time. Unchanged pages are skipped. New and changed pages are chunked, embedded and stored. Pages that have disappeared are deleted.

**Ask** (run per question): embed the question, fetch the closest chunks from the store, and have Claude answer from them, citing the page URLs.

Storage sits behind a four-method interface with two implementations: `LocalStore`, a JSON file with exact search, for development and tests, and `PgVectorStore` for PostgreSQL with the pgvector extension.

## Learning objectives

- Write a crawler that obeys `robots.txt`, rate-limits itself and stays on one site.
- Extract readable text from HTML with BeautifulSoup, removing navigation and other boilerplate.
- Use a content hash to make re-indexing incremental, and delete safely.
- Put storage behind an interface and implement it for pgvector.
- Test network code without a network, and time-dependent code without waiting.

## Requirements

- Python 3.10 or later.
- About 500 MB of disk for PyTorch and the embedding model.
- For the PostgreSQL store: PostgreSQL with pgvector (a Docker command is given below).
- An Anthropic API key only for generated answers.
- Background: [HTML parsing](/learn/advanced-document-parsing/html-parsing), [Incremental ingestion](/learn/understanding-document-ingestion/incremental-ingestion) and [PostgreSQL with pgvector](/learn/vector-database-ecosystem/postgresql-with-pgvector).

> [!WARNING]
> Crawl only sites you own or have permission to crawl. This tutorial crawls a site you serve yourself. `robots.txt` is a request, not a licence: a site's terms of use may forbid crawling even where `robots.txt` allows it.

## Technology stack

| Piece | Choice | Status in this tutorial |
| --- | --- | --- |
| HTTP | `httpx` | Run |
| robots.txt | `urllib.robotparser` (standard library) | Run |
| HTML to text | BeautifulSoup with the `lxml` parser | Run |
| Chunking | Plain Python; LangChain's splitter as an alternative | Plain Python run; LangChain not executed |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Run |
| Storage | `LocalStore` (JSON file) | Run |
| Storage | PostgreSQL + pgvector through `psycopg` | Written, not executed |
| Generation | Claude through the Anthropic SDK | Written, not executed |

## System architecture

```mermaid
flowchart TD
    R["robots.txt"] --> C["Crawler<br/>same host, rate-limited"]
    C --> X["Extract title, text, links<br/>(BeautifulSoup)"]
    X --> H{"Content hash<br/>changed?"}
    H -->|"no"| S["Skip"]
    H -->|"yes"| K["Chunk and embed"]
    K --> D[("Store<br/>LocalStore or pgvector")]
    C -->|"page gone"| DEL["Delete page"]
    DEL --> D
    Q["Question"] --> E["Embed question"]
    E --> D
    D --> G["Generator (Claude)"]
    G --> A["Answer with page links"]
```

The hash is the hinge of the design. Embedding is the expensive step, so the system does the cheap steps (fetch, parse, hash) for every page and the expensive one only where the hash differs.

## Folder structure

```text
website-knowledge-base-assistant/
├── requirements.txt
├── schema.sql
├── make_site.py
├── cli.py
├── site/                  (created by make_site.py)
├── app/
│   ├── __init__.py
│   ├── extract.py
│   ├── crawler.py
│   ├── chunking.py
│   ├── store.py
│   ├── pg_store.py
│   ├── indexer.py
│   └── assistant.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_extract_and_chunking.py
    ├── test_crawler.py
    └── test_indexer_and_assistant.py
```

The two `__init__.py` files are empty.

## Setup

```bash
mkdir website-knowledge-base-assistant && cd website-knowledge-base-assistant
python -m venv .venv && source .venv/bin/activate
mkdir app tests && touch app/__init__.py tests/__init__.py
```

Create `requirements.txt`:

```text
httpx>=0.27
beautifulsoup4>=4.12
lxml>=5.0
numpy>=1.26
sentence-transformers>=3.0
anthropic>=0.80
pytest>=8.0
# For the PostgreSQL store:
psycopg[binary]>=3.2
pgvector>=0.3
# Optional, for the LangChain splitter mentioned under "Possible improvements":
# langchain-text-splitters
```

```bash
pip install -r requirements.txt
```

The PostgreSQL store is optional. To use it, start a database that has pgvector, create the tables and export the connection string (these three commands were not run for this tutorial):

```bash
docker run --name kb-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg17
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres"
psql "$DATABASE_URL" -f schema.sql
```

Without `DATABASE_URL`, the command line uses `LocalStore` and writes `kb_index.json`.

## Implementation

### make_site.py

```python
"""Write the small sample documentation site into ./site."""
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <title>{title} | Meridian Docs</title>
  <style>body {{ font-family: sans-serif; }}</style>
  <script>window.analytics = {{ page: "{path}" }};</script>
</head>
<body>
  <nav>
    <a href="/index.html">Home</a>
    <a href="/leave.html">Leave</a>
    <a href="/remote.html">Remote work</a>
    <a href="/expenses.html">Expenses</a>
    <a href="/it/vpn.html">VPN</a>
  </nav>
  <main>
{body}
  </main>
  <footer>Meridian Ltd. Internal use only. <a href="/private/salaries.html">Staff only</a></footer>
</body>
</html>
"""

PAGES = {
    "index.html": ("Welcome", """    <h1>Meridian documentation</h1>
    <p>Policies and guides for Meridian staff. Choose a topic above.</p>
    <p>See also the <a href="it/vpn.html#setup">VPN guide</a>, the
       <a href="/old-handbook.html">old handbook</a>, our
       <a href="https://example.com/">public website</a> or
       <a href="mailto:help@meridian.example">email the support desk</a>.</p>"""),
    "leave.html": ("Annual leave", """    <h1>Annual leave</h1>
    <p>Full-time staff receive 25 days of annual leave each year, in addition to public holidays.</p>
    <h2>Carrying leave over</h2>
    <p>Up to 5 unused days carry over into the next year and must be used by 31 March.</p>"""),
    "remote.html": ("Remote work", """    <h1>Remote work</h1>
    <p>Staff may work remotely up to 3 days a week, on days agreed with their line manager.</p>
    <h2>Equipment</h2>
    <p>Meridian provides a home-office equipment budget of £400 for a desk, chair or monitor.</p>"""),
    "expenses.html": ("Expenses", """    <h1>Expenses</h1>
    <p>Expenses must be submitted within 30 days of the purchase.</p>
    <ul>
      <li>Any expense over £75 needs a receipt attached.</li>
      <li>Claims are made in the finance portal.</li>
    </ul>"""),
    "it/vpn.html": ("VPN", """    <h1 id="setup">VPN</h1>
    <p>The VPN is required whenever you connect from public Wi-Fi, such as in a café, hotel or airport.</p>
    <p>Passwords must be rotated every 90 days. The support desk is open 08:00 to 18:00 UK time.</p>"""),
    "private/salaries.html": ("Salaries", """    <h1>Salary bands</h1>
    <p>This page must never be indexed.</p>"""),
}

ROBOTS = "User-agent: *\nDisallow: /private/\nCrawl-delay: 1\n"


def main() -> None:
    site = Path("site")
    for path, (title, body) in PAGES.items():
        target = site / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(TEMPLATE.format(title=title, path=path, body=body), encoding="utf-8")
    (site / "robots.txt").write_text(ROBOTS, encoding="utf-8")
    print(f"wrote {len(PAGES)} pages and robots.txt to {site}/")


if __name__ == "__main__":
    main()
```

The sample site has the features a crawler must cope with: navigation and a footer on every page, a script, a relative link with a fragment, a link to a page that does not exist, an external link, a `mailto:` link, and a `/private/` area that `robots.txt` puts off limits but the footer links to.

### app/extract.py

```python
"""Turn an HTML page into a title, clean text and a list of links."""
import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin

from bs4 import BeautifulSoup, Comment

BOILERPLATE = ["script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form"]
BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre", "blockquote", "dt", "dd", "div", "section"]


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str

    @property
    def content_hash(self) -> str:
        """Fingerprint of what a reader sees. Markup-only changes do not alter it."""
        return hashlib.sha256(f"{self.title}\n{self.text}".encode("utf-8")).hexdigest()


def extract_links(html: str, base_url: str) -> list[str]:
    """Absolute http(s) links, fragments removed, in page order without repeats."""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        url, _fragment = urldefrag(urljoin(base_url, a["href"].strip()))
        if url.startswith(("http://", "https://")) and url not in links:
            links.append(url)
    return links


def extract_page(html: str, url: str) -> Page:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else url
    for tag in soup.find_all(BOILERPLATE):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup

    for node in main.find_all(string=True):
        if isinstance(node, Comment):
            node.extract()
        else:  # line breaks inside HTML source are just spaces
            node.replace_with(re.sub(r"\s+", " ", node))
    for br in main.find_all("br"):
        br.replace_with("\n")
    for cell in main.find_all(["td", "th"]):
        cell.insert_after(" ")
    for block in main.find_all(BLOCKS):  # one line of text per block element
        block.insert_before("\n")
        block.insert_after("\n")

    lines = (" ".join(line.split()) for line in main.get_text().splitlines())
    return Page(url=url, title=title, text="\n".join(line for line in lines if line))
```

- `extract_links` runs on the untouched HTML, because navigation links are how the crawler discovers pages. `urljoin` resolves relative links; `urldefrag` removes `#fragment`, so `vpn.html` and `vpn.html#setup` are one page.
- `extract_page` then deletes boilerplate elements and prefers `<main>` or `<article>` when present. Without this, the same navigation text would be embedded into every chunk of every page and would match every question a little.
- Text nodes have their whitespace collapsed, then a newline is inserted around each block element. The result is one line per heading, paragraph or list item, with inline links and bold text left inside their sentence. Calling `get_text("\n")` directly would break a sentence at every `<a>`.
- `content_hash` covers the title and the extracted text, not the raw HTML. A changed stylesheet link, analytics snippet or footer year therefore does not trigger re-embedding.

### app/crawler.py

```python
"""A polite, same-site, breadth-first crawler."""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from app.extract import Page, extract_links, extract_page

USER_AGENT = "MeridianKBBot/1.0"


@dataclass
class CrawlResult:
    pages: list[Page] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)    # network errors and 5xx: state unknown
    skipped: list[str] = field(default_factory=list)   # robots.txt, other sites, non-HTML, 4xx
    truncated: bool = False                             # stopped at max_pages with links left

    @property
    def complete(self) -> bool:
        """True if every reachable page was seen, so a missing page is really gone."""
        return not self.failed and not self.truncated


class Crawler:
    def __init__(self, client: httpx.Client | None = None, delay: float = 0.5, max_pages: int = 200,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=10.0, follow_redirects=False
        )
        self.delay = delay
        self.max_pages = max_pages
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def _get(self, url: str) -> httpx.Response:
        """Every request goes through here, so the rate limit cannot be bypassed."""
        if self._last_request is not None:
            wait = self.delay - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()
        return self.client.get(url)

    def _robots(self, start_url: str) -> RobotFileParser:
        parts = urlsplit(start_url)
        robots = RobotFileParser()
        try:
            response = self._get(f"{parts.scheme}://{parts.netloc}/robots.txt")
        except httpx.HTTPError:
            robots.disallow_all = True       # cannot read the rules: assume nothing is allowed
            return robots
        if response.status_code == 200:
            robots.parse(response.text.splitlines())
        elif response.status_code in (401, 403) or response.status_code >= 500:
            robots.disallow_all = True
        else:
            robots.allow_all = True          # no robots.txt (404): everything is allowed
        return robots

    def crawl(self, start_url: str) -> CrawlResult:
        site = urlsplit(start_url).netloc
        robots = self._robots(start_url)
        self.delay = max(self.delay, float(robots.crawl_delay(USER_AGENT) or 0))

        result = CrawlResult()
        queue, seen = deque([start_url]), {start_url}
        while queue:
            if len(result.pages) >= self.max_pages:
                result.truncated = True
                break
            url = queue.popleft()
            if urlsplit(url).netloc != site or not robots.can_fetch(USER_AGENT, url):
                result.skipped.append(url)
                continue
            try:
                response = self._get(url)
            except httpx.HTTPError:
                result.failed.append(url)
                continue
            if response.status_code >= 500:
                result.failed.append(url)
                continue
            html = "text/html" in response.headers.get("content-type", "")
            if response.status_code != 200 or not html:
                result.skipped.append(url)
                continue
            result.pages.append(extract_page(response.text, url))
            for link in extract_links(response.text, url):
                if link not in seen:
                    seen.add(link)
                    queue.append(link)
        return result
```

- **Identity.** The crawler sends a `User-Agent` naming itself, so a site owner can see who is calling and write a `robots.txt` rule for it.
- **robots.txt.** `RobotFileParser` does the parsing. The file is fetched through the crawler's own client so that it is rate-limited and testable. A missing file (404) means no restrictions; a file that cannot be read (401, 403, a server error or a network failure) is treated as "everything disallowed", the cautious reading.
- **Rate limit.** `_get` is the only method that touches the network, and it waits until `delay` seconds have passed since the previous request. A `Crawl-delay` in `robots.txt` raises the delay but never lowers it.
- **Same site.** A link is followed only if its host equals the start URL's host. Everything else lands in `skipped`.
- **Three outcomes per URL.** A page is a 200 response with an HTML content type. `skipped` holds things that are definitely not pages to index (other sites, disallowed paths, PDFs, 404s). `failed` holds things whose state is unknown (timeouts, 5xx). `complete` is true only when nothing failed and the crawl was not cut short by `max_pages`; the indexer uses it to decide whether deleting is safe.
- `sleep` and `clock` are parameters so that tests can replace time itself.

### app/chunking.py

```python
"""Pack a page's lines into chunks of bounded size, with one line of overlap."""
import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, max_chars: int) -> list[str]:
    """Lines of the page; a line longer than max_chars is split into sentences."""
    pieces: list[str] = []
    for line in text.splitlines():
        if len(line) <= max_chars:
            pieces.append(line)
        else:
            pieces.extend(SENTENCE_END.split(line))
    return [p for p in pieces if p.strip()]


def chunk_text(text: str, max_chars: int = 400, overlap: bool = True) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for piece in _pieces(text, max_chars):
        if current and len("\n".join(current)) + 1 + len(piece) > max_chars:
            chunks.append("\n".join(current))
            current = [current[-1]] if overlap and len(current) > 1 else []
        current.append(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks
```

The extractor already produced one line per block, so the chunker packs whole lines into chunks of up to `max_chars` characters and repeats the last line of one chunk at the start of the next, which keeps a heading attached to the paragraph after it. A single line longer than the limit is split at sentence ends.

### app/store.py

```python
"""The storage interface, and a file-backed implementation for development and tests."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.extract import Page


@dataclass(frozen=True)
class Hit:
    url: str
    title: str
    content: str
    score: float


class Store(Protocol):
    def page_hashes(self) -> dict[str, str]:
        """url -> content hash for every stored page."""
        ...

    def replace_page(self, page: Page, chunks: list[str], vectors: np.ndarray) -> None:
        """Store the page and its chunks, removing any chunks it had before."""
        ...

    def delete_page(self, url: str) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """The k chunks closest to the vector by cosine similarity, best first."""
        ...


class LocalStore:
    """Everything in memory; saved to a JSON file if a path is given. Exact search."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._pages: dict[str, dict] = {}
        if self.path and self.path.exists():
            self._pages = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self._pages), encoding="utf-8")

    def page_hashes(self) -> dict[str, str]:
        return {url: page["hash"] for url, page in self._pages.items()}

    def replace_page(self, page: Page, chunks: list[str], vectors: np.ndarray) -> None:
        self._pages[page.url] = {
            "title": page.title, "hash": page.content_hash, "chunks": chunks,
            "vectors": np.asarray(vectors, dtype=float).tolist(),
        }
        self._save()

    def delete_page(self, url: str) -> None:
        self._pages.pop(url, None)
        self._save()

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        hits = []
        for url, page in self._pages.items():
            for content, stored in zip(page["chunks"], page["vectors"]):
                hits.append(Hit(url, page["title"], content, float(np.dot(stored, vector))))
        return sorted(hits, key=lambda hit: -hit.score)[:k]
```

`Store` is the whole contract between the indexer and any database: list the hashes, replace a page, delete a page, search. `LocalStore` implements it with a dictionary and a brute-force dot product, which is exact and fast enough for a few thousand chunks, and saves to JSON so that the index survives between command-line runs.

### schema.sql

```sql
-- Run once: psql "$DATABASE_URL" -f schema.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pages (
    url          text PRIMARY KEY,
    title        text NOT NULL,
    content_hash text NOT NULL,
    fetched_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    url       text NOT NULL REFERENCES pages (url) ON DELETE CASCADE,
    position  integer NOT NULL,
    content   text NOT NULL,
    embedding vector(384) NOT NULL,
    UNIQUE (url, position)
);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

- `vector(384)` fixes the dimension to that of the embedding model; inserting a vector of another length is an error.
- `ON DELETE CASCADE` means deleting a row from `pages` removes its chunks, so a page can never be half-deleted.
- The HNSW index uses `vector_cosine_ops`, which matches the `<=>` cosine-distance operator used in the query. An index built for a different operator would simply not be used.

### app/pg_store.py (not executed)

```python
"""The same Store interface on PostgreSQL with the pgvector extension."""
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.extract import Page
from app.store import Hit


class PgVectorStore:
    def __init__(self, dsn: str):
        self.conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self.conn)   # lets psycopg send and receive NumPy arrays as vectors

    def page_hashes(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT url, content_hash FROM pages").fetchall()
        return {url: content_hash for url, content_hash in rows}

    def replace_page(self, page: Page, chunks: list[str], vectors: np.ndarray) -> None:
        with self.conn.transaction():   # all of it or none of it
            self.conn.execute(
                """
                INSERT INTO pages (url, title, content_hash) VALUES (%s, %s, %s)
                ON CONFLICT (url) DO UPDATE
                SET title = EXCLUDED.title, content_hash = EXCLUDED.content_hash, fetched_at = now()
                """,
                (page.url, page.title, page.content_hash),
            )
            self.conn.execute("DELETE FROM chunks WHERE url = %s", (page.url,))
            with self.conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO chunks (url, position, content, embedding) VALUES (%s, %s, %s, %s)",
                    [(page.url, position, content, np.asarray(vector, dtype=np.float32))
                     for position, (content, vector) in enumerate(zip(chunks, vectors))],
                )

    def delete_page(self, url: str) -> None:
        self.conn.execute("DELETE FROM pages WHERE url = %s", (url,))   # chunks go by ON DELETE CASCADE

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        query = np.asarray(vector, dtype=np.float32)
        rows = self.conn.execute(
            """
            SELECT c.url, p.title, c.content, 1 - (c.embedding <=> %s) AS score
            FROM chunks c JOIN pages p ON p.url = c.url
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (query, query, k),
        ).fetchall()
        return [Hit(url, title, content, float(score)) for url, title, content, score in rows]
```

- `register_vector` from the `pgvector` package teaches psycopg to convert NumPy arrays to and from the `vector` type. The extension must already exist in the database, which is why `schema.sql` is run first.
- `replace_page` wraps the upsert, the delete and the inserts in `conn.transaction()`, so a crash midway leaves the old version of the page intact rather than a page with no chunks.
- `<=>` returns cosine *distance*; `1 - distance` gives the similarity the rest of the code expects. Ordering by the bare operator expression is what lets PostgreSQL use the HNSW index.
- Every value is passed as a `%s` parameter. Never build this SQL with string formatting: URLs and page text come from the crawled site.

### app/indexer.py

```python
"""Bring the store in line with the site, embedding only what changed."""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.chunking import chunk_text
from app.crawler import Crawler
from app.store import Store


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    chunks_embedded: int = 0
    deletions_skipped: bool = False

    def summary(self) -> str:
        text = (f"added {len(self.added)}, updated {len(self.updated)}, unchanged {len(self.unchanged)}, "
                f"deleted {len(self.deleted)}, chunks embedded {self.chunks_embedded}")
        return text + (" (crawl incomplete: deletions skipped)" if self.deletions_skipped else "")


def sync(start_url: str, crawler: Crawler, store: Store, embedder: Embedder) -> SyncReport:
    result = crawler.crawl(start_url)
    known = store.page_hashes()
    report = SyncReport()

    for page in result.pages:
        if known.get(page.url) == page.content_hash:
            report.unchanged.append(page.url)
            continue
        chunks = chunk_text(page.text)
        if not chunks:
            continue
        vectors = embedder.encode([f"{page.title}\n{chunk}" for chunk in chunks])
        store.replace_page(page, chunks, vectors)
        report.chunks_embedded += len(chunks)
        (report.updated if page.url in known else report.added).append(page.url)

    missing = set(known) - {page.url for page in result.pages}
    if result.complete:
        for url in sorted(missing):
            store.delete_page(url)
            report.deleted.append(url)
    elif missing:
        report.deletions_skipped = True
    return report
```

- A page whose stored hash equals its current hash is skipped before any chunking or embedding.
- The title is prepended to each chunk for embedding (`f"{page.title}\n{chunk}"`) so that a chunk saying only "Up to 5 unused days carry over" still knows it is about annual leave. The stored content is the chunk alone.
- Deletion is guarded by `result.complete`. If the site returned a 500 for one page, or the crawl hit `max_pages`, "not seen" does not mean "gone", and wiping pages on a bad day would be worse than keeping stale ones for another run.

### app/assistant.py

```python
"""Answer a question from the indexed pages, with links to the pages used."""
from dataclasses import dataclass
from typing import Protocol

from app.indexer import Embedder
from app.store import Hit, Store

SYSTEM = (
    "You are AskMeridian. Answer only from the numbered sources, which are extracts "
    "from Meridian's documentation site. Cite sources as [1], [2] after each fact. "
    "If the sources do not contain the answer, say you don't know."
)
NO_ANSWER = "I could not find anything about that on the documentation site."


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[tuple[int, str, str]]   # (number, page title, url)


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(f"[{n}] {hit.title} ({hit.url})\n{hit.content}" for n, hit in enumerate(hits, 1))
    return f"Sources:\n{sources}\n\nQuestion: {question}"


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class ClaudeGenerator:
    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def retrieve(question: str, store: Store, embedder: Embedder, k: int = 3, min_score: float = 0.3) -> list[Hit]:
    hits = store.search(embedder.encode([question])[0], k)
    return [hit for hit in hits if hit.score >= min_score]


def answer(question: str, store: Store, embedder: Embedder, generator: Generator, k: int = 3) -> Answer:
    hits = retrieve(question, store, embedder, k)
    if not hits:
        return Answer(NO_ANSWER, [])
    text = generator.generate(build_prompt(question, hits))
    return Answer(text, [(n, hit.title, hit.url) for n, hit in enumerate(hits, 1)])
```

`ClaudeGenerator` is the only code that calls a model. `answer` returns the source list built from the hits, not parsed from the model's text, so the links shown are always pages that were really retrieved. When nothing scores above `min_score`, the model is not called at all.

### cli.py

```python
"""python cli.py sync <start-url>   |   python cli.py ask "<question>" """
import os
import sys

from app.assistant import NO_ANSWER, ClaudeGenerator, answer, retrieve
from app.crawler import Crawler
from app.indexer import MiniLMEmbedder, sync
from app.store import LocalStore


def make_store():
    if dsn := os.environ.get("DATABASE_URL"):
        from app.pg_store import PgVectorStore

        return PgVectorStore(dsn)
    return LocalStore("kb_index.json")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("sync", "ask"):
        sys.exit(__doc__)
    command, argument = sys.argv[1], sys.argv[2]
    store, embedder = make_store(), MiniLMEmbedder()

    if command == "sync":
        report = sync(argument, Crawler(), store, embedder)
        print(report.summary())
        for label in ("added", "updated", "deleted"):
            for url in getattr(report, label):
                print(f"  {label}: {url}")
        return

    hits = retrieve(argument, store, embedder)
    if not hits:
        print(NO_ANSWER)
        return
    for hit in hits:
        print(f"{hit.score:.2f}  {hit.url}\n      {' | '.join(hit.content.splitlines())[:100]}")
    if os.environ.get("ANTHROPIC_API_KEY"):
        result = answer(argument, store, embedder, ClaudeGenerator())
        print("\n" + result.text)
        for number, title, url in result.sources:
            print(f"[{number}] {title}: {url}")
    else:
        print("\nSet ANTHROPIC_API_KEY to generate an answer from these sources.")


if __name__ == "__main__":
    main()
```

### tests/conftest.py

```python
"""A fake website served through httpx.MockTransport, a fake clock and a fake embedder."""
import zlib

import httpx
import numpy as np
import pytest

from app.crawler import Crawler

BASE = "http://docs.test"


def html(body: str, title: str = "Page") -> str:
    return (f"<html><head><title>{title}</title><script>track()</script></head><body>"
            f"<nav><a href='/'>Home</a></nav><main>{body}</main><footer>Footer text</footer></body></html>")


class FakeSite:
    """Maps paths to (status, content type, body). Records every request made."""

    def __init__(self, pages: dict[str, str], robots: str | None = "User-agent: *\nDisallow: /private/\n"):
        self.routes: dict[str, tuple[int, str, str]] = {
            path: (200, "text/html; charset=utf-8", body) for path, body in pages.items()
        }
        if robots is not None:
            self.routes["/robots.txt"] = (200, "text/plain", robots)
        self.requested: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(str(request.url))
        if request.url.host != "docs.test":
            raise AssertionError(f"crawler left the site: {request.url}")
        status, content_type, body = self.routes.get(request.url.path, (404, "text/html", "not found"))
        return httpx.Response(status, headers={"content-type": content_type}, text=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


class FakeClock:
    """Time moves only when someone sleeps."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts_embedded = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts_embedded += len(texts)
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().replace(".", " ").replace("?", " ").split():
                out[row, zlib.crc32(word.encode()) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


@pytest.fixture
def site() -> FakeSite:
    return FakeSite({
        "/": html("<h1>Docs</h1><p>Start here.</p><a href='/leave'>Leave</a> <a href='/vpn#setup'>VPN</a> "
                  "<a href='/private/pay'>Pay</a> <a href='https://elsewhere.test/'>Elsewhere</a> "
                  "<a href='/gone'>Gone</a>", "Home"),
        "/leave": html("<h1>Annual leave</h1><p>Staff receive 25 days of annual leave each year.</p>", "Leave"),
        "/vpn": html("<h1>VPN</h1><p>The VPN is required on public Wi-Fi.</p>", "VPN"),
        "/private/pay": html("<p>Salary bands.</p>", "Pay"),
    })


def make_crawler(site: FakeSite, **kwargs) -> Crawler:
    clock = FakeClock()
    crawler = Crawler(site.client(), sleep=clock.sleep, clock=clock.clock, **kwargs)
    crawler.fake_clock = clock
    return crawler
```

`httpx.MockTransport` lets a real `httpx.Client` be answered by a Python function, so the crawler's own code runs unmodified with no sockets. `FakeSite` fails the test if any request leaves `docs.test`. `FakeClock` advances only when the crawler sleeps, which makes rate-limit tests instant and exact.

### tests/test_extract_and_chunking.py

```python
from app.chunking import chunk_text
from app.extract import extract_links, extract_page
from tests.conftest import html


def test_boilerplate_is_removed_and_blocks_become_lines():
    page = extract_page(html("<h1>Leave</h1><p>Staff receive\n   25 days.</p><ul><li>One</li><li>Two</li></ul>", "Leave"), "u")
    assert page.title == "Leave"
    assert page.text == "Leave\nStaff receive 25 days.\nOne\nTwo"       # no nav, footer or script text


def test_inline_markup_does_not_break_a_sentence():
    page = extract_page(html("<p>See the <a href='/vpn'>VPN guide</a>, then <b>restart</b>.<br>Done.</p>"), "u")
    assert page.text == "See the VPN guide, then restart.\nDone."


def test_comments_are_dropped_and_table_cells_are_separated():
    page = extract_page(html("<p>Visible<!-- secret note --></p><table><tr><td>Budget</td><td>£400</td></tr></table>"), "u")
    assert page.text == "Visible\nBudget £400"


def test_links_are_absolute_without_fragments_or_repeats():
    links = extract_links("<a href='b.html#top'>b</a> <a href='/b.html'>b again</a> "
                          "<a href='mailto:x@y.test'>mail</a> <a href='https://other.test/'>o</a>",
                          "http://docs.test/a/index.html")
    assert links == ["http://docs.test/a/b.html", "http://docs.test/b.html", "https://other.test/"]


def test_hash_ignores_markup_changes_but_not_text_changes():
    original = extract_page(html("<p>25 days of leave.</p>"), "u")
    restyled = extract_page(html("<p class='new'>25   days of leave.</p>").replace("track()", "track(2)"), "u")
    edited = extract_page(html("<p>26 days of leave.</p>"), "u")
    assert original.content_hash == restyled.content_hash
    assert original.content_hash != edited.content_hash


def test_short_pages_are_one_chunk():
    assert chunk_text("Leave\nStaff receive 25 days.") == ["Leave\nStaff receive 25 days."]
    assert chunk_text("") == []


def test_chunks_respect_the_limit_and_overlap_by_one_line():
    lines = [f"Line number {i} of the policy." for i in range(6)]
    chunks = chunk_text("\n".join(lines), max_chars=90)
    assert all(len(chunk) <= 90 for chunk in chunks)
    assert chunks[0].splitlines()[-1] == chunks[1].splitlines()[0]     # the shared line
    assert all(line in "\n".join(chunks) for line in lines)            # nothing lost


def test_a_very_long_line_is_split_into_sentences():
    chunks = chunk_text("First sentence here. Second sentence here. Third sentence here.", max_chars=45, overlap=False)
    assert chunks == ["First sentence here.\nSecond sentence here.", "Third sentence here."]
```

### tests/test_crawler.py

```python
import httpx

from tests.conftest import BASE, FakeSite, html, make_crawler


def test_crawl_follows_links_within_the_site(site):
    result = make_crawler(site).crawl(f"{BASE}/")
    assert [page.url for page in result.pages] == [f"{BASE}/", f"{BASE}/leave", f"{BASE}/vpn"]
    assert result.complete


def test_robots_txt_is_fetched_first_and_obeyed(site):
    result = make_crawler(site).crawl(f"{BASE}/")
    assert site.requested[0] == f"{BASE}/robots.txt"
    assert f"{BASE}/private/pay" not in site.requested
    assert f"{BASE}/private/pay" in result.skipped


def test_other_sites_are_never_requested(site):
    result = make_crawler(site).crawl(f"{BASE}/")      # FakeSite raises if the crawler leaves docs.test
    assert "https://elsewhere.test/" in result.skipped


def test_a_broken_link_is_skipped_and_does_not_make_the_crawl_incomplete(site):
    result = make_crawler(site).crawl(f"{BASE}/")
    assert f"{BASE}/gone" in result.skipped and result.complete


def test_requests_are_spaced_by_the_delay(site):
    crawler = make_crawler(site, delay=2.0)
    crawler.crawl(f"{BASE}/")
    assert len(crawler.fake_clock.sleeps) == len(site.requested) - 1    # no wait before the first request
    assert all(seconds == 2.0 for seconds in crawler.fake_clock.sleeps)


def test_crawl_delay_from_robots_txt_overrides_a_smaller_delay(site):
    site.routes["/robots.txt"] = (200, "text/plain", "User-agent: *\nCrawl-delay: 5\n")
    crawler = make_crawler(site, delay=0.5)
    crawler.crawl(f"{BASE}/")
    assert crawler.delay == 5.0 and crawler.fake_clock.sleeps[-1] == 5.0


def test_missing_robots_txt_allows_everything():
    site = FakeSite({"/": html("<a href='/private/pay'>p</a>"), "/private/pay": html("<p>x</p>")}, robots=None)
    assert len(make_crawler(site).crawl(f"{BASE}/").pages) == 2


def test_unreadable_robots_txt_blocks_everything(site):
    site.routes["/robots.txt"] = (503, "text/plain", "")
    result = make_crawler(site).crawl(f"{BASE}/")
    assert result.pages == [] and result.skipped == [f"{BASE}/"]


def test_server_errors_are_failures_and_make_the_crawl_incomplete(site):
    site.routes["/vpn"] = (500, "text/html", "oops")
    result = make_crawler(site).crawl(f"{BASE}/")
    assert result.failed == [f"{BASE}/vpn"] and not result.complete


def test_network_errors_are_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ConnectError("connection refused")

    crawler = make_crawler(FakeSite({}))
    crawler.client = httpx.Client(transport=httpx.MockTransport(handler))
    assert crawler.crawl(f"{BASE}/").failed == [f"{BASE}/"]


def test_non_html_responses_are_skipped(site):
    site.routes["/leave"] = (200, "application/pdf", "%PDF-1.7")
    result = make_crawler(site).crawl(f"{BASE}/")
    assert f"{BASE}/leave" in result.skipped


def test_max_pages_stops_the_crawl_and_marks_it_truncated(site):
    result = make_crawler(site, max_pages=2).crawl(f"{BASE}/")
    assert len(result.pages) == 2 and result.truncated and not result.complete
```

### tests/test_indexer_and_assistant.py

```python
from app.assistant import NO_ANSWER, answer, retrieve
from app.indexer import sync
from app.store import LocalStore
from tests.conftest import BASE, FakeEmbedder, html, make_crawler


def run_sync(site, store, embedder):
    return sync(f"{BASE}/", make_crawler(site), store, embedder)


def test_first_sync_adds_every_page(site):
    store, embedder = LocalStore(), FakeEmbedder()
    report = run_sync(site, store, embedder)
    assert report.added == [f"{BASE}/", f"{BASE}/leave", f"{BASE}/vpn"]
    assert report.chunks_embedded == embedder.texts_embedded == 3
    assert set(store.page_hashes()) == set(report.added)


def test_second_sync_embeds_nothing(site):
    store, embedder = LocalStore(), FakeEmbedder()
    run_sync(site, store, embedder)
    before = embedder.texts_embedded
    report = run_sync(site, store, embedder)
    assert len(report.unchanged) == 3 and report.chunks_embedded == 0
    assert embedder.texts_embedded == before


def test_only_the_changed_page_is_re_embedded_and_old_chunks_are_replaced(site):
    store, embedder = LocalStore(), FakeEmbedder()
    run_sync(site, store, embedder)
    site.routes["/leave"] = (200, "text/html", html("<h1>Annual leave</h1><p>Staff receive 25 days. Up to 5 carry over.</p>", "Leave"))
    report = run_sync(site, store, embedder)
    assert report.updated == [f"{BASE}/leave"] and report.chunks_embedded == 1
    contents = [hit.content for hit in store.search(embedder.encode(["annual leave"])[0], k=10)]
    assert any("Up to 5 carry over." in c for c in contents)
    assert not any("of annual leave each year" in c for c in contents)        # the stale chunk is gone


def test_a_page_that_disappears_is_deleted(site):
    store, embedder = LocalStore(), FakeEmbedder()
    run_sync(site, store, embedder)
    del site.routes["/vpn"]                       # now a 404
    report = run_sync(site, store, embedder)
    assert report.deleted == [f"{BASE}/vpn"] and f"{BASE}/vpn" not in store.page_hashes()


def test_nothing_is_deleted_when_the_crawl_was_incomplete(site):
    store, embedder = LocalStore(), FakeEmbedder()
    run_sync(site, store, embedder)
    site.routes["/vpn"] = (500, "text/html", "temporary outage")
    report = run_sync(site, store, embedder)
    assert report.deleted == [] and report.deletions_skipped
    assert f"{BASE}/vpn" in store.page_hashes()


def test_local_store_survives_a_restart(site, tmp_path):
    path = tmp_path / "index.json"
    run_sync(site, LocalStore(path), FakeEmbedder())
    reopened, embedder = LocalStore(path), FakeEmbedder()
    assert len(run_sync(site, reopened, embedder).unchanged) == 3 and embedder.texts_embedded == 0


class FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Staff receive 25 days [1]."


def test_answer_links_to_the_source_pages(site):
    store, embedder, generator = LocalStore(), FakeEmbedder(), FakeGenerator()
    run_sync(site, store, embedder)
    result = answer("How many days of annual leave do staff receive?", store, embedder, generator, k=1)
    assert result.text == "Staff receive 25 days [1]."
    assert result.sources == [(1, "Leave", f"{BASE}/leave")]
    assert f"[1] Leave ({BASE}/leave)" in generator.prompts[0]


def test_no_relevant_page_means_no_model_call(site):
    store, embedder, generator = LocalStore(), FakeEmbedder(), FakeGenerator()
    run_sync(site, store, embedder)
    result = answer("zebra crossing regulations", store, embedder, generator)
    assert result.text == NO_ANSWER and result.sources == [] and generator.prompts == []


def test_private_pages_never_reach_the_index(site):
    store, embedder = LocalStore(), FakeEmbedder()
    run_sync(site, store, embedder)
    assert all("Salary" not in hit.content for hit in retrieve("salary bands", store, embedder, k=10, min_score=0.0))
```

## Running it

Create the site and serve it in a second terminal:

```bash
python make_site.py
python -m http.server 8765 --directory site
```

First sync:

```bash
python cli.py sync http://localhost:8765/index.html
```

```output
added 5, updated 0, unchanged 0, deleted 0, chunks embedded 5
  added: http://localhost:8765/index.html
  added: http://localhost:8765/leave.html
  added: http://localhost:8765/remote.html
  added: http://localhost:8765/expenses.html
  added: http://localhost:8765/it/vpn.html
```

The server's log for that run shows the politeness rules at work: `robots.txt` first, then one request per second (the `Crawl-delay`), no request to `/private/` or to example.com, and the broken link answered with a 404:

```output
::1 - - [14/Nov/2026 10:20:43] "GET /robots.txt HTTP/1.1" 200 -
::1 - - [14/Nov/2026 10:20:44] "GET /index.html HTTP/1.1" 200 -
::1 - - [14/Nov/2026 10:20:45] "GET /leave.html HTTP/1.1" 200 -
::1 - - [14/Nov/2026 10:20:46] "GET /remote.html HTTP/1.1" 200 -
::1 - - [14/Nov/2026 10:20:47] "GET /expenses.html HTTP/1.1" 200 -
::1 - - [14/Nov/2026 10:20:48] "GET /it/vpn.html HTTP/1.1" 200 -
::1 - - [14/Nov/2026 10:20:49] code 404, message File not found
::1 - - [14/Nov/2026 10:20:49] "GET /old-handbook.html HTTP/1.1" 404 -
```

Run the same command again and nothing is embedded:

```output
added 0, updated 0, unchanged 5, deleted 0, chunks embedded 0
```

Ask a question (retrieval only, without a key):

```bash
python cli.py ask "Do I need the VPN in a hotel?"
```

```output
0.61  http://localhost:8765/it/vpn.html
      VPN | The VPN is required whenever you connect from public Wi-Fi, such as in a café, hotel or airpor
0.39  http://localhost:8765/index.html
      Meridian documentation | Policies and guides for Meridian staff. Choose a topic above. | See also th

Set ANTHROPIC_API_KEY to generate an answer from these sources.
```

Now change the site: edit one page and remove another.

```bash
sed -i.bak 's/for a desk, chair or monitor\./for a desk, chair or monitor. Claim it in the finance portal./' site/remote.html
rm site/expenses.html
python cli.py sync http://localhost:8765/index.html
```

```output
added 0, updated 1, unchanged 3, deleted 1, chunks embedded 1
  updated: http://localhost:8765/remote.html
  deleted: http://localhost:8765/expenses.html
```

One page re-embedded, one deleted, three untouched. A question about the removed page now finds nothing, where before the change it returned `expenses.html` with a score of 0.34:

```bash
python cli.py ask "When do I need a receipt?"
```

```output
I could not find anything about that on the documentation site.
```

And the private page was never indexed:

```bash
python cli.py ask "What is in the salary bands?"
```

```output
I could not find anything about that on the documentation site.
```

With `ANTHROPIC_API_KEY` set, `ask` also prints a generated answer followed by its sources. That step was not run; the wording will differ, but it looks like this (illustrative):

```text
Yes. The VPN is required whenever you connect from public Wi-Fi, such as in a café, hotel or airport [1].
[1] VPN | Meridian Docs: http://localhost:8765/it/vpn.html
[2] Welcome | Meridian Docs: http://localhost:8765/index.html
```

Stop the web server with Ctrl+C when you have finished.

## Testing

```bash
python -m pytest -q
```

```output
.............................                                            [100%]
29 passed in 0.06s
```

No test opens a socket, loads a model or sleeps. Between them they establish that:

- **Extraction** removes navigation, footer, script and comments, keeps sentences whole across inline tags, and yields a hash that ignores markup but not wording.
- **The crawler** fetches `robots.txt` first and obeys it, never requests another host, spaces its requests by exactly the configured delay, honours `Crawl-delay`, and classifies 404s, PDFs, server errors and network errors correctly.
- **The indexer** embeds everything once, nothing on an unchanged site, only the changed page after an edit (and removes the stale chunk), deletes vanished pages, and refuses to delete when the crawl was incomplete.
- **The assistant** returns the URLs of the pages used, and does not call the model when nothing relevant was found.

`PgVectorStore` has no tests here because there was no database to run them against. When you have one, run the store-level assertions from `test_indexer_and_assistant.py` with `PgVectorStore` in place of `LocalStore`; the interface is the same.

## Expected results

Against the sample site you should see the sync summaries above exactly: 5 added; then 5 unchanged with 0 chunks embedded; then 1 updated, 3 unchanged and 1 deleted with 1 chunk embedded. Retrieval scores should match too, as the embedding model is deterministic. The first sync took about 12 seconds on the machine used here, six of them deliberate waiting between its seven requests.

The cost argument for incremental sync is simple arithmetic. If a site has 2,000 pages and 20 change in a day, a full rebuild embeds 2,000 pages' worth of chunks and an incremental sync embeds 20, one hundredth of the work. The crawl itself is not reduced: every page is still fetched to compute its hash.

## Common problems

- **A fact is on the site but is not retrieved.** On the sample site, "How often must I change my password?" scores only 0.23 against the VPN page, below the 0.3 threshold, because the password sentence is a minor part of a chunk titled "VPN". Smaller chunks, or a page of its own for the topic, fix that; lowering the threshold lets in unrelated pages.
- **Every page shows as updated on every sync.** Something in the extracted text changes on each request: a timestamp, a "last visited" banner, a random recommendation. Print two extractions of the same page and compare, then add the offending element to `BOILERPLATE`.
- **The crawl finds only the start page.** The site builds its links with JavaScript. `httpx` does not run scripts; such sites need a sitemap, an API, or a headless browser.
- **Pages are skipped as non-HTML.** The server sends a content type other than `text/html`. Check the response headers before loosening the test.
- **`vector type not found in the database`** from `register_vector`. The extension has not been created in that database; run `schema.sql` first.
- **Deletions never happen.** One URL fails on every crawl, so the crawl is never `complete`. Print `result.failed`.
- **The crawl is slow.** That is the rate limit working. Raise the rate only for a site you operate.

## Possible improvements

- **Use LangChain's splitter.** The listed stack includes LangChain. Its recursive splitter can replace `chunk_text` in one line of `indexer.py`; this snippet uses the `langchain-text-splitters` package and was not executed here:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_text(page.text)
```

  See [Text splitters](/learn/langchain-for-rag/text-splitters) for how it chooses split points.
- **Send conditional requests.** Store each page's `ETag` or `Last-Modified` header and send `If-None-Match` or `If-Modified-Since`; a 304 response saves the download and the parsing as well as the embedding.
- **Read the sitemap.** `sitemap.xml` lists pages that no link reaches and often carries modification dates.
- **Hash per chunk** instead of per page, so that editing one paragraph of a long page re-embeds one chunk.
- **Add keyword search in PostgreSQL** with a `tsvector` column and fuse the two rankings, as in the [Hybrid search RAG application](/projects/hybrid-search-rag) project.
- **Schedule the sync** and record each `SyncReport`; a sudden jump in `updated` usually means a template change, not a content change.
