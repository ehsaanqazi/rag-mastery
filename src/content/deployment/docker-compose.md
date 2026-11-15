A RAG system is never one container. AskMeridian needs its API, a [Qdrant](/learn/vector-database-ecosystem/qdrant) vector database, a cache, and an ingestion job that fills the index. Starting each with a long `docker run` command, in the right order, on the right network, with the right volumes and environment variables, is slow and error-prone. **Docker Compose** replaces those commands with one declarative file that describes the whole stack, and one command that brings it up or tears it down.

This lesson covers how Compose models a multi-container application: services, the private network that lets them find each other by name, volumes that keep data alive, health-gated start-up ordering, configuration and secrets, and profiles for one-off jobs. You will write a Compose file for the AskMeridian stack and a small linter that catches the mistakes that most often break RAG stacks. It builds on the image from [Docker](/learn/deployment/docker); running across many machines is the job of [Kubernetes](/learn/deployment/kubernetes-fundamentals).

## Learning objectives

- Write a Compose file that runs a RAG API, a vector database, a cache and an ingestion job together.
- Explain how containers in a Compose project find each other, and why `localhost` inside a container refers to that container.
- Use health checks and `depends_on` conditions to order start-up, and explain what they cannot guarantee.
- Separate configuration, interpolation variables and secrets correctly.
- Decide when Compose is the right production tool and when a stack has outgrown it.

## Prerequisites

- [Docker](/learn/deployment/docker): images, containers, layers and health checks.
- [Qdrant](/learn/vector-database-ecosystem/qdrant): the vector database the stack runs.
- [Indexing pipeline](/learn/rag-architecture-fundamentals/indexing-pipeline): what the ingestion job does.

## Why this matters

Without a declarative description, a multi-container stack lives in people's heads and shell histories. The failures are predictable:

- **The new engineer cannot run it.** A README lists six `docker run` commands; one is out of date, and nobody notices because everyone else's containers are already running.
- **Start-up races.** The API starts before Qdrant is accepting connections, fails its first connection attempt, and exits. Or the ingestion job starts writing before the collection exists.
- **Lost data.** Qdrant was started without a volume, so recreating its container to change one setting silently deletes the index, and re-embedding the corpus takes an hour.
- **Environment drift.** Developers, CI and the demo server each run slightly different versions of the database, so a bug that appears in one cannot be reproduced in another.
- **Leaked secrets.** API keys pasted into commands end up in shell history.

A Compose file fixes the stack's shape in version control. Everyone, including CI, runs the same services at the same versions, wired the same way.

## The intuition

Think of a theatre production. Each actor (a container) knows their own part, but someone has to say who is on stage, in what order they enter, which props must already be in place, and which stage door leads where. That is the stage manager's running sheet.

A Compose file is the running sheet. It does not change how any actor performs; the image does that. It says: Qdrant and Redis go on first; the API enters only once both are ready; the ingestion job is a cameo that only appears when called for; and the props (the index files) stay in the props room between performances rather than being thrown away when an actor leaves the stage.

The other half of the analogy is the backstage corridor. Actors talk to each other by name ("pass this to Qdrant"), not by the street address of the theatre. That corridor is the Compose network.

## How it works

### The model: project, services, networks, volumes

A Compose file (by convention `compose.yaml`) follows the Compose Specification and has four top-level sections that matter here:

- **`services`**: each one is a container definition, with an image (or a `build` section to build one), command, environment, ports, volumes and dependencies. One service can be scaled to several identical containers.
- **`networks`**: virtual networks. If you declare none, Compose creates one called `<project>_default` and attaches every service to it.
- **`volumes`**: named, Docker-managed storage that outlives containers.
- **`secrets`** and **`configs`**: files injected into containers at known paths.

Everything belongs to a **project**, named by the top-level `name:` key or, failing that, the directory name. Compose prefixes resources with the project name (`askmeridian_qdrant_data`, `askmeridian-api-1`), so two projects on one machine do not collide.

You will still see files beginning `version: "3.8"`. That key is obsolete: the current Compose tool (Compose V2, invoked as `docker compose`) ignores it and prints a warning. The older Python tool `docker-compose`, with a hyphen, is no longer maintained.

### What `docker compose up` does

1. Reads `compose.yaml` and, if present, merges `compose.override.yaml` over it; substitutes `${VARIABLES}` from the shell environment and the project's `.env` file.
2. Creates the project network and any named volumes that do not exist yet.
3. Builds or pulls images for services that need them.
4. Works out the dependency graph from `depends_on` and creates containers in order, waiting for each dependency's condition (started, healthy, or completed successfully) before starting the services that depend on it.
5. Attaches each container to the network, where Docker's embedded DNS resolves each service name to its container's address.
6. With `-d`, returns; with `--wait`, returns only once services are running and healthy.

`docker compose down` reverses steps 1 to 5: it removes containers and networks but **keeps named volumes**. `docker compose down -v` also deletes the volumes, and with them your index.

### Service discovery and the `localhost` trap

Every container has its own network namespace, so inside the API container `localhost` means the API container itself. A setting such as `QDRANT_URL=http://localhost:6333`, which worked when Qdrant ran on your laptop, now points at a port on which nothing is listening, and you get `Connection refused`. Inside the project network, use the **service name** as the hostname: `http://qdrant:6333`.

```mermaid
flowchart LR
    U["Your browser / curl"] -->|"127.0.0.1:8000"| API
    subgraph net["Network: askmeridian_default"]
        API["api"] -->|"http://qdrant:6333"| Q["qdrant"]
        API -->|"redis://redis:6379"| R["redis"]
        ING["ingest<br/>(profile: tools)"] -->|"http://qdrant:6333"| Q
    end
    Q --- V[("volume<br/>qdrant_data")]
    D["./data/handbook"] -.->|"read-only bind mount"| ING
```

**Publishing** a port (`ports: ["127.0.0.1:8000:8000"]`) maps a port on the host to the container. Services only need to talk to each other inside the network, which needs no published ports at all. In the AskMeridian file only the API is published, and only on the host's loopback interface. Qdrant and Redis are reachable by the API and unreachable from outside.

### Volumes and bind mounts

There are two ways to give a container storage that survives it:

- A **named volume** (`qdrant_data:/qdrant/storage`) is managed by Docker. Use it for data the service owns, such as Qdrant's segments and write-ahead log.
- A **bind mount** (`./data/handbook:/data/handbook:ro`) maps a host directory into the container. Use it for inputs you manage on the host, such as source documents, or for live-editing code in development. The `:ro` suffix makes it read-only.

A cache such as Redis often needs neither. AskMeridian runs Redis with persistence switched off: if the cache is lost, the only cost is a few slower requests.

### Start-up order and health

`depends_on` controls **creation order**, and its `condition` says what "ready" means:

| Condition | Waits until the dependency… | Typical use |
| --- | --- | --- |
| `service_started` (default) | has been started | Rarely enough on its own |
| `service_healthy` | reports healthy from its health check | Databases, the API |
| `service_completed_successfully` | has exited with code 0 | Migrations, collection creation |

`service_healthy` only works if the dependency defines a `healthcheck`. A health check is a command run inside the container on an interval; its state moves from `starting` to `healthy` after a success, and to `unhealthy` after `retries` consecutive failures. `start_period` is a grace period during which failures do not count, which matters for an API that spends tens of seconds loading a model.

Two limits are easy to miss:

- **Ordering is only at start-up.** If Qdrant restarts at three in the morning, Compose does not stop the API first. Your application must still retry connections with backoff and report not-ready while a dependency is down.
- **Plain Docker does not restart an unhealthy container.** The `restart:` policy reacts to the process *exiting*. A container stuck in `unhealthy` stays running unless something else acts on it. Orchestrators such as Kubernetes do act on failed liveness checks, which is one reason to move to one.

### Configuration, interpolation and secrets

Three mechanisms look alike and are constantly confused:

1. **`environment:`** sets variables inside the container. This is where ordinary configuration goes: `QDRANT_URL`, `LOG_LEVEL`, `TOP_K`.
2. **Interpolation** with `${VAR}` happens when Compose *reads the file*, using your shell's environment and a `.env` file in the project directory. `${LOG_LEVEL:-INFO}` supplies a default; `${QDRANT_API_KEY:?set it in .env}` refuses to start with that message if the variable is missing. A `.env` file is **not** passed into containers by itself; only what the file references is.
3. **`secrets:`** mount a file at `/run/secrets/<name>` inside the container. The value never appears in `docker inspect` output for the container's environment, and the source file stays out of the image. Use this for API keys and database passwords.

`env_file:` is a fourth option: it loads a file of `KEY=value` lines into a service's environment. It is convenient, but values loaded this way are ordinary environment variables, visible to anyone who can inspect the container.

### Profiles, one-off jobs and overrides

Not every service should start with `docker compose up`. A service with `profiles: ["tools"]` is skipped unless that profile is activated or the service is named explicitly. That suits AskMeridian's ingestion job: `docker compose run --rm ingest` starts Qdrant if needed, runs the job once, and removes its container when it exits.

**Override files** layer environment-specific changes over a base file. `compose.override.yaml` is merged automatically, which makes it the natural place for development-only settings: a bind mount of the source code, a `--reload` flag, debug logging. CI or a demo server can name files explicitly instead (`docker compose -f compose.yaml -f compose.ci.yaml up`), and `docker compose config` prints the fully merged, interpolated result, which is the quickest way to see what will actually run.

### When Compose is enough, and when it is not

Compose runs containers on **one host**. That is exactly right for local development, integration tests in CI, demos, and small internal production services where one machine is enough and a few minutes of downtime during maintenance is acceptable. It is the wrong tool when you need any of the following:

- more than one machine, for capacity or to survive a machine failing;
- rolling updates that replace containers without dropping traffic;
- autoscaling on load;
- automatic replacement of unhealthy containers and rescheduling after a node failure.

Those are the problems an orchestrator such as [Kubernetes](/learn/deployment/kubernetes-fundamentals), or a managed container service such as those in [Cloud deployment](/learn/deployment/cloud-deployment), solves.

## In code

### The AskMeridian Compose file

The task: a single file that brings up the whole AskMeridian stack for development and CI, with Qdrant's data on a volume, the API gated on its dependencies being healthy, the API key delivered as a secret, and ingestion as an on-demand job. Docker was not available in the environment used to write this lesson, so the file was not executed with `docker compose up`; it was parsed and checked by the linter shown afterwards.

```yaml
name: askmeridian

services:
  qdrant:
    image: qdrant/qdrant:v1.15.0          # pin the version you test against
    volumes:
      - qdrant_data:/qdrant/storage       # the index survives container replacement
    healthcheck:
      # The Qdrant image has no curl; bash's /dev/tcp checks the port is accepting connections.
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  redis:
    image: redis:7.4-alpine
    command: ["redis-server", "--save", "", "--appendonly", "no"]   # a cache: no persistence
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  api:
    build: .
    image: askmeridian-api:dev
    ports:
      - "127.0.0.1:8000:8000"             # only the API is published, and only to this machine
    environment:
      QDRANT_URL: http://qdrant:6333      # service name, not localhost
      REDIS_URL: redis://redis:6379/0
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    secrets:
      - anthropic_api_key                 # appears as /run/secrets/anthropic_api_key
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2g
    restart: unless-stopped

  ingest:
    build: .
    image: askmeridian-api:dev
    command: ["python", "-m", "app.ingest", "--source", "/data/handbook"]
    environment:
      QDRANT_URL: http://qdrant:6333
    volumes:
      - ./data/handbook:/data/handbook:ro # read-only bind mount of the source documents
    depends_on:
      qdrant:
        condition: service_healthy
    profiles: ["tools"]                   # not started by a plain `docker compose up`

volumes:
  qdrant_data:

secrets:
  anthropic_api_key:
    file: ./secrets/anthropic_api_key.txt
```

The important lines:

- **The Qdrant health check** tests that the port accepts TCP connections, using a bash feature because the image does not ship `curl`. (If your Qdrant image version differs, check what tools it contains.) An open port is a weaker signal than a successful API call, which is one more reason the API must retry its own connections.
- **The API's health check hits `/readyz`**, the endpoint from the [Docker](/learn/deployment/docker) lesson that reports 503 until the model is loaded, and `start_period: 60s` gives model loading time before failures count.
- **`ports: "127.0.0.1:8000:8000"`** publishes only on loopback. Without the address, Docker publishes on every interface of the host.
- **`deploy.resources.limits`** caps the API at two CPUs and 2 GB, so a memory leak or an oversized batch cannot starve Qdrant on the same machine.
- **`api` and `ingest` share one image**, built from the same Dockerfile; the ingestion job differs only in its command and mounts. One image per codebase is easier to version than one per entry point.

Typical commands:

```bash
mkdir -p secrets && printf '%s' "$ANTHROPIC_API_KEY" > secrets/anthropic_api_key.txt
echo "secrets/" >> .gitignore

docker compose up -d --build --wait     # build, start, and wait until healthy
docker compose run --rm ingest          # one-off ingestion into Qdrant
docker compose ps                       # state and health of each service
docker compose logs -f api              # follow the API's logs
docker compose config                   # the merged, interpolated file
docker compose down                     # stop; the qdrant_data volume is kept
```

### Reading the secret in the application

Inside the container the key is a file, not an environment variable. `pydantic-settings` reads such files natively: with `secrets_dir` set, a field named `anthropic_api_key` is filled from a file of that name. This script simulates the container's `/run/secrets` directory with a temporary one so it can run anywhere:

```python
import os
import tempfile
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Simulate what Compose does: a secret file in a directory, plus ordinary env vars.
secrets = Path(tempfile.mkdtemp())
(secrets / "anthropic_api_key").write_text("sk-ant-demo-value\n")
os.environ["QDRANT_URL"] = "http://qdrant:6333"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(secrets_dir=str(secrets))  # "/run/secrets" in the container
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"
    anthropic_api_key: SecretStr


s = Settings()
print("qdrant_url:", s.qdrant_url)          # from the environment
print("redis_url: ", s.redis_url)           # default: nothing overrode it
print("api key:   ", s.anthropic_api_key)   # from the secret file, masked when printed
print("length:    ", len(s.anthropic_api_key.get_secret_value()))
```

```output
qdrant_url: http://qdrant:6333
redis_url:  redis://localhost:6379/0
api key:    **********
length:     17
```

Environment variables take precedence over secret files, and defaults apply when neither is set. `SecretStr` masks the value whenever it is printed or logged, so an accidental `log.info(settings)` does not leak the key. The length of 17 shows the trailing newline in the file was not kept as part of the key. The application passes the value to the SDK explicitly: `anthropic.Anthropic(api_key=s.anthropic_api_key.get_secret_value())`.

### A Compose linter

Most broken Compose files fail in the same handful of ways, and each can be detected statically. This script loads a Compose file with PyYAML, flags those mistakes, and prints the order in which Compose will start the default services, grouped into "waves" that can start in parallel. Install with `pip install pyyaml`.

```python
"""Lint a Compose file for the mistakes that bite RAG stacks, and print the start-up order."""
import re
import sys

import yaml

STATEFUL = ("qdrant", "postgres", "pgvector", "milvus", "weaviate", "elasticsearch", "opensearch")
SECRET_NAME = re.compile(r"(KEY|TOKEN|PASSWORD|SECRET)", re.I)


def deps(svc: dict) -> dict[str, str]:
    """Normalise depends_on (list or mapping form) to {service: condition}."""
    d = svc.get("depends_on", {})
    if isinstance(d, list):
        return {name: "service_started" for name in d}
    return {name: (spec or {}).get("condition", "service_started") for name, spec in d.items()}


def env_items(svc: dict):
    env = svc.get("environment", {})
    if isinstance(env, list):  # ["KEY=value", ...] form
        env = dict(item.split("=", 1) if "=" in item else (item, None) for item in env)
    return env.items()


def lint(spec: dict) -> list[str]:
    services, problems = spec.get("services", {}), []
    for name, svc in services.items():
        image = svc.get("image", "")
        if image and (":" not in image.split("/")[-1] or image.endswith(":latest")):
            problems.append(f"{name}: image '{image}' is not pinned to a version")
        for target, cond in deps(svc).items():
            if target not in services:
                problems.append(f"{name}: depends on unknown service '{target}'")
            elif cond == "service_healthy" and "healthcheck" not in services[target]:
                problems.append(f"{name}: waits for '{target}' to be healthy, but it has no healthcheck")
        for key, value in env_items(svc):
            if value and re.search(r"//(localhost|127\.0\.0\.1)[:/]", str(value)):
                problems.append(f"{name}: {key}={value} points at the container itself; use the service name")
            if value and SECRET_NAME.search(key) and not str(value).startswith("${"):
                problems.append(f"{name}: {key} has a literal value; use a secret or ${{VAR}} interpolation")
        for port in svc.get("ports", []):
            if str(port).count(":") == 1 and any(s in (image or name) for s in STATEFUL):
                problems.append(f"{name}: port {port} publishes the database on every interface")
        if any(s in image for s in STATEFUL) and not svc.get("volumes"):
            problems.append(f"{name}: stateful service with no volume; its data dies with the container")
    return problems


def start_order(spec: dict) -> list[list[str]]:
    """Group services into waves: each wave can start once all earlier waves are up."""
    services = {n: s for n, s in spec["services"].items() if not s.get("profiles")}
    remaining = {n: {d for d in deps(s) if d in services} for n, s in services.items()}
    waves = []
    while remaining:
        ready = sorted(n for n, d in remaining.items() if not d)
        if not ready:
            raise ValueError(f"dependency cycle among: {sorted(remaining)}")
        waves.append(ready)
        for n in ready:
            del remaining[n]
        for d in remaining.values():
            d.difference_update(ready)
    return waves


for path in sys.argv[1:]:
    spec = yaml.safe_load(open(path))
    print(f"== {path}")
    for p in lint(spec) or ["no problems found"]:
        print("  -", p)
    try:
        for i, wave in enumerate(start_order(spec), 1):
            print(f"  wave {i}: {', '.join(wave)}")
    except ValueError as e:
        print("  ", e)
```

To see it catch something, `compose.naive.yaml` is the kind of file people write first:

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      QDRANT_URL: http://localhost:6333
      ANTHROPIC_API_KEY: sk-ant-example-not-a-real-key
    depends_on:
      qdrant:
        condition: service_healthy
  ingest:
    build: .
    command: ["python", "-m", "app.ingest"]
    depends_on: [api, worker]
```

Running `python check_compose.py compose.yaml compose.naive.yaml`:

```output
== compose.yaml
  - no problems found
  wave 1: qdrant, redis
  wave 2: api
== compose.naive.yaml
  - qdrant: image 'qdrant/qdrant' is not pinned to a version
  - qdrant: port 6333:6333 publishes the database on every interface
  - qdrant: stateful service with no volume; its data dies with the container
  - api: waits for 'qdrant' to be healthy, but it has no healthcheck
  - api: QDRANT_URL=http://localhost:6333 points at the container itself; use the service name
  - api: ANTHROPIC_API_KEY has a literal value; use a secret or ${VAR} interpolation
  - ingest: depends on unknown service 'worker'
  wave 1: qdrant
  wave 2: api
  wave 3: ingest
```

The good file starts Qdrant and Redis together, then the API; `ingest` does not appear because its profile is not active. The naive file produces seven findings, each one a real outage or leak: an index that vanishes on the next `docker compose up --force-recreate`, a database reachable from the network with no authentication, an API that cannot reach Qdrant, and a key committed to version control. In the naive file the start order also puts `ingest` after the API, which it does not need; dependencies should express what a service actually requires.

**Errors people commonly hit.** `dependency failed to start: container … is unhealthy` means a health check never passed; run `docker compose logs <service>` and run the check command by hand with `docker compose exec`. `Bind for 0.0.0.0:6333 failed: port is already allocated` means something on the host (often a Qdrant you started earlier with `docker run`) already holds the port. A variable shown as blank in `docker compose config` was not set in the shell or `.env`.

**Extending it.** Run the linter in CI on every change to `compose.yaml`; add a rule that every service with a `build` section also sets `image` so builds are tagged; add a collection-creation service with `service_completed_successfully` so ingestion never races schema creation.

## A real-world example

When Meridian's Platform team opened AskMeridian to contributors from other teams, the onboarding document asked newcomers to start Qdrant and the API with `docker run` and then run an ingestion script. In the first month, three contributors lost an afternoon each.

The first set `QDRANT_URL=http://localhost:6333` because that was the default in the code and it had worked when the API ran outside Docker. Inside the API container the request went to the container itself. The second ran ingestion while Qdrant was still replaying its write-ahead log after a restart; the script failed on its first request and left a half-populated collection, which then produced confusing retrieval results for days. The third tidied up with `docker compose down -v` from an earlier experiment's instructions and deleted the shared demo index an hour before a stakeholder demonstration.

The team replaced the document with the Compose file in this lesson, made ingestion a profiled job that waits for a healthy Qdrant, and added a separate `create-collection` service that ingestion waits on with `service_completed_successfully`. They added the linter to CI. The demo server now uses a separate override file and project name (`docker compose -p askmeridian-demo …`), so its volume cannot be removed by commands typed in a development checkout. The API also gained connection retries with exponential backoff, because the team understood that `depends_on` protects only the first start-up.

## Common mistakes

- **Using `localhost` to reach another service.** Inside a container it means the container itself, so connections are refused.
- **Running a database without a named volume.** The index is deleted the next time the container is recreated.
- **Treating `depends_on` as a runtime guarantee.** It orders start-up only; without retries, the API fails whenever a dependency restarts later.
- **Publishing database ports.** `ports: ["6333:6333"]` exposes an unauthenticated Qdrant on every network interface of the host.
- **Putting secrets in `environment:` or committing `.env`.** The values end up in version control and in `docker inspect` output.
- **Typing `docker compose down -v` out of habit.** It deletes named volumes, including the index.
- **Using `:latest` images.** Developers and CI silently run different database versions.
- **Running production on Compose when the requirements need more than one host.** No failover, no rolling updates, no autoscaling.

## Best practices

- Keep one `compose.yaml` per stack in version control, with development tweaks in `compose.override.yaml`.
- Pin every image version, and give each stateful service a named volume.
- Define a health check for every service another service depends on, and use `service_healthy`.
- Still implement retries with backoff and a readiness endpoint in the application.
- Publish only the ports a human or external client needs, bound to `127.0.0.1` on development machines.
- Deliver secrets through `secrets:`; use `${VAR:?message}` for required configuration.
- Run one-off jobs (ingestion, migrations, evaluation runs) as profiled services with `docker compose run --rm`.
- Use Compose in CI for integration tests: `up -d --wait`, run the tests, then `down -v` in a throwaway project.

## Performance considerations

All services in a Compose project share **one machine**, so plan memory for the sum of them. Reason from the data:

- Qdrant with 200,000 chunks of 384-dimensional float32 vectors holds $200{,}000 \times 384 \times 4 \approx 307$ MB of raw vectors, plus the HNSW graph and payloads. With the default `m = 16`, the base graph layer stores up to 32 neighbour IDs of 4 bytes each per vector, roughly another 26 MB. Payloads (chunk text and metadata) are often larger than the vectors.
- The API container from the previous lesson needs roughly the model plus runtime per worker; with the 2 GB limit above, its worst case is bounded.
- Redis as a cache grows until it hits `maxmemory` if you set one; set one, or it competes with Qdrant.

If these sums approach the host's memory, the kernel's out-of-memory killer will choose a victim, and it may be Qdrant. Per-service limits under `deploy.resources` make the failure land on the service that misbehaved.

On macOS and Windows, bind mounts cross the boundary into Docker Desktop's virtual machine and can be markedly slower than on Linux for workloads that read many small files; keep large indexes on named volumes rather than bind mounts. Start-up time is dominated by the slowest health-gated chain: here, Qdrant becoming healthy plus the API's model load, which is why `start_period` must exceed the model's load time on the slowest machine that runs the stack.

## Security considerations

- **Ports and firewalls.** Docker writes its own packet-filtering rules for published ports, and on Linux these can take effect before host firewall tools such as `ufw`, so a published port may be reachable even when the firewall appears to block it. Do not publish what you do not need; bind to `127.0.0.1` when only local access is intended.
- **Secrets.** Use `secrets:` files with restrictive permissions, keep them out of version control, and use real secret stores in shared environments. Never paste keys into `compose.yaml`.
- **Vector database access.** Qdrant without an API key accepts any request that reaches it, including deletions. Inside a private Compose network that is acceptable for development; anywhere else, enable its API key (passed as a secret) and TLS.
- **Read-only inputs.** Mount source documents with `:ro` so a compromised ingestion parser cannot alter the documents it reads.

## Summary

- Compose describes a multi-container stack (services, networks, volumes, secrets) in one version-controlled file and runs it with one command.
- Services find each other by service name on the project network; `localhost` inside a container is that container.
- Named volumes keep data such as the vector index alive across container replacement; `down -v` deletes them.
- `depends_on` with `service_healthy` orders start-up but does not protect against later restarts, so applications must retry.
- Interpolation, `environment:`, `env_file:` and `secrets:` are different mechanisms; secrets belong in `secrets:`.
- Compose is ideal for development, CI and single-host services; multi-host resilience, rolling updates and autoscaling need an orchestrator.

## Exercises

1. A teammate's API logs `httpx.ConnectError: [Errno 111] Connection refused` when connecting to `http://localhost:6333`, but `curl localhost:6333` on their laptop works. Explain both observations.

<details><summary>Show solution</summary>

On the laptop, Qdrant's port is published to the host, so `localhost:6333` on the host reaches it. Inside the API container, `localhost` is the container's own network namespace, where nothing listens on 6333. Use `http://qdrant:6333`, resolved by Docker's DNS on the project network, and then stop publishing Qdrant's port, since nothing outside needs it.

</details>

2. Write a `compose.override.yaml` for development that mounts the local `app/` directory into the API container, runs Uvicorn with auto-reload, and sets debug logging, without changing `compose.yaml`.

<details><summary>Show solution</summary>

```yaml
services:
  api:
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    environment:
      LOG_LEVEL: DEBUG
    volumes:
      - ./app:/srv/app
```

Compose merges this file automatically. The bind mount replaces the code baked into the image with the working copy, and `--reload` restarts the server on changes. Because the override is development-only, CI should run with `-f compose.yaml` explicitly, so it tests the image exactly as built.

</details>

3. Extend the linter with a rule that reports any service that uses `service_completed_successfully` on a dependency that has `restart: always` or `restart: unless-stopped`. Why is that combination a problem?

<details><summary>Show solution</summary>

Add inside the dependency loop:

```python
            elif cond == "service_completed_successfully" and services[target].get("restart") in ("always", "unless-stopped"):
                problems.append(f"{name}: waits for '{target}' to complete, but '{target}' restarts after exiting")
```

A job such as collection creation is meant to run once and exit 0. With a restart policy of `always` or `unless-stopped`, Docker restarts it after it exits, so it keeps running repeatedly instead of finishing; the dependent service's wait and the job's behaviour no longer match the intent. One-off jobs should use `restart: "no"` (the default) or `on-failure`.

</details>

4. Meridian's Finance team wants to run its own small RAG assistant on one virtual machine with Compose. List the questions you would ask to decide whether Compose is acceptable for production, and the answer to each that would make you recommend an orchestrator instead.

<details><summary>Show solution</summary>

How much downtime is acceptable during a host failure or maintenance? (Minutes or more: Compose is fine; near-zero: orchestrator with multiple nodes.) Does load vary enough to need autoscaling? (Yes: orchestrator or managed container service.) Must deployments avoid dropping requests? (Yes: rolling updates, which Compose does not do.) Does the data fit on one machine with headroom? (No: distributed vector store.) Who is on call, and can they restore the volume from backup? (No one: a managed service is safer.) For an internal tool used in office hours by one team, Compose plus automated volume backups is often a sound, cheap answer.

</details>

## Check your understanding

```quiz
[
  {
    "q": "Inside the `api` container of a Compose project, which URL reaches the `qdrant` service?",
    "options": [
      "http://localhost:6333",
      "http://qdrant:6333",
      "http://0.0.0.0:6333",
      "http://host:6333"
    ],
    "answer": 1,
    "why": "Docker's embedded DNS resolves each service name to its container on the project network. localhost inside a container is the container itself."
  },
  {
    "q": "What does `docker compose down -v` do that `docker compose down` does not?",
    "options": [
      "It stops containers more gracefully.",
      "It prints verbose output.",
      "It removes named volumes, deleting data such as the Qdrant index.",
      "It also removes the built images."
    ],
    "answer": 2,
    "why": "Plain down removes containers and networks but keeps named volumes. The -v flag deletes the volumes and everything stored in them."
  },
  {
    "q": "The API depends on Qdrant with `condition: service_healthy`. Qdrant restarts at night. What happens to the API?",
    "options": [
      "Nothing is coordinated: Compose only orders the initial start-up, so the API must retry until Qdrant is back.",
      "Compose stops the API, waits for Qdrant to be healthy, then restarts the API.",
      "The API is marked unhealthy and automatically restarted by Docker.",
      "Compose refuses to restart Qdrant while the API is running."
    ],
    "answer": 0,
    "why": "depends_on conditions apply when services are started. Afterwards, resilience to a dependency restarting is the application's job: retries with backoff and a readiness endpoint."
  },
  {
    "q": "Where should the Anthropic API key for the API service go?",
    "options": [
      "In the environment section of compose.yaml, committed to the repository",
      "In the Dockerfile as an ENV instruction",
      "In a build argument",
      "In a file referenced under top-level secrets, mounted at /run/secrets in the container"
    ],
    "answer": 3,
    "why": "Compose secrets mount a file into the container without putting the value in the image, the Compose file or the container's environment."
  },
  {
    "q": "Which requirement most clearly means a RAG service has outgrown Compose?",
    "options": [
      "It has more than three services.",
      "It must keep serving when one machine fails, and deploy new versions without dropping requests.",
      "It needs environment variables.",
      "It uses a vector database."
    ],
    "answer": 1,
    "why": "Compose runs on one host and has no multi-host failover or rolling updates. Those needs call for an orchestrator or a managed container platform."
  }
]
```

## Further reading

- [Docker Compose documentation](https://docs.docker.com/compose/): the official guide, including start-up order, profiles and secrets.
- [Compose Specification](https://github.com/compose-spec/compose-spec): the formal definition of the file format.
- [Compose file reference](https://docs.docker.com/reference/compose-file/): every key, including `healthcheck`, `depends_on` and `deploy`.
- [Qdrant installation guide](https://qdrant.tech/documentation/guides/installation/): running Qdrant in Docker, storage paths and configuration.
- [pydantic-settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/): environment variables, `.env` files and secrets directories.
