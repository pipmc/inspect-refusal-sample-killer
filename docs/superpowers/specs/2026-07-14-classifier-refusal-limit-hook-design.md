# Design: inspect-fallback-run-killer — classifier-refusal limit hook

**Date:** 2026-07-14
**Status:** Approved

## Purpose

An Inspect AI hooks extension package that counts safety-classifier refusals per
sample and terminates the sample with a `LimitExceededError` when the count
exceeds a configurable limit. Claude 5 models can decline requests via safety
classifiers; Inspect surfaces these as `stop_reason == "content_filter"` with
structured `stop_details`. With `fallback_models` configured, `content_filter`
only appears when no fallback served the request — so this hook fires exactly
when a refusal actually reached the transcript.

## Configuration

- `INSPECT_MAX_CLASSIFIER_REFUSALS` — integer, default `0`.
  - `0` (default): the first refusal in a sample terminates it.
  - `N > 0`: the sample is terminated when its refusal count exceeds `N`.
  - Negative (e.g. `-1`): the hook is disabled entirely (via `enabled()`).
  - Unparseable values: treat as the default (`0`) and log a warning.

## Package structure

Distribution name `inspect-refusal-sample-killer`, import package
`inspect_refusal_sample_killer`:

```
src/inspect_refusal_sample_killer/
  __init__.py        # public exports
  _hooks.py          # Hooks subclass, detection + counting + kill logic
  _tasks.py          # registered `classifier_refusal_probe` task (+ REFUSAL_PROMPT)
  _registry.py       # imports _hooks and _tasks to register decorated objects
tests/
  test_config.py         # unit: env-var parsing
  test_detection.py      # unit: detection + message building
  test_before_generate.py  # unit: inline on_before_model_generate backstop
  test_hooks.py          # mockllm end-to-end tests
  test_integration.py    # live Fable test (marked integration)
.devcontainer/
  devcontainer.json
  Dockerfile
pyproject.toml
README.md
```

Registered via the `inspect_ai` entry-point group so both the hook and the
probe task activate on install:

```toml
[project.entry-points.inspect_ai]
inspect_refusal_sample_killer = "inspect_refusal_sample_killer._registry"
```

`_registry.py` imports both `_hooks` and `_tasks`, so the entry point exposes
the hook *and* a registered task the user can run standalone to exercise the
hook against a real model:

```bash
inspect eval inspect_refusal_sample_killer/classifier_refusal_probe \
    --model anthropic/claude-fable-5-data-retention
```

## Hook behavior

`@hooks(name="classifier_refusal_killer", description=...)` class overriding:

- `enabled()` — `False` when `INSPECT_MAX_CLASSIFIER_REFUSALS` parses to a
  negative integer; `True` otherwise.
- `on_sample_event(data: SampleEvent)` — detection, counting, async kill.
- `on_before_model_generate(data: BeforeModelGenerate)` — inline deterministic
  backstop (see Kill mechanism).
- `on_sample_end` / `on_sample_attempt_end` — clean up counting state.

### Detection

An event counts as a classifier refusal when all of:

- `data.event` is a `ModelEvent`
- the event is completed (not pending) and has a non-`None` `output`
- `event.output.stop_reason == "content_filter"`
- `event.output.stop_details is not None`
- `event.output.stop_details.type == "refusal"`

### Counting state

- Module/class-level `dict[str, int]` of refusal counts keyed by sample uuid
  (`SampleEvent.sample_id`), plus a `set[str]` of seen `ModelEvent.uuid`s so a
  re-emitted event (pending → completed) cannot double-count.
- State for a sample is removed in `on_sample_end` and `on_sample_attempt_end`
  (retry path), preventing unbounded growth. Because keys are per-attempt
  uuids, counts are naturally per sample-epoch-attempt.

### Kill mechanism

A hook cannot both observe a refusal and synchronously raise from the same
model call: the only hook carrying the model *output* is `on_sample_event`,
which is dispatched from Inspect's background emitter (it catches every
exception, including `LimitExceededError`). So termination uses two
complementary mechanisms, and whichever fires first wins:

**1. `on_sample_event` (async).** On each completed refusal `ModelEvent`, count
it; when `refusals > max_refusals`:
   - Build `LimitExceededError(type="custom", value=refusals, limit=max_refusals,
     message=…)` where the message is the verbatim base string plus the
     `. Latest refusal ({category}): {explanation}` suffix (degrading to
     `no explanation provided` / dropping `({category})` when absent).
   - Record `SampleLimitEvent(type="custom", limit=max_refusals, message=…)` to
     the transcript (once per sample) so the full message is visible in the
     log/viewer — `EvalSampleLimit` stores only type + limit.
   - Call `sample_active().limit_exceeded(err)`, which cancels the sample's task
     group. This lands while the sample is still doing work (e.g. awaiting the
     next model call), so it converts to a `custom` `EvalSampleLimit` in
     multi-step / agentic evals.

**2. `on_before_model_generate` (inline).** Before each model call, if the
sample's refusal count (keyed by `sample_active().sample_uuid`) already exceeds
the limit, raise `LimitExceededError` directly. `emit_before_model_generate`
runs inline within `generate()` and re-raises `LimitExceededError`, so it
propagates to the runner, which records a `custom` sample limit — deterministic,
and without making another (also-refused) call. It records the same
`SampleLimitEvent` first (idempotent via a per-sample "recorded" set).

Either way the sample ends as a `custom` `EvalSampleLimit`, proceeds to scoring,
and the run continues. Note that an Inspect limit is *not* an error: the eval
`status` stays `success` and the outcome is observed as
`sample.limit.type == "custom"`.

### Private-API use

`sample_active()`, `ActiveSample.limit_exceeded()`, and transcript event
recording are private Inspect APIs. This is deliberate: no public API lets a
sample-event hook terminate a sample. The mockllm end-to-end tests pin this
behavior against the installed inspect-ai version so breakage surfaces as test
failures, not silent no-ops.

### Edge cases

- **Refusal on the sample's final model call (known limitation, validated
  live):** neither mechanism can convert it to a sample limit — there is no
  subsequent in-sample work for the async cancel to interrupt, and no further
  `generate()` for the inline backstop. The breach is still recorded as a
  `SampleLimitEvent` in the transcript, but `sample.limit` stays `None` and
  `status` is `success`. This is why the probe task and live test use an
  agentic (`basic_agent`) solver rather than a single generate: the repeated
  model calls give the hook the in-sample work it needs. Single-turn tasks
  (one generate then done) are therefore not terminated — only recorded.
- No active sample or task group not running: log a warning, never crash the
  emitter.
- Multiple concurrent samples: state is keyed by sample uuid, so counts never
  bleed across samples.

## Testing

All default tests run real `eval()`s end-to-end via `mockllm/model` with
`custom_outputs` and assert on the resulting `EvalLog`. Refusal outputs are
constructed as
`ModelOutput(..., stop_reason="content_filter", stop_details=StopDetails(type="refusal", category="cyber", explanation=...))`.

Default suite (offline, must pass in devcontainer and CI):

1. **Default limit (0):** one refusal output → sample ends with
   `limit.type == "custom"`; the transcript's `SampleLimitEvent` message
   contains the model name, count, limit, category, and explanation.
2. **Limit 1, not breached:** solver generating twice; one refusal + one
   normal output → sample completes normally.
3. **Limit 1, breached:** two refusals → killed with `value == 2`,
   `limit == 1`.
4. **Disabled (`-1`):** refusal outputs flow through; sample completes.
5. **Control:** normal outputs never increment or kill.

Env var manipulation via pytest `monkeypatch`. These mockllm tests exercise the
async `on_sample_event` path; because the mockllm solver is near-instant, the
test solver `await asyncio.sleep(...)`s after each generate to give the
background emitter the scheduling point a real model provides via network I/O.
The inline `on_before_model_generate` backstop cannot be driven end-to-end via
mockllm (the async path preempts it), so it has focused unit tests in
`test_before_generate.py` that seed the hook's per-sample count, stub
`sample_active`, and assert it raises a `custom` `LimitExceededError` when over
the limit (and is a no-op when under, or when no sample is active).

Live integration test:

- Marked `integration`; excluded by default via
  `addopts = "-m 'not integration'"` in pyproject.
- Skipped unless `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` are set.
- Runs the agentic `classifier_refusal_probe` task against
  `anthropic/claude-fable-5-data-retention` (the data-retention-enabled model id
  available through METR middleman; plain `claude-fable-5` is gated) and asserts
  `limit.type == "custom"` with the refusal message in the transcript. The
  `basic_agent` solver's repeated calls give the hook the in-sample work needed
  to terminate (a single generate would only be recorded, per the known
  limitation). Validated live during implementation.
- Requires the `anthropic` package (inspect's Anthropic provider prerequisite),
  added as a dev dependency.
- Invocation:

  ```bash
  ANTHROPIC_BASE_URL=https://middleman.prd.metr.org/anthropic \
  ANTHROPIC_API_KEY=$(hawk auth access-token) \
  uv run pytest -m integration
  ```

- Expected to run on the host (needs `hawk`), not in the devcontainer.

## Tooling

- **uv** for project/dependency management; **hatchling** build backend.
- `requires-python = ">=3.11"`; runtime dependency: `inspect-ai`.
- **ruff** for formatting and linting; **basedpyright** (strict) for type
  checking; no `typing.Any`, no `# type: ignore`.
- Plain pytest functions (no test classes); fully-qualified imports per user
  conventions.

## Devcontainer

Adapted from `~/Code/task-assets/.devcontainer`:

- Same multi-stage uv Dockerfile pattern (uv binary from
  `ghcr.io/astral-sh/uv`, `python:3.11-bookworm` base, builder stage running
  `uv sync --locked`, `UV_PROJECT_ENVIRONMENT=/opt/python`), same non-root
  user/volume layout.
- Omit the AWS CLI stage and the `~/.aws` bind mount (not needed).
- Container/volume/hostname renamed to `inspect-refusal-run-killer`.
- VS Code customizations kept: ruff formatter, pytest test discovery, strict
  type checking.
