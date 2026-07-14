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

Distribution name `inspect-fallback-run-killer`, import package
`inspect_fallback_run_killer`:

```
src/inspect_fallback_run_killer/
  __init__.py        # public exports
  _hooks.py          # Hooks subclass, detection + counting + kill logic
  _registry.py       # imports _hooks to register decorated hooks
tests/
  test_hooks.py      # mockllm end-to-end tests
  test_integration.py  # live Fable test (marked integration)
.devcontainer/
  devcontainer.json
  Dockerfile
pyproject.toml
README.md
```

Registered via the `inspect_ai` entry-point group so it activates on install:

```toml
[project.entry-points.inspect_ai]
inspect_fallback_run_killer = "inspect_fallback_run_killer._registry"
```

## Hook behavior

`@hooks(name="classifier_refusal_killer", description=...)` class overriding:

- `enabled()` — `False` when `INSPECT_MAX_CLASSIFIER_REFUSALS` parses to a
  negative integer; `True` otherwise.
- `on_sample_event(data: SampleEvent)` — detection, counting, kill.
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

`on_sample_event` is dispatched from Inspect's background sample-event emitter,
which catches all exceptions including `LimitExceededError` — so the hook
cannot simply raise. Instead, when `refusals > max_refusals`:

1. Build the error:

   ```python
   LimitExceededError(
       type="custom",
       value=refusals,
       limit=max_refusals,
       message=(
           f"The model {model_name} refused {refusals} requests due to a "
           f"classifier policy trigger, which exceeds the limit of "
           f"{max_refusals} refusals. Latest refusal ({category}): {explanation}"
       ),
   )
   ```

   `model_name` comes from the `ModelEvent`. `category` and `explanation` come
   from `stop_details` of the refusal that breached the limit; when absent the
   suffix degrades gracefully (e.g. `Latest refusal: no explanation provided`).

2. Record `SampleLimitEvent(type="custom", limit=max_refusals, message=...)` to
   the transcript (mirroring what built-in token/cost limits do), so the full
   message including the explanation is visible in the log and viewer —
   `EvalSampleLimit` itself only stores type and limit.

3. Call `sample_active().limit_exceeded(err)` — the same internal path
   Inspect's working-limit monitor uses: it stores the error on the
   `ActiveSample`, fires interrupt cleanup, and cancels the sample's task
   group. The runner's cancellation handler records the result as a sample
   limit (`EvalSampleLimit`, `type="custom"`); the sample proceeds to scoring
   like any other limit-hit sample and the run continues.

### Private-API use

`sample_active()`, `ActiveSample.limit_exceeded()`, and transcript event
recording are private Inspect APIs. This is deliberate: no public API lets a
sample-event hook terminate a sample. The mockllm end-to-end tests pin this
behavior against the installed inspect-ai version so breakage surfaces as test
failures, not silent no-ops.

### Edge cases

- Refusal on the solver's final generate: the cancellation may land after the
  solver already finished; the sample then completes normally and the kill is
  a logged no-op. Documented limitation.
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

Env var manipulation via pytest `monkeypatch`.

Live integration test:

- Marked `integration`; excluded by default via
  `addopts = "-m 'not integration'"` in pyproject.
- Skipped unless `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` are set.
- Runs `anthropic/claude-fable-5` with a prompt that reliably triggers a
  classifier decline (exact prompt validated empirically during
  implementation); asserts the same `limit.type == "custom"` outcome and
  message content.
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
- Container/volume/hostname renamed to `inspect-fallback-run-killer`.
- VS Code customizations kept: ruff formatter, pytest test discovery, strict
  type checking.

## Naming note

The repo is named `inspect-fallback-run-killer` but the agreed behavior
terminates the *sample* (as a limit), not the run. Keeping the name; the README
documents the actual behavior.
