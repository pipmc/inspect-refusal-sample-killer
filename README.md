# inspect-refusal-sample-killer

An [Inspect AI](https://inspect.aisi.org.uk/) hook that counts safety-classifier
refusals per sample and terminates the sample once the count exceeds a
configurable limit.

When a Claude model declines a request via a safety classifier, Inspect surfaces
it as a model output with `stop_reason == "content_filter"` and
`stop_details.type == "refusal"`. This hook counts those refusals per sample and,
once the count exceeds the limit, ends the sample with a `custom` sample limit
whose message includes the refusal category and explanation.

> With `fallback_models` configured, Inspect only reports `content_filter` when
> no fallback served the request — so this hook fires exactly when a refusal
> actually reached the transcript.

## Install

```bash
uv add inspect-refusal-sample-killer
# or
pip install inspect-refusal-sample-killer
```

Installation registers the hook (and a probe task, see below) via the
`inspect_ai` entry point; no code changes are needed in your eval.

## Configuration

`INSPECT_MAX_CLASSIFIER_REFUSALS` (integer, default `0`):

| Value | Behavior |
|-------|----------|
| `0` (default) | The first refusal in a sample terminates it. |
| `N > 0` | The sample is terminated when its refusal count exceeds `N`. |
| negative (e.g. `-1`) | The hook is disabled. |

An unparseable value falls back to `0` and logs a warning.

## What "terminates" means

The sample ends with an `EvalSampleLimit` of type `custom` — the same graceful
mechanism as Inspect's built-in token/message/time limits. It is **not** an
error: the eval `status` stays `success`, the sample proceeds to scoring, and
the run continues. You observe it as:

```python
log.samples[0].limit.type == "custom"
```

The full message (with the refusal category and explanation) is recorded as a
`SampleLimitEvent` in the sample transcript, so it is visible in the log and the
Inspect viewer.

## How it works (and an important limitation)

A hook cannot both observe a model refusal and synchronously abort the same
call, so termination uses two complementary mechanisms:

- **Async** (`on_sample_event`): counts each refusal and, on breach, cancels the
  sample's task group. This lands while the sample is still doing work (e.g.
  awaiting its next model call) — the normal case for multi-step / agentic evals.
- **Inline** (`on_before_model_generate`): before each model call, if the
  sample's refusal count is already over the limit, raises immediately — so no
  further (also-refused) call is made.

**Limitation:** a refusal on a sample's *final* model call cannot be converted to
a sample limit — there is no subsequent in-sample work for the async cancel to
interrupt, and no further model call for the inline backstop. The breach is
still recorded as a `SampleLimitEvent` in the transcript, but `sample.limit`
stays `None` and `status` is `success`. In practice this only affects
single-turn evals (one generate, then done); agentic solvers that make repeated
model calls terminate as expected.

## Probe task

The package also registers a task, `classifier_refusal_probe`, that reliably
triggers a classifier refusal (using `basic_agent`, so its repeated model calls
let the hook terminate the sample). Run it standalone to exercise the hook
against a real model:

```bash
inspect eval inspect_refusal_sample_killer/classifier_refusal_probe \
    --model anthropic/claude-fable-5-data-retention
```

## Development

```bash
uv sync --all-groups
uv run pytest              # offline suite (mockllm + unit tests)
uv run ruff check src tests
```

Type-checking runs in the devcontainer (basedpyright needs the synced
environment):

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . uv run basedpyright
```

### Live integration test

Marked `integration` and deselected by default; it needs METR middleman
credentials and runs on the host (for `hawk`):

```bash
ANTHROPIC_BASE_URL=https://middleman.prd.metr.org/anthropic \
ANTHROPIC_API_KEY=$(hawk auth access-token) \
uv run pytest -m integration
```

It runs the probe task against `anthropic/claude-fable-5-data-retention` (plain
`claude-fable-5` is gated behind data retention on middleman) and asserts the
sample is terminated with a `custom` limit.

## Notes

- This package uses a small number of private Inspect APIs (`sample_active`,
  `ActiveSample.limit_exceeded`, transcript event recording) that no public API
  replaces. Behavior is pinned by the test suite against `inspect-ai>=0.3.246`.
- `anthropic` is a dev dependency (Inspect's Anthropic provider prerequisite),
  needed only to run the live integration test.
