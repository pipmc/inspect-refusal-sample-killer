import os

import inspect_ai
import inspect_ai.event
import pytest

import inspect_refusal_sample_killer._config as config
from inspect_refusal_sample_killer._tasks import classifier_refusal_probe


@pytest.mark.integration
def test_live_fable_refusal_terminates_sample(monkeypatch: pytest.MonkeyPatch):
    """Live end-to-end: a real classifier refusal terminates the sample.

    Requires METR middleman credentials (skipped without them). Uses the
    agentic `classifier_refusal_probe` task so the refusal is followed by
    further model calls, letting the hook convert the breach into a `custom`
    sample limit.

    Run with:
        ANTHROPIC_BASE_URL=https://middleman.prd.metr.org/anthropic \
        ANTHROPIC_API_KEY=$(hawk auth access-token) \
        uv run pytest -m integration
    """
    if not os.environ.get("ANTHROPIC_BASE_URL") or not os.environ.get(
        "ANTHROPIC_API_KEY"
    ):
        pytest.skip("ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY not set")

    monkeypatch.delenv(config.ENV_VAR, raising=False)  # default 0

    # inspect_ai.eval's own signature leaks Unknown (its `scanner` param), which
    # strict mode flags as a partially-unknown member. The type genuinely lives
    # in inspect_ai and cannot be narrowed here, so ignore this one rule only.
    logs = inspect_ai.eval(  # pyright: ignore[reportUnknownMemberType]
        classifier_refusal_probe(),
        model="anthropic/claude-fable-5-data-retention",
        display="none",
    )
    log = logs[0]

    assert log.samples is not None
    limit = log.samples[0].limit
    assert limit is not None, "expected the sample to be terminated by the hook"
    assert limit.type == "custom"

    messages = [
        event.message
        for event in log.samples[0].events
        if isinstance(event, inspect_ai.event.SampleLimitEvent)
    ]
    assert any("classifier policy trigger" in m for m in messages)
