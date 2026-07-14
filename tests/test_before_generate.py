# Seeds and inspects the hook's internal per-sample state to exercise the inline
# on_before_model_generate backstop in isolation (the async on_sample_event path
# preempts it under a real eval, so it cannot be driven end-to-end). Poking the
# private state is deliberate here; this is a file-scoped pyright directive, not
# a `# type: ignore`.
# pyright: reportPrivateUsage=false
import asyncio

import inspect_ai.hooks
import inspect_ai.log._samples
import inspect_ai.model
import inspect_ai.util
import pytest

import inspect_refusal_sample_killer._config as config
from inspect_refusal_sample_killer._hooks import ClassifierRefusalKiller


class _StubActive:
    def __init__(self, sample_uuid: str) -> None:
        self.sample_uuid = sample_uuid


def _before_data() -> inspect_ai.hooks.BeforeModelGenerate:
    return inspect_ai.hooks.BeforeModelGenerate(
        model_name="mockllm/model",
        input=[],
        tools=[],
        tool_choice="none",
        config=inspect_ai.model.GenerateConfig(),
        cache=None,
    )


def test_raises_custom_limit_when_count_over_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "0")
    hook = ClassifierRefusalKiller()
    hook._refusals["s1"] = 1
    hook._last_refusal["s1"] = ("cyber", "declined by classifier")
    monkeypatch.setattr(
        inspect_ai.log._samples, "sample_active", lambda: _StubActive("s1")
    )

    with pytest.raises(inspect_ai.util.LimitExceededError) as excinfo:
        asyncio.run(hook.on_before_model_generate(_before_data()))

    assert excinfo.value.type == "custom"
    assert excinfo.value.value == 1
    assert excinfo.value.limit == 0
    assert "declined by classifier" in (excinfo.value.message or "")


def test_noop_when_count_not_over_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "2")
    hook = ClassifierRefusalKiller()
    hook._refusals["s1"] = 2  # 2 <= 2, not breached
    monkeypatch.setattr(
        inspect_ai.log._samples, "sample_active", lambda: _StubActive("s1")
    )

    asyncio.run(hook.on_before_model_generate(_before_data()))  # must not raise
    assert hook._recorded == set()  # no limit event recorded when under threshold


def test_noop_when_no_active_sample(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "0")
    hook = ClassifierRefusalKiller()
    hook._refusals["s1"] = 5
    monkeypatch.setattr(inspect_ai.log._samples, "sample_active", lambda: None)

    asyncio.run(hook.on_before_model_generate(_before_data()))  # must not raise
    assert hook._recorded == set()  # no limit event recorded without an active sample
