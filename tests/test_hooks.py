import inspect_ai.event
import inspect_ai.log
import pytest

import inspect_refusal_sample_killer._config as config
from tests.conftest import normal_output, refusal_output, run_eval_with_outputs


def _sample_limit(log: inspect_ai.log.EvalLog) -> inspect_ai.log.EvalSampleLimit | None:
    assert log.samples is not None
    return log.samples[0].limit


def _limit_event_messages(log: inspect_ai.log.EvalLog) -> list[str]:
    assert log.samples is not None
    return [
        event.message
        for event in log.samples[0].events
        if isinstance(event, inspect_ai.event.SampleLimitEvent)
    ]


def test_default_limit_kills_on_first_refusal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)  # default 0
    log = run_eval_with_outputs([refusal_output(explanation="policy X")])

    limit = _sample_limit(log)
    assert limit is not None
    assert limit.type == "custom"

    messages = _limit_event_messages(log)
    assert any("refused 1 requests" in m for m in messages)
    assert any("policy X" in m for m in messages)
    assert any("cyber" in m for m in messages)


def test_limit_one_not_breached_by_single_refusal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "1")
    # one refusal then a normal completion; solver generates twice
    log = run_eval_with_outputs([refusal_output(), normal_output()], generate_calls=2)
    assert log.status == "success"
    assert _sample_limit(log) is None


def test_limit_one_breached_by_two_refusals(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "1")
    log = run_eval_with_outputs([refusal_output(), refusal_output()], generate_calls=2)
    limit = _sample_limit(log)
    assert limit is not None
    assert limit.type == "custom"
    assert any("refused 2 requests" in m for m in _limit_event_messages(log))


def test_disabled_when_negative(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "-1")
    log = run_eval_with_outputs([refusal_output(), normal_output()], generate_calls=2)
    assert log.status == "success"
    assert _sample_limit(log) is None


def test_normal_output_never_trips(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    log = run_eval_with_outputs([normal_output()])
    assert log.status == "success"
    assert _sample_limit(log) is None


def test_final_generate_refusal_recorded_not_terminated(
    monkeypatch: pytest.MonkeyPatch,
):
    # Documented limitation: a refusal on a sample's FINAL model call cannot
    # convert to a sample limit (no subsequent in-sample work for the async
    # cancel to interrupt, no further generate for the inline backstop). The
    # breach is still recorded in the transcript. `park_between_generates=False`
    # models the single-turn case (no trailing await after the refusal).
    monkeypatch.delenv(config.ENV_VAR, raising=False)  # default 0
    log = run_eval_with_outputs([refusal_output()], park_between_generates=False)

    assert log.status == "success"
    assert _sample_limit(log) is None  # not terminated
    # ...but the breach is recorded so it is visible in the log/viewer.
    assert any("refused 1 requests" in m for m in _limit_event_messages(log))
