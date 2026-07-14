import inspect_ai.event
import inspect_ai.model

import inspect_refusal_sample_killer._detection as detection


def _model_event(
    *,
    stop_reason: inspect_ai.model.StopReason = "stop",
    stop_details: inspect_ai.model.StopDetails | None = None,
    pending: bool | None = None,
    empty_choices: bool = False,
) -> inspect_ai.event.ModelEvent:
    if empty_choices:
        output = inspect_ai.model.ModelOutput(model="mockllm/model", choices=[])
    else:
        output = inspect_ai.model.ModelOutput.from_content(
            model="mockllm/model",
            content="hello",
            stop_reason=stop_reason,
            stop_details=stop_details,
        )
    return inspect_ai.event.ModelEvent(
        model="mockllm/model",
        input=[],
        tools=[],
        tool_choice="none",
        config=inspect_ai.model.GenerateConfig(),
        output=output,
        pending=pending,
    )


def test_detects_classifier_refusal():
    event = _model_event(
        stop_reason="content_filter",
        stop_details=inspect_ai.model.StopDetails(
            type="refusal", category="cyber", explanation="policy trigger"
        ),
    )
    assert detection.is_classifier_refusal(event) is True


def test_ignores_normal_stop():
    event = _model_event(stop_reason="stop")
    assert detection.is_classifier_refusal(event) is False


def test_ignores_content_filter_without_refusal_details():
    event = _model_event(stop_reason="content_filter", stop_details=None)
    assert detection.is_classifier_refusal(event) is False


def test_ignores_content_filter_with_non_refusal_details():
    event = _model_event(
        stop_reason="content_filter",
        stop_details=inspect_ai.model.StopDetails(type="length"),
    )
    assert detection.is_classifier_refusal(event) is False


def test_ignores_pending_event():
    event = _model_event(
        stop_reason="content_filter",
        stop_details=inspect_ai.model.StopDetails(type="refusal"),
        pending=True,
    )
    assert detection.is_classifier_refusal(event) is False


def test_ignores_empty_choices():
    event = _model_event(empty_choices=True)
    assert detection.is_classifier_refusal(event) is False


def test_refusal_details_extracts_category_and_explanation():
    event = _model_event(
        stop_reason="content_filter",
        stop_details=inspect_ai.model.StopDetails(
            type="refusal", category="bio", explanation="nope"
        ),
    )
    assert detection.refusal_details(event) == ("bio", "nope")


def test_message_includes_all_fields():
    message = detection.refusal_limit_message(
        model_name="anthropic/claude-fable-5",
        refusals=2,
        max_refusals=1,
        category="cyber",
        explanation="declined by policy",
    )
    assert "anthropic/claude-fable-5" in message
    assert "refused 2 requests" in message
    assert "limit of 1 refusals" in message
    assert "cyber" in message
    assert "declined by policy" in message


def test_message_degrades_without_explanation():
    message = detection.refusal_limit_message(
        model_name="m", refusals=1, max_refusals=0, category=None, explanation=None
    )
    assert "no explanation provided" in message
    assert "(None)" not in message  # category parenthetical omitted, not rendered
