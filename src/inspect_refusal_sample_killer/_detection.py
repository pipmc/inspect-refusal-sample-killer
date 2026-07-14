import inspect_ai.event

_REFUSAL_STOP_REASON = "content_filter"
_REFUSAL_TYPE = "refusal"


def is_classifier_refusal(event: inspect_ai.event.ModelEvent) -> bool:
    """True iff a completed model event reports a safety-classifier refusal."""
    if event.pending:
        return False
    output = event.output
    if not output.choices:
        return False
    if output.stop_reason != _REFUSAL_STOP_REASON:
        return False
    stop_details = output.choices[0].stop_details
    return stop_details is not None and stop_details.type == _REFUSAL_TYPE


def refusal_details(
    event: inspect_ai.event.ModelEvent,
) -> tuple[str | None, str | None]:
    """Return (category, explanation) from the event's refusal stop details."""
    if not event.output.choices:
        return None, None
    stop_details = event.output.choices[0].stop_details
    if stop_details is None:
        return None, None
    return stop_details.category, stop_details.explanation


def refusal_limit_message(
    model_name: str,
    refusals: int,
    max_refusals: int,
    category: str | None,
    explanation: str | None,
) -> str:
    """Build the LimitExceededError / SampleLimitEvent message."""
    base = (
        f"The model {model_name} refused {refusals} requests due to a "
        f"classifier policy trigger, which exceeds the limit of "
        f"{max_refusals} refusals"
    )
    detail = explanation if explanation else "no explanation provided"
    if category:
        return f"{base}. Latest refusal ({category}): {detail}"
    return f"{base}. Latest refusal: {detail}"
