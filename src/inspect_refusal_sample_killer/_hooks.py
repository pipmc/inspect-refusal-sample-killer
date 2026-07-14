# Deliberate private-API use (`transcript()._event`, `inspect_ai.log._samples`)
# and inspect_ai's untyped `@hooks(...)` decorator (typed `Callable[..., Type[T]]`)
# are both intentional and unavoidable here; suppress the strict-mode reports
# they trigger. These are file-scoped pyright directives, not `# type: ignore`.
# pyright: reportPrivateUsage=false, reportUntypedClassDecorator=false
import logging

import inspect_ai.event
import inspect_ai.hooks
import inspect_ai.log
import inspect_ai.log._samples
import inspect_ai.util

import inspect_refusal_sample_killer._config as config
import inspect_refusal_sample_killer._detection as detection

logger = logging.getLogger(__name__)


@inspect_ai.hooks.hooks(
    name="classifier_refusal_killer",
    description="Terminate a sample when classifier refusals exceed a limit.",
)
class ClassifierRefusalKiller(inspect_ai.hooks.Hooks):
    """Counts classifier refusals per sample and terminates the sample on breach.

    Termination uses two complementary mechanisms, because a hook cannot
    synchronously observe a model refusal and raise from the same call:

    - `on_sample_event` (async): observes each completed `ModelEvent`, counts
      refusals per sample, and on breach cancels the sample's task group via
      `ActiveSample.limit_exceeded`. This lands while the sample is still doing
      work (e.g. awaiting a subsequent model call), which is the common case in
      multi-step / agentic evals.
    - `on_before_model_generate` (inline): before each model call, if the
      sample's refusal count is already over the limit, raises
      `LimitExceededError` directly. This propagates out of `generate()` and the
      runner records a `custom` sample limit deterministically, without making
      another (also-refused) call.

    Known limitation: a refusal on a sample's FINAL model call cannot convert to
    a sample limit -- there is no subsequent in-sample work for the async cancel
    to interrupt and no further `generate()` for the inline backstop. The breach
    is still recorded as a `SampleLimitEvent` in the transcript.
    """

    def __init__(self) -> None:
        self._refusals: dict[str, int] = {}
        self._seen_events: dict[str, set[str]] = {}
        self._last_refusal: dict[str, tuple[str | None, str | None]] = {}
        self._recorded: set[str] = set()

    def enabled(self) -> bool:
        return config.hook_enabled()

    async def on_before_model_generate(
        self, data: inspect_ai.hooks.BeforeModelGenerate
    ) -> None:
        active = inspect_ai.log._samples.sample_active()
        if active is None:
            return
        sample_id = active.sample_uuid
        count = self._refusals.get(sample_id, 0)
        max_refusals = config.max_classifier_refusals()
        if count <= max_refusals:
            return
        category, explanation = self._last_refusal.get(sample_id, (None, None))
        # Raises LimitExceededError (recorded once as a SampleLimitEvent first).
        self._trip_inline(
            sample_id=sample_id,
            model_name=data.model_name,
            count=count,
            max_refusals=max_refusals,
            category=category,
            explanation=explanation,
        )

    async def on_sample_event(self, data: inspect_ai.hooks.SampleEvent) -> None:
        event = data.event
        if not isinstance(event, inspect_ai.event.ModelEvent):
            return
        if not detection.is_classifier_refusal(event):
            return

        sample_id = data.sample_id
        seen = self._seen_events.setdefault(sample_id, set())
        if event.uuid is not None:
            if event.uuid in seen:
                return
            seen.add(event.uuid)

        count = self._refusals.get(sample_id, 0) + 1
        self._refusals[sample_id] = count
        category, explanation = detection.refusal_details(event)
        self._last_refusal[sample_id] = (category, explanation)

        max_refusals = config.max_classifier_refusals()
        if count <= max_refusals:
            return

        self._record_limit_event(
            sample_id=sample_id,
            model_name=event.model,
            count=count,
            max_refusals=max_refusals,
            category=category,
            explanation=explanation,
        )

        active = inspect_ai.log._samples.sample_active()
        if active is None or active.tg is None:
            logger.warning(
                "Classifier refusal limit exceeded but no active sample task "
                "group is available to terminate; the breach is recorded in the "
                "transcript. model=%s refusals=%d limit=%d",
                event.model,
                count,
                max_refusals,
            )
            return

        active.limit_exceeded(
            inspect_ai.util.LimitExceededError(
                type="custom",
                value=count,
                limit=max_refusals,
                message=self._message(
                    event.model, count, max_refusals, category, explanation
                ),
            )
        )

    async def on_sample_end(self, data: inspect_ai.hooks.SampleEnd) -> None:
        self._forget(data.sample_id)

    async def on_sample_attempt_end(
        self, data: inspect_ai.hooks.SampleAttemptEnd
    ) -> None:
        self._forget(data.sample_id)

    def _forget(self, sample_id: str) -> None:
        self._refusals.pop(sample_id, None)
        self._seen_events.pop(sample_id, None)
        self._last_refusal.pop(sample_id, None)
        self._recorded.discard(sample_id)

    def _message(
        self,
        model_name: str,
        count: int,
        max_refusals: int,
        category: str | None,
        explanation: str | None,
    ) -> str:
        return detection.refusal_limit_message(
            model_name=model_name,
            refusals=count,
            max_refusals=max_refusals,
            category=category,
            explanation=explanation,
        )

    def _record_limit_event(
        self,
        *,
        sample_id: str,
        model_name: str,
        count: int,
        max_refusals: int,
        category: str | None,
        explanation: str | None,
    ) -> None:
        # Record the breach as a SampleLimitEvent once per sample so the full
        # message (incl. explanation) is visible in the log/viewer; the runner's
        # EvalSampleLimit stores only type+limit. Idempotent because both the
        # async and inline paths may fire for the same breach.
        if sample_id in self._recorded:
            return
        self._recorded.add(sample_id)
        inspect_ai.log.transcript()._event(
            inspect_ai.event.SampleLimitEvent(
                type="custom",
                limit=max_refusals,
                message=self._message(
                    model_name, count, max_refusals, category, explanation
                ),
            )
        )

    def _trip_inline(
        self,
        *,
        sample_id: str,
        model_name: str,
        count: int,
        max_refusals: int,
        category: str | None,
        explanation: str | None,
    ) -> None:
        self._record_limit_event(
            sample_id=sample_id,
            model_name=model_name,
            count=count,
            max_refusals=max_refusals,
            category=category,
            explanation=explanation,
        )
        raise inspect_ai.util.LimitExceededError(
            type="custom",
            value=count,
            limit=max_refusals,
            message=self._message(
                model_name, count, max_refusals, category, explanation
            ),
        )
