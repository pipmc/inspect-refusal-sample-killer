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
    """Counts classifier refusals per sample and terminates on breach."""

    def __init__(self) -> None:
        self._refusals: dict[str, int] = {}
        self._seen_events: dict[str, set[str]] = {}

    def enabled(self) -> bool:
        return config.hook_enabled()

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

        max_refusals = config.max_classifier_refusals()
        if count <= max_refusals:
            return

        category, explanation = detection.refusal_details(event)
        message = detection.refusal_limit_message(
            model_name=event.model,
            refusals=count,
            max_refusals=max_refusals,
            category=category,
            explanation=explanation,
        )
        self._terminate_sample(message=message, count=count, max_refusals=max_refusals)

    async def on_sample_end(self, data: inspect_ai.hooks.SampleEnd) -> None:
        self._forget(data.sample_id)

    async def on_sample_attempt_end(
        self, data: inspect_ai.hooks.SampleAttemptEnd
    ) -> None:
        self._forget(data.sample_id)

    def _forget(self, sample_id: str) -> None:
        self._refusals.pop(sample_id, None)
        self._seen_events.pop(sample_id, None)

    def _terminate_sample(self, *, message: str, count: int, max_refusals: int) -> None:
        active = inspect_ai.log._samples.sample_active()
        if active is None or active.tg is None:
            logger.warning(
                "Classifier refusal limit exceeded but no active sample task "
                "group is available to terminate; skipping. (%s)",
                message,
            )
            return

        error = inspect_ai.util.LimitExceededError(
            type="custom",
            value=count,
            limit=max_refusals,
            message=message,
        )
        # Record a transcript event so the full message (incl. explanation) is
        # visible in the log/viewer; EvalSampleLimit itself stores only type+limit.
        inspect_ai.log.transcript()._event(
            inspect_ai.event.SampleLimitEvent(
                type="custom", limit=max_refusals, message=message
            )
        )
        active.limit_exceeded(error)
