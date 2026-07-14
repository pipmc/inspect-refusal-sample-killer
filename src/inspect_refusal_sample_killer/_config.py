import logging
import os

logger = logging.getLogger(__name__)

ENV_VAR = "INSPECT_MAX_CLASSIFIER_REFUSALS"
_DEFAULT = 0


def max_classifier_refusals() -> int:
    """Return the configured classifier-refusal limit.

    Unset or unparseable values fall back to the default of 0 (an unparseable
    value additionally logs a warning). Negative values are returned as-is and
    signal that the hook is disabled (see `hook_enabled`).
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return _DEFAULT
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; defaulting to %d", ENV_VAR, raw, _DEFAULT
        )
        return _DEFAULT


def hook_enabled() -> bool:
    """The hook is disabled when the configured limit is negative."""
    return max_classifier_refusals() >= 0
