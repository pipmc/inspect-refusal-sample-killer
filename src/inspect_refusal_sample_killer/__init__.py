"""Inspect AI hook that terminates a sample when classifier refusals exceed a limit."""

from inspect_refusal_sample_killer._hooks import ClassifierRefusalKiller

__version__ = "0.1.0"

__all__ = ["ClassifierRefusalKiller", "__version__"]
