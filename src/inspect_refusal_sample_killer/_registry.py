# Importing these modules runs their decorators, which register the hook and
# the probe task with Inspect. This module is the target of the `inspect_ai`
# entry point declared in pyproject.toml. The imports exist purely for their
# registration side effect, so the "unused import" reports are expected: the
# noqa on each import line silences ruff and the directive below silences pyright.
# pyright: reportUnusedImport=false
import inspect_refusal_sample_killer._hooks  # noqa: F401
import inspect_refusal_sample_killer._tasks  # noqa: F401
