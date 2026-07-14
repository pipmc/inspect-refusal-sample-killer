# Importing the hooks module runs the @hooks decorator, which registers the
# hook instance with Inspect. This module is the target of the `inspect_ai`
# entry point declared in pyproject.toml. The import exists purely for its
# registration side effect, so the "unused import" reports are expected: the
# noqa on the import line silences ruff and the directive below silences pyright.
# pyright: reportUnusedImport=false
import inspect_refusal_sample_killer._hooks  # noqa: F401
