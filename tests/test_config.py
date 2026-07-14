import pytest

import inspect_refusal_sample_killer._config as config


def test_default_is_zero_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    assert config.max_classifier_refusals() == 0


def test_parses_positive_int(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "3")
    assert config.max_classifier_refusals() == 3


def test_parses_negative_int(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "-1")
    assert config.max_classifier_refusals() == -1


def test_unparseable_falls_back_to_zero_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv(config.ENV_VAR, "not-a-number")
    with caplog.at_level("WARNING"):
        assert config.max_classifier_refusals() == 0
    assert any(config.ENV_VAR in record.message for record in caplog.records)


def test_enabled_when_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    assert config.hook_enabled() is True


def test_disabled_when_negative(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(config.ENV_VAR, "-1")
    assert config.hook_enabled() is False
