from types import SimpleNamespace

import pytest

from rubric import openai_client


def test_responses_request_uses_environment_key_and_disables_storage(monkeypatch):
    seen = {}

    class FakeResponses:
        def create(self, **kwargs):
            seen["request"] = kwargs
            return SimpleNamespace(output_text="  result  ")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setattr(openai_client, "OpenAI", FakeOpenAI)

    result = openai_client.call_openai_sync(
        [{"role": "user", "content": "hello"}],
        model="test-model",
        max_tokens=128,
        temperature=0.2,
        timeout=30,
    )

    assert result == "result"
    assert seen["client"] == {
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "max_retries": 0,
        "timeout": 30,
    }
    assert seen["request"] == {
        "model": "test-model",
        "input": [{"role": "user", "content": "hello"}],
        "max_output_tokens": 128,
        "store": False,
        "temperature": 0.2,
    }


def test_temperature_can_be_omitted(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = SimpleNamespace(
                create=lambda **kwargs: seen.setdefault("response", SimpleNamespace(output_text="ok"))
            )
            original = self.responses.create

            def create(**kwargs):
                seen["request"] = kwargs
                return original(**kwargs)

            self.responses.create = create

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_OMIT_TEMPERATURE", "1")
    monkeypatch.setattr(openai_client, "OpenAI", FakeOpenAI)

    openai_client.call_openai_sync([], model="test-model", max_tokens=8, temperature=0.7, timeout=5)

    assert "temperature" not in seen["request"]


def test_missing_api_key_fails_before_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_client.call_openai_sync([], model="test-model", max_tokens=8, temperature=0, timeout=5)
