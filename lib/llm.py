"""LLM backend clients for Ada.

Two providers are supported with the same public interface:
    - LlamaServerLLM: HTTP client for a local llama-server (OpenAI-compatible SSE).
    - ClaudeLLM:      Anthropic Claude API client (streaming).

Both expose:
    __init__(*, system_prompt, history_size, temperature, ...)
    check_connection() -> Optional[str]    # None if OK, error string otherwise
    stream(user_message) -> Iterator[str]  # yields text deltas
    self.history: list[dict]               # sliding window {role, content}

Use make_llm(llm_cfg) to instantiate the right backend from the config dict.
"""

import json
import os
from typing import Iterator, Optional

import requests


class LlamaServerLLM:
    """Client for llama-server's /v1/chat/completions endpoint.

    Maintains a sliding window of conversation history (excluding system prompt).
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080",
        system_prompt: str = "You are a helpful assistant.",
        history_size: int = 20,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.system_prompt = system_prompt
        self.history_size = history_size
        self.temperature = temperature
        self.timeout = timeout
        self.history: list[dict] = []

    def check_connection(self) -> Optional[str]:
        """Check if llama-server is reachable. Returns error string or None."""
        try:
            r = requests.get(f"{self.endpoint}/health", timeout=3)
            if r.status_code != 200:
                return f"llama-server returned {r.status_code}"
            return None
        except requests.exceptions.ConnectionError:
            return f"Cannot connect to llama-server at {self.endpoint}. Is it running?"
        except Exception as e:
            return f"llama-server check failed: {e}"

    def stream(self, user_message: str) -> Iterator[str]:
        """Send a user message and yield content deltas as they arrive.

        After the stream ends, the user message and full response are appended
        to history (with sliding window applied).
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }

        full_response = ""
        try:
            with requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            full_response += delta
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except requests.exceptions.RequestException as e:
            yield f"\n[LLM error: {e}]"
            return

        # Update history with sliding window
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": full_response})
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]


class ClaudeLLM:
    """Client for Anthropic's Claude API (Messages API with streaming).

    Same public interface as LlamaServerLLM. The system prompt is passed as a
    top-level parameter (not inside messages — that's how Anthropic's API works).
    History is kept client-side as a sliding window and resent on every request,
    since the Messages API is stateless.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        system_prompt: str = "You are a helpful assistant.",
        history_size: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        # Import lazily so users on the llama-server backend don't need anthropic installed.
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for the Claude LLM backend. "
                "Install it with: pip install anthropic"
            ) from e

        self._anthropic = anthropic
        self.model = model
        self.system_prompt = system_prompt
        self.history_size = history_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.history: list[dict] = []

        # If api_key is None, the SDK reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def check_connection(self) -> Optional[str]:
        """Verify an API key is available. Returns error string or None.

        Does not make a network call — that would burn tokens on every startup.
        Real errors (auth, model, network) surface on the first stream() call.
        """
        if not self.api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            return (
                "ANTHROPIC_API_KEY env var is not set and no api_key in config. "
                "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        return None

    def stream(self, user_message: str) -> Iterator[str]:
        """Send a user message and yield text deltas as they arrive.

        After the stream ends, the user message and full response are appended
        to history (with sliding window applied).
        """
        messages = list(self.history)
        messages.append({"role": "user", "content": user_message})

        full_response = ""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        full_response += text
                        yield text
        except self._anthropic.APIError as e:
            yield f"\n[claude error: {e}]"
            return
        except Exception as e:
            yield f"\n[claude error: {e}]"
            return

        # Update history with sliding window
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": full_response})
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]


def make_llm(llm_cfg: dict):
    """Construct the right LLM backend from the `llm:` config dict.

    Reads `provider` (default: "llama-server") and dispatches to the matching
    class. Backwards-compatible with the legacy flat layout where `endpoint`
    sat directly under `llm:` (no `llama_server:` sub-dict).
    """
    provider = llm_cfg.get("provider", "llama-server")
    common = dict(
        system_prompt=llm_cfg.get("system_prompt", "You are a helpful assistant."),
        history_size=llm_cfg.get("history_size", 20),
        temperature=llm_cfg.get("temperature", 0.7),
    )

    if provider == "llama-server":
        sub = llm_cfg.get("llama_server", {}) or {}
        # Backwards compat: fall back to top-level `endpoint` (old config layout).
        endpoint = sub.get("endpoint") or llm_cfg.get("endpoint", "http://localhost:8080")
        return LlamaServerLLM(endpoint=endpoint, **common)

    if provider == "claude":
        sub = llm_cfg.get("claude", {}) or {}
        return ClaudeLLM(
            model=sub.get("model", "claude-opus-4-6"),
            max_tokens=sub.get("max_tokens", 1024),
            api_key=sub.get("api_key"),
            **common,
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Expected 'llama-server' or 'claude'."
    )
