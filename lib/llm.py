"""HTTP client for llama-server (OpenAI-compatible) with SSE streaming."""

import json
from typing import Iterator, Optional

import requests


class LLMClient:
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
