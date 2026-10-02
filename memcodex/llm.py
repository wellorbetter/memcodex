"""OpenAI-compatible chat client used for all LLM calls."""
from __future__ import annotations

import time

import requests


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, base_url: str, api_key: str = "local", model: str = "",
                 max_tokens: int = 2048, max_retries: int = 3, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
                self.last_completion_tokens = int(usage.get("completion_tokens", 0))
                self.total_prompt_tokens += self.last_prompt_tokens
                self.total_completion_tokens += self.last_completion_tokens
                return text
            except (requests.RequestException, KeyError, ValueError) as err:
                last_err = err
                time.sleep(min(2 ** attempt, 15))
        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_err}")