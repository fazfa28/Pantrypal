"""One OpenAI-compatible client for every model (OpenRouter, Gemini, a local vLLM...).

`chat()` optionally caches each response on disk, keyed by a hash of the full
request (model + messages incl. images + params). Reruns then cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import time

from openai import OpenAI

from core.config import cfg, path, secret

_clients: dict[tuple[str, str], OpenAI] = {}


def client(base_url: str | None = None, api_key_env: str | None = None) -> OpenAI:
    base_url = base_url or cfg("llm", "base_url")
    api_key_env = api_key_env or cfg("llm", "api_key_env")
    key = (base_url, api_key_env)
    if key not in _clients:
        api_key = secret(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"No API key: set {api_key_env} in your environment or in .streamlit/secrets.toml"
            )
        _clients[key] = OpenAI(base_url=base_url, api_key=api_key, timeout=120)   # seconds; a hung call retries
    return _clients[key]


def _cache_file(payload: dict):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    d = path("cache")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{digest}.json"


def chat(
    messages: list[dict],
    model: str | None = None,
    *,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    response_format: dict | None = None,
    use_cache: bool = False,
    base_url: str | None = None,
    api_key_env: str | None = None,
    retries: int = 3,
    **params,
) -> dict:
    """Call a chat model. Returns a plain dict:
    {content, tool_calls: [{id, name, arguments}], usage: {prompt_tokens, completion_tokens},
     model, latency_s, cached}
    """
    payload = {
        "model": model or cfg("llm", "chat_model"),
        "messages": messages,
        "temperature": cfg("llm", "temperature", default=0),
        **params,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if response_format:
        payload["response_format"] = response_format

    cache_file = _cache_file({**payload, "_base_url": base_url}) if use_cache else None
    if cache_file and cache_file.exists():
        out = json.loads(cache_file.read_text(encoding="utf-8"))
        out["cached"] = True
        return out

    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = client(base_url, api_key_env).chat.completions.create(**payload)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)  # simple backoff for rate limits

    msg = resp.choices[0].message
    usage = resp.usage
    out = {
        "content": msg.content or "",
        "tool_calls": [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ],
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        },
        "model": payload["model"],
        "latency_s": round(time.time() - t0, 3),
        "cached": False,
    }
    if cache_file:
        cache_file.write_text(json.dumps(out), encoding="utf-8")
    return out


def cost_usd(usage: dict, price: list | None) -> float | None:
    """price = [input, output] US$ per 1M tokens, from config.yaml eval_models."""
    if not price or price[0] is None or price[1] is None:
        return None
    return (usage["prompt_tokens"] * price[0] + usage["completion_tokens"] * price[1]) / 1e6
