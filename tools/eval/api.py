"""Request building and response parsing, mirroring src/mod-ollama-chat_api.cpp.

One builder and one parser per provider, selected by ``OllamaChat.Provider``.
Think mode is always off here, as it is for the module's default
(``ThinkMode = auto`` with ``ThinkModeEnableForModule = 0``), so the reasoning
reserve is 0 and Anthropic requests carry no ``thinking`` block.

Tested live: anthropic. The ollama and openai paths follow the C++ but have
not been run against a real server from this tool.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


def f32(x: float) -> float:
    """Round to float32, the type the module's config globals use."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def sanitize_utf8(s: str) -> str:
    """SanitizeUTF8 replaces invalid bytes; a Python str is already valid unicode."""
    return s


@dataclass
class Settings:
    provider: str = "ollama"
    url: str = "http://localhost:11434/api/generate"
    model: str = "llama3.2:1b"
    system_prompt: str = ""
    stop: str = ""
    seed: str = ""
    num_predict: int = 40
    num_ctx: int = 0
    num_threads: int = 0
    temperature: float = 0.8
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    top_k: int = -1
    min_p: float = -1.0
    presence_penalty: float = -1000.0
    frequency_penalty: float = -1000.0
    api_key: str = ""
    api_key_header: str = ""
    http_timeout: int = 120
    rate_limit_retries: int = 2
    rate_limit_max_wait: int = 15

    def __repr__(self) -> str:          # never leak the key through a stray print
        masked = "<set>" if self.api_key else "<empty>"
        return f"Settings(provider={self.provider!r}, model={self.model!r}, url={self.url!r}, api_key={masked})"


_PROVIDER_ALIASES = {
    "": "ollama", "ollama": "ollama",
    "openai": "openai", "openrouter": "openai", "openai-compatible": "openai", "openai_compatible": "openai",
    "anthropic": "anthropic", "claude": "anthropic",
}


def parse_provider(text: str) -> str:
    """Mirror of OllamaProvider_Parse (config.cpp): ASCII case-fold, strip spaces, tabs and quotes
    from both ends, accept the alias names, fall back to ollama for anything unknown."""
    lowered = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)
    return _PROVIDER_ALIASES.get(lowered.strip(" \t\"'"), "ollama")


def settings_from_config(cfg: dict[str, str], api_key: str = "") -> Settings:
    def g(key: str, default: str) -> str:
        return cfg.get("OllamaChat." + key, default)

    def num(key: str, default, cast):
        raw = g(key, "")
        if raw == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    provider = parse_provider(g("Provider", "ollama"))
    return Settings(
        provider=provider,
        url=g("Url", "http://localhost:11434/api/generate"),
        model=g("Model", "llama3.2:1b"),
        system_prompt=g("SystemPrompt", ""),
        stop=g("Stop", ""),
        seed=g("Seed", ""),
        num_predict=num("NumPredict", 40, int),
        num_ctx=num("NumCtx", 0, int),
        num_threads=num("NumThreads", 0, int),
        temperature=num("Temperature", 0.8, float),
        top_p=num("TopP", 0.95, float),
        repeat_penalty=num("RepeatPenalty", 1.1, float),
        top_k=num("TopK", -1, int),
        min_p=num("MinP", -1.0, float),
        presence_penalty=num("PresencePenalty", -1000.0, float),
        frequency_penalty=num("FrequencyPenalty", -1000.0, float),
        api_key=api_key or g("ApiKey", "").strip(),
        api_key_header=g("ApiKeyHeader", ""),
        http_timeout=num("HttpTimeoutSeconds", 120, int),
        rate_limit_retries=num("RateLimitRetries", 2, int),
        rate_limit_max_wait=num("RateLimitMaxWaitSeconds", 15, int),
    )


def resolve_api_key(env_name: str, provider: str, environ=os.environ) -> str:
    key = environ.get(env_name, "")
    if not key and provider == "anthropic":
        key = environ.get("ANTHROPIC_API_KEY", "")
    return key.strip()


def parse_stop_sequences(stop: str) -> list[str]:
    return [item.strip(" \t") for item in stop.split(",") if item.strip(" \t")]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_ollama_request(cfg: Settings, prompt: str) -> dict:
    request: dict = {"model": cfg.model, "prompt": sanitize_utf8(prompt), "stream": False}
    options: dict = {}
    if cfg.num_predict > 0:
        options["num_predict"] = cfg.num_predict
    if f32(cfg.temperature) != f32(0.8):
        options["temperature"] = cfg.temperature
    if f32(cfg.top_p) != f32(0.95):
        options["top_p"] = cfg.top_p
    if f32(cfg.repeat_penalty) != f32(1.1):
        options["repeat_penalty"] = cfg.repeat_penalty
    if cfg.num_ctx > 0:
        options["num_ctx"] = cfg.num_ctx
    if cfg.num_threads > 0:
        options["num_thread"] = cfg.num_threads
    if cfg.top_k >= 0:
        options["top_k"] = cfg.top_k
    if cfg.min_p >= 0.0:
        options["min_p"] = cfg.min_p
    if cfg.presence_penalty > -999.0:
        options["presence_penalty"] = cfg.presence_penalty
    if cfg.frequency_penalty > -999.0:
        options["frequency_penalty"] = cfg.frequency_penalty
    if cfg.seed:
        try:
            options["seed"] = int(cfg.seed)
        except ValueError:
            pass
    if options:
        request["options"] = options
    if cfg.stop:
        stops = parse_stop_sequences(cfg.stop)
        if stops:
            request["stop"] = stops
    if cfg.system_prompt:
        request["system"] = sanitize_utf8(cfg.system_prompt)
    request["think"] = False
    return request


def build_openai_request(cfg: Settings, prompt: str) -> dict:
    messages = []
    if cfg.system_prompt:
        messages.append({"role": "system", "content": sanitize_utf8(cfg.system_prompt)})
    messages.append({"role": "user", "content": sanitize_utf8(prompt)})
    request: dict = {"model": cfg.model, "messages": messages, "stream": False}
    if cfg.num_predict > 0:
        request["max_tokens"] = cfg.num_predict
    request["temperature"] = cfg.temperature
    request["top_p"] = cfg.top_p
    if cfg.presence_penalty > -999.0:
        request["presence_penalty"] = cfg.presence_penalty
    if cfg.frequency_penalty > -999.0:
        request["frequency_penalty"] = cfg.frequency_penalty
    if cfg.seed:
        try:
            request["seed"] = int(cfg.seed)
        except ValueError:
            pass
    if cfg.stop:
        stops = parse_stop_sequences(cfg.stop)
        if stops:
            request["stop"] = stops
    return request


def build_anthropic_request(cfg: Settings, prompt: str) -> dict:
    max_tokens = cfg.num_predict if cfg.num_predict > 0 else 1024
    request: dict = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": sanitize_utf8(prompt)}],
        "stream": False,
    }
    if cfg.system_prompt:
        request["system"] = sanitize_utf8(cfg.system_prompt)
    if cfg.stop:
        stops = parse_stop_sequences(cfg.stop)
        if stops:
            request["stop_sequences"] = stops
    request["temperature"] = cfg.temperature
    if cfg.top_k >= 0:
        request["top_k"] = cfg.top_k
    return request


def build_request(cfg: Settings, prompt: str) -> dict:
    if cfg.provider == "openai":
        return build_openai_request(cfg, prompt)
    if cfg.provider == "anthropic":
        return build_anthropic_request(cfg, prompt)
    return build_ollama_request(cfg, prompt)


def build_request_headers(cfg: Settings) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    if cfg.provider == "anthropic":
        headers.append(("anthropic-version", "2023-06-01"))
    if not cfg.api_key:
        return headers
    if cfg.api_key_header:
        value = cfg.api_key
        if cfg.api_key_header.lower() == "authorization" and not value.lower().startswith("bearer "):
            value = "Bearer " + value
        headers.append((cfg.api_key_header, value))
        return headers
    if cfg.provider == "anthropic":
        headers.append(("x-api-key", cfg.api_key))
    else:
        headers.append(("Authorization", "Bearer " + cfg.api_key))
    return headers


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


@dataclass
class Parsed:
    text: str = ""
    thinking: str = ""
    error: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


def extract_error_message(body: str) -> str:
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed["error"]
            if isinstance(err, str):
                return err
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                return err["message"]
    except ValueError:
        pass
    return body if len(body) <= 400 else body[:400] + "..."


def parse_anthropic_body(body: str) -> Parsed:
    out = Parsed()
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        out.error = f"JSON parse failure: {exc}"
        return out
    if isinstance(parsed, dict) and ("error" in parsed or parsed.get("type") == "error"):
        out.error = extract_error_message(body)
        return out
    if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), list):
        out.error = "no content array in response"
        return out
    text, thinking = [], []
    for block in parsed["content"]:
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            continue
        if block["type"] == "text" and isinstance(block.get("text"), str):
            text.append(block["text"])
        elif block["type"] == "thinking" and isinstance(block.get("thinking"), str):
            thinking.append(block["thinking"])
    out.text, out.thinking = "".join(text), "".join(thinking)
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        out.input_tokens = usage.get("input_tokens")
        out.output_tokens = usage.get("output_tokens")
    return out


def parse_openai_body(body: str) -> Parsed:
    out = Parsed()
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        out.error = f"JSON parse failure: {exc}"
        return out
    if isinstance(parsed, dict) and "error" in parsed:
        out.error = extract_error_message(body)
        return out
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not isinstance(choices, list) or not choices:
        out.error = "no choices in response"
        return out
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        out.error = "first choice has no message"
        return out
    content = msg.get("content")
    if isinstance(content, str):
        out.text = content
    elif isinstance(content, list):
        out.text = "".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    for key in ("reasoning", "reasoning_content"):
        if isinstance(msg.get(key), str):
            out.thinking = msg[key]
            break
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        out.input_tokens = usage.get("prompt_tokens")
        out.output_tokens = usage.get("completion_tokens")
    return out


def parse_ollama_body(body: str) -> Parsed:
    """Single JSON object, or one object per line (streaming shape behind some proxies)."""
    out = Parsed()
    text, thinking = [], []
    parsed_any = False
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except ValueError as exc:
            if not out.error:
                out.error = f"JSON parse failure: {exc}"
            continue
        parsed_any = True
        if not isinstance(parsed, dict):
            continue
        if isinstance(parsed.get("error"), str):
            out.error = parsed["error"]
            continue
        if isinstance(parsed.get("response"), str):
            text.append(parsed["response"])
        if isinstance(parsed.get("thinking"), str):
            thinking.append(parsed["thinking"])
        if isinstance(parsed.get("prompt_eval_count"), int):
            out.input_tokens = parsed["prompt_eval_count"]
        if isinstance(parsed.get("eval_count"), int):
            out.output_tokens = parsed["eval_count"]
    if not parsed_any and not out.error:
        out.error = "no JSON object in response body"
    out.text, out.thinking = "".join(text), "".join(thinking)
    return out


def parse_body(provider: str, body: str) -> Parsed:
    if provider == "openai":
        return parse_openai_body(body)
    if provider == "anthropic":
        return parse_anthropic_body(body)
    return parse_ollama_body(body)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


@dataclass
class ApiResult:
    ok: bool = False
    text: str = ""
    error: str = ""
    status: int = 0
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts: int = 1


def is_rate_limited(status: int) -> bool:
    return status in (429, 503, 529)


def _post_once(cfg: Settings, prompt: str, opener: Callable) -> tuple[ApiResult, int]:
    """One HTTP round trip. Returns (result, retry_after_seconds)."""
    body = json.dumps(build_request(cfg, prompt), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(cfg.url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for name, value in build_request_headers(cfg):
        req.add_header(name, value)

    result = ApiResult()
    retry_after = 0
    started = time.monotonic()
    try:
        with opener(req, timeout=cfg.http_timeout) as resp:
            result.status = resp.status
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        result.status = exc.code
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        try:
            retry_after = int(exc.headers.get("Retry-After", "0") or 0)
        except (ValueError, AttributeError):
            retry_after = 0
        result.latency_ms = int((time.monotonic() - started) * 1000)
        result.error = f"HTTP {exc.code}"
        if raw:
            result.error += ": " + extract_error_message(raw)
        return result, retry_after
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.latency_ms = int((time.monotonic() - started) * 1000)
        result.error = f"transport error: {exc}"
        return result, 0

    result.latency_ms = int((time.monotonic() - started) * 1000)
    parsed = parse_body(cfg.provider, raw)
    if parsed.error:
        result.error = parsed.error
        return result, 0
    result.ok = True
    result.text = parsed.text
    result.input_tokens, result.output_tokens = parsed.input_tokens, parsed.output_tokens
    return result, 0


def query(cfg: Settings, prompt: str, *, opener: Callable = urllib.request.urlopen,
          sleep: Callable[[float], None] = time.sleep) -> ApiResult:
    """One request plus the module's bounded back-off on 429/503/529."""
    result, retry_after = _post_once(cfg, prompt, opener)
    attempt = 0
    while attempt < cfg.rate_limit_retries and not result.ok and is_rate_limited(result.status):
        wait = retry_after if retry_after > 0 else (2 << attempt)
        wait = min(wait, cfg.rate_limit_max_wait)
        if wait == 0:
            break
        sleep(wait)
        result, retry_after = _post_once(cfg, prompt, opener)
        attempt += 1
        result.attempts = attempt + 1
    return result
