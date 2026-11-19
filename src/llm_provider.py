"""Swappable LLM providers + a cached, validated, retrying JSON call.

Every provider implements one method: ``complete(system, messages) -> LLMResult``.
Swapping models is an environment change (LLM_PROVIDER / LLM_MODEL), not a code change:

  gemini             Google AI Studio REST API (free tier)      GEMINI_API_KEY
  openai_compatible  any /chat/completions endpoint: OpenAI, Groq, OpenRouter, Ollama,
                     vLLM, an internal gateway ...              LLM_BASE_URL + LLM_API_KEY
  anthropic          Claude Messages API via the official SDK  ANTHROPIC_API_KEY
  claude_cli         local `claude -p` (Claude Code) - no API key needed
  replay             cache only: re-runs a previous execution offline and FAILS on any
                     cache miss. It never invents output.

There is no silent fallback to fake results: a missing key or unreachable API raises.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

import config


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)


Message = dict  # {"role": "user"|"assistant", "content": str}


class Provider:
    name = "base"
    default_model = ""

    def __init__(self, model: str = ""):
        self.model = model or self.default_model

    def complete(self, system: str, messages: list[Message]) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


# --------------------------------------------------------------------------- providers
class GeminiProvider(Provider):
    name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self, model: str = ""):
        super().__init__(model)
        self.key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set (get a free key at https://aistudio.google.com/apikey)")

    def complete(self, system, messages):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]} for m in messages],
            "generationConfig": {"temperature": config.LLM_TEMPERATURE,
                                 "responseMimeType": "application/json"},
        }
        r = _post_with_backoff(url, body, headers={"x-goog-api-key": self.key})
        data = r.json()
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected Gemini response: {json.dumps(data)[:500]}") from exc
        return LLMResult(text, self.model, self.name, data.get("usageMetadata", {}))


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"
    default_model = "gpt-4o-mini"

    def __init__(self, model: str = ""):
        super().__init__(model)
        self.base = (config.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        self.key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not self.key and "localhost" not in self.base and "127.0.0.1" not in self.base:
            raise LLMError("LLM_API_KEY is not set for the OpenAI-compatible endpoint")
        self.json_mode = os.getenv("LLM_JSON_MODE", "1") == "1"

    def complete(self, system, messages):
        body = {"model": self.model, "temperature": config.LLM_TEMPERATURE,
                "messages": [{"role": "system", "content": system}, *messages]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = _post_with_backoff(f"{self.base}/chat/completions", body, headers=headers)
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {json.dumps(data)[:500]}") from exc
        return LLMResult(text, data.get("model", self.model), self.name, data.get("usage", {}))


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-opus-5-5"

    def __init__(self, model: str = ""):
        super().__init__(model)
        try:
            import anthropic  # optional dependency
        except ImportError as exc:
            raise LLMError("pip install anthropic to use LLM_PROVIDER=anthropic") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S)  # reads ANTHROPIC_API_KEY

    def complete(self, system, messages):
        a = self._anthropic
        kwargs = dict(model=self.model, max_tokens=16000, system=system, messages=messages)
        if os.getenv("LLM_EFFORT"):
            kwargs["output_config"] = {"effort": os.getenv("LLM_EFFORT")}
        try:
            resp = self.client.messages.create(**kwargs)
        except a.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limit: {exc}") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request (stop_reason=refusal)")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(text, resp.model, self.name,
                         {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens})


class ClaudeCLIProvider(Provider):
    """Runs the locally installed Claude Code CLI in print mode with all tools disabled."""
    name = "claude_cli"
    default_model = "sonnet"

    def __init__(self, model: str = ""):
        super().__init__(model)
        self.exe = shutil.which("claude.cmd") or shutil.which("claude")
        if not self.exe:
            raise LLMError("`claude` CLI not found on PATH")

    def complete(self, system, messages):
        prompt = _flatten_messages(messages)
        # Observed on Windows: the CLI replaces non-ASCII characters in its output with U+FFFD
        # (input arrives intact). JSON \uXXXX escapes are pure ASCII and decode to the exact text.
        system += ("\n\nOUTPUT ENCODING: write only ASCII characters. Inside JSON strings encode every "
                   "non-ASCII character as a \\uXXXX escape (e.g. é -> \\u00e9, – -> \\u2013).")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(system)
            sys_file = fh.name
        cmd = [self.exe, "-p", "--model", self.model, "--output-format", "json", "--tools", "",
               "--system-prompt-file", sys_file, "--no-session-persistence"]
        try:
            # Transient CLI failures were observed under parallel load (e.g. a spurious
            # "unrecognized_model" for a model that succeeded seconds earlier), so retry with backoff.
            for attempt in range(4):
                try:
                    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                          errors="replace", timeout=config.LLM_TIMEOUT_S, cwd=tempfile.gettempdir())
                except subprocess.TimeoutExpired as exc:
                    if attempt == 3:
                        raise LLMError("claude CLI timed out") from exc
                    continue
                if proc.returncode == 0:
                    break
                if attempt < 3:
                    time.sleep(10 * (attempt + 1))
        finally:
            os.unlink(sys_file)
        if proc.returncode != 0:
            raise LLMError(f"claude CLI failed ({proc.returncode}) after retries: {(proc.stderr or proc.stdout)[:500]}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON envelope: {proc.stdout[:300]}") from exc
        if data.get("is_error"):
            raise LLMError(f"claude CLI error: {data.get('result', '')[:300]}")
        usage = data.get("usage", {}) | {"total_cost_usd": data.get("total_cost_usd")}
        model = ",".join(data.get("modelUsage", {}).keys()) or self.model
        return LLMResult(data.get("result", ""), model, self.name, usage)


class ReplayProvider(Provider):
    name = "replay"
    default_model = "cache-only"

    def complete(self, system, messages):
        raise LLMError("replay mode: no cached response for this request. Run once with a real "
                       "provider (see .env.example) before using replay.")


PROVIDERS: dict[str, type[Provider]] = {p.name: p for p in
                                        (GeminiProvider, OpenAICompatibleProvider, AnthropicProvider,
                                         ClaudeCLIProvider, ReplayProvider)}


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    name = (name or config.LLM_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise LLMError(f"unknown LLM_PROVIDER '{name}'. Options: {', '.join(PROVIDERS)}")
    return PROVIDERS[name](model if model is not None else config.LLM_MODEL)


# --------------------------------------------------------------------------- helpers
_rate_lock = threading.Lock()
_last_call = [0.0]


def _throttle():
    if config.LLM_MIN_INTERVAL_S <= 0:
        return
    with _rate_lock:
        wait = _last_call[0] + config.LLM_MIN_INTERVAL_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _post_with_backoff(url: str, body: dict, headers: dict, attempts: int = 5) -> requests.Response:
    delay = 5.0
    for i in range(attempts):
        _throttle()
        try:
            r = requests.post(url, json=body, headers=headers, timeout=config.LLM_TIMEOUT_S)
        except requests.RequestException as exc:
            if i == attempts - 1:
                raise LLMError(f"network error calling {url.split('?')[0]}: {exc}") from exc
            time.sleep(delay); delay *= 2
            continue
        if r.status_code in (429, 500, 502, 503, 504) and i < attempts - 1:
            retry_after = r.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay)
            delay *= 2
            continue
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code} from {url.split('?')[0]}: {r.text[:400]}")
        return r
    raise LLMError("exhausted retries")  # pragma: no cover


def _flatten_messages(messages: list[Message]) -> str:
    if len(messages) == 1:
        return messages[0]["content"]
    parts = []
    for m in messages:
        tag = "YOUR PREVIOUS RESPONSE" if m["role"] == "assistant" else "USER"
        parts.append(f"=== {tag} ===\n{m['content']}")
    return "\n\n".join(parts)


def extract_json(text: str):
    """Parse a JSON object from model text, tolerating ``` fences and leading prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", text, 0)


# --------------------------------------------------------------------------- cached JSON call
@dataclass
class CallRecord:
    name: str
    cache_key: str
    cached: bool
    attempts: int
    raw_responses: list[str]
    model: str
    usage: list[dict]


class LLMClient:
    """Provider + on-disk cache + JSON validation with error-feedback retries."""

    def __init__(self, provider: Provider, cache_dir: Path = config.CACHE_DIR,
                 raw_dir: Path | None = None):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = raw_dir
        if raw_dir:
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()

    def _key(self, system: str, user: str) -> str:
        # replay mode must hit entries written by any real provider, so the provider is not
        # part of the key in that case; otherwise provider+model are.
        ident = "" if self.provider.name == "replay" else self.provider.describe()
        h = hashlib.sha256(json.dumps([config.PROMPT_VERSION, ident, system, user]).encode()).hexdigest()
        return h[:32]

    def _lookup(self, key: str) -> dict | None:
        direct = self.cache_dir / f"{key}.json"
        return json.loads(direct.read_text(encoding="utf-8")) if direct.exists() else None

    def call_json(self, name: str, system: str, user: str,
                  validate: Callable[[object], list[str]] | None = None) -> object:
        """Return parsed JSON. ``validate`` returns a list of problems ([] = OK); problems are fed
        back to the model for up to LLM_MAX_RETRIES corrective retries."""
        key = self._key(system, user)
        if self.provider.name == "replay":
            key = self._replay_key(system, user)
        hit = self._lookup(key) if key else None
        if hit is not None:
            parsed = extract_json(hit["final_text"])
            rec = CallRecord(name, key, True, 0, hit["raw_responses"], hit["model"], hit.get("usage", []))
            self._record(rec, hit)
            return parsed
        if self.provider.name == "replay":
            raise LLMError(f"replay mode: no cached response for '{name}'")

        messages: list[Message] = [{"role": "user", "content": user}]
        raws, usages = [], []
        problems: list[str] = []
        model = self.provider.model
        history: list[list[str]] = []
        last_attempt = config.LLM_MAX_RETRIES + 1
        for attempt in range(1, last_attempt + 1):
            res = self.provider.complete(system, messages)
            raws.append(res.text); usages.append(res.usage); model = res.model
            try:
                parsed = extract_json(res.text)
                problems = validate(parsed) if validate else []
            except json.JSONDecodeError as exc:
                parsed, problems = None, [f"response was not valid JSON ({exc.msg})"]
            except Exception as exc:  # noqa: BLE001 - validator crashed on odd structure
                parsed, problems = None, [f"response did not match the required structure ({exc})"]
            history.append(problems)
            hard = [p for p in problems if not p.startswith("SOFT:")]
            # Soft problems (rule violations such as duration) are fed back for self-correction,
            # but if they persist the response is accepted and validation rejects that promo later.
            if not problems or (attempt == last_attempt and not hard):
                entry = {"name": name, "key": key, "provider": self.provider.name, "model": model,
                         "prompt_version": config.PROMPT_VERSION,
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "system_sha": hashlib.sha256(system.encode()).hexdigest()[:16],
                         "user_sha": hashlib.sha256(user.encode()).hexdigest()[:16],
                         "attempts": attempt, "raw_responses": raws, "final_text": res.text,
                         "feedback_history": history, "usage": usages}
                (self.cache_dir / f"{key}.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
                self._write_replay_index(system, user, key)
                self._record(CallRecord(name, key, False, attempt, raws, model, usages), entry)
                return parsed
            messages += [{"role": "assistant", "content": res.text},
                         {"role": "user", "content": "Your JSON had these problems:\n- "
                          + "\n- ".join(p.removeprefix("SOFT:").strip() for p in problems)
                          + "\nFix them (adjust block ranges using the timestamps; drop a promo only if it "
                            "cannot satisfy the rules). Return the COMPLETE corrected JSON object only."}]
        failure = {"name": name, "key": key, "attempts": len(raws), "raw_responses": raws, "problems": problems}
        if self.raw_dir:
            (Path(self.raw_dir) / f"{name}.FAILED.json").write_text(json.dumps(failure, indent=1), encoding="utf-8")
        raise LLMError(f"{name}: still invalid after {len(raws)} attempts: {problems[:3]}")

    # replay index: provider-independent pointer from prompt hash -> cache entry
    def _replay_key(self, system: str, user: str) -> str | None:
        idx = self.cache_dir / "replay_index.json"
        if not idx.exists():
            return None
        h = hashlib.sha256(json.dumps([config.PROMPT_VERSION, system, user]).encode()).hexdigest()[:32]
        return json.loads(idx.read_text(encoding="utf-8")).get(h)

    def _write_replay_index(self, system: str, user: str, key: str):
        with self._lock:
            idx = self.cache_dir / "replay_index.json"
            data = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else {}
            h = hashlib.sha256(json.dumps([config.PROMPT_VERSION, system, user]).encode()).hexdigest()[:32]
            data[h] = key
            idx.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def _record(self, rec: CallRecord, entry: dict):
        with self._lock:
            self.calls.append(rec)
        if self.raw_dir:
            out = dict(entry)
            out["served_from_cache"] = rec.cached
            (Path(self.raw_dir) / f"{rec.name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                encoding="utf-8")
