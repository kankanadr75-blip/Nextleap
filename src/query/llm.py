"""Phase 5b - provider-agnostic LLM adapter.

One protocol, four implementations. ``StubLLM`` is the default whenever no API
key is configured, and it is **extractive**: it builds the answer out of the
retrieved chunk's own sentences and never invents a word. That is what lets the
demo, the test suite and ``sample_qa.md`` run with zero credentials - no phase
of this project requires a key.

Design rules for the networked clients:

* temperature 0 and JSON mode, so answers are as reproducible as the stub's;
* the model returns ``{"answer", "source_chunk_id"}`` and **never a URL** - the
  link is re-derived from a code-side map in ``format.py``;
* nothing raises into the chat path. A provider error, a timeout or malformed
  JSON falls back to ``StubLLM``, because a guarded refusal is always a better
  outcome than a stack trace in front of a user.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Protocol, runtime_checkable

from src.query import config

logger = logging.getLogger("mf_faq.llm")

SYSTEM_PROMPT = """\
You answer factual questions about 5 HDFC Mutual Fund schemes.

Rules you must follow:
1. Use ONLY the numbered CONTEXT supplied in the user message. If the context
   does not contain the answer, say so plainly instead of guessing.
2. Answer in at most 3 sentences. Be direct; no preamble or summary.
3. Never give investment advice, opinions, recommendations or comparisons.
4. Never state returns, NAV, AUM, performance or any profit figure.
5. Never output a URL, a link, or a citation. A citation is added afterwards
   from our own records; you only return the id of the chunk you used.
6. Never repeat or request personal data (PAN, Aadhaar, folio numbers, OTPs).

Return only this JSON object:
{"answer": "<your answer, max 3 sentences>", "source_chunk_id": "<id of the chunk you used>"}
"""


@runtime_checkable
class LLMClient(Protocol):
    """What ``generate.py`` needs from any provider."""

    name: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return the parsed JSON object. Never raise into the chat path."""
        ...


# --------------------------------------------------------------------------
# Robust JSON parsing
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_loose(raw: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply, tolerating the usual junk.

    Handles code fences, leading prose, and trailing text by falling back to the
    first balanced ``{...}`` block. Returns ``{}`` rather than raising, because
    every caller treats an unparseable reply as "fall back to the stub".
    """
    if not raw:
        return {}
    text = _FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {}
    return {}


# --------------------------------------------------------------------------
# Stub (default, offline)
# --------------------------------------------------------------------------

# A chunk document is "<display> | <section> | <source type>. <display>: <body>".
_HEADER_RE = re.compile(r"^(?P<display>[^|]+)\|\s*(?P<section>[^|]+)\|\s*(?P<label>[^.]+)\.\s*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹(])")


def _split_body(document: str) -> tuple[str, str]:
    """Split a chunk document into (display_name, fact_body)."""
    match = _HEADER_RE.match(document)
    display = match.group("display").strip() if match else ""
    body = _HEADER_RE.sub("", document, count=1).strip()
    if not display and ": " in body:
        display, _, body = body.partition(": ")
        display, body = display.strip(), body.strip()
    return display, body


class StubLLM:
    """Deterministic, offline, extractive. No network, no credentials.

    The answer is assembled from the top chunk's own sentences, so it cannot
    introduce a fact the corpus does not contain. It is the floor the test suite
    runs against: if a guardrail only holds with a real model, it is a prompt
    and not a guarantee.
    """

    name = "stub"

    def __init__(self, max_sentences: int | None = None) -> None:
        self.max_sentences = max_sentences or config.MAX_SENTENCES

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        del system  # the stub has no use for the prompt; it reads the context
        blocks = parse_context_blocks(user)
        if not blocks:
            return {
                "answer": "I don't have that in my sources.",
                "source_chunk_id": "",
            }

        block = blocks[0]
        answer = self._compose(block["text"], block.get("chunk_type", "fact_card"))
        return {"answer": answer, "source_chunk_id": block["chunk_id"]}

    def _compose(self, document: str, chunk_type: str) -> str:
        """Build an answer from the chunk's own words.

        For a ``steps`` chunk each numbered step is one unit, so at most three
        steps are shown. That is an intentional trade-off: a procedure split
        mid-way is worse than a partial one, and the single citation points at
        the full procedure.
        """
        _, body = _split_body(document)

        if chunk_type == "steps":
            steps = re.findall(r"Step \d+:[^|]*?(?=Step \d+:|$)", body, re.DOTALL)
            if not steps:
                return _cap_sentences(body, self.max_sentences)
            joined = " ".join(s.strip() for s in steps[: self.max_sentences])
            return re.sub(r"\s+", " ", joined).strip()

        return _cap_sentences(body, self.max_sentences)


def _cap_sentences(text: str, max_sentences: int) -> str:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    if not parts:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


def parse_context_blocks(user_turn: str) -> list[dict[str, str]]:
    """Pull the ``[n]`` context blocks back out of a built user turn.

    The stub shares the prompt-builder's format, so the two can never drift.
    """
    blocks: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\[(\d+)\]\s*chunk_id:\s*(?P<cid>\S+)\s*type:\s*(?P<ctype>\S+)\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(user_turn))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user_turn)
        text = user_turn[start:end]
        text = re.sub(r"^\s*source_url:.*$", "", text, flags=re.MULTILINE)
        blocks.append(
            {
                "chunk_id": match.group("cid"),
                "chunk_type": match.group("ctype"),
                "text": text.strip(),
            }
        )
    return blocks


# --------------------------------------------------------------------------
# Networked providers
# --------------------------------------------------------------------------


class _HttpProvider:
    """Shared plumbing: key lookup, model selection, temperature 0, never raise.

    Model resolution order is explicit argument, then the provider's ``*_MODEL``
    environment variable, then ``default_model``. The env var is read in
    ``__init__`` rather than at import, so editing ``.env`` and restarting is
    enough, and so a test can override it per instance.
    """

    provider_name = "http"
    key_env = ""
    model_env = ""
    default_model = ""

    def __init__(self, model: str = "") -> None:
        self.model = self._resolve_model(model)
        self._key = ""

    @property
    def name(self) -> str:
        """Provider plus resolved model, e.g. ``groq:openai/gpt-oss-20b``.

        ``LLMClient`` declares ``name`` and ``generate.py`` records it on every
        ``GenerationResult``, so this is the one place you can confirm *which*
        model actually answered. That matters more than it looks: a bad key or a
        missing package silently degrades to ``StubLLM``, whose answers are
        plausible enough to be mistaken for a real model's.
        """
        return f"{self.provider_name}:{self.model}"

    def _resolve_model(self, explicit: str) -> str:
        if explicit:
            return explicit
        if self.model_env:
            from_env = os.environ.get(self.model_env, "").strip()
            if from_env:
                return from_env
        return self.default_model

    def _require_key(self) -> str:
        if not self._key:
            self._key = os.environ.get(self.key_env, "").strip()
        return self._key

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        key = self._require_key()
        if not key:
            logger.warning("%s selected but %s is unset; using stub",
                           self.provider_name, self.key_env)
            return StubLLM().complete_json(system, user)
        try:
            return parse_json_loose(self._raw_call(key, system, user))
        except Exception as exc:  # noqa: BLE001 - the chat path must never raise
            logger.warning("%s call failed (%s); falling back to stub",
                           self.provider_name, type(exc).__name__)
            return StubLLM().complete_json(system, user)

    def _raw_call(self, key: str, system: str, user: str) -> str:
        raise NotImplementedError


class GroqClient(_HttpProvider):
    provider_name = "groq"
    key_env = "GROQ_API_KEY"
    model_env = "GROQ_MODEL"
    default_model = "llama-3.3-70b-versatile"

    def _raw_call(self, key: str, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: not a hard dependency

        client = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class OpenAIClient(_HttpProvider):
    provider_name = "openai"
    key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    default_model = "gpt-4o-mini"

    def _raw_call(self, key: str, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: not a hard dependency

        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class GeminiClient(_HttpProvider):
    provider_name = "gemini"
    key_env = "GEMINI_API_KEY"
    model_env = "GEMINI_MODEL"
    default_model = "gemini-1.5-flash"

    def _raw_call(self, key: str, system: str, user: str) -> str:
        import requests  # lazy: not a hard dependency

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            url, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

_CLIENTS: dict[str, type[_HttpProvider]] = {
    "groq": GroqClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def get_llm(provider: str | None = None) -> LLMClient:
    """Return the configured client, defaulting to the offline stub.

    An unknown provider name warns once and falls back rather than raising, so a
    typo in ``.env`` cannot take the app down.
    """
    chosen = (provider or config.llm_provider()).strip().lower()
    if chosen in ("", "stub"):
        return StubLLM()
    client_cls = _CLIENTS.get(chosen)
    if client_cls is None:
        logger.warning("unknown LLM_PROVIDER %r; using stub", chosen)
        return StubLLM()
    return client_cls()
