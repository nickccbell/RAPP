"""
llm_backends.py — pluggable LLM backend adapters for RAPP Brainstem.

All backends expose:
    call(messages, tools=None, model=None)
    → OpenAI-compatible response dict:
      {"choices": [{"message": {"role": "assistant", "content": ..., "tool_calls": [...]},
                    "finish_reason": ...}]}

Factory:
    get_llm_backend(name=None) → backend instance
    set_runtime_backend(name, model=None) → update in-memory override
    get_active_backend_name() → str
    get_backend_status() → list[dict]
"""

import os
import json
import requests

# ── Base ──────────────────────────────────────────────────────────────────────

class LLMBackend:
    name = "base"

    def call(self, messages, tools=None, model=None):
        raise NotImplementedError

# ── GitHub Copilot ────────────────────────────────────────────────────────────

class GitHubCopilotBackend(LLMBackend):
    name = "github_copilot"

    def call(self, messages, tools=None, model=None):
        # Deferred import to avoid circular import at module load time.
        # brainstem is fully initialised before any .call() is ever invoked.
        import brainstem as bs

        active_model = model or os.getenv("GITHUB_MODEL", "gpt-4o")
        copilot_token, endpoint = bs.get_copilot_token()

        url = f"{endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {copilot_token}",
            "Content-Type": "application/json",
            "Editor-Version": "vscode/1.95.0",
            "Copilot-Integration-Id": "vscode-chat",
        }
        body = {"model": active_model, "messages": messages}
        if tools:
            body["tools"] = tools
            if active_model not in bs._NO_TOOL_CHOICE_MODELS:
                body["tool_choice"] = "auto"

        print(f"[llm/copilot] model={active_model}, tools={len(tools) if tools else 0}, "
              f"tool_choice={body.get('tool_choice', 'NONE')}")

        resp = requests.post(url, headers=headers, json=body, timeout=60)

        # Fallback through other available models on transient errors
        if resp.status_code in (400, 429, 500, 502, 503):
            tried = {active_model}
            fallback_ids = [m["id"] for m in bs.AVAILABLE_MODELS if m["id"] != active_model]
            for fb in fallback_ids:
                if fb in tried:
                    continue
                tried.add(fb)
                print(f"[llm/copilot] Retrying with {fb}...")
                body["model"] = fb
                if fb in bs._NO_TOOL_CHOICE_MODELS:
                    body.pop("tool_choice", None)
                elif tools and "tool_choice" not in body:
                    body["tool_choice"] = "auto"
                resp = requests.post(url, headers=headers, json=body, timeout=60)
                if resp.status_code == 200:
                    break
                print(f"[llm/copilot] {fb} also failed ({resp.status_code})")

        if resp.status_code != 200:
            error_detail = resp.text[:500] if resp.text else "No details"
            bs._tlog("api.error", {"model": active_model, "status": resp.status_code,
                                   "detail": error_detail[:300]}, level="error")
            print(f"[llm/copilot] API error {resp.status_code}: {error_detail}")

        resp.raise_for_status()
        result = resp.json()

        # Merge multi-choice responses (Claude via Copilot API can split text and
        # tool_calls into separate choices — normalise to a single choice).
        choices = result.get("choices", [])
        if len(choices) > 1:
            merged = {"role": "assistant", "content": None, "tool_calls": []}
            for c in choices:
                m = c.get("message", {})
                if m.get("content"):
                    merged["content"] = (merged["content"] or "") + m["content"]
                if m.get("tool_calls"):
                    merged["tool_calls"].extend(m["tool_calls"])
            if not merged["tool_calls"]:
                del merged["tool_calls"]
            fr = "tool_calls" if merged.get("tool_calls") else choices[0].get("finish_reason", "stop")
            result["choices"] = [{"message": merged, "finish_reason": fr}]

        # Debug logging
        choice = result.get("choices", [{}])[0]
        msg    = choice.get("message", {})
        fr     = choice.get("finish_reason", "")
        has_tools = bool(msg.get("tool_calls"))
        print(f"[llm/copilot] finish_reason={fr}, has_tool_calls={has_tools}, "
              f"content_len={len(msg.get('content') or '')}")
        if has_tools:
            print(f"[llm/copilot]   tool_calls: "
                  f"{[tc.get('function', {}).get('name', '?') for tc in msg['tool_calls']]}")

        return result


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIBackend(LLMBackend):
    name = "openai"

    def call(self, messages, tools=None, model=None):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

        base_url     = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        active_model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {"model": active_model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        print(f"[llm/openai] model={active_model}, tools={len(tools) if tools else 0}")
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicBackend(LLMBackend):
    name = "anthropic"

    # ── Schema converters ─────────────────────────────────────────────────────

    def _tools_to_anthropic(self, tools):
        """Convert OpenAI tool schema → Anthropic tool schema (input_schema)."""
        result = []
        for t in (tools or []):
            fn = t.get("function", {})
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _messages_to_anthropic(self, messages):
        """Convert OpenAI-format messages → Anthropic messages + system string."""
        system_parts = []
        chat = []

        for m in messages:
            role = m.get("role", "")

            if role == "system":
                system_parts.append(m.get("content", ""))

            elif role == "tool":
                # tool result → user message with tool_result content block
                chat.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": str(m.get("content", "")),
                    }],
                })

            elif role == "assistant" and m.get("tool_calls"):
                # assistant tool_calls → assistant message with tool_use blocks
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                chat.append({"role": "assistant", "content": blocks})

            else:
                # Plain user or assistant message — pass through
                chat.append({"role": role, "content": m.get("content", "")})

        return "\n".join(system_parts), chat

    def _response_to_openai(self, response):
        """Normalise Anthropic response → OpenAI-compatible dict."""
        content_blocks = response.get("content", [])
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        content      = "".join(text_parts) if text_parts else None
        stop_reason  = response.get("stop_reason", "end_turn")
        finish_reason = ("tool_calls" if tool_calls
                         else ("stop" if stop_reason in ("end_turn", "max_tokens") else stop_reason))

        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        return {"choices": [{"message": msg, "finish_reason": finish_reason}]}

    # ── call ──────────────────────────────────────────────────────────────────

    def call(self, messages, tools=None, model=None):
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

        active_model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        system_str, chat_messages = self._messages_to_anthropic(messages)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": active_model,
            "max_tokens": 8096,
            "messages": chat_messages,
        }
        if system_str:
            body["system"] = system_str
        if tools:
            body["tools"] = self._tools_to_anthropic(tools)

        print(f"[llm/anthropic] model={active_model}, tools={len(tools) if tools else 0}")
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return self._response_to_openai(resp.json())


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    name = "ollama"

    def call(self, messages, tools=None, model=None):
        host         = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        active_model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        url          = f"{host}/v1/chat/completions"

        body = {"model": active_model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        print(f"[llm/ollama] model={active_model}, tools={len(tools) if tools else 0}, host={host}")

        try:
            resp = requests.post(url, json=body, timeout=120)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama is not running at {host}. "
                "Start it with:  ollama serve"
            )

        # Some older Ollama builds return 404 when tools are included;
        # fall back to a tool-free request so we at least get a text reply.
        if resp.status_code == 404 and tools:
            print(f"[llm/ollama] 404 with tools — retrying without tools (model may not support them)")
            body_notool = {"model": active_model, "messages": messages}
            resp = requests.post(url, json=body_notool, timeout=120)

        resp.raise_for_status()
        return resp.json()


# ── Factory & runtime state ───────────────────────────────────────────────────

_runtime_backend: str | None = None
_runtime_model:   str | None = None


def get_llm_backend(backend_name=None) -> LLMBackend:
    """Return the appropriate backend instance.

    Priority: explicit arg → runtime override → LLM_BACKEND env var → github_copilot
    """
    name = (backend_name or _runtime_backend or
            os.getenv("LLM_BACKEND", "github_copilot")).lower()
    if name == "openai":
        return OpenAIBackend()
    if name == "anthropic":
        return AnthropicBackend()
    if name == "ollama":
        return OllamaBackend()
    return GitHubCopilotBackend()


def get_active_backend_name() -> str:
    return (_runtime_backend or os.getenv("LLM_BACKEND", "github_copilot")).lower()


def get_runtime_model() -> str | None:
    return _runtime_model


def set_runtime_backend(backend: str, model: str | None = None):
    """Update in-memory backend / model overrides (survives until process restart)."""
    global _runtime_backend, _runtime_model
    _runtime_backend = backend.lower()
    if model is not None:
        _runtime_model = model
    print(f"[llm_backends] Runtime backend → {_runtime_backend}"
          + (f", model → {_runtime_model}" if _runtime_model else ""))


VALID_BACKENDS = {"github_copilot", "openai", "anthropic", "ollama"}


def get_backend_status() -> list:
    """Return every backend with availability flags (based on env vars)."""
    return [
        {
            "id":          "github_copilot",
            "name":        "GitHub Copilot",
            "configured":  True,           # auth is checked lazily at call time
            "note":        "Requires GitHub Copilot subscription + gh auth login",
        },
        {
            "id":          "openai",
            "name":        "OpenAI",
            "configured":  bool(os.getenv("OPENAI_API_KEY")),
            "model":       os.getenv("OPENAI_MODEL", "gpt-4o"),
            "base_url":    os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        },
        {
            "id":          "anthropic",
            "name":        "Anthropic",
            "configured":  bool(os.getenv("ANTHROPIC_API_KEY")),
            "model":       os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        },
        {
            "id":          "ollama",
            "name":        "Ollama (local)",
            "configured":  True,           # always reachable if `ollama serve` is running
            "host":        os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "model":       os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        },
    ]
