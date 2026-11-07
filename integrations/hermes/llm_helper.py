#!/usr/bin/env python3
"""LLM helper for KB maintenance — reads Hermes config.yaml + .env to auto-follow
the user's current chat model, no manual config needed.

Resolves ${VAR} references in config.yaml against ~/.hermes/.env, then calls
the OpenAI-compatible API (works with any provider Hermes supports via custom endpoint).

Automatically follows whatever model the user has configured in config.yaml.
When the user changes their chat model, this helper picks it up on next call (60s cache)."""
import os, re, yaml, httpx, sys, threading
from pathlib import Path

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
_CONFIG_PATH = _HERMES_HOME / "config.yaml"
_ENV_PATH = _HERMES_HOME / ".env"

_cache = None
_cache_time = 0
_cache_lock = threading.Lock()


def _load_env_file() -> dict:
    """Load ~/.hermes/.env into a dict (KEY=VALUE lines).
    Handles quoted values with embedded = signs correctly."""
    env = {}
    if not _ENV_PATH.exists():
        return env
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        # Strip surrounding quotes (single or double), handle values with embedded =
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        env[k] = v
    return env


def _resolve_env_vars(value: str, env_dict: dict) -> str:
    """Resolve ${VAR_NAME} references in a string using env dict."""
    def replacer(m):
        var_name = m.group(1)
        return env_dict.get(var_name, os.environ.get(var_name, m.group(0)))
    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replacer, value)


def get_model_config() -> dict:
    """Read Hermes config.yaml model section, resolve env vars.
    Returns {model, provider, base_url, api_key, max_tokens} or {} if unavailable.
    Cached for 60s — automatically picks up model changes. Thread-safe."""
    import time
    global _cache, _cache_time
    now = time.time()
    with _cache_lock:
        if _cache and now - _cache_time < 60:
            return _cache

    if not _CONFIG_PATH.exists():
        return {}

    try:
        config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    model_cfg = config.get("model", {})
    if not model_cfg:
        return {}

    env_dict = _load_env_file()

    result = {
        "model": model_cfg.get("default", ""),
        "provider": model_cfg.get("provider", "custom"),
        "base_url": _resolve_env_vars(str(model_cfg.get("base_url", "")), env_dict),
        "api_key": _resolve_env_vars(str(model_cfg.get("api_key", "")), env_dict),
        "max_tokens": model_cfg.get("max_tokens", 16384),
    }

    with _cache_lock:
        _cache = result
        _cache_time = now
    return result


def llm_summarize(system_prompt: str, user_prompt: str, max_tokens: int = 8192,
                  temperature: float = 0.3) -> str | None:
    """Call the Hermes-configured LLM (OpenAI-compatible) to summarize/compact text.
    Automatically uses whatever model the user has configured in config.yaml.
    Returns the LLM's text response, or None on failure.

    Reasoning models that return content=null (all tokens spent on
    reasoning_content) now return None — the reasoning chain must NEVER be
    used as output (it once polluted the KB with "Let me analyze..." drafts)."""
    cfg = get_model_config()
    if not cfg.get("base_url") or not cfg.get("api_key"):
        print("[LLM-Helper] No model config available (base_url or api_key missing)", file=sys.stderr, flush=True)
        return None

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    # Use generous max_tokens — reasoning models need room for both reasoning + output
    effective_max = max(max_tokens, 4096)
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": effective_max,
        "temperature": temperature,
        "stream": False,
    }

    try:
        # 2026-10-16: 120s → 45s，必须小于 MCP 客户端工具超时(60s)，否则手动 kb_compact 必超时
        with httpx.Client(timeout=45) as client:
            resp = client.post(
                f"{cfg['base_url']}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
            # Some reasoning models return content=null when they run out of tokens
            # (all tokens spent on reasoning_content). Use content first, fall back to reasoning.
            content = message.get("content")
            if content and len(content.strip()) > 0:
                return content
            # Fallback: reasoning_content — 2026-09-09 已禁用。
            # 之前 reasoning 草稿被当摘要写进 MEMORY.md → compact 拆段入库,
            # LLM 内部推理("Let me analyze...")污染 KB。content=null 一律视为失败。
            print(f"[LLM-Helper] Empty content (finish_reason={choice.get('finish_reason')}), "
                  f"refusing reasoning_content fallback to prevent KB pollution",
                  file=sys.stderr, flush=True)
            return None
    except httpx.HTTPStatusError as e:
        print(f"[LLM-Helper] HTTP {e.response.status_code}: {e.response.text[:200]}", file=sys.stderr, flush=True)
        return None
    except Exception as e:
        print(f"[LLM-Helper] LLM call failed: {e}", file=sys.stderr, flush=True)
        return None
