"""Load config.yaml once and resolve paths relative to the repo root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg(*keys: str, default=None):
    """cfg("llm", "vision_model") -> value, or default if any key is missing."""
    node = load_config()
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in config.yaml."""
    return ROOT / cfg("paths", key)


@lru_cache(maxsize=None)
def load_yaml(key: str) -> dict:
    """Load one of the small data/*.yaml files named in `paths:`."""
    p = path(key)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def secret(name: str) -> str | None:
    """Environment variable first, then Streamlit secrets when running in the app."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None
