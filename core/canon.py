"""Map any free-text food name to a canonical food id (see data/aliases.yaml).

Every module keys on these ids: shelf life, storage rules, recipe matching and
evaluation. Matching order: exact alias -> fuzzy (rapidfuzz) -> embeddings (optional).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from core.config import cfg, path

_WORD = re.compile(r"[a-z]+")


def _singular(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    if w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def normalize(text: str) -> str:
    """'Cherry Tomatoes!' -> 'cherry tomato'"""
    return " ".join(_singular(w) for w in _WORD.findall(str(text).lower()))


@dataclass
class Match:
    id: str | None
    score: float          # 0-1
    method: str           # exact | fuzzy | embedding | none


class Canonicalizer:
    def __init__(self, aliases_file: str | Path | None = None, use_embeddings: bool | None = None):
        with open(aliases_file or path("aliases"), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.category: dict[str, str] = {}
        self._alias_to_id: dict[str, str] = {}
        for cid, info in data.items():
            info = info or {}
            self.category[cid] = info.get("category", "other")
            for alias in [cid.replace("_", " "), *info.get("aliases", [])]:
                self._alias_to_id.setdefault(normalize(alias), cid)
        self._aliases = list(self._alias_to_id)
        self.fuzzy_min = cfg("canon", "fuzzy_min", default=90)
        self.embedding_min = cfg("canon", "embedding_min", default=0.80)
        self.use_embeddings = (
            cfg("canon", "use_embeddings", default=False) if use_embeddings is None else use_embeddings
        )
        self._memo: dict[str, Match] = {}
        self._model = None
        self._emb = None

    @property
    def ids(self) -> list[str]:
        return list(self.category)

    def match(self, name: str) -> Match:
        q = normalize(name)
        if q not in self._memo:
            self._memo[q] = self._match(q)
        return self._memo[q]

    def _match(self, q: str) -> Match:
        if not q:
            return Match(None, 0.0, "none")
        if q in self._alias_to_id:
            return Match(self._alias_to_id[q], 1.0, "exact")
        best, best_score = None, 0.0
        for alias in self._aliases:
            s = fuzz.token_set_ratio(q, alias)
            # On ties prefer the longer alias ("tomato sauce" over "tomato").
            if s > best_score or (s == best_score and best and len(alias) > len(best)):
                best, best_score = alias, s
        if best and best_score >= self.fuzzy_min:
            return Match(self._alias_to_id[best], best_score / 100, "fuzzy")
        if self.use_embeddings:
            m = self._embedding_match(q)
            if m:
                return m
        return Match(None, best_score / 100, "none")

    def _embedding_match(self, q: str) -> Match | None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        if self._emb is None:
            self._model = SentenceTransformer(cfg("canon", "embedding_model", default="all-MiniLM-L6-v2"))
            self._emb = self._model.encode(self._aliases, normalize_embeddings=True)
        v = self._model.encode([q], normalize_embeddings=True)[0]
        sims = self._emb @ v
        i = int(sims.argmax())
        if sims[i] >= self.embedding_min:
            return Match(self._alias_to_id[self._aliases[i]], float(sims[i]), "embedding")
        return None


@lru_cache(maxsize=1)
def get_canon() -> Canonicalizer:
    return Canonicalizer()


def canonical_id(name: str) -> str | None:
    return get_canon().match(name).id
