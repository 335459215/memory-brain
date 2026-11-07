"""Hermes KB adapter — memory-brain ideas adapted to a zvec knowledge base.

The upstream project (`memory-brain`) models long-term memory as *typed*
records (episode / fact / preference / relation / skill) and recalls them with
a multi-signal score: semantic similarity, keyword match, importance, recency
and entity graph.  It also treats forgetting as a graduated process
(decay → archive → merge) instead of a single age cutoff.

This module keeps those ideas but stays storage-agnostic: every function takes
plain "views" (dicts with id/fields/vector) so it can be unit-tested without
zvec.  `integrations/hermes/README.md` explains how the MCP server wires them.

Design rules borrowed from memory-brain:
1. Hybrid recall must fuse *ranks*, not raw scores.  Cosine similarity and
   BM25 live on different scales; max() silently turns hybrid search into
   "whichever scorer returns bigger numbers".
2. Stable knowledge must not rot just because it is old.  Trust describes how
   well a fact was verified; recency is a separate, much weaker signal.
3. Duplicates are merged only above a high similarity threshold and inside the
   same category by default.  Merges are reversible (soft-deprecate losers).
4. Forgetting is conservative: navigation artefacts and low-trust stale notes
   go first; verified rules/infrastructure never expire merely by age.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# Memory kinds (memory-brain's five kinds + two Hermes-specific ones)
# --------------------------------------------------------------------------- #

KIND_UNSPECIFIED = 0
KIND_FACT = 1
KIND_EPISODE = 2
KIND_PREFERENCE = 3
KIND_LESSON = 4
KIND_RULE = 5
KIND_PROFILE = 6
KIND_TASK = 7

KIND_NAMES = {
    KIND_UNSPECIFIED: "",
    KIND_FACT: "fact",
    KIND_EPISODE: "episode",
    KIND_PREFERENCE: "preference",
    KIND_LESSON: "lesson",
    KIND_RULE: "rule",
    KIND_PROFILE: "profile",
    KIND_TASK: "task",
}
KIND_CODES = {v: k for k, v in KIND_NAMES.items() if v}
VALID_KINDS = tuple(KIND_CODES)

# Categories whose content is durable by nature.  Used by the health report
# and by the conservative forgetting policy.
DURABLE_CATEGORIES = {
    "rules", "infrastructure", "services", "development", "architecture",
    "network", "newapi",
}


def kind_name(code: Any) -> str:
    try:
        return KIND_NAMES.get(int(code), "")
    except (TypeError, ValueError):
        return ""


def kind_code(kind: Any) -> int:
    if isinstance(kind, int):
        return kind if kind in KIND_NAMES else KIND_UNSPECIFIED
    return KIND_CODES.get(str(kind or "").strip().lower(), KIND_UNSPECIFIED)


def infer_kind(category: str = "", title: str = "", content: str = "",
               scene: str = "") -> int:
    """Infer a memory kind from metadata.  Deterministic, no model call.

    Rules are intentionally narrow: only explicit normative documents or
    sentences that *start* with a normative marker.  A casual note that merely
    mentions "必须" somewhere deep in the text should stay a fact.
    """
    cat = str(category or "").strip().lower()
    title_l = str(title or "").strip().lower()
    head = str(content or "")[:200]
    text = f"{title or ''}\n{head}"
    low = text.lower()
    head_low = head.lower()

    if cat == "rules" or any(title_l.startswith(w) for w in ("铁律", "规则", "约定", "规范")):
        return KIND_RULE
    if any(w in head_low[:120] for w in (
        "必须", "禁止", "不允许", "只能", "一定要", "严禁", "铁律", "硬约束",
    )):
        return KIND_RULE
    if title_l.startswith("[") and ("-exp]" in title_l[:20] or "-lesson]" in title_l[:20]):
        return KIND_LESSON
    if any(w in head_low[:160] for w in (
        "经验", "教训", "踩坑", "踩雷", "最佳实践", "复用", "根因", "复盘",
        "坑：", "坑:", "的做法是",
    )):
        return KIND_LESSON
    if cat == "user" or any(w in head_low[:120] for w in (
        "偏好", "喜欢", "讨厌", "习惯", "风格", "口味", "介意", "接受不了",
        "用户要求", "用户原话",
    )):
        return KIND_PREFERENCE
    if cat == "profile" or any(w in head_low[:80] for w in ("画像", "个人档案")):
        return KIND_PROFILE
    if any(w in head_low[:120] for w in ("任务:", "结果:", "进度", "已完成", "待办", "交付")):
        return KIND_TASK
    if any(w in head_low[:160] for w in (
        "今天", "昨天", "上次", "那天", "当时", "之前", "后来", "发生", "聊天", "对话",
    )):
        return KIND_EPISODE
    return KIND_FACT


# --------------------------------------------------------------------------- #
# Query intent → kind preference (memory-brain INTENT_HINTS, Chinese version)
# --------------------------------------------------------------------------- #

_INTENT_HINTS: dict[int, tuple[str, ...]] = {
    KIND_PREFERENCE: (
        "偏好", "喜欢", "讨厌", "习惯", "风格", "口味", "介意", "接受", "想要",
    ),
    KIND_RULE: (
        "规则", "铁律", "必须", "禁止", "允许", "要求", "约定", "规范", "约束",
    ),
    KIND_LESSON: (
        "经验", "教训", "坑", "怎么做", "怎么弄", "方法", "最佳实践", "踩雷", "复用",
    ),
    KIND_EPISODE: (
        "上次", "那天", "当时", "以前", "记得", "发生过", "昨天", "今天", "之前",
    ),
    KIND_TASK: ("任务", "进度", "做完", "提交", "交付", "处理完"),
    KIND_FACT: (
        "是什么", "在哪", "端口", "地址", "配置", "版本", "多少", "谁", "怎么",
        "如何", "为什么", "路径", "命令",
    ),
}


def query_kinds(query: str) -> set[int]:
    q = str(query or "")
    out: set[int] = set()
    for kind, words in _INTENT_HINTS.items():
        if any(w in q for w in words):
            out.add(kind)
    return out


# --------------------------------------------------------------------------- #
# Text normalisation / fingerprints
# --------------------------------------------------------------------------- #

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lowercase and drop whitespace/punctuation (Chinese-friendly)."""
    return _STRIP_RE.sub("", str(text or "")).lower()


def content_fingerprint(text: str) -> str:
    """Stable fingerprint tolerant to punctuation/whitespace changes."""
    import hashlib

    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()[:12]


def bigrams(text: str) -> set[str]:
    t = normalize_text(text)
    if len(t) <= 1:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def bigram_similarity(a: str, b: str) -> float:
    ga, gb = bigrams(a), bigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


# --------------------------------------------------------------------------- #
# Navigation-artefact detection (a real Hermes KB failure mode)
# --------------------------------------------------------------------------- #

_POINTER_TITLE_PREFIXES = ("指针头", "kb索引", "kb导航索引", "kb 导航索引", "导航索引")


def is_pointer_doc(fields: dict) -> bool:
    """Detect kb_compact pointer/navigation files that must not be search hits.

    Those files are generated *from* the KB (topic → `kb搜'…'` lines).  When
    compacted again they recursively grow ("- - - 指针头 …") and their many
    keywords make them outrank real memories.  They are derived artefacts,
    not knowledge.

    Beware: old MEMORY.md/USER.md snapshots also start with "← …", but they
    often contain substantive profile text.  Only treat a "←" doc as pointer
    junk when its body is mostly navigation lines / very short.
    """
    title = str(fields.get("title") or "")
    content = str(fields.get("content") or "")
    title_clean = re.sub(r"^[-\s]+", "", title).strip().lower()
    if any(title_clean.startswith(p) for p in _POINTER_TITLE_PREFIXES):
        return True
    if "已由 kb_compact 全部迁入KB" in content[:160] and len(content) < 300:
        return True
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if lines:
        pointer_lines = sum(1 for ln in lines if "→KB: kb搜" in ln or "→KB:" in ln)
        if pointer_lines >= 3 and pointer_lines / len(lines) >= 0.5:
            return True
    return False


def is_reasoning_pollution(fields: dict) -> bool:
    """Detect model reasoning drafts accidentally stored as knowledge."""
    text = str(fields.get("content") or "")
    head = text[:400].lower()
    markers = (
        "let me analyze", "the original file contains", "i need to compress",
        "let me think", "we need to compress", "the user wants me to",
    )
    return any(m in head for m in markers)



# --------------------------------------------------------------------------- #
# Hybrid retrieval: Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #

RRF_K = 60


def rrf_fuse(
    vector_docs: Iterable[Any],
    fts_docs: Iterable[Any],
    *,
    k: int = RRF_K,
) -> dict[str, dict[str, float | int]]:
    """Fuse two ranked result lists with Reciprocal Rank Fusion.

    Returns ``{doc_id: {rrf, relevance, vector_rank, fts_rank}}`` where
    ``relevance`` is min-max normalised to 0..1 (best fused hit = 1.0).
    Items are objects with an ``id`` attribute (zvec Doc) or dicts with ``id``.
    """
    def _id(item: Any) -> str:
        return str(item["id"] if isinstance(item, dict) else item.id)

    fused: dict[str, dict[str, float | int]] = {}

    def _add(items: Iterable[Any], field: str) -> None:
        for rank, item in enumerate(items, 1):
            did = _id(item)
            row = fused.setdefault(did, {"rrf": 0.0, "vector_rank": 0, "fts_rank": 0})
            row["rrf"] = float(row["rrf"]) + 1.0 / (k + rank)
            row[field] = rank

    _add(vector_docs, "vector_rank")
    _add(fts_docs, "fts_rank")

    if not fused:
        return {}

    max_rrf = max(float(v["rrf"]) for v in fused.values()) or 1.0
    for row in fused.values():
        row["relevance"] = round(float(row["rrf"]) / max_rrf, 6)
    return fused


# --------------------------------------------------------------------------- #
# Composite scoring (memory-brain five-way blend, Hermes weights)
# --------------------------------------------------------------------------- #

WEIGHTS = {
    "relevance": 0.55,
    "trust": 0.15,
    "recency": 0.12,
    "hit": 0.08,
    "kind": 0.10,
}
RECENCY_HALF_LIFE_DAYS = 60.0
TRUST_DECAY_START_DAYS = 30
TRUST_DECAY_RATE = 0.02


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def recency_score(updated_at: Any, now: float | None = None,
                  half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    ts = _f(updated_at, 0.0)
    if ts <= 0:
        return 0.5
    age_days = max(0.0, ((now or time.time()) - ts) / 86400.0)
    return 0.5 ** (age_days / max(1.0, half_life_days))


def hit_score(hit_count: Any) -> float:
    """Diminishing returns; unlike a linear boost this doesn't lock in winners."""
    n = max(0, _i(hit_count))
    if n <= 0:
        return 0.0
    return min(1.0, 1.0 - 0.5 ** min(n, 10))


def trust_with_floor(trust: Any, updated_at: Any, now: float | None = None) -> float:
    """Trust decays for age but never below 50% of its verified value.

    The old KB multiplied trust by a factor that reached 0, so a two-month-old
    verified fact looked like garbage.  Trust is evidence, not freshness.
    """
    t = max(0.0, min(1.0, _f(trust, 1.0)))
    ts = _f(updated_at, 0.0)
    if ts <= 0:
        return t
    age_days = max(0.0, ((now or time.time()) - ts) / 86400.0)
    if age_days <= TRUST_DECAY_START_DAYS:
        return t
    decay = max(0.0, 1.0 - (age_days - TRUST_DECAY_START_DAYS) * TRUST_DECAY_RATE)
    return round(t * (0.5 + 0.5 * decay), 4)


def score_memory(
    fields: dict,
    relevance: float,
    *,
    query: str = "",
    now: float | None = None,
    scene_boost: float = 0.0,
) -> dict[str, Any]:
    """Return final score plus an explainable breakdown."""
    trust = trust_with_floor(fields.get("trust_score", 1.0), fields.get("updated_at"), now)
    rec = recency_score(fields.get("updated_at"), now)
    hit = hit_score(fields.get("hit_count", 0))
    kind = _i(fields.get("kind", 0))
    intents = query_kinds(query)
    if not intents:
        kind_comp = 0.5
        kind_match = None
    elif kind in intents:
        kind_comp = 1.0
        kind_match = True
    else:
        kind_comp = 0.0
        kind_match = False

    final = (
        WEIGHTS["relevance"] * max(0.0, min(1.0, _f(relevance)))
        + WEIGHTS["trust"] * trust
        + WEIGHTS["recency"] * rec
        + WEIGHTS["hit"] * hit
        + WEIGHTS["kind"] * kind_comp
        + scene_boost
    )
    return {
        "score": round(min(1.0, final), 4),
        "why": {
            "relevance": round(max(0.0, min(1.0, _f(relevance))), 3),
            "trust": round(trust, 3),
            "recency": round(rec, 3),
            "hit": round(hit, 3),
            "kind": kind_name(kind) or None,
            "kind_match": kind_match,
            "scene_boost": round(scene_boost, 3),
        },
    }


def dedupe_results(rows: list[dict], threshold: float = 0.88,
                   text_key: str = "content") -> list[dict]:
    """Drop near-identical search hits, keeping the higher-ranked one.

    Compared on both content and title: two notes can share the same title
    but add different details, and showing both wastes context.
    """
    kept: list[dict] = []
    for row in rows:
        text = str(row.get(text_key) or "")
        title = str(row.get("title") or "")
        is_dup = False
        for k in kept:
            ktext = str(k.get(text_key) or "")
            ktitle = str(k.get("title") or "")
            if (bigram_similarity(text, ktext) >= threshold
                    or (title and ktitle and bigram_similarity(title, ktitle) >= 0.95)):
                is_dup = True
                break
        if not is_dup:
            kept.append(row)
    return kept


# --------------------------------------------------------------------------- #
# Consolidation: find and merge near-duplicate memories
# --------------------------------------------------------------------------- #

def _as_view(item: Any) -> dict:
    if isinstance(item, dict):
        v = dict(item)
    else:
        v = {
            "id": str(getattr(item, "id", "")),
            "fields": dict(getattr(item, "fields", {}) or {}),
        }
        vectors = getattr(item, "vectors", None) or {}
        v["vector"] = vectors.get("embedding") if vectors else None
    v.setdefault("fields", {})
    v.setdefault("id", "")
    if "vector" not in v:
        v["vector"] = None
    return v


def _norm_rows(entries: Iterable[Any]) -> list[dict]:
    return [_as_view(e) for e in entries]


def duplicate_clusters(
    entries: Iterable[Any],
    *,
    threshold: float = 0.92,
    same_category: bool = True,
    max_docs: int = 6000,
) -> list[list[int]]:
    """Union-find clusters of near-duplicate active memories.

    ``entries`` are dicts (id, fields, vector) or zvec Docs.  Returns lists of
    indices into the normalised entry list.  Documents without vectors are
    skipped (they cannot be compared reliably).
    """
    rows = _norm_rows(entries)
    idx = [i for i, r in enumerate(rows)
           if _i(r["fields"].get("deprecated", 0)) < 1 and r.get("vector") is not None]
    idx = idx[:max_docs]
    if len(idx) < 2:
        return []

    mat = np.asarray([np.asarray(rows[i]["vector"], dtype=np.float32) for i in idx])
    norms = np.linalg.norm(mat, axis=1) + 1e-8
    mat = mat / norms[:, None]
    sim = mat @ mat.T
    np.fill_diagonal(sim, 0.0)

    parent = list(range(len(idx)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_i, pair_j = np.where(np.triu(sim, k=1) >= float(threshold))
    for a, b in zip(pair_i.tolist(), pair_j.tolist()):
        ra, rb = rows[idx[a]], rows[idx[b]]
        if same_category and str(ra["fields"].get("category", "")) != str(rb["fields"].get("category", "")):
            continue
        union(a, b)

    groups: dict[int, list[int]] = {}
    for a in range(len(idx)):
        groups.setdefault(find(a), []).append(idx[a])
    return [sorted(g) for g in groups.values() if len(g) > 1]


def merge_plan(cluster: list[int], entries: Iterable[Any]) -> dict[str, Any]:
    """Choose a survivor and describe the merge of one duplicate cluster."""
    rows = _norm_rows(entries)

    def richness(r: dict) -> tuple:
        f = r["fields"]
        return (
            _f(f.get("trust_score", 1.0), 1.0),
            _i(f.get("hit_count", 0)),
            len(str(f.get("content", ""))),
            _f(f.get("updated_at", 0)),
        )

    members = [rows[i] for i in cluster]
    survivor = max(members, key=richness)
    best_content = max((str(m["fields"].get("content", "")) for m in members), key=len)
    best_title = str(survivor["fields"].get("title", "")) or best_content[:30]
    trust = max(_f(m["fields"].get("trust_score", 1.0), 1.0) for m in members)
    hits = sum(_i(m["fields"].get("hit_count", 0)) for m in members)
    updated = max(_f(m["fields"].get("updated_at", 0)) for m in members)
    kind = next((_i(m["fields"].get("kind", 0)) for m in members if _i(m["fields"].get("kind", 0))), 0)
    scene = next((_i(m["fields"].get("scene", 0)) for m in members if _i(m["fields"].get("scene", 0))), 0)
    return {
        "keep_id": survivor["id"],
        "keep_fields": {
            "title": best_title,
            "content": best_content,
            "trust_score": round(min(1.0, trust), 3),
            "hit_count": hits,
            "updated_at": int(updated or time.time()),
            "kind": kind,
            "scene": scene,
        },
        "drop_ids": [m["id"] for m in members if m["id"] != survivor["id"]],
        "size": len(members),
    }


# --------------------------------------------------------------------------- #
# Forgetting: conservative, value-aware
# --------------------------------------------------------------------------- #

def select_forgetting(entries: Iterable[Any], *,
                      now: float | None = None,
                      stale_days: int = 365,
                      min_trust: float = 0.5) -> list[dict[str, Any]]:
    """Return docs that may be auto-deprecated.

    Rules:
    - pointer/navigation artefacts always go;
    - otherwise only stale (> stale_days), low-trust (< min_trust), never-hit
      documents go.  Verified rules and infrastructure are never selected
      merely for age (their trust/hit history protects them).
    """
    now = now or time.time()
    out: list[dict[str, Any]] = []
    for entry in _norm_rows(entries):
        f = entry["fields"]
        if _i(f.get("deprecated", 0)) >= 1:
            continue
        if is_pointer_doc(f):
            out.append({"id": entry["id"], "reason": "pointer_artifact"})
            continue
        if is_reasoning_pollution(f):
            out.append({"id": entry["id"], "reason": "reasoning_pollution"})
            continue
        age_days = max(0.0, (now - _f(f.get("updated_at", 0))) / 86400.0)
        trust = _f(f.get("trust_score", 1.0), 1.0)
        hits = _i(f.get("hit_count", 0))
        if age_days > stale_days and trust < min_trust and hits == 0:
            out.append({"id": entry["id"], "reason": "stale_low_trust"})
    return out


# --------------------------------------------------------------------------- #
# Health report
# --------------------------------------------------------------------------- #

def health_report(
    entries: Iterable[Any],
    *,
    now: float | None = None,
    dup_threshold: float = 0.92,
    stale_days: int = 365,
) -> dict[str, Any]:
    now = now or time.time()
    rows = _norm_rows(entries)
    active = [r for r in rows if _i(r["fields"].get("deprecated", 0)) < 1]
    deprecated = len(rows) - len(active)

    categories: dict[str, int] = {}
    kinds: dict[str, int] = {}
    missing_kind = 0
    pointer_ids: list[str] = []
    reasoning_ids: list[str] = []
    oversized: list[str] = []
    no_updated = 0
    total_chars = 0
    for r in active:
        f = r["fields"]
        cat = str(f.get("category", "general") or "general")
        categories[cat] = categories.get(cat, 0) + 1
        kn = kind_name(f.get("kind", 0)) or "unspecified"
        kinds[kn] = kinds.get(kn, 0) + 1
        if not _i(f.get("kind", 0)):
            missing_kind += 1
        if is_pointer_doc(f):
            pointer_ids.append(r["id"])
        if is_reasoning_pollution(f):
            reasoning_ids.append(r["id"])
        length = len(str(f.get("content", "")))
        total_chars += length
        if length > 5000:
            oversized.append(r["id"])
        if _f(f.get("updated_at", 0)) <= 0:
            no_updated += 1

    clusters = duplicate_clusters(active, threshold=dup_threshold)
    dup_docs = sum(len(c) for c in clusters)
    forget = select_forgetting(active, now=now, stale_days=stale_days)

    issues: list[dict[str, Any]] = []
    if pointer_ids:
        issues.append({"level": "warn", "code": "pointer_artifacts",
                       "message": f"{len(pointer_ids)} 条指针/导航文件混在知识库里, 会霸占搜索结果",
                       "ids": pointer_ids[:20]})
    if reasoning_ids:
        issues.append({"level": "warn", "code": "reasoning_pollution",
                       "message": f"{len(reasoning_ids)} 条模型推理草稿被当成知识入库",
                       "ids": reasoning_ids[:20]})
    if clusters:
        issues.append({"level": "warn", "code": "near_duplicates",
                       "message": f"{len(clusters)} 组近重复, 涉及 {dup_docs} 条 (阈值 {dup_threshold})",
                       "groups": len(clusters)})
    if missing_kind:
        issues.append({"level": "info", "code": "missing_kind",
                       "message": f"{missing_kind} 条未标注记忆类型, 运行 kb_reindex 可回填"})
    if forget:
        issues.append({"level": "info", "code": "forget_candidates",
                       "message": f"{len(forget)} 条可安全废弃 (指针或长期低信任未命中)"})
    if oversized:
        issues.append({"level": "info", "code": "oversized",
                       "message": f"{len(oversized)} 条超过 5000 字, 建议拆分"})

    # Simple health score: start at 100, subtract meaningful problems.
    score = 100
    score -= min(25, len(pointer_ids) * 2)
    score -= min(10, len(reasoning_ids) * 5)
    score -= min(30, len(clusters) * 3)
    score -= min(10, missing_kind // 50)
    score = max(0, score)

    return {
        "total": len(rows),
        "active": len(active),
        "deprecated": deprecated,
        "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
        "kinds": kinds,
        "missing_kind": missing_kind,
        "pointer_artifacts": len(pointer_ids),
        "reasoning_pollution": len(reasoning_ids),
        "duplicate_groups": len(clusters),
        "duplicate_docs": dup_docs,
        "forget_candidates": len(forget),
        "oversized": len(oversized),
        "no_updated_at": no_updated,
        "avg_chars": round(total_chars / max(1, len(active)), 1),
        "health_score": score,
        "issues": issues,
    }
