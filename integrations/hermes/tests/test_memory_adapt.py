"""Unit tests for the Hermes KB adapter (no zvec required)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory_adapt as ma  # noqa: E402


def entry(id_, content="", category="general", title="", trust=1.0, hits=0,
          updated=None, kind=0, vector=None, deprecated=0, scene=0):
    return {
        "id": id_,
        "fields": {
            "content": content,
            "category": category,
            "title": title or content[:20],
            "trust_score": trust,
            "hit_count": hits,
            "updated_at": updated if updated is not None else 1_700_000_000,
            "kind": kind,
            "deprecated": deprecated,
            "scene": scene,
        },
        "vector": vector,
    }


class TestKinds(unittest.TestCase):
    def test_rule_beats_user_prefix(self):
        self.assertEqual(ma.infer_kind("rules", "", "生产容器必须用 compose"), ma.KIND_RULE)

    def test_preference_from_category(self):
        self.assertEqual(ma.infer_kind("user", "", "喜欢简洁的回答"), ma.KIND_PREFERENCE)

    def test_lesson_from_title_tag(self):
        self.assertEqual(ma.infer_kind("memory", "[coding-exp] 子代理死法", "x"), ma.KIND_LESSON)

    def test_task(self):
        self.assertEqual(ma.infer_kind("memory", "", "任务: 修复登录 | 结果: 成功"), ma.KIND_TASK)

    def test_default_fact(self):
        self.assertEqual(ma.infer_kind("services", "demo-svc", "容器 192.0.2.25"), ma.KIND_FACT)

    def test_query_intent_preference(self):
        self.assertIn(ma.KIND_PREFERENCE, ma.query_kinds("用户喜欢什么回答风格"))

    def test_query_intent_rule(self):
        self.assertIn(ma.KIND_RULE, ma.query_kinds("部署铁律是什么"))


class TestText(unittest.TestCase):
    def test_fingerprint_ignores_punctuation(self):
        a = "Docker 端口映射：192.0.2.18"
        b = "docker端口映射 192.0.2.18"
        self.assertEqual(ma.content_fingerprint(a), ma.content_fingerprint(b))

    def test_pointer_detection(self):
        fields = {"title": "- - 指针头: 功能投入判据",
                  "content": "- 功能投入判据 →KB: kb搜'功能'\n- 容器铁律 →KB: kb搜'容器'"}
        self.assertTrue(ma.is_pointer_doc(fields))

    def test_normal_note_not_pointer(self):
        fields = {"title": "nginx 配置", "content": "upstream -> 192.0.2.33:9081"}
        self.assertFalse(ma.is_pointer_doc(fields))


class TestFusion(unittest.TestCase):
    def test_rrf_prefers_both_lists(self):
        a = type("D", (), {"id": "a"})()
        b = type("D", (), {"id": "b"})()
        c = type("D", (), {"id": "c"})()
        fused = ma.rrf_fuse([a, b], [a, c])
        self.assertGreater(fused["a"]["relevance"], fused["b"]["relevance"])
        self.assertGreater(fused["a"]["relevance"], fused["c"]["relevance"])
        self.assertEqual(fused["a"]["vector_rank"], 1)
        self.assertEqual(fused["a"]["fts_rank"], 1)

    def test_rrf_empty(self):
        self.assertEqual(ma.rrf_fuse([], []), {})


class TestScoring(unittest.TestCase):
    def test_trust_has_floor(self):
        old = 1_500_000_000
        v = ma.trust_with_floor(1.0, old, now=1_800_000_000)
        self.assertGreaterEqual(v, 0.5)

    def test_recency_decreases(self):
        now = 1_800_000_000
        self.assertGreater(ma.recency_score(now - 86400, now=now),
                           ma.recency_score(now - 86400 * 365, now=now))

    def test_kind_match_boosts(self):
        fields = {"trust_score": 1.0, "updated_at": 1_800_000_000,
                  "hit_count": 0, "kind": ma.KIND_PREFERENCE}
        matched = ma.score_memory(fields, 0.8, query="用户喜欢什么风格", now=1_800_000_000)
        unmatched = ma.score_memory(fields, 0.8, query="nginx 端口", now=1_800_000_000)
        self.assertGreater(matched["score"], unmatched["score"])

    def test_dedupe_results(self):
        rows = [
            {"content": "nginx upstream demo_backend -> 192.0.2.33:9081 配置错误"},
            {"content": "nginx upstream demo_backend -> 192.0.2.33:9081 配置错误!!"},
            {"content": "完全不同的内容 docker compose 部署"},
        ]
        out = ma.dedupe_results(rows, threshold=0.85)
        self.assertEqual(len(out), 2)


class TestConsolidation(unittest.TestCase):
    @staticmethod
    def vec(x, y):
        return np.asarray([x, y, 0.0], dtype=np.float32)

    def test_cluster_same_category(self):
        rows = [
            entry("a", "docker 端口映射", "infrastructure", vector=self.vec(1, 0)),
            entry("b", "docker 端口映射说明", "infrastructure", vector=self.vec(0.99, 0.05)),
            entry("c", "完全无关", "user", vector=self.vec(0, 1)),
        ]
        clusters = ma.duplicate_clusters(rows, threshold=0.95)
        self.assertEqual(len(clusters), 1)
        self.assertEqual({rows[i]["id"] for i in clusters[0]}, {"a", "b"})

    def test_cross_category_not_merged_by_default(self):
        rows = [
            entry("a", "docker 端口映射", "infrastructure", vector=self.vec(1, 0)),
            entry("b", "docker 端口映射", "memory", vector=self.vec(1, 0)),
        ]
        self.assertEqual(ma.duplicate_clusters(rows, threshold=0.95), [])
        self.assertEqual(len(ma.duplicate_clusters(rows, threshold=0.95, same_category=False)), 1)

    def test_merge_plan_keeps_rich_doc(self):
        rows = [
            entry("a", "短内容", "infrastructure", trust=1.0, hits=0),
            entry("b", "更长的内容, 包含端口 192.0.2.18 和路径 /srv/demo", "infrastructure",
                  trust=0.8, hits=5),
        ]
        plan = ma.merge_plan([0, 1], rows)
        self.assertEqual(plan["keep_id"], "a")  # highest trust wins the slot
        self.assertEqual(plan["drop_ids"], ["b"])
        self.assertEqual(plan["keep_fields"]["hit_count"], 5)
        self.assertIn("192.0.2.18", plan["keep_fields"]["content"])


class TestForgetting(unittest.TestCase):
    def test_pointer_always_selected(self):
        rows = [entry("p", "- 指针头 →KB: kb搜'x'", title="指针头", updated=1_700_000_000)]
        out = ma.select_forgetting(rows, now=1_800_000_000)
        self.assertEqual(out[0]["reason"], "pointer_artifact")

    def test_high_trust_never_selected_by_age(self):
        rows = [entry("a", "生产配置", "infrastructure", trust=1.0, updated=1_500_000_000)]
        self.assertEqual(ma.select_forgetting(rows, now=1_800_000_000), [])

    def test_low_trust_stale_selected(self):
        rows = [entry("a", "过时快照", "memory", trust=0.2, updated=1_500_000_000)]
        out = ma.select_forgetting(rows, now=1_800_000_000)
        self.assertEqual(out[0]["reason"], "stale_low_trust")


class TestHealth(unittest.TestCase):
    def test_health_report_shape(self):
        rows = [
            entry("a", "docker 端口", "infrastructure", vector=np.array([1.0, 0, 0], dtype=np.float32)),
            entry("b", "docker 端口", "infrastructure", vector=np.array([1.0, 0, 0], dtype=np.float32)),
            entry("c", "x", "memory", vector=np.array([0, 1.0, 0], dtype=np.float32)),
        ]
        rep = ma.health_report(rows, dup_threshold=0.95)
        self.assertEqual(rep["active"], 3)
        self.assertEqual(rep["duplicate_groups"], 1)
        self.assertIn("health_score", rep)


if __name__ == "__main__":
    unittest.main()
