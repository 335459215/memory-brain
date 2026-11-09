"""real 集失败形态诊断: 每条 query 输出期望文档排名 + 实际 top3, 按失败模式归类。
每跑用全新库副本 (hit_count 回写污染, 见 tools_eval_retrieval 注释)。"""
import os, shutil, sys, tempfile, time
sys.path.insert(0, "/root/hermes-kb")
import tools_eval_retrieval as E
import mcp_server as kb  # noqa: E402  (harness 同款导入)

cleanup = E.prepare_db()
kb.get_collection()
try:
    miss, low = [], []
    for query, expect in E.REAL_QUERIES:
        res = kb.kb_search(query, topk=5)
        rows = res["results"]
        r = E.rank_of(rows, expect)
        titles = [x["title"][:26] for x in rows[:3]]
        if r == 1:
            continue
        if r is None:
            miss.append((query, expect, titles))
        else:
            low.append((query, expect, r, titles))
    print(f"\n===== MISS (期望文档不在 top5, 共 {len(miss)}/20) =====")
    for q, e, t in miss:
        print(f"  Q: {q}")
        print(f"     期望含: {e!r}")
        for i, x in enumerate(t, 1):
            print(f"     top{i}: {x}")
    print(f"\n===== LOW (在 top5 但非第1, 共 {len(low)}/20) =====")
    for q, e, r, t in low:
        print(f"  Q: {q}  (期望 rank={r}, 含 {e!r})")
        for i, x in enumerate(t, 1):
            print(f"     top{i}: {x}")
    print(f"\n汇总: hit@1={20 - len(miss) - len(low)}/20, 需提排名={len(low)}, 需提召回={len(miss)}")
finally:
    kb._col.close()
    cleanup()
