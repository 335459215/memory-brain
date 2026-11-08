"""KB 检索质量评估 harness (2026-11-08, 移植 memory-brain eval 思路到 Hermes KB)。

用途: 改动检索链路 (时间过滤/实体扩展/权重) 前后各跑一次, 用 hit@k + MRR + 延迟
证明"变好还是变坏", 不靠感觉。查询集两块:
  A. 真实 agent query (20 条): 按本机 KB 实际沉淀内容构造, 每条带期望文档
     (已知 id 或标题/正文唯一子串), 模拟"我/操盘手真会问的问题"
  B. 自指 query (30 条): 确定性抽样库存文档, 用 title 当 query, 期望该文档
     本身被召回 — 量基础检索健全性

用法:
  venv/bin/python3 tools_eval_retrieval.py                 # 跑并打印
  venv/bin/python3 tools_eval_retrieval.py -o result.json  # 存结果
  venv/bin/python3 tools_eval_retrieval.py --compare a.json b.json  # 前后对比
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, "/root/hermes-kb")
import mcp_server as kb  # noqa: E402  (__main__ guarded, import 不启动服务)

# A. 真实 query: (query, 期望判定) — id 精确匹配 或 标题/正文唯一子串
REAL_QUERIES = [
    ("LunaTV 播放页切后台回来不播 root cause", "playerEpoch"),
    ("VPS swap 满了怎么归因 /tmp tmpfs 真凶", "tmpfs"),
    ("zvec 段不合并 optimize 内存占用", "FtsRocksdbReducer"),
    ("chromium cmdline 是空格分隔不是 NUL", "空格分隔"),
    ("hermes-kb ONNX 编码器 parity 余弦", "onnxruntime"),
    ("CDP 泄漏实例占 9222 端口 22 小时", "9222"),
    ("kb_compact 指针文件被当知识递归入库", "指针"),
    ("hls.js 测速并发 bufferAddCodecError", "bufferAddCodecError"),
    ("new-api 日志无限增长 轮转 copytruncate", "copytruncate"),
    ("用户看片画质优先 先按分辨率挑源", "分辨率"),
    ("MoonTV LibreTV FongMi 参考源码 浅克隆", "LibreTV"),
    ("Emby 插件 C# HarmonyLib ARM64 部署", "HarmonyLib"),
    ("tmdb 镜像 401 language discover", "tmdb"),
    ("容器该不该加限额 memory.events oom_kill", "memory.events"),
    ("前端预取去重 模块级变量 重复 rsc", "window.__"),
    ("memory-brain 适配 hermes RRF 融合", "rrf_fuse"),
    ("MySQL binlog 堆积 PURGE BINARY LOGS", "PURGE BINARY LOGS"),
    ("LunaTV 缓存键 0 基 集数 差一", "0 基"),
    ("x86 容器跑在 ARM 上 qemu 内存膨胀", "qemu"),
    ("swap 回收标准动作 swapoff swapon", "swapoff"),
]
SELF_N = 30
SELF_SEED = 20261108


def hit_at(rows, expect, k):
    for r in rows[:k]:
        if expect in (r["id"],) or expect in r.get("title", "") or expect in r.get("content", ""):
            return True
    return False


def rank_of(rows, expect):
    for i, r in enumerate(rows, 1):
        if expect in (r["id"],) or expect in r.get("title", "") or expect in r.get("content", ""):
            return i
    return 0


def run_set(name, cases):
    hits = {1: 0, 3: 0, 5: 0}
    mrr, lat = [], []
    detail = []
    for query, expect in cases:
        t0 = time.time()
        res = kb.kb_search(query, topk=5)
        lat.append((time.time() - t0) * 1000)
        rows = res["results"]
        r = rank_of(rows, expect)
        for k in (1, 3, 5):
            if hit_at(rows, expect, k):
                hits[k] += 1
        mrr.append(1.0 / r if r else 0.0)
        detail.append({"query": query, "expect": expect, "rank": r,
                       "top1_title": rows[0]["title"] if rows else ""})
        print(f"  [{'HIT@1' if r == 1 else f'rank={r}' if r else 'MISS':>7}] {query[:38]:<40} -> {(rows[0]['title'][:34] if rows else '-')}")
    n = len(cases)
    lat.sort()
    agg = {"set": name, "n": n,
           "hit@1": round(hits[1] / n, 3), "hit@3": round(hits[3] / n, 3),
           "hit@5": round(hits[5] / n, 3), "mrr": round(sum(mrr) / n, 3),
           "latency_p50_ms": round(lat[n // 2]), "latency_p90_ms": round(lat[int(n * 0.9)])}
    print(f"  == {name}: hit@1={agg['hit@1']} hit@3={agg['hit@3']} hit@5={agg['hit@5']} "
          f"MRR={agg['mrr']} p50={agg['latency_p50_ms']}ms p90={agg['latency_p90_ms']}ms")
    return agg, detail


def self_cases():
    col = kb.get_collection()
    docs = list(col.iter_docs())
    rng = random.Random(SELF_SEED)
    sample = rng.sample(docs, min(SELF_N, len(docs)))
    return [(d.fields.get("title", "")[:60], d.id) for d in sample if d.fields.get("title")]


def prepare_db():
    """live 服务独占 zvec_db LOCK — 拷一份到 scratch 再评, 不打断服务也不污染
    生产 hit_count。KB_EVAL_DB 指定时直接用该冻结副本 (受控 A/B 用)。
    返回清理函数。"""
    import shutil
    import tempfile
    frozen = os.environ.get("KB_EVAL_DB")
    if frozen:
        kb.DB_PATH = frozen
        print(f"[eval] 冻结库 {frozen}")
        return lambda: None
    tmp = tempfile.mkdtemp(prefix="kb-eval-", dir="/root/.hermes/cache/scratch")
    dst = os.path.join(tmp, "zvec_db")
    shutil.copytree(kb.DB_PATH, dst)
    kb.DB_PATH = dst  # get_collection 懒打开, 之后所有查询走副本
    print(f"[eval] 副本库 {dst} ({sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(dst) for f in fs)//1024} KB)")

    def cleanup():
        shutil.rmtree(tmp, ignore_errors=True)
    return cleanup


def main():
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else None
    cleanup = prepare_db()
    try:
        print("== A. 真实 agent query (20) ==")
        agg_a, det_a = run_set("real", REAL_QUERIES)
        print("== B. 自指 query (title→本体, 30) ==")
        agg_b, det_b = run_set("self", self_cases())
    finally:
        cleanup()
    result = {"ts": time.strftime("%F %T"), "real": agg_a, "self": agg_b,
              "detail_real": det_a, "detail_self": det_b}
    if out:
        with open(out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"saved -> {out}")
    return result


if __name__ == "__main__":
    main()
