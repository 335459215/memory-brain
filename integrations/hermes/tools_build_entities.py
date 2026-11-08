"""实体图谱全量回填 (2026-11-08): 遍历 zvec 库挖实体 → 重建 entities.db。

zvec 活库被 MCP server 独占 LOCK, 照 tools_eval_retrieval 的套路拷副本到
scratch 再挖 (只读 iter_docs, 不碰活库)。产出的 entities.db 是独立 SQLite,
server 重启后生效。

用法: venv/bin/python3 tools_build_entities.py [--min-df 3]
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "/root/hermes-kb")
import entity_graph as eg  # noqa: E402

SCRATCH = "/root/.hermes/cache/scratch"


def main():
    min_df = 3
    if "--min-df" in sys.argv:
        min_df = int(sys.argv[sys.argv.index("--min-df") + 1])

    import mcp_server as kb
    tmp = tempfile.mkdtemp(prefix="kb-entity-", dir=SCRATCH)
    dst = os.path.join(tmp, "zvec_db")
    print(f"[entities] 拷贝活库 -> {dst}")
    shutil.copytree(kb.DB_PATH, dst)
    kb.DB_PATH = dst
    try:
        col = kb.get_collection()
        n_docs, n_ent, n_link = eg.rebuild_from_col(col, min_df=min_df)
        s = eg.stats()
        print(f"[entities] 回填完成: {n_docs} 文档 -> {n_ent} 实体, {n_link} 条关联")
        print(f"[entities] df>=3 词典 top15: "
              + ", ".join(f"{e['entity']}:{e['docs']}" for e in s["top"][:15]))
        # 冒烟: 几条真实 query 的实体识别
        for q in ["Emby 插件 C# HarmonyLib ARM64 部署", "zvec 段不合并 optimize 内存占用",
                  "new-api 日志无限增长 轮转 copytruncate", "LunaTV 播放页切后台回来不播",
                  "hermes-kb ONNX 编码器 parity 余弦", "VPS swap 满了怎么归因 /tmp tmpfs"]:
            print(f"[entities]   {q[:34]:<36} -> {eg.entities_in(q)}")
        col.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[entities] 重启 kb-http.service 后生效")


if __name__ == "__main__":
    main()
