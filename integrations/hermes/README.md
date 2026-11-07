# Hermes Agent 适配层

把 [memory-brain](../../README.md) 的记忆设计接到 **Hermes Agent 的 zvec 知识库**（`~/hermes-kb`）上，不改上游算法，只补工程层：

| memory-brain 的做法 | Hermes 原来的问题 | 本适配层 |
|---|---|---|
| 五路混合检索 + RRF | 向量分与 BM25 分直接 `max()` 合并，量纲不同，实际退化成 FTS 优先 | `rrf_fuse()` 只融合**排名**，再与 trust / recency / hit / kind 加权 |
| 记忆分 5 类 | 所有条目一锅粥，问偏好和问端口用同一套权重 | `infer_kind()` + 查询意图 → kind 加权 |
| 睡觉整理（相似合并） | 只按 md5 精确去重；`kb_import` 完全绕过语义闸门 | `duplicate_clusters()` + `merge_plan()`，支持 dry-run、同分类合并、软废弃 |
| 体检（重复/僵尸/冲突） | 只有统计数，没有可行动报告 | `health_report()`，新增 `kb_health` 工具 |
| 衰减 → 归档 → 合并 | `_auto_deprecate` 一刀切：90 天未更新就废，稳定知识也杀 | `select_forgetting()`：指针垃圾优先，其次「长期 + 低信任 + 零命中」；高信任规则/基础设施永不因年龄淘汰 |
| 重要性/新鲜度分开 | `effective_trust = trust × 时间衰减` 可降到 0；两个月前的已验证事实像垃圾 | `trust_with_floor()`：信任最低保留 50%，新鲜度单独只占 12% |

## 额外修的一个真实故障

Hermes 的 `kb_compact` 会把 MEMORY.md 压成「主题 → `kb搜'…'`」指针文件。旧逻辑下一轮 compact 又把指针文件当知识入库，标题递归长成 `- - - 指针头…`，靠关键词霸占搜索第一名。

`is_pointer_doc()` 识别这类派生产物，`kb_health(fix=True)` / `kb_optimize()` 自动软废弃，搜索时直接过滤。

## 目录内容

```
integrations/hermes/
├── memory_adapt.py      # 适配核心：RRF / kind / 去重 / 体检 / 保守遗忘
├── mcp_server.py        # 已适配的完整 Hermes KB MCP server（可直接替换）
├── llm_helper.py        # 语义闸门/概览/精简共用的 LLM 配置读取
├── requirements.txt     # server 依赖
├── patches/mcp_server.patch
├── tests/test_memory_adapt.py
└── README.md
```

`mcp_server.py` 是 live 版本（2026-11-07，2181 行），包含 `kb_health` / `kb_dedup` / `kb_reindex` 和全部检索改造；如果只想最小改动，用 `patches/mcp_server.patch`。

## 安装到 Hermes KB

```bash
cd ~/hermes-kb
cp mcp_server.py mcp_server.py.bak-$(date +%Y%m%d_%H%M%S)
patch -p1 < /path/to/memory-brain/integrations/hermes/patches/mcp_server.patch
cp /path/to/memory-brain/integrations/hermes/memory_adapt.py .
systemctl --user restart kb-http.service
```

补丁基于 2026-11-07 的 `~/hermes-kb/mcp_server.py`（2178 行版本）。如果上游/本机版本不同，按下面的接线点手动合并：

1. schema 增加 `kind`（INT32，0=未标注）；
2. `kb_search` 用 `rrf_fuse` + `score_memory` + `dedupe_results`；
3. `kb_add` / `kb_import` 增加 kind 推断与指纹级去重；
4. 新增 `kb_health`、`kb_dedup`、`kb_reindex`；
5. `_auto_deprecate` 换成 `select_forgetting`；
6. `_compact_memory_file` 过滤指针行，避免递归入库；
7. zvec 单次写入上限 1024 条：全库更新用 `_batch_update` / `_batch_delete` 分片。

## 数据边界

本目录只包含代码、单元测试和文档，不包含任何 Hermes 知识库条目、聊天记录、用户名、账号、密码、Token 或真实服务配置；示例中的 IP 使用 RFC 5737 文档地址（192.0.2.0/24），服务名均为占位符。

## 测试

```bash
cd integrations/hermes
python -m unittest discover -s tests -v
```

测试只依赖标准库 + numpy，不需要 zvec、不碰生产库。

## 设计边界

- **不自动合并跨分类条目**：`user` 的偏好和 `memory` 的任务记录即使语义相近也可能是两件事；需要显式 `same_category=False`。
- **合并是软删除**：loser 标 `deprecated=1` 且 trust 归零，可用 `kb_trust restore` 恢复。
- **不改人格/原始数据**：适配层只读写 KB 元数据，不触碰 MEMORY.md/USER.md 的正文（compact 逻辑除外）。
