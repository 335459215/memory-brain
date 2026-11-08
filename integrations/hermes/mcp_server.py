#!/usr/bin/env python3
"""
Hermes Knowledge Base MCP Server v3.0
ZVec + ONNX ARM64-int8 + 信任评分 — 搜索延迟~10ms

新增信任评分机制:
- trust_score: 1.0=初始, >0.7=可信, 0.3~0.7=待确认, <0.3=已过时, 0.0=已废弃
- updated_at: unix timestamp, 用于时间衰减
- deprecated: 0=正常, 1=已废弃(被新事实替代)

工具: kb_search(混合搜索+信任过滤) kb_add(写入) kb_stats(状态) kb_trust(信任操作)
启动时预加载模型+DB + 自动schema迁移 → 首次工具调用即达热延迟
"""

import os, sys, warnings, hashlib, time, re, math, threading
from contextlib import contextmanager
warnings.filterwarnings("ignore")
from pathlib import Path

# 模型已本地缓存(~/hermes-kb/models 或 HF 缓存), 强制离线加载 —
# 境内网络访问 huggingface.co 每次 10-20s 超时, 是 MCP server 反复
# spawn 超时/连环崩溃的真根因 (2026-08-18 实测: 在线检查 404/307 慢)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import zvec
from mcp.server.mcpserver import MCPServer
# 注意: sentence_transformers (torch) 不在此处导入 — 2026-11-08 改为 onnxruntime
# 主路径, torch 仅在 ONNX 加载失败时懒加载兜底 (见 get_onnx_embedder)。
# 顶层导入 torch 会使无 GPU 的 ARM 机白背 ~800MB (libtorch_cpu/cublas 栈)。

# memory-brain 适配层 (RRF 混合检索 / 记忆类型 / 合并 / 体检 / 保守遗忘)
import memory_adapt as ma

# 时间感知召回 (2026-11-08): query 时间表达 → 时间窗后过滤。
# 防御式导入: 模块缺失/损坏不拖死 server (照 onnx_embedder 的兜底教训)。
try:
    import time_recall as tr
except Exception as _e:
    tr = None
    print(f"[MCP-KB] time_recall 不可用 ({_e}), 时间窗召回禁用", file=sys.stderr, flush=True)

# 实体图谱 (2026-11-08): SQLite 边车库, query 实体 → 关联文档加分 + kb_graph 查询。
# 防御式导入: 模块缺失/损坏不拖死 server。
try:
    import entity_graph as eg
except Exception as _e:
    eg = None
    print(f"[MCP-KB] entity_graph 不可用 ({_e}), 实体加分/图谱禁用", file=sys.stderr, flush=True)

ENTITY_BOOST = float(os.environ.get("KB_ENTITY_BOOST", "0.06"))  # query 命中词典实体的关联文档加分 (与 scene_boost 同量级); KB_ENTITY_BOOST=0 可关 (受控 A/B 用)
ENTITY_ON = os.environ.get("KB_ENTITY_BOOST", "1") not in ("0", "false", "off")
TIME_ON = os.environ.get("KB_TIME_FILTER", "1") not in ("0", "false", "off")

# Obsidian 兼容的 markdown 镜像目录 (kb_export 导出 / kb_add 同步)
NOTES_ROOT = Path(os.environ.get("KB_NOTES_ROOT", "/root/hermes-kb/notes"))

DB_PATH = os.path.expanduser("~/hermes-kb/zvec_db")

mcp = MCPServer(name="hermes-kb")

# === 全局单例 ===
_onnx_model = None
_col = None

# 请求指标 — 记录调用次数/延迟
_metrics = {"search_calls": 0, "add_calls": 0, "trust_calls": 0, "optimize_calls": 0, "total_ms": 0}
_metrics_lock = threading.Lock()

# ZVec线程安全锁 — 虽然stdio模式串行，加锁防未来SSE transport并发
_zvec_lock = threading.Lock()

# 2026-10-24 僵死修复: 带超时的锁获取。
# 背景: kb_compact 工具曾持 _zvec_lock 跑 LLM 摘要(45s/文件), 期间所有工具
# 阻塞在锁上 → MCP 客户端 60s 超时 → 表现为"server 活着但全部卡死"。
# 现在: 读工具用 _db_lock(超时后快速报错), kb_compact 不再持锁跑 LLM。
class KBBusyError(Exception):
    """KB 锁在限定时间内拿不到 → 工具快速失败, 避免 MCP 客户端 60s 超时挂死"""
    pass

@contextmanager
def _db_lock(timeout: float = 15.0):
    """带超时的 _zvec_lock 获取: 超时抛 KBBusyError (FastMCP 作为工具错误返回, 不再是挂死)"""
    acquired = _zvec_lock.acquire(timeout=timeout)
    if not acquired:
        raise KBBusyError(f"KB 忙: {timeout:.0f}s 内未拿到数据库锁, 请稍后重试")
    try:
        yield
    finally:
        _zvec_lock.release()

# 信任评分参数
TRUST_CONFIRM_BOOST = 0.15      # 每次确认提升
TRUST_CONTRADICT_DECAY = 0.3    # 每次矛盾降低
TRUST_TIME_DECAY_DAYS = 30      # N天后开始衰减
TRUST_TIME_DECAY_RATE = 0.02    # 每天衰减量
TRUST_DEPRECATED_THRESHOLD = 0.3  # 低于此值视为废弃
TRUST_DEFAULT = 1.0

def get_onnx_embedder():
    """all-MiniLM-L6-v2 编码器 — onnxruntime CPU backend, 384 维, mean pooling + L2 归一化。

    2026-11-08 优化: 从 sentence-transformers(torch) 切到 onnxruntime。
    本机是无 NVIDIA GPU 的 ARM64 VPS: torch 路径稳态 RSS ~800MB (模型本身仅
    90M, 其余全是 libtorch_cpu/libtorch_cuda/libcublasLt 栈与分配器 arena),
    且启动要 15s; onnxruntime + tokenizers 均为 venv 已有依赖, RSS ~200M,
    加载 0.3s。编码接口对齐 SentenceTransformer.encode (str→(384,), list→(n,384))。

    兼容性 (实测 10 条中英混合/截断/空串边界): fp32 ONNX 与 torch 输出
    余弦相似度 = 1.0000, 与库内既有 embedding 完全兼容。
    量化版 model_quantized.onnx 余弦仅 0.974~0.992 (空串 0.9742), 不采用。

    历史: 最初就是 onnx 实现, 因 optimum 未安装 + 模型文件不存在启动即崩
    (492 次/天崩溃循环), 2026-11-07 改 torch 兜底; 本次直接用 onnxruntime
    (无需 optimum) + 本地模型 (/root/hermes-kb/models) 解决。torch 仅作
    ONNX 加载失败时的兜底, 正常不再加载。
    """
    global _onnx_model
    if _onnx_model is not None:
        return _onnx_model
    try:
        from onnx_embedder import OnnxEmbedder
        _model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        _onnx_model = OnnxEmbedder(_model_dir, model_file="model.onnx")
        _onnx_model.encode("warmup knowledge base search query")  # 图优化+分配器预热
        print("[MCP-KB] onnxruntime 模型加载完成 (fp32, 较 torch 路径省 ~600MB)",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[MCP-KB] onnxruntime 加载失败 ({e!r}), 回退 torch CPU 路径",
              file=sys.stderr, flush=True)
        from sentence_transformers import SentenceTransformer
        _onnx_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _onnx_model.encode("warmup knowledge base search query")
        _onnx_model.encode("Docker端口映射规则Emby插件编译")
        _onnx_model.encode("防火墙iptables配置VPS IPv6地址")
        print("[MCP-KB] torch CPU模型预加载完成", file=sys.stderr, flush=True)
    return _onnx_model


def _ensure_schema(col):
    """自动检测并迁移缺失字段(信任评分)"""
    schema = col.schema
    field_names = [f.name for f in schema.fields]
    
    if "trust_score" not in field_names:
        print("[MCP-KB] Schema迁移: 添加 trust_score (FLOAT, 默认1.0)...", file=sys.stderr, flush=True)
        col.add_column(
            zvec.FieldSchema(name="trust_score", data_type=zvec.DataType.FLOAT),
            expression="1.0"
        )
    
    if "updated_at" not in field_names:
        now_ts = int(time.time())
        print(f"[MCP-KB] Schema迁移: 添加 updated_at (INT64, 默认{now_ts})...", file=sys.stderr, flush=True)
        col.add_column(
            zvec.FieldSchema(name="updated_at", data_type=zvec.DataType.INT64),
            expression=str(now_ts)
        )

    if "hit_count" not in field_names:
        print("[MCP-KB] Schema迁移: 添加 hit_count (INT64, 默认0)...", file=sys.stderr, flush=True)
        col.add_column(
            zvec.FieldSchema(name="hit_count", data_type=zvec.DataType.INT64),
            expression="0"
        )

    if "deprecated" not in field_names:
        print("[MCP-KB] Schema迁移: 添加 deprecated (INT32, 默认0)...", file=sys.stderr, flush=True)
        try:
            col.add_column(
                zvec.FieldSchema(name="deprecated", data_type=zvec.DataType.BOOL),
                expression="false"
            )
        except Exception:
            col.add_column(
                zvec.FieldSchema(name="deprecated", data_type=zvec.DataType.INT32),
                expression="0"
            )

    if "scene" not in field_names:
        # 2026-10-23: 场景路由字段 (Hy-Memory scene 借鉴)。zvec add_column 仅支持数值型,
        # 用 INT32 枚举编码: 0=未标注(存量863条兼容), 1=chat, 2=working, 3=coding
        print("[MCP-KB] Schema迁移: 添加 scene (INT32 枚举: 0空/1chat/2working/3coding)...", file=sys.stderr, flush=True)
        col.add_column(
            zvec.FieldSchema(name="scene", data_type=zvec.DataType.INT32),
            expression="0"
        )

    if "kind" not in field_names:
        # 2026-11-07: 记忆类型 (memory-brain 五类记忆借鉴)。
        # INT32 枚举: 0=未标注, 1=fact, 2=episode, 3=preference, 4=lesson, 5=rule, 6=profile, 7=task
        print("[MCP-KB] Schema迁移: 添加 kind (INT32 记忆类型)...", file=sys.stderr, flush=True)
        col.add_column(
            zvec.FieldSchema(name="kind", data_type=zvec.DataType.INT32),
            expression="0"
        )
    
    # 验证迁移结果
    schema = col.schema
    field_names = [f.name for f in schema.fields]
    print(f"[MCP-KB] Schema就绪, 字段: {field_names}", file=sys.stderr, flush=True)


_col_init_lock = threading.Lock()

def get_collection():
    global _col
    if _col is None:
        with _col_init_lock:  # 2026-10-24: 防后台线程与工具并发首开
            if _col is None:
                _col = zvec.open(DB_PATH)
                _ensure_schema(_col)  # 自动迁移
                _repair_fts_if_empty(_col)  # FTS 索引自愈 (2026-08-18: 重建后 FTS 未落盘 → 搜索全 0)
                print(f"[MCP-KB] ZVec知识库已打开: {DB_PATH} ({_col.stats.doc_count}条)", file=sys.stderr, flush=True)
    return _col


def _repair_fts_if_empty(col):
    """打开后验证 FTS 索引健康; 索引为空(重建/异常关闭后 WAL 未落盘)时 optimize 重建。
    背景: 2026-08-18 重建 zvec_db 后 fts.1.rocksdb 只有 WAL 无 sst, kb_search fts 全 0 结果,
    vector 正常 → 真根因是 FTS 索引未持久化, 非网络/模型/锁。
    """
    try:
        probe = col.query(zvec.Query("content", fts=zvec.Fts(query_string="nginx")), topk=1)
        if len(probe) > 0:
            return  # FTS 索引正常
    except Exception as e:
        print(f"[MCP-KB] FTS 探测异常: {e}", file=sys.stderr, flush=True)
        return  # 查询报错不是索引问题, 交给其他修复
    try:
        print("[MCP-KB] FTS 索引为空, 触发 optimize 重建...", file=sys.stderr, flush=True)
        t0 = time.time()
        col.optimize()
        col.flush()
        print(f"[MCP-KB] FTS 索引重建完成 ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[MCP-KB] FTS 重建失败: {e}", file=sys.stderr, flush=True)


def embed(text: str) -> list:
    return get_onnx_embedder().encode(text, normalize_embeddings=True).tolist()


def _calc_time_decay(updated_at: int) -> float:
    """计算时间衰减因子 (0.0~1.0)"""
    if updated_at <= 0:
        return 1.0
    now = int(time.time())
    days_old = (now - updated_at) / 86400.0
    if days_old < TRUST_TIME_DECAY_DAYS:
        return 1.0
    decay = 1.0 - (days_old - TRUST_TIME_DECAY_DAYS) * TRUST_TIME_DECAY_RATE
    return max(0.0, decay)


def _effective_trust(trust_score: float, updated_at: int) -> float:
    """有效信任 = 基础信任 × 时间衰减"""
    return trust_score * _calc_time_decay(updated_at)


# === 复合重排序 (借鉴 Mnemosyne compound-cue) ===
# final_score = 0.55*semantic + 0.20*recency + 0.15*trust + 0.10*hit_freq
RECENCY_HALF_LIFE_DAYS = 30  # 30天半衰期
COMPOSITE_WEIGHTS = {
    "semantic": 0.55,
    "recency": 0.20,
    "trust": 0.15,
    "hit_freq": 0.10,
}


def _calc_recency_score(updated_at: int) -> float:
    """时间近度评分 (0~1): 半衰期模型, 越近越高"""
    if updated_at <= 0:
        return 0.5
    days_old = (int(time.time()) - updated_at) / 86400.0
    if days_old <= 0:
        return 1.0
    return 0.5 ** (days_old / RECENCY_HALF_LIFE_DAYS)


def _calc_hit_freq_score(hit_count: int) -> float:
    """命中频率评分 (0~1): 对数缩放, 频繁命中的条目得分更高"""
    if hit_count <= 0:
        return 0.0
    return min(1.0, 0.3 + 0.7 * (1 - 0.5 ** hit_count))


def _composite_rescore(raw_score: float, trust: float, updated_at: int, hit_count: int) -> float:
    """复合评分: semantic + recency + trust + hit_freq"""
    recency = _calc_recency_score(updated_at)
    eff_trust = trust * _calc_time_decay(updated_at)
    hit_freq = _calc_hit_freq_score(hit_count)
    w = COMPOSITE_WEIGHTS
    return (
        w["semantic"] * max(0.0, min(1.0, raw_score)) +
        w["recency"] * recency +
        w["trust"] * eff_trust +
        w["hit_freq"] * hit_freq
    )


# === 场景路由 (2026-10-23, 借鉴 Hy-Memory scene) ===
# scene ∈ chat(闲聊/偏好/个人事实) / working(生产力任务事实+经验) / coding(软件工程, 含版本演进)
# 空字符串 = 未标注(存量 863 条兼容, 检索时按"无场景过滤"处理)
VALID_SCENES = ("chat", "working", "coding")
# zvec 数值编码: 0=未标注, 1=chat, 2=working, 3=coding
_SCENE_CODE = {"": 0, "chat": 1, "working": 2, "coding": 3}
_SCENE_NAME = {v: k for k, v in _SCENE_CODE.items()}


def _norm_scene(scene) -> str:
    """归一化 scene 字段; 非法值回退空串(=未标注, 兼容存量)"""
    if scene is None:
        return ""
    if isinstance(scene, int):
        return _SCENE_NAME.get(scene, "")
    s = str(scene).strip().lower()
    return s if s in VALID_SCENES else ""


def _scene_code(scene) -> int:
    """scene 字符串 → zvec INT8 存储码"""
    return _SCENE_CODE.get(_norm_scene(scene), 0)


# zvec FTS 解析器是带括号分组的表达式语法: ':' '^' 直接语法错误, 不平衡的
# '(' ')' 报 missing ')', 任何"用户原话"当 query 都可能让整个 kb_search 崩
# (实测含冒号的标题/query 批量触发)。白名单消毒: 只留 字母/数字/CJK/下划线/
# 空格, 其余一律变空格 — 标点损失由向量检索腿补, 消毒后为空则跳过 FTS 腿。
_FTS_BREAK_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff\u3400-\u4dbf_ ]+")


def _sanitize_fts_query(q: str) -> str:
    return " ".join(_FTS_BREAK_CHARS.sub(" ", q or "").split())


def _read_scene_field(fields: dict) -> str:
    """从 zvec 行读 scene 字段(可能是 int 码/str), 归一化为字符串"""
    raw = fields.get("scene", "")
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return ""
    if isinstance(raw, int):
        return _SCENE_NAME.get(raw, "")
    if isinstance(raw, str):
        if raw.isdigit():
            return _SCENE_NAME.get(int(raw), "")
        return _norm_scene(raw)
    return ""


# === L0 摘要生成 (2026-10-23, 借鉴 OpenViking L0/L2 分层) ===
L0_ABSTRACT_CHARS = 200  # 摘要长度上限

def _make_l0(content: str) -> str:
    """生成 L0 摘要: 首句优先, 截断到完整句/换行; 无句读则硬截"""
    text = (content or "").strip()
    if len(text) <= L0_ABSTRACT_CHARS:
        return text
    head = text[:L0_ABSTRACT_CHARS]
    for sep in ("\n", "。", "！", "？", "；", ". ", "! ", "? "):
        idx = head.rfind(sep)
        if idx >= 40:  # 至少保住 40 字, 避免碎句
            return head[:idx + len(sep)].strip()
    return head.strip()


def _record_metric(key: str, elapsed_ms: float):
    """记录请求指标"""
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0) + 1
        _metrics["total_ms"] += elapsed_ms


def _auto_deprecate(col, stale_days: int = 365) -> int:
    """保守遗忘 (memory-brain 衰减策略适配版)。

    旧逻辑「90 天未更新就废弃」会杀掉稳定知识 (规则/基础设施/服务文档),
    且每天后台跑一次, 三个月后整个 KB 被清空。现在只废弃:
      1. kb_compact 产生的指针/导航文件 (派生垃圾, 还会递归繁殖);
      2. 长期 (> stale_days) 低信任 (<0.5) 且从未被检索到的条目。
    高信任、有命中、规则类知识永不因年龄淘汰。
    """
    docs = list(_iter_all_docs(col))
    candidates = ma.select_forgetting(docs, stale_days=stale_days, min_trust=0.5)
    if not candidates:
        return 0
    now = int(time.time())
    by_reason = {}
    batch = []
    for c in candidates:
        by_reason[c["reason"]] = by_reason.get(c["reason"], 0) + 1
        batch.append(zvec.Doc(id=c["id"], fields={
            "deprecated": 1, "trust_score": 0.0, "updated_at": now
        }))
    _batch_update(col, batch)
    print(f"[MCP-KB] 保守遗忘: {by_reason} (stale_days={stale_days})", file=sys.stderr, flush=True)
    return len(batch)


def _cleanup_deprecated(col, days: int = 730) -> int:
    """物理删除: 已废弃超过N天的条目, 返回删除数"""
    now = int(time.time())
    threshold_ts = now - days * 86400
    all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
    to_delete = []
    for r in all_docs:
        dep = r.fields.get("deprecated", 0)
        if isinstance(dep, str): dep = int(dep) if dep.isdigit() else 0
        if dep < 1:
            continue
        updated_at = r.fields.get("updated_at", 0)
        if isinstance(updated_at, str):
            try: updated_at = int(updated_at)
            except: updated_at = 0
        if updated_at > 0 and updated_at < threshold_ts:
            to_delete.append(r.id)
    if to_delete:
        _batch_delete(col, to_delete)
    return len(to_delete)


# === Obsidian 兼容 markdown 镜像 (目录树) ===

def _safe_filename(title: str) -> str:
    """文件名安全化: 去路径分隔符/控制字符, 截断"""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", title.strip())
    name = name.strip(". ")
    return name[:80] or "untitled"


def _doc_to_markdown(r) -> str:
    """ZVec 文档 → Obsidian markdown 文件内容"""
    f = r.fields
    title = f.get("title", "") or f.get("content", "")[:30]
    content = f.get("content", "")
    category = f.get("category", "general")
    trust = f.get("trust_score", TRUST_DEFAULT)
    try:
        trust = float(trust)
    except Exception:
        trust = TRUST_DEFAULT
    dep = f.get("deprecated", 0)
    try:
        dep = int(dep)
    except Exception:
        dep = 0
    updated = f.get("updated_at", 0)
    try:
        updated = int(updated)
    except Exception:
        updated = 0
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(updated)) if updated else "unknown"
    status = "deprecated" if dep >= 1 else "active"
    md = [
        "---",
        f"id: {r.id}",
        f"category: {category}",
        f"trust: {round(trust, 2)}",
        f"status: {status}",
        f"updated: {ts}",
        "---",
        "",
        f"# {title}",
        "",
        content,
        "",
    ]
    kind = ma.kind_name(f.get("kind", 0))
    if kind:
        md.insert(6, f"kind: {kind}")
    scene = _read_scene_field(f)
    if scene:
        md.insert(6, f"scene: {scene}")
    return "\n".join(md)


def _sync_doc_to_md(r) -> Path:
    """单条 ZVec 文档同步为 notes/<category>/<title>.md, 返回路径。废弃条目跳过并删除旧镜像。"""
    f = r.fields
    dep = f.get("deprecated", 0)
    if isinstance(dep, str):
        dep = int(dep) if dep.isdigit() else 0
    category = f.get("category", "general") or "general"
    title = f.get("title", "") or f.get("content", "")[:30]
    if dep >= 1:
        # 删除已存在的废弃镜像
        for old in NOTES_ROOT.glob(f"{category}/*.md"):
            if old.stem == title:
                try:
                    old.unlink()
                except Exception:
                    pass
        return NOTES_ROOT / category / "DEPRECATED_SKIPPED.md"  # 哨兵, 调用方忽略
    cat_dir = NOTES_ROOT / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    path = cat_dir / f"{_safe_filename(title)}.md"
    path.write_text(_doc_to_markdown(r), encoding="utf-8")
    return path


def _iter_all_docs(col):
    """遍历全部文档(零向量查询 topk=10000 是 ZVec 全量枚举的惯用法)"""
    return col.query(zvec.Query("embedding", vector=[0.0] * 384), topk=10000)


# === MCP 工具 ===

@mcp.tool()
def kb_search(query: str, mode: str = "hybrid", topk: int = 5, min_trust: float = 0.0,
              scene: str = "", level: str = "L2", explain: bool = False,
              dedupe: bool = True, include_pointer: bool = False,
              time_filter: str = "") -> dict:
    """搜索Hermes知识库。mode: hybrid(默认,向量+FTS RRF融合)/vector(语义)/fts(全文)。min_trust: 最低信任分阈值(默认0.0不过滤; 已废弃条目一律不返回)。
    scene: 场景过滤 chat/working/coding(默认""不过滤)。同场景条目加分; 未打标条目兜底保留(scene_fallback=true); 已打标但场景不符的跳过。
    time_filter: 时间窗过滤。自动识别 query 里的相对时间表达(昨天/上周/最近3天/3天前/上半年...); 绝对日期不自动识别(文档标题常嵌日期会误判), 需显式指定: '7d'/'2w'/'3m'/'1y'/'2026-11-08'/'2026-11'/'all'(不过滤)。窗口内命中0条才回退全量并打标 time_fallback=true。
    level: 加载层级 L2=全文(默认) / L0=首句摘要+content_len+truncated标记(省token)。
    explain: 返回 why 评分分解(向量排名/FTS排名/信任/新鲜度/命中/类型匹配)。
    dedupe: 合并近重复结果(默认true)。include_pointer: 是否返回 kb_compact 指针文件(默认false, 它们不是知识)。"""
    t0 = time.time()
    with _db_lock(15):  # 2026-10-24: 读工具带超时, 锁忙快速报错不挂死
        col = get_collection()
        candidate_k = max(30, min(120, topk * 6))

        vec_raw, fts_raw = [], []
        if mode in ("hybrid", "vector"):
            vec = embed(query)
            vec_raw = list(col.query(zvec.Query("embedding", vector=vec), topk=candidate_k))
        if mode in ("hybrid", "fts"):
            fts_q = _sanitize_fts_query(query)
            if fts_q:
                fts_raw = list(col.query(
                    zvec.Query("content", fts=zvec.Fts(query_string=fts_q)), topk=candidate_k))

        # RRF: 只融合排名, 不混用量纲不同的余弦分与 BM25 分 (memory-brain DESIGN §4)
        fused = ma.rrf_fuse(vec_raw, fts_raw)
        doc_by_id = {}
        for r in vec_raw + fts_raw:
            doc_by_id[r.id] = r

        # 实体图谱 boost (2026-11-08): query 命中的词典实体 (最多2个, 罕见的优先)
        # → 其关联文档按 idf 加权加分 (罕见实体强信号, 烂大街实体几乎不加 —
        # 平铺加分实测会把时间窗内的相邻文档顶上来, time 集 hit@1 -0.17)。
        # 异常/无词典时静默跳过, 不影响检索主路径。
        q_entities, boost_by_doc = [], {}
        if eg is not None and ENTITY_ON:
            try:
                detailed = eg.entities_in_detailed(query)
                q_entities = [n for n, _ in detailed]
                for name, df in detailed:
                    w = ENTITY_BOOST / (1.0 + math.log2(max(2, df)))
                    for did in eg.docs_for(name, limit=400):
                        if w > boost_by_doc.get(did, 0.0):
                            boost_by_doc[did] = w
            except Exception:
                q_entities, boost_by_doc = [], {}

        results = []
        for doc_id, fusion in fused.items():
            r = doc_by_id.get(doc_id)
            if r is None:
                continue
            fields = r.fields
            deprecated = fields.get("deprecated", 0)
            if isinstance(deprecated, str):
                try: deprecated = int(deprecated)
                except: deprecated = 0
            if deprecated >= 1:
                continue
            if not include_pointer and ma.is_pointer_doc(fields):
                continue

            trust = fields.get("trust_score", TRUST_DEFAULT)
            if isinstance(trust, str):
                try: trust = float(trust)
                except: trust = TRUST_DEFAULT
            updated_at = fields.get("updated_at", 0)
            if isinstance(updated_at, str):
                try: updated_at = int(updated_at)
                except: updated_at = 0
            hit_count = fields.get("hit_count", 0)
            if isinstance(hit_count, str):
                try: hit_count = int(hit_count)
                except: hit_count = 0

            # 信任是证据强度, 不是新鲜度; 最低保留 50%, 不再衰减到 0
            eff_trust = ma.trust_with_floor(trust, updated_at)
            if eff_trust < min_trust:
                continue

            # 场景软过滤: 同场景加分, 不匹配打标条目跳过, 未打标兜底
            entry_scene = _read_scene_field(fields)
            scene_fallback = False
            scene_boost = 0.0
            if scene and _norm_scene(scene):
                if entry_scene and entry_scene != _norm_scene(scene):
                    continue
                if not entry_scene:
                    scene_fallback = True
                else:
                    scene_boost = 0.05

            scored = ma.score_memory(fields, fusion["relevance"], query=query,
                                     scene_boost=scene_boost,
                                     entity_boost=boost_by_doc.get(r.id, 0.0))
            content_full = fields.get("content", "")
            row = {
                "id": r.id,
                "score": round(float(r.score or 0.0), 4),
                "composite_score": scored["score"],
                "trust_score": round(float(trust), 2),
                "effective_trust": round(eff_trust, 2),
                "hit_count": hit_count,
                "updated_at": updated_at,
                "deprecated": deprecated,
                "title": fields.get("title", ""),
                "content": content_full,
                "content_len": len(content_full),
                "scene": entry_scene,
                "scene_fallback": scene_fallback,
                "category": fields.get("category", ""),
                "kind": ma.kind_name(fields.get("kind", 0)) or None,
            }
            if explain:
                row["why"] = {
                    **scored["why"],
                    "vector_rank": fusion.get("vector_rank", 0),
                    "fts_rank": fusion.get("fts_rank", 0),
                    "rrf": round(float(fusion.get("rrf", 0.0)), 5),
                }
            results.append(row)

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        if dedupe:
            results = ma.dedupe_results(results, threshold=0.88, text_key="content")

        # 时间窗召回 (2026-11-08): 显式 time_filter 优先, 否则自动识别 query 里的
        # 相对时间表达 (昨天/上周/最近N天...)。后过滤 (zvec Query 无 filter);
        # 绝对日期不自动识别 (文档标题大量嵌日期会误判, 实测 self MRR -0.15),
        # 需要时由调用方显式传 time_filter='2026-11-08'。
        # 窗口内命中 0 条才回退全量并打标 (命中 1-2 条是正确回答, 不饿死也不粉饰)。
        time_window = None
        if tr is not None and TIME_ON:
            try:
                spec = str(time_filter).strip()
                if spec:
                    tw = tr.parse_explicit(spec)
                else:
                    tw = tr.parse_time_window(query, absolute=False)
                if tw:
                    kept = [r for r in results if tw["start"] <= r.get("updated_at", 0) <= tw["end"]]
                    if kept:
                        tw["matched"] = len(kept)
                        tw["fallback"] = False
                        results = kept
                    else:
                        tw["matched"] = 0
                        tw["fallback"] = True
                    time_window = tw
            except Exception:
                time_window = None  # 解析/过滤异常绝不影响搜索主路径

        top = results[:topk]
        for row in top:
            if level == "L0":
                row["content"] = _make_l0(row["content"])
                row["truncated"] = len(row["content"]) < row["content_len"]
            else:
                row["truncated"] = False

        # 命中频率追踪: 只更新 hit_count, 不动 updated_at
        try:
            col.update([
                zvec.Doc(id=row["id"], fields={"hit_count": row["hit_count"] + 1})
                for row in top
            ])
        except Exception:
            pass

    elapsed = round((time.time() - t0) * 1000)
    _record_metric("search_calls", elapsed)
    return {"results": top, "elapsed_ms": elapsed, "total": len(results), "mode": mode,
            "min_trust": min_trust, "scene": _norm_scene(scene) or None, "level": level,
            "fusion": "rrf" if mode == "hybrid" else mode,
            "time_window": time_window, "entities": q_entities}


@mcp.tool()
def kb_graph(entity: str, topk: int = 10) -> dict:
    """实体图谱查询 (2026-11-08): 某实体关联的知识文档 + 共现实体。
    entity: 实体名, 大小写不敏感 (如 Emby / new-api / zvec / kb_search)。
    返回: 实体统计 (文档数/类型), 关联文档 (按信任+新鲜度排序, 含标题/摘要),
    以及经常与该实体同现的其它实体 (related, 用于顺藤摸瓜)。"""
    if eg is None:
        return {"error": "entity_graph 模块不可用"}
    name = str(entity or "").strip().lower()
    if not name:
        return {"error": "entity 不能为空"}
    try:
        eg.init_db()
        with eg._lock, eg._conn() as con:
            row = con.execute(
                "SELECT display, type, doc_count FROM entities WHERE name=?", (name,)).fetchone()
        if not row:
            return {"entity": name, "found": False, "doc_count": 0, "docs": [], "related": []}
        doc_ids = eg.docs_for(name, limit=300)
        docs = []
        try:
            with _db_lock(10):
                col = get_collection()
                fetched = col.fetch(doc_ids, include_vector=False) or {}
                docs = list(fetched.values())
        except Exception:
            docs = []
        if not docs:  # fetch 不可用 → 回退全量扫描过滤
            try:
                with _db_lock(15):
                    col = get_collection()
                    want = set(doc_ids)
                    docs = [d for d in col.iter_docs() if d.id in want]
            except Exception:
                docs = []
        rows = []
        for d in docs:
            f = d.fields
            if int(f.get("deprecated", 0) or 0) >= 1:
                continue
            content = f.get("content", "") or ""
            rows.append({
                "id": d.id,
                "title": f.get("title", ""),
                "snippet": content[:120],
                "trust_score": round(float(f.get("trust_score", 0) or 0), 2),
                "updated_at": f.get("updated_at", 0),
                "kind": ma.kind_name(f.get("kind", 0)) or None,
            })
        rows.sort(key=lambda x: (x["trust_score"], x["updated_at"]), reverse=True)
        return {"entity": name, "display": row[0], "type": row[1], "found": True,
                "doc_count": row[2], "docs": rows[:topk],
                "related": eg.related(name)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}



def _semantic_gate_judge(new_content: str, candidates: list):
    """语义闸门仲裁: 新内容 vs 库内最相似条目。
    返回 {verdict: duplicate|conflict|complement, winner, conflict_with, reason, confidence}
    或 None (LLM 不可用/解析失败/低置信 → 一律放行, 宁可漏不误杀; 漏的由 03:30 cron 兜底)。"""
    try:
        from llm_helper import llm_summarize
        if not candidates:
            return None
        cands_text = "\n\n".join(
            f"[候选{i+1}] id={c['id']} title={c['title']}\n{c['content']}"
            for i, c in enumerate(candidates)
        )
        system = (
            "你是 KB 写入闸门仲裁员。给定新内容和库内最相似的候选条目, 判定关系。"
            '输出严格 JSON: {"verdict": "duplicate"|"conflict"|"complement", "winner": "existing"|"new"|"", '
            '"conflict_with": "<最相关候选id或空>", "reason": "<50字内>", "confidence": 0.0-1.0}。'
            "判定标准: duplicate=同一结论换措辞(winner 选信息更全的一方); "
            "conflict=同一事实/参数的数值或结论矛盾(winner=正确的那个, 依据数值合理性与内容具体性, 与检索分数无关); "
            "complement=不同主题或互为补充。conflict_with=最相关候选的 id。"
            "不确定时判 complement。不要寒暄, 不要输出 JSON 之外内容。"
        )
        user = f"新内容:\n{new_content[:500]}\n\n库内候选:\n{cands_text}"
        raw = llm_summarize(system, user, max_tokens=4096, temperature=0.1)
        if not raw:
            return None
        import re, json as _json
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not m:
            return None
        result = _json.loads(m.group(0))
        if result.get("verdict") not in ("duplicate", "conflict", "complement"):
            return None
        if result.get("verdict") == "conflict" and result.get("winner") not in ("existing", "new"):
            return None
        try:
            if float(result.get("confidence", 0)) < 0.75:
                return None
        except (TypeError, ValueError):
            return None
        return result
    except Exception as e:
        print(f"[MCP-KB] semantic gate error: {e}", file=sys.stderr, flush=True)
        return None


@mcp.tool()
def kb_add(content: str, category: str = "general", title: str = "", suppress_duplicates: bool = True,
           semantic_gate: bool = True, gate_topk: int = 3, scene: str = "", kind: str = "") -> dict:
    """写入新知识到Hermes知识库。content:知识内容, category:分类(rules/services/infrastructure/development/general), title:标题(可选,默认取content前30字), suppress_duplicates:是否抑制重复内容(默认true)
    semantic_gate: 写入前语义对比闸门(默认true) — 检索最相似 top-k, LLM 判定 duplicate(重复)/conflict(矛盾,给winner)/complement(互补)。
    conflict 且新内容为输 → 拒绝写入并返回裁决; gate_topk: 语义对比的候选数(默认3)。会增加 2-5s LLM 延迟。
    scene: 场景标注 chat/working/coding(默认""不标注, 兼容存量)。生产力任务结果写 working, 代码修复/技术决策写 coding, 偏好/闲聊事实写 chat。
    kind: 记忆类型 fact/episode/preference/lesson/rule/profile/task(默认""自动推断, 借鉴 memory-brain 五类记忆)。"""
    t0 = time.time()
    col = get_collection()
    if not title:
        title = content[:30]
    kind_code = ma.kind_code(kind) or ma.infer_kind(category, title, content, scene)

    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    norm_hash = ma.content_fingerprint(content)
    pre_vec = None
    
    # 去重检查: 相同content_hash的条目已存在
    if suppress_duplicates:
        pre_vec = embed(title + " " + content)
    with _zvec_lock:
        if suppress_duplicates:
            all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
            for r in all_docs:
                same = r.fields.get("content_hash") == content_hash
                if not same:
                    # 标点/空白差异也算同一条 (memory-brain: 指纹比 md5 原文更稳)
                    same = ma.content_fingerprint(r.fields.get("content", "")) == norm_hash
                if same:
                    existing_trust = r.fields.get("trust_score", TRUST_DEFAULT)
                    # 相同内容再次写入 → 确认强化 (boost trust)
                    try:
                        new_trust = min(1.0, float(existing_trust) + TRUST_CONFIRM_BOOST)
                    except:
                        new_trust = TRUST_DEFAULT
                    col.update([zvec.Doc(
                        id=r.id,
                        fields={"trust_score": new_trust, "updated_at": int(time.time())}
                    )])
                    elapsed = round((time.time() - t0) * 1000)
                    _record_metric("add_calls", elapsed)
                    return {"id": r.id, "title": title, "category": category, "status": "confirmed", "trust": round(new_trust, 2)}

            # 语义近重复早停: 向量 >=0.96 直接确认, 不进 LLM 闸门 (防 kb_import 绕过)
            if pre_vec is not None:
                try:
                    near = col.query(zvec.Query("embedding", vector=pre_vec), topk=3)
                    for r in near:
                        dep = r.fields.get("deprecated", 0)
                        if isinstance(dep, str) and dep.isdigit():
                            dep = int(dep)
                        if dep >= 1:
                            continue
                        if float(r.score) >= 0.96:
                            old_trust = r.fields.get("trust_score", TRUST_DEFAULT)
                            try:
                                new_trust = min(1.0, float(old_trust) + TRUST_CONFIRM_BOOST)
                            except:
                                new_trust = TRUST_DEFAULT
                            col.update([zvec.Doc(id=r.id, fields={
                                "trust_score": new_trust, "updated_at": int(time.time())})])
                            elapsed = round((time.time() - t0) * 1000)
                            _record_metric("add_calls", elapsed)
                            return {"id": r.id, "title": r.fields.get("title", ""), "category": category,
                                    "status": "confirmed_near_duplicate", "trust": round(new_trust, 2),
                                    "similarity": round(float(r.score), 4)}
                except Exception:
                    pass

        # === 语义闸门 (2026-09-09): 持锁检索相似候选 (LLM 仲裁在锁外) ===
        gate_result = None
        gate_candidates = []
        if semantic_gate:
            vec = pre_vec if pre_vec is not None else embed(title + " " + content)
            sim_raw = col.query(zvec.Query("embedding", vector=vec), topk=gate_topk)
            for r in sim_raw:
                dep = r.fields.get("deprecated", 0)
                if isinstance(dep, str) and dep.isdigit():
                    dep = int(dep)
                if dep >= 1:
                    continue
                gate_candidates.append({
                    "id": r.id,
                    "title": r.fields.get("title", "")[:60],
                    "content": r.fields.get("content", "")[:500],
                    "score": float(r.score),
                })

    # LLM 仲裁在锁外 (不阻塞 kb_search 等工具; threading.Lock 非重入, 严禁在持锁时再取)
    gate_verdict = None
    if gate_candidates:
        gate_verdict = _semantic_gate_judge(content, gate_candidates)

    # === 闸门裁决处理 ===
    if gate_verdict:
        if gate_verdict.get("verdict") == "conflict":
            winner = gate_verdict.get("winner", "")
            if winner == "existing":
                # 新内容是错误方 → 拒绝写入
                elapsed = round((time.time() - t0) * 1000)
                _record_metric("add_calls", elapsed)
                return {
                    "status": "rejected_conflict",
                    "reason": gate_verdict.get("reason", ""),
                    "conflict_with": gate_verdict.get("conflict_with"),
                    "confidence": gate_verdict.get("confidence", 0.0),
                    "hint": "库内已有权威结论与此矛盾; 若确需覆盖, 用 kb_trust deprecate 旧条目后重写",
                }
            # winner=new → 放行写入, 冲突留给 03:30 cron 废弃旧条目
            gate_result = {"verdict": "conflict", "winner": "new", "conflict_with": gate_verdict.get("conflict_with")}
        elif gate_verdict.get("verdict") == "duplicate":
            # 语义重复(非hash) → 确认强化已有条目, 不再新增
            dup_id = gate_verdict.get("conflict_with")
            with _zvec_lock:
                try:
                    docs = [d for d in _iter_all_docs(col) if d.id == dup_id]
                    if docs:
                        old_trust = docs[0].fields.get("trust_score", TRUST_DEFAULT)
                        try: old_trust = float(old_trust)
                        except: old_trust = TRUST_DEFAULT
                        boosted = min(1.0, old_trust + TRUST_CONFIRM_BOOST)
                        col.update([zvec.Doc(id=dup_id, fields={
                            "trust_score": boosted,
                            "updated_at": int(time.time()),
                        })])
                        elapsed = round((time.time() - t0) * 1000)
                        _record_metric("add_calls", elapsed)
                        return {"id": dup_id, "title": docs[0].fields.get("title", ""), "category": category,
                                "status": "confirmed_semantic_duplicate", "trust": round(boosted, 2),
                                "reason": gate_verdict.get("reason", "")}
                except Exception:
                    pass  # 强化失败则继续走新条目写入
            gate_result = {"verdict": "duplicate", "conflict_with": dup_id}
        else:  # complement
            gate_result = {"verdict": "complement"}

    # 新条目 (回锁落库)
    full_text = title + " " + content
    emb = pre_vec if pre_vec is not None else embed(full_text)
    
    doc_id = f"kb_{int(time.time())}_{content_hash}"
    doc = zvec.Doc(
        id=doc_id, vectors={"embedding": emb},
        fields={
            "title": title, "content": content, "category": category,
            "content_hash": content_hash,
            "trust_score": TRUST_DEFAULT, "updated_at": int(time.time()),
            "deprecated": 0, "hit_count": 0,
            "scene": _scene_code(scene),
            "kind": kind_code,
        }
    )
    with _zvec_lock:
        col.insert([doc])
    # 实体图谱增量关联 (2026-11-08): 新文档抽取实体入库。失败不影响写入本身。
    if eg is not None:
        try:
            eg.init_db()
            eg.link_doc(doc_id, title, content)
        except Exception as e:
            print(f"[MCP-KB] 实体关联失败(不致命): {e}", file=sys.stderr, flush=True)
    # 同步 Obsidian md 镜像
    try:
        _sync_doc_to_md(zvec.Doc(id=doc_id, fields={
            "title": title, "content": content, "category": category,
            "trust_score": TRUST_DEFAULT, "updated_at": int(time.time()), "deprecated": 0,
            "scene": _scene_code(scene), "kind": kind_code,
        }))
    except Exception as e:
        print(f"[MCP-KB] md镜像同步失败: {e}", file=sys.stderr, flush=True)
    
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("add_calls", elapsed)
    return {"id": doc_id, "title": title, "category": category, "status": "ok", "trust": TRUST_DEFAULT,
            "scene": _norm_scene(scene) or None, "kind": ma.kind_name(kind_code) or None}


@mcp.tool()
def kb_commit(task: str, outcome: str, approach: str = "", scene: str = "working",
              experience: str = "", experience_conditions: str = "",
              workspace: str = "", files: str = "") -> dict:
    """提交一次任务的「事实+经验」双层记忆 (借鉴 Hy-Memory working/coding 双层)。
    task: 本次做了什么(一句话) | outcome: 结果(成功/失败/关键产出, 一句话) | approach: 采取的做法/关键命令/决策依据(可选)
    scene: working(默认,生产力任务) / coding(代码修复/技术决策; 建议带 workspace+files)
    experience: 可复用经验(可选; 0-N条, 无价值不写) | experience_conditions: 经验适用条件/边界(可选, 复用前判断)
    workspace: 项目/仓库标识(如 github.com/x/y, 可选) | files: 涉及文件路径列表(逗号分隔, 可选)
    事实层每会话必记1条; 经验层需确有复用价值才写。自动走语义闸门去重。"""
    t0 = time.time()
    sc = _norm_scene(scene) or "working"
    if sc == "chat":
        sc = "working"  # chat 无双层概念, 纠正为 working
    # 事实层: 每次提交 1 条 (Hy-Memory: exactly 1, no threshold)
    fact_parts = [f"任务: {task}", f"结果: {outcome}"]
    if approach:
        fact_parts.append(f"做法: {approach}")
    if workspace:
        fact_parts.append(f"项目: {workspace}")
    if files:
        fact_parts.append(f"文件: {files}")
    fact_content = " | ".join(fact_parts)
    fact_title = f"[{sc}-fact] {task}"[:60]
    fact_result = kb_add(content=fact_content, category="memory", title=fact_title,
                         suppress_duplicates=True, semantic_gate=True, gate_topk=3, scene=sc)
    # 经验层: 可选 0-N 条 (Hy-Memory: must pass a value threshold — 由调用方判断价值)
    exp_result = None
    if experience.strip():
        exp_parts = [f"经验: {experience}"]
        if experience_conditions:
            exp_parts.append(f"适用条件: {experience_conditions}")
        if workspace:
            exp_parts.append(f"项目: {workspace}")
        exp_content = " | ".join(exp_parts)
        exp_title = f"[{sc}-exp] {experience}"[:60]
        exp_result = kb_add(content=exp_content, category="memory", title=exp_title,
                            suppress_duplicates=True, semantic_gate=True, gate_topk=3, scene=sc)
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("commit_calls", elapsed)
    return {
        "scene": sc,
        "fact": {"id": fact_result.get("id"), "status": fact_result.get("status"),
                 "title": fact_result.get("title")},
        "experience": ({"id": exp_result.get("id"), "status": exp_result.get("status"),
                        "title": exp_result.get("title")} if exp_result else None),
        "elapsed_ms": elapsed,
    }


@mcp.tool()
def kb_import(entries: list, category: str = "general", auto_title: bool = True, scene: str = "",
              kind: str = "", dup_threshold: float = 0.94) -> dict:
    """批量写入多条知识到知识库(每条独立去重)。entries: 字符串列表(每条一段内容, 共享category) 或 [{"content":..., "category":..., "title":..., "scene":..., "kind":...}] 列表。
    auto_title: 未提供title时取content前30字(默认true)。scene: 场景标注 chat/working/coding(默认""不标注), 条目内 scene 可逐条覆盖。
    kind: 记忆类型 fact/episode/preference/lesson/rule/profile/task(默认""自动推断)。
    dup_threshold: 批量写入的向量近重复阈值(默认0.94; 高于此值视为同一条并 confirm, 不进 LLM)。
    返回每条的写入结果。"""
    t0 = time.time()
    results = []
    added = 0
    confirmed = 0
    with _zvec_lock:
        col = get_collection()
        existing_docs = list(_iter_all_docs(col))
        hash_map = {r.fields.get("content_hash", ""): r.id for r in existing_docs if r.fields.get("content_hash")}
        fp_map = {}
        for r in existing_docs:
            fp = ma.content_fingerprint(r.fields.get("content", ""))
            if fp:
                fp_map.setdefault(fp, r.id)

        for e in entries:
            if isinstance(e, dict):
                content = e.get("content", "")
                cat = e.get("category", category)
                title = e.get("title", "")
                ent_scene = e.get("scene", scene)
                ent_kind = e.get("kind", kind)
            else:
                content = str(e)
                cat = category
                title = ""
                ent_scene = scene
                ent_kind = kind
            ent_scene = _norm_scene(ent_scene)
            if not content.strip():
                continue
            if not title and auto_title:
                title = content[:30]
            content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
            fingerprint = ma.content_fingerprint(content)
            kind_code = ma.kind_code(ent_kind) or ma.infer_kind(cat, title, content, ent_scene)

            # 1) 精确 hash / 标点归一指纹去重
            dup_id = hash_map.get(content_hash) or fp_map.get(fingerprint)
            if dup_id:
                r = next((d for d in existing_docs if d.id == dup_id), None)
                existing_trust = (r.fields.get("trust_score", TRUST_DEFAULT) if r else TRUST_DEFAULT)
                try:
                    new_trust = min(1.0, float(existing_trust) + TRUST_CONFIRM_BOOST)
                except Exception:
                    new_trust = TRUST_DEFAULT
                col.update([zvec.Doc(id=dup_id, fields={"trust_score": new_trust, "updated_at": int(time.time())})])
                results.append({"title": title, "category": cat, "status": "confirmed", "id": dup_id})
                confirmed += 1
                continue

            # 2) 向量近重复: 批量导入最容易绕过语义闸门，这里做一道廉价闸
            full_text = title + " " + content
            emb = embed(full_text)
            dup_id = None
            try:
                near = col.query(zvec.Query("embedding", vector=emb), topk=1)
                if near and float(near[0].score) >= dup_threshold:
                    dep = near[0].fields.get("deprecated", 0)
                    if isinstance(dep, str) and dep.isdigit():
                        dep = int(dep)
                    if dep < 1:
                        dup_id = near[0].id
            except Exception:
                pass
            if dup_id:
                col.update([zvec.Doc(id=dup_id, fields={"trust_score": TRUST_DEFAULT, "updated_at": int(time.time())})])
                results.append({"title": title, "category": cat, "status": "confirmed_near_duplicate", "id": dup_id})
                confirmed += 1
                continue

            # 3) 新条目
            doc_id = f"kb_{int(time.time())}_{content_hash}"
            col.insert([zvec.Doc(
                id=doc_id, vectors={"embedding": emb},
                fields={
                    "title": title, "content": content, "category": cat,
                    "content_hash": content_hash,
                    "trust_score": TRUST_DEFAULT, "updated_at": int(time.time()),
                    "deprecated": 0, "hit_count": 0,
                    "scene": _scene_code(ent_scene),
                    "kind": kind_code,
                }
            )])
            hash_map[content_hash] = doc_id
            fp_map[fingerprint] = doc_id
            try:
                _sync_doc_to_md(zvec.Doc(id=doc_id, fields={
                    "title": title, "content": content, "category": cat,
                    "trust_score": TRUST_DEFAULT, "updated_at": int(time.time()), "deprecated": 0, "hit_count": 0,
                    "scene": _scene_code(ent_scene), "kind": kind_code,
                }))
            except Exception as e:
                print(f"[MCP-KB] md镜像同步失败: {e}", file=sys.stderr, flush=True)
            results.append({"title": title, "category": cat, "status": "ok", "id": doc_id,
                            "kind": ma.kind_name(kind_code) or None})
            added += 1
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("import_calls", elapsed)
    return {"total": len(results), "added": added, "confirmed": confirmed, "results": results, "elapsed_ms": elapsed}



@mcp.tool()
def kb_trust(action: str, fact_id: str = "", query: str = "", category: str = "") -> dict:
    """管理知识库中条目的信任评分。action: confirm(确认强化)/contradict(矛盾降权)/deprecate(标记废弃)/restore(恢复)/list_deprecated(列出废弃)/list_low_trust(列出低信任)/reset_all(紧急重置所有)/restore_all(恢复所有到1.0)。fact_id:针对单条操作 / query:按语义匹配操作 / category:按分类批量操作"""
    t0 = time.time()
    with _db_lock(15):  # 2026-10-24: 读工具带超时, 锁忙快速报错不挂死
        col = get_collection()

        if action == "list_deprecated":
            all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
            results = []
            for r in all_docs:
                dep = r.fields.get("deprecated", 0)
                if isinstance(dep, str): dep = int(dep) if dep.isdigit() else 0
                if dep >= 1:
                    results.append({"id": r.id, "title": r.fields.get("title", ""), "trust": r.fields.get("trust_score", "?")})
            elapsed = round((time.time() - t0) * 1000)
            _record_metric("trust_calls", elapsed)
            return {"deprecated": results, "count": len(results)}
        
        elif action == "list_low_trust":
            all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
            results = []
            for r in all_docs:
                trust = r.fields.get("trust_score", TRUST_DEFAULT)
                try: trust = float(trust)
                except: trust = TRUST_DEFAULT
                if trust < TRUST_DEPRECATED_THRESHOLD:
                    results.append({"id": r.id, "title": r.fields.get("title", ""), "trust": round(trust, 2)})
            elapsed = round((time.time() - t0) * 1000)
            _record_metric("trust_calls", elapsed)
            return {"low_trust": results, "count": len(results)}
        
        elif action == "reset_all":
            """紧急重置: 所有条目trust_score=1.0, deprecated=0"""
            all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
            now = int(time.time())
            batch = []
            for r in all_docs:
                batch.append(zvec.Doc(id=r.id, fields={"trust_score": 1.0, "deprecated": 0, "updated_at": now}))
            if batch:
                col.update(batch)
            elapsed = round((time.time() - t0) * 1000)
            _record_metric("trust_calls", elapsed)
            return {"action": "reset_all", "count": len(batch)}
        
        elif action == "restore_all":
            """恢复所有: 所有条目trust_score=1.0, deprecated=0 (同reset_all但语义更清晰)"""
            all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
            now = int(time.time())
            batch = []
            for r in all_docs:
                batch.append(zvec.Doc(id=r.id, fields={"trust_score": 1.0, "deprecated": 0, "updated_at": now}))
            if batch:
                col.update(batch)
            elapsed = round((time.time() - t0) * 1000)
            _record_metric("trust_calls", elapsed)
            return {"action": "restore_all", "count": len(batch)}
        
        # 找到目标条目
        targets = []
        if fact_id:
            targets = [fact_id]
        elif query:
            vec = embed(query)
            hits = col.query(zvec.Query("embedding", vector=vec), topk=5)
            targets = [r.id for r in hits]
        elif category:
            all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
            targets = [r.id for r in all_docs if r.fields.get("category", "") == category]
        
        if not targets:
            elapsed = round((time.time() - t0) * 1000)
            _record_metric("trust_calls", elapsed)
            return {"error": "未找到目标条目", "action": action}
        
        updated = []
        for tid in targets:
            # 先获取当前值
            fetch = col.fetch([tid])
            if not fetch or tid not in fetch:
                continue
            
            doc_data = fetch[tid]
            current_trust = doc_data.fields.get("trust_score", TRUST_DEFAULT)
            try: current_trust = float(current_trust)
            except: current_trust = TRUST_DEFAULT
            
            if action == "confirm":
                new_trust = min(1.0, current_trust + TRUST_CONFIRM_BOOST)
                col.update([zvec.Doc(id=tid, fields={"trust_score": new_trust, "updated_at": int(time.time())})])
                updated.append({"id": tid, "trust": round(new_trust, 2), "action": "confirmed"})
            
            elif action == "contradict":
                new_trust = max(0.0, current_trust - TRUST_CONTRADICT_DECAY)
                dep = 1 if new_trust < TRUST_DEPRECATED_THRESHOLD else 0
                col.update([zvec.Doc(id=tid, fields={"trust_score": new_trust, "updated_at": int(time.time()), "deprecated": dep})])
                updated.append({"id": tid, "trust": round(new_trust, 2), "deprecated": dep, "action": "contradicted"})
            
            elif action == "deprecate":
                col.update([zvec.Doc(id=tid, fields={"trust_score": 0.0, "deprecated": 1, "updated_at": int(time.time())})])
                updated.append({"id": tid, "trust": 0.0, "deprecated": 1, "action": "deprecated"})
            
            elif action == "restore":
                col.update([zvec.Doc(id=tid, fields={"trust_score": 0.5, "deprecated": 0, "updated_at": int(time.time())})])
                updated.append({"id": tid, "trust": 0.5, "deprecated": 0, "action": "restored"})
    
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("trust_calls", elapsed)
    return {"action": action, "updated": updated, "count": len(updated)}


@mcp.tool()
def kb_stats() -> dict:
    """查看Hermes知识库统计信息(含信任评分分布+性能指标)"""
    t0 = time.time()
    with _db_lock(15):  # 2026-10-24: 读工具带超时, 锁忙快速报错不挂死
        col = get_collection()
        s = col.stats
        
        # 计算信任分布
        all_docs = col.query(zvec.Query("embedding", vector=[0.0]*384), topk=10000)
        trust_buckets = {"high_0.7+": 0, "medium_0.3-0.7": 0, "low_0.0-0.3": 0, "deprecated": 0}
        categories = {}
        scene_counts = {"chat": 0, "working": 0, "coding": 0, "unlabeled": 0}
        kind_counts = {}
        pointer_count = 0
        
        for r in all_docs:
            trust = r.fields.get("trust_score", TRUST_DEFAULT)
            try: trust = float(trust)
            except: trust = TRUST_DEFAULT
            
            dep = r.fields.get("deprecated", 0)
            if isinstance(dep, str): dep = int(dep) if dep.isdigit() else 0
            
            if dep >= 1:
                trust_buckets["deprecated"] += 1
            elif trust >= 0.7:
                trust_buckets["high_0.7+"] += 1
            elif trust >= 0.3:
                trust_buckets["medium_0.3-0.7"] += 1
            else:
                trust_buckets["low_0.0-0.3"] += 1
            
            cat = r.fields.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            sc = _read_scene_field(r.fields)
            scene_counts[sc if sc else "unlabeled"] += 1
            kn = ma.kind_name(r.fields.get("kind", 0)) or "unspecified"
            kind_counts[kn] = kind_counts.get(kn, 0) + 1
            if dep < 1 and ma.is_pointer_doc(r.fields):
                pointer_count += 1
    
    elapsed = round((time.time() - t0) * 1000)
    
    # 附带性能指标
    with _metrics_lock:
        metrics_copy = dict(_metrics)
    
    return {
        "doc_count": s.doc_count,
        "db_path": DB_PATH,
        "trust_distribution": trust_buckets,
        "categories": categories,
        "scene_distribution": scene_counts,
        "kind_distribution": kind_counts,
        "pointer_artifacts": pointer_count,
        "metrics": metrics_copy,
    }


def _load_views(col) -> list:
    """Load all docs with vectors for consolidation/health analysis."""
    return list(col.iter_docs(include_vector=True))


def _batch_update(col, docs: list, size: int = 500) -> None:
    """zvec 单次写入上限 1024 条; 全库回填/批量废弃必须分片."""
    for i in range(0, len(docs), max(1, int(size))):
        col.update(docs[i:i + size])


def _batch_delete(col, ids: list, size: int = 500) -> None:
    for i in range(0, len(ids), max(1, int(size))):
        col.delete(ids[i:i + size])


@mcp.tool()
def kb_health(dup_threshold: float = 0.92, stale_days: int = 365, fix: bool = False) -> dict:
    """知识库体检 (memory-brain health.py 适配版): 重复/指针垃圾/低信任/类型缺失/超大条目。

    dup_threshold: 近重复判定阈值(默认0.92)
    stale_days: 低信任条目的“陈旧”天数(默认365, 只影响报告与 fix 的候选)
    fix: 是否顺手废弃安全候选(指针文件 + 长期低信任零命中条目), 默认false只报告。
    返回 health_score(0-100) + issues + 分类/类型分布。"""
    t0 = time.time()
    with _db_lock(60):
        col = get_collection()
        docs = _load_views(col)
        report = ma.health_report(docs, dup_threshold=dup_threshold, stale_days=stale_days)
        actions = []
        if fix:
            candidates = ma.select_forgetting(docs, stale_days=stale_days, min_trust=0.5)
            if candidates:
                now = int(time.time())
                _batch_update(col, [zvec.Doc(id=c["id"], fields={
                    "deprecated": 1, "trust_score": 0.0, "updated_at": now,
                }) for c in candidates])
                actions = [{"id": c["id"], "reason": c["reason"]} for c in candidates]
                # refresh report so the caller sees post-fix numbers, not stale ones
                report = ma.health_report(_load_views(col), dup_threshold=dup_threshold,
                                          stale_days=stale_days)
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("health_calls", elapsed)
    return {**report, "actions": actions, "fix": fix,
            "elapsed_ms": elapsed}


@mcp.tool()
def kb_dedup(threshold: float = 0.92, dry_run: bool = True, same_category: bool = True,
             purge: bool = False, max_groups: int = 100) -> dict:
    """语义去重/合并 (memory-brain consolidate.py 适配版)。

    threshold: 余弦相似度阈值(默认0.92; 0.95=几乎同文, 0.88=同主题不同措辞)
    dry_run: 默认true只返回合并计划, 不改库; false 才执行
    same_category: 默认true只合并同分类, 避免把 user 偏好并进 memory 任务记录
    purge: 合并后是否物理删除 loser(默认false, 软废弃可恢复)
    保留策略: trust 最高 → hit_count 最多 → 内容最长; 合并 hit_count 并保留最长正文。
    """
    t0 = time.time()
    with _db_lock(120):
        col = get_collection()
        docs = _load_views(col)
        clusters = ma.duplicate_clusters(docs, threshold=threshold, same_category=same_category)
        clusters = clusters[:max(1, int(max_groups))]
        plans = []
        for cluster in clusters:
            plan = ma.merge_plan(cluster, docs)
            titles = {str(docs[i].fields.get("title", ""))[:50] for i in cluster}
            plans.append({
                "keep_id": plan["keep_id"],
                "drop_ids": plan["drop_ids"],
                "size": plan["size"],
                "titles": list(titles)[:6],
                "keep_preview": plan["keep_fields"]["content"][:120],
            })
            if not dry_run:
                keep = zvec.Doc(id=plan["keep_id"], fields=plan["keep_fields"])
                losers = [zvec.Doc(id=i, fields={"deprecated": 1, "trust_score": 0.0,
                                                 "updated_at": int(time.time())})
                          for i in plan["drop_ids"]]
                col.update([keep] + losers)
                try:
                    _sync_doc_to_md(keep)
                except Exception:
                    pass
                for drop_id in plan["drop_ids"]:
                    if purge:
                        try:
                            col.delete([drop_id])
                        except Exception:
                            pass
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("dedup_calls", elapsed)
    return {
        "threshold": threshold, "same_category": same_category, "dry_run": dry_run,
        "groups": len(plans), "docs_in_groups": sum(p["size"] for p in plans),
        "would_merge" if dry_run else "merged": sum(p["size"] - 1 for p in plans),
        "plans": plans, "elapsed_ms": elapsed,
    }


@mcp.tool()
def kb_reindex(limit: int = 5000, force: bool = False) -> dict:
    """回填记忆类型 kind 字段: 按 category/title/content 规则推断 (fact/episode/preference/lesson/rule/profile/task)。

    limit: 单次最多处理条数(默认5000)。幂等, 可重复运行; 默认只处理 kind=0 的条目。
    force: true 时对所有条目重新推断(规则升级后修正旧标签用)。
    """
    t0 = time.time()
    with _db_lock(120):
        col = get_collection()
        docs = list(_iter_all_docs(col))
        updates = []
        counts = {}
        for r in docs[:max(1, int(limit))]:
            if not force and int(r.fields.get("kind", 0) or 0) != 0:
                continue
            code = ma.infer_kind(r.fields.get("category", ""), r.fields.get("title", ""),
                                 r.fields.get("content", ""), _read_scene_field(r.fields))
            if not force and code == int(r.fields.get("kind", 0) or 0):
                continue
            updates.append(zvec.Doc(id=r.id, fields={"kind": code}))
            name = ma.kind_name(code) or "unspecified"
            counts[name] = counts.get(name, 0) + 1
        if updates:
            _batch_update(col, updates)
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("reindex_calls", elapsed)
    return {"updated": len(updates), "kinds": counts, "elapsed_ms": elapsed}


@mcp.tool()
def kb_optimize(deprecate_days: int = 365, cleanup_days: int = 730,
                dedupe: bool = False, dedupe_threshold: float = 0.92) -> dict:
    """优化知识库: 保守遗忘(指针垃圾 + 长期低信任零命中) + 物理清理已废弃条目 + 可选语义去重。
    deprecate_days: 低信任条目的陈旧阈值(默认365天; 不再按90天一刀切)
    cleanup_days: 物理删除已废弃条目的天数(默认730天)
    dedupe: 是否同时执行语义合并(默认false; 合并前建议先 kb_dedup dry_run 预览)
    """
    t0 = time.time()
    dedupe_result = None
    with _db_lock(60):  # kb_optimize 是全库扫描+清理, 给长超时; 超时仍快速报错不挂死
        col = get_collection()
        deprecated_count = _auto_deprecate(col, stale_days=deprecate_days)
        cleaned_count = _cleanup_deprecated(col, days=cleanup_days)
        if dedupe:
            docs = _load_views(col)
            clusters = ma.duplicate_clusters(docs, threshold=dedupe_threshold, same_category=True)
            merged = 0
            for cluster in clusters:
                plan = ma.merge_plan(cluster, docs)
                col.update([zvec.Doc(id=plan["keep_id"], fields=plan["keep_fields"])] +
                           [zvec.Doc(id=i, fields={"deprecated": 1, "trust_score": 0.0,
                                                   "updated_at": int(time.time())})
                            for i in plan["drop_ids"]])
                merged += len(plan["drop_ids"])
            dedupe_result = {"groups": len(clusters), "merged": merged,
                             "threshold": dedupe_threshold}
        final_count = col.stats.doc_count
    
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("optimize_calls", elapsed)
    return {
        "auto_deprecated": deprecated_count,
        "cleaned": cleaned_count,
        "dedupe": dedupe_result,
        "final_doc_count": final_count,
        "deprecate_threshold_days": deprecate_days,
        "cleanup_threshold_days": cleanup_days,
        "elapsed_ms": elapsed,
    }


@mcp.tool()
def kb_tree(include_deprecated: bool = False, include_abstract: bool = False) -> dict:
    """返回知识库的 Obsidian 式目录树: 按 category 分层, 每层列出条目标题。include_deprecated: 是否包含已废弃条目(默认false)。
    include_abstract: 附带每分类的 L0 抽取式摘要(高频命中条目首句拼接, 零LLM成本, 借鉴OpenViking include_abstract)"""
    t0 = time.time()
    with _db_lock(15):  # 2026-10-24: 读工具带超时, 锁忙快速报错不挂死
        col = get_collection()
        tree = {}
        cat_hits = {}
        for r in _iter_all_docs(col):
            dep = r.fields.get("deprecated", 0)
            try:
                dep = int(dep)
            except Exception:
                dep = 0
            if dep >= 1 and not include_deprecated:
                continue
            cat = r.fields.get("category", "general") or "general"
            tree.setdefault(cat, []).append({
                "id": r.id,
                "title": r.fields.get("title", "") or r.fields.get("content", "")[:30],
                "trust": round(float(r.fields.get("trust_score", TRUST_DEFAULT) or TRUST_DEFAULT), 2),
                "deprecated": dep,
                "scene": _read_scene_field(r.fields) or None,
            })
            if include_abstract:
                try:
                    hits = int(r.fields.get("hit_count", 0) or 0)
                except Exception:
                    hits = 0
                first_sent = _make_l0(r.fields.get("content", ""))[:60]
                cat_hits.setdefault(cat, []).append((hits, first_sent))
        abstracts = {}
        if include_abstract:
            for cat, items in cat_hits.items():
                items.sort(key=lambda x: x[0], reverse=True)
                abstracts[cat] = " / ".join(s for _, s in items[:8] if s)
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("tree_calls", elapsed)
    out = {"tree": tree, "categories": list(tree.keys()), "total": sum(len(v) for v in tree.values()), "elapsed_ms": elapsed}
    if include_abstract:
        out["abstracts"] = abstracts
    return out


@mcp.tool()
def kb_overview(category: str, force: bool = False) -> dict:
    """生成/读取某分类的 L1 概览 (借鉴 OpenViking overview, 按需LLM非后台)。
    category: 分类名; force: 强制重新生成(默认false)。
    首次调用走一次 LLM(约2-5s), 结果缓存到 notes/<category>/.overview.md; 内容未变时缓存命中零LLM成本。
    缓存失效条件: 分类下条目集合或内容变化(content hash)。"""
    t0 = time.time()
    cat = (category or "").strip()
    if not cat:
        return {"error": "category required"}
    with _zvec_lock:
        col = get_collection()
        entries = []
        for r in _iter_all_docs(col):
            dep = r.fields.get("deprecated", 0)
            try:
                dep = int(dep)
            except Exception:
                dep = 0
            if dep >= 1:
                continue
            if (r.fields.get("category", "general") or "general") != cat:
                continue
            entries.append({
                "id": r.id,
                "title": r.fields.get("title", "") or r.fields.get("content", "")[:30],
                "l0": _make_l0(r.fields.get("content", "")),
                "updated_at": int(r.fields.get("updated_at", 0) or 0),
                "hit_count": int(r.fields.get("hit_count", 0) or 0),
            })
    if not entries:
        return {"error": f"category '{cat}' not found or empty"}
    entries.sort(key=lambda x: x["updated_at"], reverse=True)
    cache_key = hashlib.md5(
        ("|".join(e["id"] + e["l0"] for e in sorted(entries, key=lambda x: x["id"]))).encode()
    ).hexdigest()[:16]
    ov_path = NOTES_ROOT / cat / ".overview.md"
    if ov_path.exists() and not force:
        try:
            text = ov_path.read_text(encoding="utf-8")
            if f"cache_key: {cache_key}" in text:
                # 缓存命中: 正文 = frontmatter 之后部分
                body = text.split("---", 2)[-1].strip()
                elapsed = round((time.time() - t0) * 1000)
                _record_metric("overview_calls", elapsed)
                return {"category": cat, "overview": body, "entries": len(entries),
                        "cache": "hit", "elapsed_ms": elapsed}
        except Exception:
            pass
    # 缓存未命中 → LLM 生成 (锁外调用, 参照语义闸门纪律)
    listing = "\n".join(f"- {e['title']}: {e['l0']}" for e in entries[:40])
    system = (
        "你是知识库目录概览生成器。给定一个分类下的条目清单(标题+首句), 生成该分类的 L1 概览:\n"
        "1. 第一段: 一句话 L0 概览(80字内), 概括这个分类存了什么、适合查什么。\n"
        "2. 第二段起: 按主题归组列出条目导航(每条一行, 格式 `- 标题 — 一句话说明`), 按重要性排序。\n"
        "不要寒暄, 不要输出清单之外的内容, 总长 500 字内。"
    )
    try:
        from llm_helper import llm_summarize
        body = llm_summarize(system, f"分类: {cat}\n条目({len(entries)}条):\n{listing}",
                             max_tokens=2048, temperature=0.2)
    except Exception as e:
        body = ""
        print(f"[MCP-KB] overview LLM failed: {e}", file=sys.stderr, flush=True)
    if not body:
        # LLM 不可用 → 抽取式兜底(零LLM), 不阻塞
        top = sorted(entries, key=lambda x: x["hit_count"], reverse=True)[:10]
        body = f"{cat} 分类概览(抽取式): " + " / ".join(e["l0"][:50] for e in top if e["l0"])
        llm_used = False
    else:
        llm_used = True
    try:
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        ov_path.write_text(
            "---\n"
            f"category: {cat}\n"
            f"cache_key: {cache_key}\n"
            f"entries: {len(entries)}\n"
            f"generated: {time.strftime('%Y-%m-%d %H:%M')}\n"
            "---\n\n" + body + "\n",
            encoding="utf-8")
    except Exception as e:
        print(f"[MCP-KB] overview 写缓存失败: {e}", file=sys.stderr, flush=True)
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("overview_calls", elapsed)
    return {"category": cat, "overview": body, "entries": len(entries),
            "cache": "miss", "llm": llm_used, "elapsed_ms": elapsed}


@mcp.tool()
def kb_export(notes_root: str = "") -> dict:
    """导出知识库为 Obsidian 兼容的 markdown 目录树: notes/<category>/<title>.md。notes_root: 导出根目录(默认/root/hermes-kb/notes, 可用KB_NOTES_ROOT覆盖)。返回导出统计与文件清单。"""
    t0 = time.time()
    root = Path(notes_root) if notes_root else NOTES_ROOT
    with _zvec_lock:
        col = get_collection()
        exported = []
        for r in _iter_all_docs(col):
            dep = r.fields.get("deprecated", 0)
            if isinstance(dep, str):
                dep = int(dep) if dep.isdigit() else 0
            if dep >= 1:
                # 废弃条目: 删除其旧镜像, 不导出
                old_title = r.fields.get("title", "") or r.fields.get("content", "")[:30]
                old_cat = r.fields.get("category", "general") or "general"
                for old in (root / old_cat).glob(f"{_safe_filename(old_title)}.md"):
                    try:
                        old.unlink()
                    except Exception:
                        pass
                continue
            path = _sync_doc_to_md(r)
            if path.parent.name != r.fields.get("category", "general"):
                # 类别变更残留: 旧文件由 sync 阶段统一清理
                pass
            exported.append(str(path))
    # 清理孤儿: 删除 notes/ 下不再对应任何条目的 md (跳过 .overview.md 等隐藏系统文件)
    live_paths = set(exported)
    removed = 0
    for p in root.rglob("*.md"):
        if p.name.startswith("."):
            continue
        if str(p) not in live_paths:
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    # 生成 README 索引
    try:
        _write_notes_readme(root)
    except Exception:
        pass
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("export_calls", elapsed)
    return {"root": str(root), "exported": len(exported), "removed_orphans": removed, "elapsed_ms": elapsed}


def _write_notes_readme(root: Path):
    """在 notes 根生成 README.md 目录索引 (Obsidian 打开即可见层级)"""
    lines = ["# Hermes 知识库目录树", "", "> 由 `kb_export` 自动生成, 每 3h 后台同步。分类目录如下:", ""]
    for cat_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = sorted(p for p in cat_dir.glob("*.md") if p.name != "README.md")
        if not files:
            continue
        lines.append(f"## {cat_dir.name} ({len(files)})")
        for f in files:
            lines.append(f"- [[{f.stem}]]")
        lines.append("")
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


# === KB 冲突检测 (2026-09-04 用户架构需求) ===
# 目的: 找"同一 topic 但结论矛盾"的两条高 trust 条目
# 策略: 检索 top-N → 过滤 deprecated+低 trust → 两两比对 score 接近度 → LLM 判语义冲突
# 输出: 冲突报告(不修改任何条目, 由用户/agent 决定 deprecate 谁)


def _llm_judge_conflict(topic: str, text_a: str, text_b: str) -> dict:
    """调 LLM 判两条结论是否矛盾并裁决谁对谁错.
    返回 {conflict, reason, severity, winner: 'a'|'b'|''}"""
    try:
        from llm_helper import llm_summarize
        system = (
            "你是 KB 知识冲突仲裁员。给定 topic 和两条候选结论, 判断是否矛盾, 矛盾时裁决哪个正确。"
            "输出严格 JSON: {\"conflict\": true|false, \"reason\": \"<50字内>\", \"severity\": \"high|medium|low|none\", \"winner\": \"a\"|\"b\"|\"\"}。"
            "severity 含义: high=数值/事实直接矛盾(如 60% vs 70%); medium=部分重叠但侧重不同(分工而非冲突); low=看起来矛盾但 context 不同; none=不冲突。"
            "winner: conflict=true 时必填, 指结论正确的那个(依据=数值合理性/内容具体性, 与检索分数无关); conflict=false 时留空。"
            "不要寒暄, 不要解释 JSON 之外内容。"
        )
        user = (
            f"Topic: {topic}\n\n"
            f"结论 A:\n{text_a[:500]}\n\n"
            f"结论 B:\n{text_b[:500]}\n\n"
            f"判断 A 和 B 是否矛盾? 矛盾时哪个正确?"
        )
        raw = llm_summarize(system, user, max_tokens=300, temperature=0.1)
        if not raw:
            return {"conflict": False, "reason": "LLM unavailable, skipped", "severity": "none", "winner": ""}
        # 提取 JSON
        import re
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not m:
            return {"conflict": False, "reason": f"LLM parse fail: {raw[:80]}", "severity": "none", "winner": ""}
        import json as _json
        result = _json.loads(m.group(0))
        result.setdefault("winner", "")
        return result
    except Exception as e:
        return {"conflict": False, "reason": f"judge error: {e}", "severity": "none", "winner": ""}


async def _llm_judge_conflict_async(topic: str, text_a: str, text_b: str,
                                    semaphore: "asyncio.Semaphore" = None) -> dict:
    """异步 LLM 判冲突 (httpx.AsyncClient). 与 _llm_judge_conflict 同步版逻辑一致.
    semaphore: 传入用于限流并发, 避免拖死 server."""
    if semaphore:
        await semaphore.acquire()
    try:
        from llm_helper import get_model_config
        cfg = get_model_config()
        if not cfg.get("base_url") or not cfg.get("api_key"):
            return {"conflict": False, "reason": "no model config", "severity": "none", "confidence": 0.0}

        system = (
            "你是 KB 知识冲突仲裁员。给定 topic 和两条候选结论, 判断是否矛盾, 矛盾时裁决哪个正确。"
            "输出严格 JSON: {\"conflict\": true|false, \"reason\": \"<50字内>\", \"severity\": \"high|medium|low|none\", \"confidence\": 0.0-1.0, \"winner\": \"a\"|\"b\"|\"\"}。"
            "severity 含义: high=数值/事实直接矛盾(如 60% vs 70%); medium=部分重叠但侧重不同(分工而非冲突); low=看起来矛盾但 context 不同; none=不冲突。"
            "confidence: 你对这个判断的把握(0.0-1.0, 模糊情况写 0.5 以下)。"
            "winner: conflict=true 时必填, 指结论正确的那个(依据=数值合理性/内容具体性, 与检索分数无关); conflict=false 时留空。"
            "不要寒暄, 不要解释 JSON 之外内容。"
        )
        user = (
            f"Topic: {topic}\n\n"
            f"结论 A:\n{text_a[:500]}\n\n"
            f"结论 B:\n{text_b[:500]}\n\n"
            f"判断 A 和 B 是否矛盾? 矛盾时哪个正确?"
        )

        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{cfg['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    # 2026-09-09: 512 对 reasoning 模型不够(并发时全部 token 耗在思考链,
                    # content=null → parse fail)。对齐 llm_helper 的 max(max_tokens, 4096) 经验值。
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            # 2026-09-09: 不回退 reasoning_content — 仲裁 JSON 必须来自正式输出,
            # 思考草稿无法产出有效裁决 JSON(与 llm_helper 污染修复同一原则)。

        import re, json as _json
        m = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if not m:
            return {"conflict": False, "reason": f"parse fail: {content[:60]}", "severity": "none", "confidence": 0.0, "winner": ""}
        result = _json.loads(m.group(0))
        # 标准化 confidence
        if "confidence" not in result:
            result["confidence"] = 0.8  # 兼容旧 LLM, 默认高
        result.setdefault("winner", "")
        return result
    except Exception as e:
        return {"conflict": False, "reason": f"async err: {e}", "severity": "none", "confidence": 0.0}
    finally:
        if semaphore:
            semaphore.release()


# === 缓存层 (24h TTL, 文件持久化) ===
_CONFLICTS_CACHE_PATH = "/tmp/kb_conflicts_cache.json"
_CONFLICTS_CACHE_TTL = 24 * 3600  # 24h


def _cache_load() -> dict:
    import json as _json
    try:
        with open(_CONFLICTS_CACHE_PATH, "r", encoding="utf-8") as f:
            return _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        return {}


def _cache_save(cache: dict) -> None:
    import json as _json
    try:
        with open(_CONFLICTS_CACHE_PATH, "w", encoding="utf-8") as f:
            _json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[KB-Conflicts] cache save failed: {e}", file=sys.stderr, flush=True)


def _cache_pair_key(a_id: str, b_id: str) -> str:
    return "|".join(sorted([a_id, b_id]))


def _cache_get(a_id: str, b_id: str) -> dict | None:
    cache = _cache_load()
    key = _cache_pair_key(a_id, b_id)
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > _CONFLICTS_CACHE_TTL:
        return None
    return entry.get("judge")


def _cache_put(a_id: str, b_id: str, judge: dict) -> None:
    cache = _cache_load()
    key = _cache_pair_key(a_id, b_id)
    cache[key] = {"ts": time.time(), "judge": judge}
    # 限制缓存大小 (防无限增长)
    if len(cache) > 5000:
        # 删除最旧的 1000 条
        sorted_keys = sorted(cache.keys(), key=lambda k: cache[k].get("ts", 0))
        for k in sorted_keys[:1000]:
            cache.pop(k, None)
    _cache_save(cache)


# === KB 趋势统计 ===
def _kb_stats_snapshot(col) -> dict:
    """返回 KB 当前快照 + 30 天新增/废弃趋势"""
    now = int(time.time())
    cutoff_30d = now - 30 * 86400
    total = 0
    deprecated = 0
    added_30d = 0
    deprecated_30d = 0
    all_docs = col.query(zvec.Query("embedding", vector=[0.0] * 384), topk=10000)
    for r in all_docs:
        total += 1
        dep = r.fields.get("deprecated", 0)
        if isinstance(dep, str):
            try: dep = int(dep)
            except: dep = 0
        updated_at = r.fields.get("updated_at", 0)
        if isinstance(updated_at, str):
            try: updated_at = int(updated_at)
            except: updated_at = 0

        if dep >= 1:
            deprecated += 1
            if updated_at >= cutoff_30d:
                deprecated_30d += 1
        else:
            if updated_at >= cutoff_30d:
                added_30d += 1
    return {
        "total_docs": total,
        "active_docs": total - deprecated,
        "deprecated_docs": deprecated,
        "added_30d": added_30d,
        "deprecated_30d": deprecated_30d,
        "net_growth_30d": added_30d - deprecated_30d,
    }


# === 自动 deprecate 策略 (C 方案) ===
def _should_auto_deprecate(pair: dict, confidence_threshold: float = 0.9,
                           age_diff_days: int = 7,
                           score_sanity_pct: float = 0.30) -> tuple[bool, str, str]:
    """判断一对冲突是否可以自动 deprecate (2026-09-09 winner 裁决版).
    旧版按「检索 score 低者输」定向 — 实测会方向反转: 新写入的错误条目措辞更贴 query
    score 反而更高, 会把权威结论废掉保留错误条目 (09-09 用 99% vs 70% 实测复现)。
    新规则: HIGH severity + confidence>=阈值 + LLM 给出 winner 裁决 → loser=裁决输家。
    防误杀门槛:
    - loser 比 winner 旧 → 直接允许 (废弃旧的错误结论, 低风险)
    - loser 比 winner 新 (新条目输) → 仅当 winner 检索 score 与 loser 相差 <= score_sanity_pct
      才允许 (score 只是检索相似度不是正确性, 仅作 sanity check; winner 几乎检不到时留给用户)
    age_diff_days 参数保留兼容 cron 调用, winner 裁决版不再使用。
    返回 (should_deprecate, reason, loser_side: 'a'|'b'|'')"""
    a, b = pair["a"], pair["b"]
    severity = pair.get("severity", "none")
    confidence = pair.get("confidence", 0.0)
    winner = (pair.get("winner") or "").strip().lower()
    if severity != "high":
        return False, f"severity={severity} != high", ""
    if confidence < confidence_threshold:
        return False, f"confidence {confidence} < {confidence_threshold}", ""
    if winner not in ("a", "b"):
        return False, f"无 winner 裁决 (winner={winner!r}), 需用户判断", ""
    loser_side = "b" if winner == "a" else "a"
    loser, win = pair[loser_side], pair[winner]
    if loser["updated_at"] >= win["updated_at"]:
        denom = max(loser["score"], 1e-6)
        gap_pct = abs(loser["score"] - win["score"]) / denom
        if gap_pct > score_sanity_pct:
            return False, (f"loser({loser_side}) 较新且 winner 检索分差 {gap_pct:.0%} > {score_sanity_pct:.0%}, "
                           f"winner 可能难以检索到, 需用户判断"), ""
        return True, (f"LLM 裁决 winner={winner}, loser={loser_side} 较新但 winner 可检索到(分差{gap_pct:.0%}), "
                      f"conf={confidence}"), loser_side
    return True, (f"LLM 裁决 winner={winner}, loser={loser_side} 较旧, conf={confidence}"), loser_side


@mcp.tool()
def kb_conflicts(query: str, topk: int = 15, score_gap_pct: float = 30.0,
                 min_trust: float = 0.5, use_llm: bool = True,
                 use_cache: bool = True, max_concurrent_llm: int = 5,
                 auto_deprecate: bool = False, confidence_threshold: float = 0.9,
                 age_diff_days: int = 7) -> dict:
    """检测 KB 内同一 topic 的高 trust 结论冲突。
    query: 检索词(主题/问题)
    topk: 检索候选数(默认15)
    score_gap_pct: 候选间 score 差距阈值 % (默认30)
    min_trust: 最低信任分(默认0.5)
    use_llm: 是否调 LLM 判语义冲突(默认true)
    use_cache: 是否用 24h 缓存(默认true, 避免重复 LLM 调用)
    max_concurrent_llm: LLM 并发上限(默认5, 防拖死 server)
    auto_deprecate: 是否自动 deprecate loser (B/C 方案, 默认 false)
    confidence_threshold: LLM confidence 阈值才自动动(默认0.9)
    age_diff_days: updated_at 差距>此天数才自动 deprecate(默认7)

    返回 {query, candidates, conflict_pairs, auto_actions, warnings, kb_stats, elapsed_ms, note}
    本工具默认只检测, auto_deprecate=True 时才修改 KB (HIGH + 高 confidence + 时间差明显)."""
    t0 = time.time()
    with _zvec_lock:
        col = get_collection()
        stats_snapshot = _kb_stats_snapshot(col)
        vec = embed(query)
        vec_raw = col.query(zvec.Query("embedding", vector=vec), topk=topk)
        fts_q = _sanitize_fts_query(query)
        fts_raw = col.query(zvec.Query("content", fts=zvec.Fts(query_string=fts_q)), topk=topk) if fts_q else []
        merged = {}
        for r in vec_raw:
            merged[r.id] = r
        for r in fts_raw:
            if r.id in merged:
                if r.score > merged[r.id].score:
                    merged[r.id] = r
            else:
                merged[r.id] = r
        raw = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:topk]

        candidates = []
        for r in raw:
            trust = r.fields.get("trust_score", 0.5)
            if isinstance(trust, str):
                try: trust = float(trust)
                except: trust = 0.5
            dep = r.fields.get("deprecated", 0)
            if isinstance(dep, str):
                try: dep = int(dep)
                except: dep = 0
            if dep >= 1 or trust < min_trust:
                continue
            candidates.append({
                "id": r.id,
                "score": round(float(r.score), 4),
                "trust": trust,
                "title": r.fields.get("title", "")[:80],
                "content": r.fields.get("content", "")[:500],
                "category": r.fields.get("category", ""),
                "updated_at": r.fields.get("updated_at", 0),
            })

    warnings = []
    if len(candidates) < 2:
        return {
            "query": query,
            "candidates": len(candidates),
            "comparable": 0,
            "conflict_pairs": [],
            "auto_actions": [],
            "warnings": ["候选不足 2 条, 无冲突可判"],
            "kb_stats": stats_snapshot,
            "elapsed_ms": round((time.time() - t0) * 1000),
        }

    # score_gap 过滤
    top1_score = candidates[0]["score"]
    if top1_score > 0:
        threshold = top1_score * (1 - score_gap_pct / 100.0)
    else:
        threshold = float("-inf")
    comparable = [c for c in candidates if c["score"] >= threshold]
    if len(comparable) < 2:
        warnings.append(f"score 差距 > {score_gap_pct}%, 只比较 {len(comparable)} 条, 无 pair")

    # 构建候选对 (先查缓存, 再判 quick_flag)
    candidate_pairs = []
    cache_hits = 0
    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            a, b = comparable[i], comparable[j]
            import re as _re
            nums_a = set(_re.findall(r'\b\d+\.?\d*%?\b', a["content"]))
            nums_b = set(_re.findall(r'\b\d+\.?\d*%?\b', b["content"]))
            shared = nums_a & nums_b
            quick_flag = bool(shared) and nums_a != nums_b
            cached_judge = _cache_get(a["id"], b["id"]) if use_cache else None
            if cached_judge:
                cache_hits += 1
            candidate_pairs.append((a, b, quick_flag, shared, cached_judge))

    if cache_hits:
        warnings.append(f"缓存命中 {cache_hits}/{len(candidate_pairs)} 对 (24h TTL)")

    # LLM 仲裁 (并发, 限流)
    pairs = []
    if use_llm:
        uncached = [(a, b, qf, sh) for a, b, qf, sh, cj in candidate_pairs if cj is None]
        judge_results = {}
        if uncached:
            import asyncio
            sem = asyncio.Semaphore(max_concurrent_llm)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def judge_all():
                    tasks = [_llm_judge_conflict_async(query, a["content"], b["content"], sem)
                             for a, b, _, _ in uncached]
                    return await asyncio.gather(*tasks, return_exceptions=True)
                llm_results = loop.run_until_complete(judge_all())
                loop.close()
                for (a, b, _, _), r in zip(uncached, llm_results):
                    if isinstance(r, Exception):
                        r = {"conflict": False, "reason": f"err: {r}", "severity": "none", "confidence": 0.0, "winner": ""}
                    judge_results[(a["id"], b["id"])] = r
                    # 2026-09-09: 失败判定不进缓存(24h TTL 会把 parse fail 钉死, 阻碍重试)
                    _reason = str(r.get("reason", ""))
                    if not _reason.startswith(("parse fail", "async err", "err:", "no model")):
                        if use_cache:
                            _cache_put(a["id"], b["id"], r)
            except Exception as e:
                warnings.append(f"并发 LLM 失败, 回退串行: {e}")
                for a, b, _, _ in uncached:
                    try:
                        r = _llm_judge_conflict(query, a["content"], b["content"])
                        judge_results[(a["id"], b["id"])] = r
                        if use_cache:
                            _cache_put(a["id"], b["id"], r)
                    except Exception as ex:
                        judge_results[(a["id"], b["id"])] = {"conflict": False, "reason": f"err: {ex}", "severity": "none", "confidence": 0.0}

        for a, b, quick_flag, shared, cached in candidate_pairs:
            judge = cached if cached is not None else judge_results.get((a["id"], b["id"]), {"conflict": False, "reason": "no result", "severity": "none", "confidence": 0.0})
            if judge.get("conflict"):
                pairs.append({
                    "a": {"id": a["id"], "score": a["score"], "trust": a["trust"], "title": a["title"], "updated_at": a["updated_at"]},
                    "b": {"id": b["id"], "score": b["score"], "trust": b["trust"], "title": b["title"], "updated_at": b["updated_at"]},
                    "score_gap": round(a["score"] - b["score"], 4),
                    "conflict": True,
                    "severity": judge.get("severity", "low"),
                    "reason": judge.get("reason", ""),
                    "confidence": judge.get("confidence", 0.0),
                    "winner": judge.get("winner", ""),
                    "quick_flag": quick_flag,
                })
    else:
        # 纯字符串/数值快速判
        for a, b, quick_flag, shared, _ in candidate_pairs:
            if quick_flag:
                pairs.append({
                    "a": {"id": a["id"], "score": a["score"], "trust": a["trust"], "title": a["title"], "updated_at": a["updated_at"]},
                    "b": {"id": b["id"], "score": b["score"], "trust": b["trust"], "title": b["title"], "updated_at": b["updated_at"]},
                    "score_gap": round(a["score"] - b["score"], 4),
                    "conflict": True,
                    "severity": "low",
                    "reason": f"string-only: shared_nums={shared}",
                    "confidence": 0.0,
                    "quick_flag": True,
                })

    # 按 severity 排序 (高置信度排前)
    sev_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    pairs.sort(key=lambda p: (sev_order.get(p["severity"], 3), -p.get("confidence", 0)), reverse=False)

    # 自动 deprecate (B/C 方案, 2026-09-09 winner 裁决版)
    auto_actions = []
    if auto_deprecate and pairs:
        from llm_helper import get_model_config as _gmc  # for sanity check
        for pair in pairs:
            should, reason, loser_side = _should_auto_deprecate(
                pair, confidence_threshold=confidence_threshold, age_diff_days=age_diff_days
            )
            if should and loser_side in ("a", "b"):
                # loser = LLM 裁决的错误方 (不再按检索 score 定向)
                loser_id, loser_title = pair[loser_side]["id"], pair[loser_side]["title"]
                winner_id = pair["b" if loser_side == "a" else "a"]["id"]
                # 实际 deprecate
                with _zvec_lock:
                    try:
                        col.update([zvec.Doc(
                            id=loser_id,
                            fields={"deprecated": 1, "trust_score": 0.0, "updated_at": int(time.time())}
                        )])
                        auto_actions.append({
                            "action": "deprecated",
                            "id": loser_id,
                            "title": loser_title,
                            "winner": winner_id,
                            "reason": reason,
                        })
                    except Exception as e:
                        auto_actions.append({"action": "failed", "id": loser_id, "error": str(e)})
        if auto_actions:
            stats_snapshot = _kb_stats_snapshot(col)  # 刷新

    note_parts = []
    if auto_deprecate:
        note_parts.append(f"auto_deprecate=True: 已自动 deprecate {len([a for a in auto_actions if a.get('action') == 'deprecated'])} 条")
    note_parts.append("建议: HIGH severity + confidence≥0.9 + 时间/score 差距明显 → deprecate loser")
    if use_cache:
        note_parts.append("24h 缓存生效, 相同对不重复 LLM")
    if not auto_deprecate:
        note_parts.append("调用时设 auto_deprecate=True 才会自动动 KB (C 方案, 谨慎)")

    return {
        "query": query,
        "candidates": len(candidates),
        "comparable": len(comparable),
        "conflict_pairs": pairs,
        "auto_actions": auto_actions,
        "warnings": warnings,
        "kb_stats": stats_snapshot,
        "elapsed_ms": round((time.time() - t0) * 1000),
        "note": " | ".join(note_parts),
    }


# === MEMORY.md / USER.md 自动精简 (Iron Rule 5: 先KB化再压缩) ===

MEMORY_MD = Path(os.path.expanduser("~/.hermes/memories/MEMORY.md"))
USER_MD = Path(os.path.expanduser("~/.hermes/memories/USER.md"))
MEMORY_CHAR_LIMIT = 8000
USER_CHAR_LIMIT = 4000


def _compact_memory_file(col, path: Path, category: str, limit: int, pointer_header: str, force: bool = False) -> dict:
    """把一个记忆文件的内容用 LLM 精简摘要后入库 KB, 成功后压缩为指针。
    
    内部用 _zvec_lock 保护 DB 操作; LLM 调用在锁外执行(避免阻塞其他工具)。
    
    返回 {checked, added, confirmed, compressed, method}
    """
    if not path.exists():
        return {"checked": 0, "added": 0, "confirmed": 0, "compressed": False}
    size = path.stat().st_size
    # 2026-09-04 OOB 用户改: 软阈值 60% → 70% (延迟自动精简, 给手写更多缓冲)
    # MEMORY 8000 * 0.7 = 5600 触发, USER 4000 * 0.7 = 2800 触发, 两个文件独立计算
    # 2026-09-09 修 force bug: 原代码声明 force 参数但从未传递/使用, force=True 无法跳过阈值
    if not force and size <= limit * 0.7:
        return {"checked": 0, "added": 0, "confirmed": 0, "compressed": False}
    content = path.read_text(encoding="utf-8").strip()
    # 2026-10-24: 剔除指针头行(← 开头, 历史 compact 产物) — 否则指针文件本身
    # 被当知识反复入库, 且标题以'←'开头霸占各类查询榜首
    # 2026-11-07: 同时剔除 "→KB: kb搜..." 导航行 — 旧逻辑会把指针文件再压一次,
    # 标题递归长成 "- - - 指针头…", 成为搜索噪声。派生索引不是知识。
    content = "\n".join(
        l for l in content.splitlines()
        if not l.lstrip().startswith("←") and "→KB: kb搜" not in l
    )
    if not content.strip():
        return {"checked": 0, "added": 0, "confirmed": 0, "compressed": False}

    segs = [s.strip() for s in content.split("§") if len(s.strip()) >= 10]

    # Step 1: 拆段入库 KB — zvec DB 操作, 需要锁
    with _zvec_lock:
        existing = {r.fields.get("content_hash", "") for r in _iter_all_docs(col)}
        added = 0
        confirmed = 0
        for seg in segs:
            chash = hashlib.md5(seg.encode()).hexdigest()[:12]
            if chash in existing:
                confirmed += 1
                continue
            existing.add(chash)
            title = seg[:30]
            full_text = title + " " + seg
            emb = embed(full_text)
            doc_id = f"kb_{int(time.time())}_{chash}"
            kind_code = ma.infer_kind(category, title, seg)
            col.insert([zvec.Doc(
                id=doc_id, vectors={"embedding": emb},
                fields={
                    "title": title, "content": seg, "category": category,
                    "content_hash": chash,
                    "trust_score": TRUST_DEFAULT, "updated_at": int(time.time()),
                    "deprecated": 0, "hit_count": 0,
                    "scene": 0,  # 2026-10-24: schema 要求 scene 必填(10-23 加的列), 缺了 insert 直接失败 → 自动精简自 10-23 起全挂
                    "kind": kind_code,
                }
            )])
            try:
                _sync_doc_to_md(zvec.Doc(id=doc_id, fields={
                    "title": title, "content": seg, "category": category,
                    "trust_score": TRUST_DEFAULT, "updated_at": int(time.time()), "deprecated": 0, "hit_count": 0,
                    "scene": 0, "kind": kind_code,
                }))
            except Exception:
                pass
            added += 1

    if added + confirmed != len(segs) or len(segs) == 0:
        return {"checked": len(segs), "added": added, "confirmed": confirmed, "compressed": False}

    # Step 2: LLM 摘要精简 — 网络调用, 在锁外执行(不阻塞 kb_search 等工具)
    compacted = False
    method = "none"
    try:
        from llm_helper import llm_summarize
        
        segments_text = "\n\n---\n".join(segs)
        system_prompt = (
            "你是一个记忆精简助手。请把以下记忆条目精简压缩为一个索引指针文件。"
            "要求:\n"
            "1. 合并重复/相似的条目为一个条目\n"
            "2. 保留所有关键信息, 不要遗漏任何事实(但可以压缩措辞)\n"
            "3. 每个条目一行, 格式: - 主题关键词: 精简后的关键信息\n"
            "4. 第一行写指针头(我会提供)\n"
            "5. 总大小控制在原文件的 30% 以下\n"
            "6. 用中文, 简洁直接, 不要寒暄"
        )
        user_prompt = f"指针头:\n{pointer_header}\n\n记忆条目 (用---分隔):\n{segments_text}"
        
        llm_result = llm_summarize(system_prompt, user_prompt, max_tokens=4096, temperature=0.2)
        
        if llm_result and len(llm_result.strip()) > 10:
            if pointer_header.split('\n')[0] not in llm_result:
                llm_result = pointer_header + "\n" + llm_result
            path.write_text(llm_result.strip() + "\n", encoding="utf-8")
            compacted = True
            method = "llm"
    except Exception as e:
        print(f"[MCP-KB] LLM 精简失败, 回退到代码指针: {e}", file=sys.stderr, flush=True)
    
    # Step 3: 回退 — 如果 LLM 不可用, 用纯代码生成索引指针
    if not compacted:
        index_lines = [pointer_header]
        for seg in segs:
            title = seg[:30].replace("\n", " ").strip()
            if title:
                index_lines.append(f"- {title} →KB: kb搜'{title[:12]}'")
        pointer = "\n".join(index_lines) + "\n"
        path.write_text(pointer, encoding="utf-8")
        compacted = True
        method = "code"
    
    return {"checked": len(segs), "added": added, "confirmed": confirmed,
            "compressed": compacted, "size_before": size, "method": method}


@mcp.tool()
def kb_compact(force: bool = False) -> dict:
    """自动精简 MEMORY.md/USER.md: 拆段写入KB(去重) → 压缩为指针。仅当文件超过80%限制时执行(force=true强制)。返回各文件处理结果。"""
    t0 = time.time()
    # 2026-10-24 僵死修复: 去掉外层 _zvec_lock — 曾持锁跑 LLM 摘要(45s/文件),
    # 阻塞所有其他工具到 MCP 客户端 60s 超时。_compact_memory_file 内部
    # 自行管理锁(DB 操作持锁, LLM 在锁外), 这里无需再包一层。
    col = get_collection()
    mem = _compact_memory_file(
        col, MEMORY_MD, "memory", MEMORY_CHAR_LIMIT,
        "← MEMORY.md 已由 kb_compact 全部迁入KB。KB入口: kb搜'MEMORY' 或 kb_search 任意主题。",
        force=force
    )
    usr = _compact_memory_file(
        col, USER_MD, "user", USER_CHAR_LIMIT,
        "← USER.md 已由 kb_compact 全部迁入KB。KB入口: kb搜'USER偏好' 或 kb_search 任意偏好。",
        force=force
    )
    elapsed = round((time.time() - t0) * 1000)
    _record_metric("compact_calls", elapsed)
    return {"MEMORY.md": mem, "USER.md": usr, "elapsed_ms": elapsed}


# === 后台自动维护线程 ===
_bg_thread = None
_bg_stop_event = threading.Event()

def _background_maintenance():
    """后台线程: 每15min自动运行 optimize (自动废弃+物理清理+记忆精简)"""
    # 2026-10-16: 1h → 15min。高频会话日 memory 写入速度曾超过压缩速度
    # (2026-10-16 MEMORY.md 一晚涨破硬上限触发 health-check 告警)。
    INTERVAL = 15 * 60
    # 首次延迟5分钟 (等 MCP server 完全就绪)
    _bg_stop_event.wait(300)
    
    while not _bg_stop_event.is_set():
        try:
            t0 = time.time()
            with _zvec_lock:
                col = get_collection()
                deprecated_count = _auto_deprecate(col)
                cleaned_count = _cleanup_deprecated(col, days=730)
                # 2026-11-08: 补上从未在维护路径执行过的 zvec optimize (合并段/重建索引)。
                # 之前只有 _repair_fts_if_empty 在 FTS 探测为空时才调, 而 FTS 从不空 →
                # 段永不合并: schema 迁移每次 add_column 开新段, 每段各带一个 RocksDB
                # FTS 实例, 11 个段时仅开库就 +763MB RSS (实测)。旧段 FTS postings
                # 格式还会让 optimize 报 "source postings is not BitPacked" 合并失败
                # (已用 tools_rebuild_db.py 重建为单段新库解决)。
                # optimize 失败不致命 (查询走读路径), 单独 try 保住后续维护步骤。
                try:
                    ot0 = time.time()
                    col.optimize()
                    print(f"[MCP-KB] optimize 完成 (合并段/重建索引, {time.time()-ot0:.1f}s)",
                          file=sys.stderr, flush=True)
                except Exception as oe:
                    print(f"[MCP-KB] optimize 失败 (不致命, 下个周期重试): {oe}",
                          file=sys.stderr, flush=True)
                final_count = col.stats.doc_count
            # 同步 Obsidian md 目录树镜像 (清理孤儿 + 生成README索引)
            try:
                for r in _iter_all_docs(col):
                    _sync_doc_to_md(r)
                _write_notes_readme(NOTES_ROOT)
            except Exception as e:
                print(f"[MCP-KB] 目录树镜像同步失败: {e}", file=sys.stderr, flush=True)
            # 自动精简 MEMORY.md/USER.md (溢出才触发)
            try:
                mem = _compact_memory_file(
                    col, MEMORY_MD, "memory", MEMORY_CHAR_LIMIT,
                    "← MEMORY.md 已由 kb_compact 全部迁入KB。KB入口: kb搜'MEMORY' 或 kb_search 任意主题。"
                )
                usr = _compact_memory_file(
                    col, USER_MD, "user", USER_CHAR_LIMIT,
                    "← USER.md 已由 kb_compact 全部迁入KB。KB入口: kb搜'USER偏好' 或 kb_search 任意偏好。"
                )
                if mem.get("compressed") or usr.get("compressed"):
                    mem_method = mem.get("method", "?")
                    usr_method = usr.get("method", "?")
                    print(f"[MCP-KB] 记忆文件自动精简: MEMORY.md={mem} (method={mem_method}) USER.md={usr} (method={usr_method})", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[MCP-KB] 记忆文件精简失败: {e}", file=sys.stderr, flush=True)
            elapsed = round((time.time() - t0) * 1000)
            print(f"[MCP-KB] 后台维护: 废弃{deprecated_count}条, 清理{cleaned_count}条, 剩余{final_count}条 ({elapsed}ms)", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[MCP-KB] 后台维护出错: {e}", file=sys.stderr, flush=True)
        
        _bg_stop_event.wait(INTERVAL)


def _start_background_maintenance():
    """启动后台维护线程 (仅启动一次)"""
    global _bg_thread
    if _bg_thread is not None:
        return
    _bg_thread = threading.Thread(target=_background_maintenance, daemon=True, name="kb-maintenance")
    _bg_thread.start()
    print("[MCP-KB] 后台自动维护线程已启动 (每15min optimize)", file=sys.stderr, flush=True)


# === 启动时预加载 ===
if __name__ == "__main__":
    get_onnx_embedder()   # ~15s, 一次性
    get_collection()      # ~2s + 自动迁移
    _start_background_maintenance()  # 启动后台自动维护
    print("[MCP-KB] 服务就绪 (v3.0 信任评分 + 后台自动维护), 首次工具调用即达热速度", file=sys.stderr, flush=True)
    # 2026-09-04 架构改造: 启动模式由 MCP_TRANSPORT 环境变量控制
    # - "stdio" (默认, 旧 hermes-gateway 走这条路; watchdog 起; 父进程卡时全挂)
    # - "streamable-http" (新架构; 独立进程, hermes-gateway 通过 HTTP 调; 父进程卡不影响)
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport in ("streamable-http", "http", "sse"):
        # FastMCP 支持 streamable-http 协议 (MCP 2025 规范)
        host = os.environ.get("MCP_HTTP_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_HTTP_PORT", "8765"))
        print(f"[MCP-KB] 启动 HTTP 模式: http://{host}:{port}/mcp", file=sys.stderr, flush=True)
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run(transport="stdio")
