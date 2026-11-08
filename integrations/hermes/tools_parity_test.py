"""torch vs onnx 向量一致性实测 (2026-11-08)。

用法: /usr/local/lib/hermes-agent/venv/bin/python /root/hermes-kb/tools_parity_test.py
判定: 每个测试文本 torch 与 onnx 的余弦相似度 >= 0.99 才允许上线切换。
"""
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "/root/hermes-kb")
from onnx_embedder import OnnxEmbedder

MODEL_DIR = "/root/hermes-kb/models"

TEXTS = [
    "LunaTV 播放页 root cause cleanupPlayer",
    "Docker容器内存限额 memory.max cgroup v2 配置",
    "知识库向量检索 onnx 模型量化 ARM64",
    "swap 100% 满 /tmp tmpfs Shmem 归因",
    "Hermes gateway room-driver 多代理团队调度",
    "短文本",
    "特殊字符 !@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "混合 mixed 中英文 text with numbers 12345 and symbols #hash",
    ("长文本测试 " * 120).strip(),  # ~600 字, 触发 256 截断
    "",
]

print("=== 加载 torch 路径 (当前线上实现) ===", flush=True)
t0 = time.time()
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
m.encode(["warmup knowledge base search query", "Docker端口映射规则"])
print(f"torch 加载+预热: {time.time()-t0:.1f}s", flush=True)
print(f"torch 峰值 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB", flush=True)

v_torch = m.encode(TEXTS, normalize_embeddings=True)
n_torch = np.linalg.norm(v_torch, axis=1)
print(f"torch 归一化校验 (应全为 1.0): min={n_torch.min():.6f} max={n_torch.max():.6f}", flush=True)

for variant in ["model_quantized.onnx", "model.onnx"]:
    print(f"\n=== ONNX 路径: {variant} ===", flush=True)
    t0 = time.time()
    e = OnnxEmbedder(MODEL_DIR, model_file=variant)
    print(f"onnx 加载: {time.time()-t0:.1f}s (inputs={list(e._input_meta)}, outputs={e._output_names})", flush=True)
    t0 = time.time()
    v_onnx = e.encode(TEXTS)
    dt = time.time() - t0
    print(f"onnx 编码 {len(TEXTS)} 条: {dt:.2f}s", flush=True)
    print(f"进程峰值 RSS (含 torch, 仅参考): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB", flush=True)

    cos = (v_torch * v_onnx).sum(axis=1)  # 双方均已 L2 归一化
    ok = bool((cos >= 0.99).all())
    for t, c in zip(TEXTS, cos):
        shown = t if len(t) <= 30 else t[:30] + "…"
        print(f"  cos={c:.4f}  {'OK ' if c >= 0.99 else 'FAIL'}  {shown!r}")
    print(f"  >>> {variant}: min_cos={cos.min():.4f} mean_cos={cos.mean():.4f} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    # 单条接口兼容性 (mcp_server.py 197 行的用法)
    one = e.encode(TEXTS[1])
    assert one.shape == (384,), one.shape
    assert isinstance(one.tolist(), list) and len(one.tolist()) == 384
    print(f"  单条接口: shape={one.shape} tolist OK", flush=True)
