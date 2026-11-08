"""onnxruntime 版 all-MiniLM-L6-v2 编码器 (2026-11-08 优化)。

替代 sentence-transformers(torch) 路径的原因: 本机是无 NVIDIA GPU 的 ARM64 VPS,
torch 版稳态 RSS ~1.6G —— 模型本身只有 90M, 其余全是 libtorch_cpu/libtorch_cuda/
libcublasLt 等栈与分配器 arena (2026-11-08 实测 smaps: Anonymous 1.4G)。
onnxruntime + tokenizers 均为 venv 已有依赖, 无需新增安装, RSS ~200M。

输出兼容性: 与 torch 路径同样输出 384 维 L2 归一化向量 (mean pooling),
实测余弦相似度 >0.99 (见 tools 目录 parity 测试), 与库内既有 embedding 兼容。

接口对齐 SentenceTransformer.encode: 传 str 返回 (384,) ndarray,
传 list 返回 (n, 384) ndarray。
"""

from __future__ import annotations

import os

import numpy as np


class OnnxEmbedder:
    def __init__(self, model_dir: str, model_file: str = "model.onnx",
                 max_seq_length: int = 256, intra_op_threads: int = 2):
        """加载 tokenizer + ONNX 会话。

        model_file 默认 fp32 (87M): 实测与 torch 路径余弦相似度 = 1.0000;
        量化版 (model_quantized.onnx, 22M) 仅 0.974~0.992 (空串/长文本偏差大),
        与库内既有 embedding 兼容性不够, 不采用。
        intra_op_threads 默认 2: KB 检索是低频交互, 不跟容器抢满核。
        """
        from tokenizers import Tokenizer
        import onnxruntime as ort

        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3  # 静默 warning
        self.session = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"])

        tok_path = os.path.join(model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_seq_length)
        self.tokenizer.enable_padding()
        self.max_seq_length = max_seq_length

        # 输入名/ dtype 以会话为准 (不同导出的命名可能不同)
        self._input_meta = {i.name: i.type for i in self.session.get_inputs()}
        self._output_names = [o.name for o in self.session.get_outputs()]
        self.model_path = model_path

    def _build_feeds(self, texts):
        encs = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        types = np.array([e.type_ids for e in encs], dtype=np.int64)
        feeds = {}
        for name, typ in self._input_meta.items():
            if "input_ids" in name or name == "ids":
                arr = ids
            elif "attention" in name or "mask" in name:
                arr = mask
            elif "token_type" in name or "segment" in name:
                arr = types
            else:
                continue
            # tensor(int64)/tensor(int32) 按会话要求转换
            feeds[name] = arr.astype(np.int32 if "int32" in str(typ) else np.int64)
        return feeds, mask

    def encode(self, sentences, normalize_embeddings: bool = True, **kwargs):
        """对齐 SentenceTransformer.encode 的返回形状与归一化行为。"""
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        feeds, mask = self._build_feeds(texts)
        outputs = self.session.run(None, feeds)

        # 优先 last_hidden_state; 若导出方已池化 (ndim==2) 则直接用
        token_embeddings = None
        for name, out in zip(self._output_names, outputs):
            if "last_hidden" in name:
                token_embeddings = out
                break
        if token_embeddings is None:
            for out in outputs:
                if out.ndim == 3:
                    token_embeddings = out
                    break
        if token_embeddings is None:
            token_embeddings = outputs[0]

        if token_embeddings.ndim == 2:
            embeddings = token_embeddings.astype(np.float32)
        else:
            # mean pooling (与 sentence-transformers 1_Pooling 一致)
            m = mask[..., None].astype(np.float32)
            summed = (token_embeddings * m).sum(axis=1)
            counts = np.clip(m.sum(axis=1), 1e-9, None)
            embeddings = (summed / counts).astype(np.float32)

        if normalize_embeddings:
            norms = np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
            embeddings = embeddings / norms
        return embeddings[0] if single else embeddings
