# -*- coding: utf-8 -*-
"""
三种检索策略对照：
- vector：bge-m3 向量余弦相似度
- bm25：关键词 BM25
- hybrid：RRF（Reciprocal Rank Fusion）融合向量与 BM25 排名
"""
import json
import os

import numpy as np
import ollama

from bm25 import BM25Index

EMBED_MODEL = "bge-m3:latest"
TOP_K = 3
RRF_K = 60

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_NPZ = os.path.join(BASE, "embeddings.npz")
CACHE_JSON = os.path.join(BASE, "chunks.json")

OLLAMA_API = os.environ.get("OLLAMA_API", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_API)


def cache_signature(chunks):
    return json.dumps([[c["source"], len(c["text"])] for c in chunks], ensure_ascii=False)


def embed(text: str) -> np.ndarray:
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(resp["embedding"], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class VectorIndex:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.embs = self._build(chunks)

    def _build(self, chunks):
        if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_JSON):
            try:
                with open(CACHE_JSON, encoding="utf-8") as f:
                    saved = json.load(f)
                if saved["signature"] == cache_signature(chunks):
                    data = np.load(CACHE_NPZ)
                    return data["embeddings"]
            except Exception as e:
                print(f"向量缓存读取失败，重建索引：{e}")
        print(f"正在向量化 {len(chunks)} 个文本块（{EMBED_MODEL}）...")
        embs = np.vstack([embed(c["text"]) for c in chunks])
        np.savez(CACHE_NPZ, embeddings=embs)
        with open(CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump({"signature": cache_signature(chunks), "chunks": chunks},
                      f, ensure_ascii=False)
        return embs

    def rank(self, query: str, top_k: int):
        q_emb = embed(query)
        scores = [_cosine(q_emb, e) for e in self.embs]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"index": i, "source": self.chunks[i]["source"],
                 "text": self.chunks[i]["text"], "score": round(scores[i], 3)}
                for i in order]


def _rrf_merge(vector_hits: list, bm25_hits: list, top_k: int):
    scores: dict[int, float] = {}
    for rank, h in enumerate(vector_hits, start=1):
        scores[h["index"]] = scores.get(h["index"], 0.0) + 1.0 / (RRF_K + rank)
    for rank, h in enumerate(bm25_hits, start=1):
        scores[h["index"]] = scores.get(h["index"], 0.0) + 1.0 / (RRF_K + rank)
    order = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    merged = []
    by_index = {h["index"]: h for h in vector_hits + bm25_hits}
    for i in order:
        h = by_index[i]
        merged.append({"index": i, "source": h["source"], "text": h["text"],
                       "score": round(scores[i], 4)})
    return merged


class RetrievalBenchmark:
    """统一检索入口，按 strategy 返回 Top-K 命中。"""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.vector = VectorIndex(chunks)
        self.bm25 = BM25Index(chunks)

    def retrieve(self, query: str, strategy: str = "hybrid", top_k: int = TOP_K):
        if strategy == "vector":
            return self.vector.rank(query, top_k)
        if strategy == "bm25":
            return self.bm25.rank(query, top_k)
        if strategy == "hybrid":
            v_hits = self.vector.rank(query, top_k)
            b_hits = self.bm25.rank(query, top_k)
            return _rrf_merge(v_hits, b_hits, top_k)
        raise ValueError(f"未知检索策略：{strategy}")
