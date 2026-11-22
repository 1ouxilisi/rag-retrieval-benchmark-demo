# -*- coding: utf-8 -*-
"""
纯 Python BM25 检索（无 jieba / rank_bm25 重依赖）。
中文按单字切分并拼接相邻 bigram，英文与数字按整词提取；统一小写。
"""
import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str):
    text = (text or "").lower()
    tokens = []
    # 英文 / 数字整词
    tokens += _WORD_RE.findall(text)
    # 中文单字 + 相邻 bigram
    cjk = _CJK_RE.findall(text)
    tokens += cjk
    tokens += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return tokens


class BM25Index:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.n = len(chunks)
        self.docs_tokens = [tokenize(c["text"]) for c in chunks]
        self.tf = [Counter(toks) for toks in self.docs_tokens]
        self.dl = [len(toks) for toks in self.docs_tokens]
        self.avgdl = sum(self.dl) / self.n if self.n else 0.0
        self.df = Counter()
        for counter in self.tf:
            for term in counter:
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        scores = [0.0] * self.n
        for idx, counter in enumerate(self.tf):
            dl = self.dl[idx]
            norm = 1 - B + B * (dl / (self.avgdl or 1.0))
            for term in q_tokens:
                tf = counter.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                scores[idx] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)
        return scores

    def rank(self, query: str, top_k: int):
        scores = self.score(query)
        order = sorted(range(self.n), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"index": i, "source": self.chunks[i]["source"],
                 "text": self.chunks[i]["text"], "score": round(scores[i], 3)}
                for i in order]
