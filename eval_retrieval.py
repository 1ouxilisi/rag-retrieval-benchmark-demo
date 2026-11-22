# -*- coding: utf-8 -*-
"""
检索策略对照评测：vector / bm25 / hybrid(RRF)。
指标：Recall@K（期望来源是否进入 Top-K）、MRR@K（首个命中位置的倒数排名）。
用法：python eval_retrieval.py
输出：控制台汇总 + retrieval_report.md
"""
import json
import os

import chunking
from retrievers import RetrievalBenchmark, TOP_K

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(BASE, "eval_set.json")
REPORT_FILE = os.path.join(BASE, "retrieval_report.md")
STRATEGIES = ["vector", "bm25", "hybrid"]


def hit_rank(expect: str, hits: list):
    """返回期望来源在命中列表中的 1-based 排名；未命中返回 None。"""
    for rank, h in enumerate(hits, start=1):
        if expect in h["source"]:
            return rank
    return None


def main():
    chunks = chunking.load_chunks(os.path.join(BASE, "docs"))
    bench = RetrievalBenchmark(chunks)
    with open(EVAL_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    summary = {s: {"hit": 0, "mrr": 0.0} for s in STRATEGIES}

    for item in cases:
        row = {"id": item["id"], "q": item["question"], "expect": item["expect_source"], "strategies": {}}
        for s in STRATEGIES:
            hits = bench.retrieve(item["question"], strategy=s, top_k=TOP_K)
            rank = hit_rank(item["expect_source"], hits)
            row["strategies"][s] = {"rank": rank, "sources": [h["source"] for h in hits]}
            if rank is not None:
                summary[s]["hit"] += 1
                summary[s]["mrr"] += 1.0 / rank
        rows.append(row)

    n = len(cases)
    lines = [
        "# 检索策略对照评测报告",
        "",
        f"- 评测题数：{n}；Top-K = {TOP_K}；向量模型：bge-m3:latest",
        "",
        "| 检索策略 | Recall@%d | MRR@%d |" % (TOP_K, TOP_K),
        "| --- | --- | --- |",
    ]
    for s in STRATEGIES:
        recall = summary[s]["hit"] / n if n else 0
        mrr = summary[s]["mrr"] / n if n else 0
        lines.append(f"| {s} | {summary[s]['hit']}/{n} = {recall:.0%} | {mrr:.3f} |")

    lines += ["", "## 逐题明细", ""]
    lines.append("| # | 问题 | 期望来源 | vector | bm25 | hybrid |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        def cell(s):
            rank = r["strategies"][s]["rank"]
            return f"#{rank}" if rank is not None else "✗"
        lines.append(
            f"| {r['id']} | {r['q']} | {r['expect']} | {cell('vector')} | {cell('bm25')} | {cell('hybrid')} |"
        )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    for s in STRATEGIES:
        recall = summary[s]["hit"] / n if n else 0
        mrr = summary[s]["mrr"] / n if n else 0
        print(f"{s:>8}  Recall@{TOP_K}={recall:.0%}  MRR@{TOP_K}={mrr:.3f}")
    print(f"报告已写入：{REPORT_FILE}")


if __name__ == "__main__":
    main()
