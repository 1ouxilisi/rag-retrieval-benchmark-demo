# -*- coding: utf-8 -*-
"""文档切块：按段落切分，相邻块借尾部重叠，提升检索召回。"""
import os
import re
import glob

CHUNK_SIZE = 320
CHUNK_OVERLAP = 60


def split_chunks(text: str, source: str):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) <= CHUNK_SIZE:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append({"source": source, "text": buf})
            buf = p
    if buf:
        chunks.append({"source": source, "text": buf})

    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0 and CHUNK_OVERLAP > 0:
            tail = chunks[i - 1]["text"][-CHUNK_OVERLAP:]
            c = {"source": c["source"], "text": tail + "\n" + c["text"]}
        overlapped.append(c)
    return overlapped


def load_chunks(docs_dir: str):
    chunks = []
    files = []
    for pat in ["**/*.md", "**/*.txt"]:
        files += glob.glob(os.path.join(docs_dir, pat), recursive=True)
    for path in sorted(set(files)):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks += split_chunks(text, os.path.basename(path))
    return chunks
