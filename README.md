# RAG 检索策略对照基线（vector / BM25 / Hybrid RRF）

一个聚焦**检索层**的可运行演示基线：在同一批样例知识库上，对比三种检索策略的命中表现，并用 Recall@K 与 MRR@K 量化对照。不接对话生成，便于单独评估和调优检索质量。

- **vector**：bge-m3 向量嵌入 + numpy 余弦相似度；
- **bm25**：纯 Python 实现的 BM25（中文按字 + bigram 分词，无 jieba 重依赖）；
- **hybrid**：用 RRF（Reciprocal Rank Fusion）融合向量与 BM25 两路排名。

样例场景为新能源汽车门店资料（保养质保、充电 FAQ、销售接待话术、异议跟进），可替换为任意行业知识库。

## 评测说明

- 评测集 `eval_set.json`：12 道题，每题标注期望命中的来源文档。
- 指标：**Recall@3**（期望来源是否进入 Top-3）、**MRR@3**（首个命中位置的倒数排名）。
- 本机实测（bge-m3:latest，Top-K=3）：

| 检索策略 | Recall@3 | MRR@3 |
| --- | --- | --- |
| vector（向量） | 12/12 = 100% | 0.944 |
| bm25（关键词） | 12/12 = 100% | 1.000 |
| hybrid（RRF 混合） | 12/12 = 100% | 1.000 |

> 对照案例：题 12「安装家充桩的条件与审批以什么为准？」中，向量检索把期望来源排到 #3，BM25 与混合策略均排到 #1——精确术语（家充桩、审批、物业）在关键词路径上更稳定，混合检索通过 RRF 融合了两路优势。
> 以上结果仅针对本样例题集；更换模型、文档或题集后指标会变化，以本机 `python eval_retrieval.py` 复跑结果为准。

## 功能特性

- **三条检索路径独立可换**：统一 `retrieve(query, strategy, top_k)` 入口，便于在线下对比或按场景选型。
- **纯轻量依赖**：检索用 numpy 余弦 + 标准库实现 BM25，无 FAISS / Chroma / jieba 重依赖，便于轻量起步与离线复现。
- **可复现检索评测**：脚本输出三策略 Recall@K、MRR@K 与逐题排名明细到 `retrieval_report.md`。
- **检索演示界面**：标准库 `http.server`，可在三种策略间切换，查看每条命中的来源、得分与文本片段。

## 技术栈

- Python 3.10+，依赖 `ollama`、`numpy`
- 本地模型服务：[Ollama](https://ollama.com)；向量模型 `bge-m3:latest`
- 检索：numpy 余弦相似度；BM25（k1=1.5, b=0.75）；RRF（k=60）

## 目录结构

```
rag-retrieval-benchmark-demo/
├── chunking.py          # 文档切块（段落 + 重叠）
├── bm25.py              # 分词与纯 Python BM25 索引
├── retrievers.py        # vector / bm25 / hybrid(RRF) 三策略统一入口
├── eval_retrieval.py    # 检索对照评测：Recall@K、MRR@K，输出报告
├── eval_set.json        # 评测集（问题、期望来源）
├── app.py               # 本地检索演示服务（/ 页面，POST /api/retrieve）
├── index.html           # 检索演示界面（策略切换 + 命中列表）
├── questions.txt        # 演示检索问题
├── docs/                # 样例知识库（4 份虚构演示文档）
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## 快速开始

### 1. 前置条件

- 安装并启动 [Ollama](https://ollama.com)，拉取向量化模型：

```bash
ollama pull bge-m3
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行

```bash
# 检索演示网页（默认 http://127.0.0.1:8000），可切换策略
python app.py

# 跑三策略对照评测，输出 retrieval_report.md
python eval_retrieval.py
```

### 连接远程或非默认 Ollama 地址

```bash
# Linux / macOS
export OLLAMA_API=http://localhost:11434
# Windows PowerShell
$env:OLLAMA_API="http://localhost:11434"
```

> 注意：`OLLAMA_HOST` 是 Ollama 服务端监听地址，不要直接当作客户端连接地址；客户端地址用独立的 `OLLAMA_API` 环境变量。

## 关键配置（`retrievers.py` / `bm25.py` 顶部）

```python
EMBED_MODEL = "bge-m3:latest"
TOP_K       = 3
RRF_K       = 60      # RRF 融合常数
# BM25: K1 = 1.5, B = 0.75
```

## 接入自己的资料

1. 把真实文档放入 `docs/`；
2. 重启服务，向量缓存按文档变化自动重建；
3. 在 `eval_set.json` 补充问题与期望来源，作为检索质量验收依据；
4. 可在此基线之上接入重排模型（cross-encoder）或混合权重调优，再用同一套 Recall/MRR 指标对比。

## 已知边界与局限

- 样例数据均为虚构演示，不代表任何真实品牌的政策与参数；正式交付以客户正式资料为准。
- 中文分词为字 + bigram 的轻量方案，未接 jieba / 词典；专业术语较多的场景可替换分词器后复跑评测。
- 本仓库只评估检索层，不含对话生成；生产链路可在命中片段之上再接大模型作答。
- 第三方模型调用费用按实际用量结算，建议设置用量封顶；服务器 / GPU 等硬件由使用方自备，本项目只交付软件层。

## License

MIT（发布时按需确认）。
