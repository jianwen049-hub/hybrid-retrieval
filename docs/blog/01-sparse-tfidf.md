# 01 · 稀疏检索：为什么还要 TF-IDF / 类 BM25

## 当时我怎么看这个问题

刚做 RAG 时，我一度以为「上了向量库就可以扔掉关键词检索」。现实很快打脸：用户问「订单号 ORD-2024-881」，稠密检索常把语义相近的「订单」「退款」文档排上来，却漏掉真正带该 ID 的那一页。稀疏检索对**稀有 token / 专有名词 / 错误拼写的部分重合**仍然极其有效。

另一面：纯倒排 + TF 也会把「的 / 是 / 什么」这类停用词抬高噪声。需要 IDF 压低频、TF 饱和，才接近生产里 BM25 的味道。

## 试过什么

1. **只用向量库 top-k**：专有名词与精确匹配全面崩。
2. **朴素词频排序（纯 TF）**：长文档、停用词霸榜。
3. **布尔 AND 倒排**：召回太脆，一个同义词就空结果。
4. **把查询整句当一个 token**：中英混排几乎没法命中。

## 为什么否决

否决「稠密替代稀疏」。混合检索的前提是两路**互补**：稀疏保精确，稠密保语义。也否决「纯 TF」——没有 IDF 与长度归一，排行榜不可信。本仓用**可解释的类 BM25 公式**（k1 / b 可调），而不是黑盒商业引擎，方便博客对照数字。

## 最终怎么解

`lab/src/hybrid_lab/tokenize.py` 做轻量分词（英文按非字母切、中文按字 bigram + 单字），`sparse.py` 建 DF / 平均文档长，查询时打 BM25 风格分：

```
score(q,d) = Σ_t IDF(t) · [ f(t,d)·(k1+1) ] / [ f(t,d) + k1·(1-b+b·|d|/avgdl) ]
```

输出是 `(doc_id, score)` 的有序列表，供融合层消费。语料来自 `lab/data/corpus.jsonl`，离线可复现。

## 用到的具体 skill / 技巧清单

| Skill | 用法 |
|-------|------|
| **BM25 饱和 TF** | `k1` 控制词频边际收益 |
| **长度归一** | `b` + `avgdl` 抑制长文刷分 |
| **IDF 平滑** | `log(1 + (N-df+0.5)/(df+0.5))` |
| **中英混排分词** | 英文词 + 中文 bigram |
| **可复现索引** | 纯 dict / Counter，无外部库 |
| **与稠密解耦** | sparse 只吐 rank 列表 |

## 代码解析（按文件路径）

### `lab/src/hybrid_lab/tokenize.py`

`tokenize(text) -> list[str]`：小写、去标点；ASCII 词按 regex；CJK 连续段切成 unigram + bigram。单测会锁「订单号」类串至少产出可命中的 token。

### `lab/src/hybrid_lab/sparse.py`

`SparseIndex.from_docs(docs)` 预计算 `df`、`doc_len`、`avgdl`；`search(query, k)` 返回 top-k。`k1`/`b` 默认 1.5 / 0.75，与常见 BM25 默认同量级。

### `lab/data/corpus.jsonl`

每行 `{"id","title","text"}`。故意塞了订单号、同义退款表述、无关闲聊，方便对比稀疏/稠密谁先召回。

### `lab/scripts/smoke_check.py`

先建索引再断言：精确 ID 查询时，带该 ID 的文档进入 sparse top-3。
