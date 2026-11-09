# 02 · 稠密检索：玩具字符袋也能讲清余弦

## 当时我怎么看这个问题

稠密检索的「正确打开方式」通常是 sentence-transformers / OpenAI embeddings。但实验室若一上来就绑 GPU 与 API Key，博客读者跑不动，CI 也难绿。我想先用**可解释的玩具向量**把「嵌入 → 余弦 → top-k」链路钉死，再在真实项目里替换成正经编码器——接口不变。

## 试过什么

1. **直接调线上 embedding API**：密钥、配额、网络一坏，演示全挂。
2. **随机投影假向量**：能跑，但与文本内容无关，融合实验没有教学意义。
3. **按词 one-hot 再平均**：词表爆炸，中文更惨。
4. **只用标题不做正文**：短文本余弦全挤在一起，区分度差。

## 为什么否决

否决「实验室必须上真模型」。教学与回归测试需要**确定性、零依赖**的 dense 路径。也否决随机向量——分数必须随文本重叠单调变化，读者才能在博客里对上数字。字符 n-gram 袋（bag-of-chars）足够小、足够稳，换真编码器时只替换 `embed()`。

## 最终怎么解

`lab/src/hybrid_lab/dense.py`：

- `embed(text) -> tuple[float, ...]`：固定维（默认 128）的字符 3-gram 哈希桶，累加后 L2 归一。
- `DenseIndex.from_docs(docs)`：预计算每篇向量。
- `search(query, k)`：余弦相似度 top-k。

哈希桶用 `hash(ngram) % dim`，Python 字符串 hash 在进程内稳定（本仓 smoke 不跨进程比绝对 hash，只比相对排序）。对近义改写（「多久退款」vs「退款几天到账」）通常比纯稀疏更友好。

## 用到的具体 skill / 技巧清单

| Skill | 用法 |
|-------|------|
| **哈希桶嵌入** | 固定维、无词表文件 |
| **L2 归一** | 余弦 = 点积 |
| **字符 3-gram** | 中英都能吃一点形态变化 |
| **接口对齐 sparse** | 同为 `(doc_id, score)` 列表 |
| **可替换编码器** | 生产换真模型只改 `embed` |
| **离线可测** | 无 torch / 无网络 |

## 代码解析（按文件路径）

### `lab/src/hybrid_lab/dense.py`

`_char_ngrams` + `_hash_embed` + `cosine`。`DenseIndex.search` 对语料做线性扫描——玩具语料足够；真库再换 ANN。

### `lab/src/hybrid_lab/corpus.py`

`load_corpus(path)` 读 jsonl；`demo_corpus()` 内置一小撮文档，smoke / CLI 不依赖外部文件也能跑。

### `lab/src/hybrid_lab/cli.py`

`search` 子命令同时打印 sparse / dense / fused 三列，方便肉眼看分歧。
