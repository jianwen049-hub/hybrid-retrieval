# 03 · 融合重排：RRF 与线性分数该怎么配

## 当时我怎么看这个问题

两路召回后，最诱人的写法是「分数 min-max 归一再加权求和」。问题：稀疏 BM25 与稠密余弦的量纲、分布完全不同；今天语料上调好的权重，换一批文档就漂。另一种极端是只取并集再交给交叉编码器——玩具仓没有 cross-encoder，且会丢掉「谁先排到第几」这一强信号。

## 试过什么

1. **裸加分**：BM25≈10、cosine≈0.7，稀疏永远碾压。
2. **min-max 后 0.5/0.5**：对离群分数过敏，小候选集更抖。
3. **只取两路交集**：召回骤降，长尾查询常空。
4. **稠密结果直接覆盖稀疏**：专有名词场景回退。

## 为什么否决

否决「依赖绝对分数可比」。**倒数排名融合（RRF）**只看名次：`1 / (k + rank)`，天然跨量纲。线性融合仍保留为可选项，用于演示「何时分数融合会翻车」。最终 pipeline 默认 RRF，CLI 可切 `linear`。

## 最终怎么解

`lab/src/hybrid_lab/fusion.py`：

- `rrf_fuse(rank_lists, k=60)`：多路 `(doc_id, score)` → 按名次累加 RRF 分 → 重排。
- `linear_fuse(rank_lists, weights)`：先对每路 score 做 min-max，再加权。

`pipeline.py` 的 `HybridSearcher`：sparse top-n ∪ dense top-n → fuse → 截断 k。博客与测试断言：含精确订单号的查询，fused 结果仍保住 sparse 冠军；近义改写查询，dense 贡献名次提升。

## 用到的具体 skill / 技巧清单

| Skill | 用法 |
|-------|------|
| **RRF** | 跨量纲名次融合，`k=60` 常用 |
| **线性融合对照** | 教学用，默不用 |
| **候选并集** | 先扩召回再压排 |
| **可观测中间层** | CLI 打印三路 |
| **确定性** | 同输入同 fused 序 |
| **与评测仓对齐思路** | 名次列表可丢进 nDCG |

## 代码解析（按文件路径）

### `lab/src/hybrid_lab/fusion.py`

`rrf_fuse` / `linear_fuse`。RRF 忽略原始 score，只枚举 enumerate(rank, start=1)。

### `lab/src/hybrid_lab/pipeline.py`

`HybridSearcher.search(query, k)` 编排 sparse → dense → fuse。`explain=True` 时附带每路原始命中，方便博客截图。

### `lab/tests/test_fusion.py`

锁定：两路名次互换时 RRF 合并对称；空列表不炸；线性融合权重和为 1 时分数落在 [0,1] 邻域。

### `lab/scripts/smoke_check.py`

跑 demo 语料 + 断言 fused 非空 + 打印 `smoke_check OK`。
