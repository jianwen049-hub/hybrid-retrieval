# hybrid-retrieval — Sparse + Dense + Rerank Fusion

> Open lab on **hybrid retrieval**: BM25-like sparse scoring, toy bag-of-chars dense embeddings, and reciprocal-rank / linear fusion reranking — fully offline, stdlib-first (no API key, no vector DB).

**Author:** 马健文 / Evans Owen ([jianwen049-hub](https://github.com/jianwen049-hub))

| | |
|---|---|
| **Lab** | [`lab/`](./lab/) — sparse · dense · fusion · pipeline · tests |
| **Blog** | [`docs/blog/`](./docs/blog/) |
| **Bio** | [`PROFILE.md`](./PROFILE.md) |

## 中文速览

混合检索不是「把两路分数相加」那么简单：稀疏路吃关键词，稠密路吃语义近似，重排要把**排序位置**与**原始分数**都用上。本仓用玩具语料把整条链路跑通，方便对照博客里的取舍。

## Reading guide (EN)

```bash
cd lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # empty / stdlib only
export PYTHONPATH=src
python3 scripts/smoke_check.py
python3 -m hybrid_lab.cli search "退款多久到账"
python3 -m unittest discover -s tests -v
```

## License

MIT — see [`LICENSE`](./LICENSE).
