# hybrid_lab

Offline hybrid retrieval: sparse (BM25-like) + dense (bag-of-chars) + RRF/linear fusion.

```bash
export PYTHONPATH=src
python3 scripts/smoke_check.py
python3 -m hybrid_lab.cli search "退款多久到账"
```
