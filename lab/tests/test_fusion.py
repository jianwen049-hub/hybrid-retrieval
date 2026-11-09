import unittest

from hybrid_lab.fusion import linear_fuse, rrf_fuse
from hybrid_lab.pipeline import HybridSearcher
from hybrid_lab.corpus import demo_corpus
from hybrid_lab.tokenize import tokenize
from hybrid_lab.sparse import SparseIndex


class TestFusion(unittest.TestCase):
    def test_rrf_symmetric(self):
        a = [("d1", 10.0), ("d2", 5.0)]
        b = [("d2", 0.9), ("d1", 0.8)]
        fused = dict(rrf_fuse([a, b], k=60))
        self.assertIn("d1", fused)
        self.assertIn("d2", fused)
        # both appear once in each list → equal RRF
        self.assertAlmostEqual(fused["d1"], fused["d2"])

    def test_rrf_empty(self):
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([[]]), [])

    def test_linear_weights(self):
        a = [("d1", 10.0), ("d2", 0.0)]
        b = [("d2", 1.0), ("d1", 0.0)]
        fused = dict(linear_fuse([a, b], weights=[0.5, 0.5]))
        self.assertGreaterEqual(fused["d1"], 0.0)
        self.assertLessEqual(fused["d1"], 1.0)

    def test_sparse_exact_id(self):
        docs = demo_corpus()
        idx = SparseIndex.from_docs(docs)
        hits = idx.search("ORD-2024-881", k=3)
        ids = [d for d, _ in hits]
        self.assertIn("d2", ids)

    def test_tokenize_cjk(self):
        toks = tokenize("退款 ORD-1")
        self.assertTrue(any("退" in t or t == "退" for t in toks))
        self.assertIn("ord-1", toks)

    def test_pipeline_nonempty(self):
        s = HybridSearcher.from_docs(demo_corpus())
        fused = s.search("退款多久到账", k=3)
        self.assertTrue(len(fused) >= 1)


if __name__ == "__main__":
    unittest.main()
