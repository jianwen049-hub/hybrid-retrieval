"""hybrid_lab — sparse + dense + fusion retrieval (offline)."""
from hybrid_lab.pipeline import HybridSearcher
from hybrid_lab.corpus import demo_corpus, load_corpus

__all__ = ["HybridSearcher", "demo_corpus", "load_corpus"]
__version__ = "0.1.0"
