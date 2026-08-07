"""LATEC: SZ-style error-bounded lossy compression with a pluggable predictor
(SZ3-style multilevel interpolation by default, or a trained GNN)."""

from .codec import compress, decompress
from .gnn_codec import GNNCodec, GNNCompressorCodec

__all__ = ["compress", "decompress", "GNNCompressorCodec", "GNNCodec"]
__version__ = "0.1.0"
