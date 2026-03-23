"""
Root conftest.py — pytest session configuration.

Establishes the correct C library loading order to prevent BLAS conflicts
between faiss-cpu and torch on macOS (both bundle BLAS; torch must load first).
"""
# Pre-load torch before faiss to establish PyTorch's BLAS setup first.
# faiss-cpu loaded after torch uses the already-initialized BLAS without conflict.
try:
    import torch  # noqa: F401
except ImportError:
    pass

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    pass
