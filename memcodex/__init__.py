"""MemCodex: hierarchical memory as evolving programs.

Reference implementation of arXiv:2609.39765v1 (Wang & Liu, 2026).
"""
from .config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "__version__"]