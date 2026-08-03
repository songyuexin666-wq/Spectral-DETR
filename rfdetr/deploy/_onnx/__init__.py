# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Spectral-DETR
# ------------------------------------------------------------------------

"""
onnx optimizer and symbolic registry
"""
from . import optimizer
from . import symbolic

from .optimizer import OnnxOptimizer
from .symbolic import CustomOpSymbolicRegistry
