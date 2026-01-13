"""
LBCode - Locally Balanced Constraint Coding Library
Based on Ge22 Paper: "Coding for Locally Balanced Constraints"

Author: Nguyen Tuan Ngoc

M4 Verifier Implementation
"""

from .verifier import is_locally_balanced, is_rll
from .graph_alg1 import algorithm1_find_core, CoreResult

__version__ = "1.0.0"
__all__ = ["is_locally_balanced", "is_rll", "algorithm1_find_core", "CoreResult"]
