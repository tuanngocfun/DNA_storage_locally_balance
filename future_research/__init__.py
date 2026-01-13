"""
Future Research Module - General Recurrence Relations for Locally Balanced Constraints

This module extends the work in Section V of the Ge22 paper by:
1. Discovering linear recurrence relations for general (ℓ, δ) parameters
2. Deriving the characteristic polynomial from the transfer matrix
3. Exploring bounds and inequalities on |Σ^n(ℓ,δ)|

Reference: "Coding schemes for locally balanced constraints" - Ge et al., 2022
Section VI states: "finding the recurrence relation (or proper forms of inequalities) 
on the size of Σ^n(ℓ,δ) for general (ℓ,δ) is an interesting and challenging direction."

Author: M4 Verifier Team (Nguyễn Tuấn Ngọc)
"""

from .general_recurrence import (
    discover_recurrence_relation,
    get_characteristic_polynomial,
    compute_recurrence_coefficients,
    verify_recurrence_general,
)

from .recurrence_analysis import (
    analyze_recurrence_patterns,
    compare_recurrence_complexity,
)

__version__ = "1.0.0"
__all__ = [
    "discover_recurrence_relation",
    "get_characteristic_polynomial", 
    "compute_recurrence_coefficients",
    "verify_recurrence_general",
    "analyze_recurrence_patterns",
    "compare_recurrence_complexity",
]
