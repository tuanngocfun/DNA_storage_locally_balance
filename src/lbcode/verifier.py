"""
Verifier Module - Definition 1: Locally Balanced Constraint Check
Based on Ge22 Paper Section II

Author: Nguyen Tuan Ngoc

A binary string x is (ℓ,δ)-locally balanced if:
- ℓ is even
- Every substring of length ℓ has Hamming weight in [ℓ/2 - δ, ℓ/2 + δ]
"""

from __future__ import annotations
from typing import Tuple


def is_locally_balanced(x: str, ell: int, delta: int) -> bool:
    """
    Check if binary string x satisfies the (ℓ, δ)-locally balanced constraint.
    
    Args:
        x: Binary string (e.g., "01010101")
        ell: Window length (must be even)
        delta: Allowed deviation from ℓ/2
        
    Returns:
        True if all windows of length ℓ have weight in [ℓ/2 - δ, ℓ/2 + δ]
        
    Raises:
        ValueError: If ℓ is not even
        
    Example:
        >>> is_locally_balanced("01010", 4, 1)
        True
        >>> is_locally_balanced("00000", 4, 1)
        False  # Contains "0000" with weight 0
    """
    n = len(x)
    
    if ell % 2 != 0:
        raise ValueError("ℓ (ell) must be even per Definition 1")
    
    # If string is shorter than window, no windows exist -> vacuously true
    if n < ell:
        return True
    
    # Valid weight bounds
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    # Use prefix sums for O(1) window weight computation
    prefix = [0] * (n + 1)
    for i, ch in enumerate(x):
        prefix[i + 1] = prefix[i] + (1 if ch == '1' else 0)
    
    # Check all windows
    for i in range(n - ell + 1):
        weight = prefix[i + ell] - prefix[i]
        if weight < lo or weight > hi:
            return False
    
    return True


def is_locally_balanced_with_info(x: str, ell: int, delta: int) -> Tuple[bool, str]:
    """
    Same as is_locally_balanced but returns failure info.
    
    Returns:
        (is_valid, message) tuple
    """
    n = len(x)
    
    if ell % 2 != 0:
        return False, "ℓ must be even"
    
    if n < ell:
        return True, "Pass (vacuously true: n < ℓ)"
    
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    prefix = [0] * (n + 1)
    for i, ch in enumerate(x):
        prefix[i + 1] = prefix[i] + (1 if ch == '1' else 0)
    
    for i in range(n - ell + 1):
        weight = prefix[i + ell] - prefix[i]
        if weight < lo or weight > hi:
            window = x[i:i + ell]
            return False, f"Fail at index {i}: window '{window}' has weight {weight} (valid: [{lo},{hi}])"
    
    return True, "Pass"


def is_rll(x: str, k: int) -> bool:
    """
    Check if binary string x satisfies run-length-limited (RLL) constraint.
    
    A string is k-RLL if no more than k consecutive identical bits appear.
    
    Args:
        x: Binary string
        k: Maximum allowed run length
        
    Returns:
        True if max run length ≤ k
        
    Example:
        >>> is_rll("01010", 3)
        True
        >>> is_rll("0000", 3)
        False  # Run of 4 zeros
    """
    if len(x) == 0:
        return True
    
    run = 1
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            run += 1
            if run > k:
                return False
        else:
            run = 1
    
    return True


def is_rll_with_info(x: str, k: int) -> Tuple[bool, str]:
    """
    Same as is_rll but returns failure info.
    """
    if len(x) == 0:
        return True, "Pass (empty string)"
    
    run = 1
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            run += 1
            if run > k:
                return False, f"Fail at index {i}: run of {run} '{x[i]}'s (max allowed: {k})"
        else:
            run = 1
    
    return True, "Pass"


# Convenience function combining both checks
def is_lb_and_rll(x: str, ell: int, delta: int, k: int) -> bool:
    """Check both (ℓ,δ)-locally balanced and k-RLL constraints."""
    return is_locally_balanced(x, ell, delta) and is_rll(x, k)


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        ("01010", 4, 1, True),
        ("00110", 4, 1, True),
        ("00000", 4, 1, False),  # weight 0
        ("11111", 4, 1, False),  # weight 4
        ("10001", 4, 1, True),
    ]
    
    print("Self-test for is_locally_balanced:")
    for x, ell, delta, expected in test_cases:
        result = is_locally_balanced(x, ell, delta)
        status = "✓" if result == expected else "✗"
        print(f"  {status} is_locally_balanced('{x}', {ell}, {delta}) = {result} (expected {expected})")
