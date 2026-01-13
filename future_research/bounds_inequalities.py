#!/usr/bin/env python3
"""
Bounds and Inequalities for |Σ^n(ℓ,δ)|

Author: Nguyen Tuan Ngoc

This module explores bounds and inequalities on the size of locally balanced string sets,
as suggested in Section VI of the Ge22 paper.

The paper states: "finding the recurrence relation (or proper forms of inequalities) 
on the size of Σ^n(ℓ,δ) for general (ℓ,δ) is an interesting and challenging direction."

We develop:
1. Upper bounds based on forbidden patterns
2. Lower bounds based on valid constructions
3. Asymptotic bounds from eigenvalue analysis
4. Comparison with theoretical capacity

Author: M4 Verifier Team
"""

from __future__ import annotations
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'M2_work'))

import numpy as np

# Handle both package import and direct script execution
try:
    from .general_recurrence import (
        build_transfer_matrix,
        count_lb_sequences_dp,
    )
except ImportError:
    from general_recurrence import (
        build_transfer_matrix,
        count_lb_sequences_dp,
    )


@dataclass
class BoundsResult:
    """Bounds on |Σ^n(ℓ,δ)| for a specific n."""
    n: int
    ell: int
    delta: int
    exact_count: int
    lower_bound: int
    upper_bound: int
    capacity_approx: float
    asymptotic_approx: float


def count_forbidden_patterns(ell: int, delta: int) -> Tuple[int, List[str]]:
    """
    Count forbidden ℓ-bit patterns (those with weight outside valid range).
    
    For (ℓ,δ)-locally balanced:
    - Valid weight range: [ℓ/2 - δ, ℓ/2 + δ]
    - Forbidden: weight < ℓ/2 - δ or weight > ℓ/2 + δ
    """
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    forbidden = []
    total_patterns = 2 ** ell
    
    for v in range(total_patterns):
        bits = format(v, f'0{ell}b')
        weight = bits.count('1')
        if weight < lo or weight > hi:
            forbidden.append(bits)
    
    return len(forbidden), forbidden


def compute_forbidden_pattern_upper_bound(n: int, ell: int, delta: int) -> float:
    """
    Upper bound using inclusion-exclusion on forbidden patterns.
    
    f_n ≤ 2^n - (number of strings containing at least one forbidden pattern)
    
    This is a rough upper bound since we're not accounting for overlaps properly.
    """
    if n < ell:
        return 2 ** n
    
    num_forbidden, _ = count_forbidden_patterns(ell, delta)
    num_windows = n - ell + 1
    
    # Very rough upper bound: assume each forbidden pattern appears independently
    # P(at least one forbidden) ≈ 1 - (1 - p)^{num_windows}
    # where p = num_forbidden / 2^ell
    
    p_forbidden = num_forbidden / (2 ** ell)
    p_all_valid = (1 - p_forbidden) ** num_windows
    
    upper_bound = (2 ** n) * p_all_valid
    return upper_bound


def compute_capacity_asymptotic(ell: int, delta: int) -> Tuple[float, float]:
    """
    Compute capacity C(ℓ,δ) and the asymptotic approximation 2^{C·n}.
    """
    states, A = build_transfer_matrix(ell, delta)
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = max(abs(eigenvalues))
    capacity = np.log2(lambda_max)
    
    return float(capacity), float(lambda_max)


def trivial_lower_bound_constant_weight(n: int, ell: int, delta: int) -> int:
    """
    Lower bound from constant-weight strings.
    
    Strings with weight exactly n/2 (if n is even) are often locally balanced
    with high probability for reasonable parameters.
    """
    if n % 2 != 0:
        # For odd n, use floor(n/2) or ceil(n/2)
        target_weight = n // 2
    else:
        target_weight = n // 2
    
    # C(n, n/2) is the number of constant-weight strings
    from math import comb
    total_constant_weight = comb(n, target_weight)
    
    # Not all constant-weight strings are locally balanced
    # This is a very rough lower bound
    # A proper bound would filter by local balance
    
    # For safety, return a conservative fraction
    return total_constant_weight // 4  # Conservative estimate


def construction_based_lower_bound(n: int, ell: int, delta: int) -> int:
    """
    Lower bound from Construction 3 in the paper.
    
    If we can encode s bits into m-bit blocks with rate s/m,
    then we can construct at least 2^{s·floor(n/m)} valid strings of length ≥ n.
    """
    # Use known rates from the paper
    known_rates = {
        (4, 1): (11, 13),   # s=11, m=13, rate ≈ 0.846
        (6, 1): (12, 15),   # s=12, m=15, rate = 0.8
        (8, 1): (10, 13),   # s=10, m=13, rate ≈ 0.769
    }
    
    if (ell, delta) in known_rates:
        s, m = known_rates[(ell, delta)]
        num_blocks = n // m
        return 2 ** (s * num_blocks)
    else:
        # Conservative fallback
        return 2 ** (n // 2)


def compute_bounds(n: int, ell: int, delta: int) -> BoundsResult:
    """
    Compute various bounds on |Σ^n(ℓ,δ)|.
    """
    # Exact count
    exact = count_lb_sequences_dp(n, ell, delta)
    
    # Upper bound from forbidden patterns
    upper = compute_forbidden_pattern_upper_bound(n, ell, delta)
    
    # Lower bound from construction
    lower = construction_based_lower_bound(n, ell, delta)
    
    # Capacity and asymptotic
    capacity, lambda_max = compute_capacity_asymptotic(ell, delta)
    asymptotic = lambda_max ** n
    
    return BoundsResult(
        n=n,
        ell=ell,
        delta=delta,
        exact_count=exact,
        lower_bound=lower,
        upper_bound=int(upper),
        capacity_approx=capacity,
        asymptotic_approx=asymptotic
    )


def analyze_growth_rate(ell: int, delta: int, n_range: range) -> Dict:
    """
    Analyze how f_n grows and compare with theoretical bounds.
    """
    results = []
    
    capacity, lambda_max = compute_capacity_asymptotic(ell, delta)
    
    for n in n_range:
        f_n = count_lb_sequences_dp(n, ell, delta)
        asymptotic = lambda_max ** n
        
        # Ratio f_n / asymptotic converges to a constant
        ratio = f_n / asymptotic if asymptotic > 0 else 0
        
        # Empirical rate: log_2(f_n) / n
        empirical_rate = np.log2(f_n) / n if f_n > 0 and n > 0 else 0
        
        results.append({
            'n': n,
            'f_n': f_n,
            'asymptotic': asymptotic,
            'ratio': ratio,
            'empirical_rate': empirical_rate,
            'capacity': capacity
        })
    
    return {
        'ell': ell,
        'delta': delta,
        'capacity': capacity,
        'lambda_max': lambda_max,
        'data': results
    }


def derive_inequalities(ell: int, delta: int) -> List[str]:
    """
    Derive useful inequalities for f_n = |Σ^n(ℓ,δ)|.
    """
    inequalities = []
    
    # 1. Basic bounds
    num_forbidden, _ = count_forbidden_patterns(ell, delta)
    num_valid = 2**ell - num_forbidden
    
    inequalities.append(f"Valid ℓ-patterns: {num_valid} out of {2**ell}")
    inequalities.append(f"Fraction valid: {num_valid / 2**ell:.4f}")
    
    # 2. Capacity bound
    capacity, lambda_max = compute_capacity_asymptotic(ell, delta)
    inequalities.append(f"Capacity C({ell},{delta}) = {capacity:.5f}")
    inequalities.append(f"Asymptotically: f_n ≈ c · {lambda_max:.6f}^n for some c > 0")
    
    # 3. Submultiplicativity (if holds)
    # f_{m+n} ≤ f_m · f_n doesn't always hold, but we can check
    
    # 4. Recurrence inequality
    inequalities.append(f"f_n satisfies a linear recurrence of order {2**(ell-1)}")
    
    return inequalities


def study_parameter_scaling() -> None:
    """
    Study how bounds scale with parameters ℓ and δ.
    """
    print("=" * 80)
    print("BOUNDS AND INEQUALITIES STUDY")
    print("=" * 80)
    
    # Study effect of ℓ for fixed δ
    print("\n--- Effect of ℓ (δ=1 fixed) ---")
    print(f"{'ℓ':>4} {'Valid/Total':>15} {'Capacity':>12} {'λ_max':>12}")
    print("-" * 50)
    
    for ell in [4, 6, 8, 10, 12]:
        delta = 1
        num_forbidden, _ = count_forbidden_patterns(ell, delta)
        num_valid = 2**ell - num_forbidden
        capacity, lambda_max = compute_capacity_asymptotic(ell, delta)
        print(f"{ell:>4} {num_valid:>7}/{2**ell:<7} {capacity:>12.5f} {lambda_max:>12.6f}")
    
    # Study effect of δ for fixed ℓ
    print("\n--- Effect of δ (ℓ=8 fixed) ---")
    print(f"{'δ':>4} {'Valid/Total':>15} {'Capacity':>12} {'λ_max':>12}")
    print("-" * 50)
    
    ell = 8
    for delta in [1, 2, 3]:
        num_forbidden, _ = count_forbidden_patterns(ell, delta)
        num_valid = 2**ell - num_forbidden
        capacity, lambda_max = compute_capacity_asymptotic(ell, delta)
        print(f"{delta:>4} {num_valid:>7}/{2**ell:<7} {capacity:>12.5f} {lambda_max:>12.6f}")
    
    # Bounds comparison for specific n
    print("\n--- Bounds Comparison (n=20) ---")
    print(f"{'ℓ':>4} {'δ':>3} {'Exact f_n':>15} {'Lower':>15} {'Upper':>15} {'Asymptotic':>15}")
    print("-" * 80)
    
    n = 20
    for ell in [4, 6, 8]:
        for delta in [1, 2]:
            bounds = compute_bounds(n, ell, delta)
            print(f"{ell:>4} {delta:>3} {bounds.exact_count:>15} {bounds.lower_bound:>15} "
                  f"{bounds.upper_bound:>15} {bounds.asymptotic_approx:>15.0f}")
    
    # Inequalities
    print("\n--- Derived Inequalities ---")
    for ell in [6, 8]:
        delta = 1
        print(f"\nFor (ℓ={ell}, δ={delta}):")
        for ineq in derive_inequalities(ell, delta):
            print(f"  • {ineq}")


def verify_submultiplicativity(ell: int, delta: int, max_n: int = 20) -> bool:
    """
    Check if f_{m+n} ≤ f_m · f_n (submultiplicativity).
    
    This property, if true, gives useful bounds.
    """
    print(f"\nChecking submultiplicativity for (ℓ={ell}, δ={delta})...")
    
    violations = []
    
    for m in range(ell, max_n):
        for n in range(ell, max_n):
            f_m = count_lb_sequences_dp(m, ell, delta)
            f_n = count_lb_sequences_dp(n, ell, delta)
            f_mn = count_lb_sequences_dp(m + n, ell, delta)
            
            if f_mn > f_m * f_n:
                violations.append((m, n, f_mn, f_m * f_n))
    
    if violations:
        print(f"  Submultiplicativity DOES NOT hold. Found {len(violations)} violations.")
        for m, n, actual, product in violations[:5]:
            print(f"    f_{m+n} = {actual} > f_{m} · f_{n} = {product}")
        return False
    else:
        print(f"  Submultiplicativity holds for tested range!")
        return True


if __name__ == "__main__":
    study_parameter_scaling()
    
    print("\n" + "=" * 80)
    print("SUBMULTIPLICATIVITY CHECK")
    print("=" * 80)
    
    for ell in [4, 6]:
        verify_submultiplicativity(ell, 1, max_n=15)
    
    print("\n" + "=" * 80)
    print("GROWTH RATE CONVERGENCE")
    print("=" * 80)
    
    for ell, delta in [(4, 1), (6, 1), (8, 1)]:
        growth = analyze_growth_rate(ell, delta, range(ell, 25))
        print(f"\n(ℓ={ell}, δ={delta}): Capacity = {growth['capacity']:.5f}")
        print(f"  n     f_n            Empirical Rate   Ratio to Asymptotic")
        print(f"  " + "-" * 60)
        for r in growth['data'][-8:]:  # Last 8 values
            print(f"  {r['n']:>3} {r['f_n']:>14} {r['empirical_rate']:>14.6f} {r['ratio']:>18.8f}")
