#!/usr/bin/env python3
"""
Fast Recurrence Discovery using Competitive Programming Techniques

Author: Nguyen Tuan Ngoc

This module uses advanced algorithms to discover recurrence relations
for large (ℓ, δ) parameters that would be infeasible with symbolic computation.

Key Techniques:
1. Berlekamp-Massey Algorithm - O(n²) to find minimal linear recurrence from sequence
2. Matrix Exponentiation - O(d³ log n) to compute f_n for large n
3. Kitamasa's Algorithm - O(d² log n) to compute f_n given recurrence
4. Optimized DP with state compression

Complexity Analysis:
- Naive sympy det: O(n!) for n×n symbolic matrix - INFEASIBLE for n > 10
- Numerical np.poly: O(n³) - works for n ≤ 1000, but rounding errors
- Berlekamp-Massey: O(d²) where d is recurrence order - EXACT integers!

For (ℓ, δ) locally balanced constraints:
- Matrix size d = 2^(ℓ-1)
- ℓ=8: d=128, BM needs ~256 terms, runs in milliseconds
- ℓ=10: d=512, BM needs ~1024 terms, runs in seconds
- ℓ=12: d=2048, BM needs ~4096 terms, runs in minutes

Author: M4 Verifier Team
"""

from __future__ import annotations
import sys
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Try to import numba for JIT compilation (optional speedup)
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator
    def njit(func):
        return func


@dataclass
class FastRecurrenceResult:
    """Result of fast recurrence discovery."""
    ell: int
    delta: int
    order: int
    coefficients: List[int]
    sequence_computed: int  # How many f_n terms were computed
    computation_time: float
    verified: bool
    
    def recurrence_formula(self) -> str:
        """Return human-readable recurrence formula."""
        terms = []
        d = self.order
        for i, c in enumerate(self.coefficients):
            if c != 0:
                sign = "+" if c > 0 else "-"
                coef = abs(c)
                idx = d - 1 - i
                if coef == 1:
                    terms.append(f"{sign} f_{{n+{idx}}}" if idx > 0 else f"{sign} f_n")
                else:
                    terms.append(f"{sign} {coef}·f_{{n+{idx}}}" if idx > 0 else f"{sign} {coef}·f_n")
        
        rhs = " ".join(terms).lstrip("+ ").replace("+ -", "- ")
        return f"f_{{n+{d}}} = {rhs}"


# ============================================================================
# BERLEKAMP-MASSEY ALGORITHM
# ============================================================================

def berlekamp_massey(sequence: List[int]) -> List[int]:
    """
    Berlekamp-Massey algorithm using Chinese Remainder Theorem.
    
    For large sequences where values can be hundreds of digits,
    we use modular arithmetic with multiple large primes and CRT.
    
    Key insight: The true minimal recurrence order O is found by all "good" primes.
    Some primes may give smaller orders due to coincidental relationships mod p.
    We use majority voting on the order, then CRT with primes that agree.
    
    Complexity: O(n² × k) where k is number of primes
    """
    # Pool of large primes - we'll use many to get robust majority
    PRIME_POOL = [
        10**18 + 9,
        10**18 + 7,
        10**18 + 3,
        10**18 + 31,
        10**18 + 51,
        10**18 + 81,
        10**18 + 111,
        10**18 + 117,
        10**18 + 127,
        10**18 + 177,
        10**18 + 207,
        10**18 + 213,
    ]
    
    # Run BM for each prime
    results = []
    for p in PRIME_POOL:
        coeffs_mod = _bm_mod(sequence, p)
        results.append((p, coeffs_mod, len(coeffs_mod)))
    
    # Find the most common order (should be the true order)
    orders = [r[2] for r in results]
    order_counts = {}
    for o in orders:
        order_counts[o] = order_counts.get(o, 0) + 1
    
    # Get the order with most votes (tie-break: larger order is safer)
    best_order = max(order_counts.keys(), key=lambda o: (order_counts[o], o))
    
    # Filter to primes that gave this order
    good_results = [(p, c) for p, c, o in results if o == best_order]
    
    if not good_results:
        return _bm_fraction_fallback(sequence)
    
    # Use up to 8 agreeing primes for CRT
    primes_to_use = [p for p, c in good_results[:8]]
    coeffs_by_prime = {p: c for p, c in good_results[:8]}
    
    if len(primes_to_use) < 3:
        # Not enough agreement - use fraction fallback
        print(f"  WARNING: Only {len(primes_to_use)} primes agree on order {best_order}")
        return _bm_fraction_fallback(sequence)
    
    # Reconstruct using CRT
    order = best_order
    if order == 0:
        return []
    
    result = []
    M = 1
    for p in primes_to_use:
        M *= p
    
    for i in range(order):
        remainders = [coeffs_by_prime[p][i] for p in primes_to_use]
        val = _crt_general(remainders, primes_to_use, M)
        result.append(val)
    
    return result


def _modinv(a: int, m: int) -> int:
    """Modular inverse using extended GCD."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        return gcd, y1 - (b // a) * x1, x1
    
    _, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m


def _bm_mod(sequence: List[int], mod: int) -> List[int]:
    """Berlekamp-Massey modulo a prime."""
    n = len(sequence)
    seq = [x % mod for x in sequence]
    
    C = [1]  
    B = [1]  
    L = 0
    m = 1
    b = 1
    
    for i in range(n):
        d = seq[i]
        for j in range(1, len(C)):
            d = (d + C[j] * seq[i - j]) % mod
        
        if d == 0:
            m += 1
        else:
            T = C.copy()
            coef = (d * _modinv(b, mod)) % mod
            
            while len(C) < len(B) + m:
                C.append(0)
            
            for j in range(len(B)):
                C[j + m] = (C[j + m] - coef * B[j]) % mod
            
            if 2 * L <= i:
                L = i + 1 - L
                B = T.copy()
                b = d
                m = 1
            else:
                m += 1
    
    return [(-c) % mod for c in C[1:]]


def _crt_general(remainders: List[int], primes: List[int], M: int) -> int:
    """Chinese Remainder Theorem for any number of primes, returning signed integer."""
    val = 0
    for r, p in zip(remainders, primes):
        Mi = M // p
        val += r * Mi * _modinv(Mi, p)
    val %= M
    
    # Convert to signed: if val > M/2, it's negative
    if val > M // 2:
        val -= M
    
    return val


def _bm_fraction_fallback(sequence: List[int]) -> List[int]:
    """Fraction-based fallback when CRT method fails."""
    from fractions import Fraction
    
    n = len(sequence)
    seq = [Fraction(x) for x in sequence]
    
    C = [Fraction(1)]
    B = [Fraction(1)]
    L, m = 0, 1
    b = Fraction(1)
    
    for i in range(n):
        d = seq[i]
        for j in range(1, len(C)):
            d += C[j] * seq[i - j]
        
        if d == 0:
            m += 1
        else:
            T = C.copy()
            coef = d / b
            
            while len(C) < len(B) + m:
                C.append(Fraction(0))
            for j in range(len(B)):
                C[j + m] -= coef * B[j]
            
            if 2 * L <= i:
                L, B, b, m = i + 1 - L, T.copy(), d, 1
            else:
                m += 1
    
    return [int(-c) for c in C[1:]]


def berlekamp_massey_mod(sequence: List[int], mod: int) -> List[int]:
    """
    Berlekamp-Massey over finite field Z/modZ.
    Useful when working with modular arithmetic.
    """
    def modinv(a, m):
        """Modular inverse using extended Euclidean algorithm."""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        _, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m
    
    n = len(sequence)
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    
    for i in range(n):
        d = sequence[i] % mod
        for j in range(1, len(C)):
            d = (d + C[j] * sequence[i - j]) % mod
        
        if d == 0:
            m += 1
        else:
            T = C.copy()
            coef = (d * modinv(b, mod)) % mod
            
            while len(C) < len(B) + m:
                C.append(0)
            
            for j in range(len(B)):
                C[j + m] = (C[j + m] - coef * B[j]) % mod
            
            if 2 * L <= i:
                L = i + 1 - L
                B = T.copy()
                b = d
                m = 1
            else:
                m += 1
    
    return [(-c) % mod for c in C[1:]]


# ============================================================================
# OPTIMIZED SEQUENCE COMPUTATION
# ============================================================================

def compute_fn_sequence_fast(ell: int, delta: int, n_max: int) -> List[int]:
    """
    Compute f_1, f_2, ..., f_{n_max} using optimized DP.
    
    Uses state compression and efficient transitions.
    Uses Python's arbitrary precision integers to avoid overflow.
    
    Complexity: O(n_max * d) where d = 2^(ℓ-1)
    """
    # Build state space: all valid (ℓ-1)-bit suffixes
    half = ell // 2
    lo, hi = half - delta, half + delta
    
    # State = (ℓ-1)-bit suffix represented as integer
    num_states = 1 << (ell - 1)
    
    # Precompute valid transitions
    valid_next = [[] for _ in range(num_states)]
    
    for s in range(num_states):
        for b in [0, 1]:
            window = ((s << 1) | b) & ((1 << ell) - 1)
            weight = bin(window).count('1')
            
            if lo <= weight <= hi:
                next_state = ((s << 1) | b) & ((1 << (ell - 1)) - 1)
                valid_next[s].append(next_state)
    
    results = []
    
    # Initialize for n < ℓ: all strings are valid
    for n in range(1, ell):
        results.append(1 << n)  # 2^n strings
    
    # For n = ℓ, initialize DP using Python dict for arbitrary precision
    dp = {}
    for s in range(1 << ell):
        weight = bin(s).count('1')
        if lo <= weight <= hi:
            suffix = s & ((1 << (ell - 1)) - 1)
            dp[suffix] = dp.get(suffix, 0) + 1
    
    results.append(sum(dp.values()))
    
    # Extend to larger n using Python integers (arbitrary precision)
    for n in range(ell + 1, n_max + 1):
        new_dp = {}
        for s, count in dp.items():
            if count > 0:
                for next_s in valid_next[s]:
                    new_dp[next_s] = new_dp.get(next_s, 0) + count
        dp = new_dp
        results.append(sum(dp.values()))
    
    return results


def compute_fn_matrix_exp(ell: int, delta: int, n: int) -> int:
    """
    Compute f_n using matrix exponentiation.
    
    Uses the fact that f_{n+1} = sum of A^n[s,t] for valid initial/final states.
    
    Complexity: O(d³ log n) where d = 2^(ℓ-1)
    """
    from general_recurrence import build_transfer_matrix
    
    states, A = build_transfer_matrix(ell, delta)
    d = len(states)
    
    if n < ell:
        return 1 << n
    
    # Compute A^(n-ℓ+1)
    def matrix_power(M, p):
        result = np.eye(len(M), dtype=np.int64)
        base = M.copy()
        while p > 0:
            if p & 1:
                result = result @ base
            base = base @ base
            p >>= 1
        return result
    
    # Initial vector: count of valid ℓ-bit strings for each state
    half = ell // 2
    lo, hi = half - delta, half + delta
    
    init = np.zeros(d, dtype=np.int64)
    for i, s in enumerate(states):
        # s is the (ℓ-1)-bit suffix
        # Count valid ℓ-bit strings with this suffix
        for prefix_bit in [0, 1]:
            full = (prefix_bit << (ell - 1)) | s
            if lo <= bin(full).count('1') <= hi:
                init[i] += 1
    
    if n == ell:
        return int(np.sum(init))
    
    power = n - ell
    result = matrix_power(A.astype(np.int64), power) @ init
    return int(np.sum(result))


# ============================================================================
# MAIN FAST DISCOVERY FUNCTION
# ============================================================================

def discover_recurrence_fast(ell: int, delta: int, 
                             verbose: bool = True,
                             verify_length: int = 20) -> FastRecurrenceResult:
    """
    Discover recurrence relation using Berlekamp-Massey algorithm.
    
    This is MUCH faster than symbolic computation:
    - Symbolic det: infeasible for ℓ > 6
    - Numerical: rounding errors for ℓ > 10
    - Berlekamp-Massey: exact integers, works for any ℓ!
    
    Args:
        ell: Window length
        delta: Deviation
        verbose: Print progress
        verify_length: Number of extra terms to verify
        
    Returns:
        FastRecurrenceResult with discovered recurrence
    """
    start_time = time.time()
    
    if verbose:
        print(f"Fast recurrence discovery for (ℓ={ell}, δ={delta})...")
    
    # Expected order is d = 2^(ℓ-1)
    expected_order = 1 << (ell - 1)
    
    # Need at least 2*d terms for Berlekamp-Massey to find order-d recurrence
    # Add some extra for safety
    n_terms = 2 * expected_order + verify_length + 10
    
    if verbose:
        print(f"  Expected order: {expected_order}")
        print(f"  Computing {n_terms} terms of f_n sequence...")
    
    # Compute sequence
    seq_start = time.time()
    sequence = compute_fn_sequence_fast(ell, delta, n_terms)
    seq_time = time.time() - seq_start
    
    if verbose:
        print(f"  Sequence computed in {seq_time:.3f}s")
        print(f"  f_1..f_10: {sequence[:10]}")
    
    # Apply Berlekamp-Massey
    bm_start = time.time()
    coeffs = berlekamp_massey(sequence)
    bm_time = time.time() - bm_start
    
    order = len(coeffs)
    
    if verbose:
        print(f"  Berlekamp-Massey found order {order} in {bm_time:.3f}s")
    
    # Verify the recurrence
    verified = True
    for i in range(order, len(sequence)):
        expected = sum(c * sequence[i - j - 1] for j, c in enumerate(coeffs))
        if sequence[i] != expected:
            verified = False
            if verbose:
                print(f"  WARNING: Mismatch at n={i+1}: got {sequence[i]}, expected {expected}")
            break
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"  Verified: {'✓' if verified else '✗'}")
        print(f"  Total time: {total_time:.3f}s")
    
    return FastRecurrenceResult(
        ell=ell,
        delta=delta,
        order=order,
        coefficients=coeffs,
        sequence_computed=n_terms,
        computation_time=total_time,
        verified=verified
    )


# ============================================================================
# KITAMASA'S ALGORITHM - Fast computation of f_n given recurrence
# ============================================================================

def kitamasa(coeffs: List[int], initial: List[int], n: int) -> int:
    """
    Kitamasa's algorithm: compute f_n in O(d² log n) given recurrence.
    
    Given: f_i = c_1*f_{i-1} + ... + c_d*f_{i-d} for i >= d
    And: initial values f_0, f_1, ..., f_{d-1}
    Compute: f_n
    
    This is faster than matrix exponentiation O(d³ log n) for large d.
    
    Args:
        coeffs: [c_1, c_2, ..., c_d] recurrence coefficients
        initial: [f_0, f_1, ..., f_{d-1}] initial values
        n: Index to compute
        
    Returns:
        f_n
    """
    d = len(coeffs)
    
    if n < d:
        return initial[n]
    
    # We want to express x^n mod (x^d - c_1*x^{d-1} - ... - c_d)
    # as a_0 + a_1*x + ... + a_{d-1}*x^{d-1}
    # Then f_n = a_0*f_0 + a_1*f_1 + ... + a_{d-1}*f_{d-1}
    
    def poly_mod(p: List[int]) -> List[int]:
        """Reduce polynomial p modulo the characteristic polynomial."""
        # p = [p_0, p_1, ..., p_k] represents p_0 + p_1*x + ... + p_k*x^k
        while len(p) >= d + 1:
            if p[-1] != 0:
                lead = p[-1]
                for i in range(d):
                    p[len(p) - d + i - 1] += lead * coeffs[d - 1 - i]
            p.pop()
        return p
    
    def poly_mult(a: List[int], b: List[int]) -> List[int]:
        """Multiply two polynomials."""
        result = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                result[i + j] += ai * bj
        return poly_mod(result)
    
    # Compute x^n mod characteristic polynomial using binary exponentiation
    result = [1] + [0] * (d - 1)  # 1
    base = [0, 1] + [0] * (d - 2)  # x
    
    exp = n
    while exp > 0:
        if exp & 1:
            result = poly_mult(result, base)
        base = poly_mult(base, base)
        exp >>= 1
    
    # f_n = sum(result[i] * initial[i] for i in range(d))
    return sum(result[i] * initial[i] for i in range(min(len(result), d)))


# ============================================================================
# BATCH ANALYSIS FOR MULTIPLE PARAMETERS
# ============================================================================

def analyze_parameters_batch(params: List[Tuple[int, int]], 
                             verbose: bool = True) -> Dict[Tuple[int, int], FastRecurrenceResult]:
    """
    Analyze multiple (ℓ, δ) parameters efficiently.
    
    Args:
        params: List of (ℓ, δ) pairs
        verbose: Print progress
        
    Returns:
        Dictionary mapping (ℓ, δ) -> FastRecurrenceResult
    """
    results = {}
    
    for ell, delta in params:
        try:
            result = discover_recurrence_fast(ell, delta, verbose=verbose)
            results[(ell, delta)] = result
        except Exception as e:
            if verbose:
                print(f"  Error for ({ell}, {delta}): {e}")
    
    return results


def print_comparison_table(results: Dict[Tuple[int, int], FastRecurrenceResult]):
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'(ℓ,δ)':<10} {'Order':<10} {'Non-zero':<12} {'Time (s)':<12} {'Verified':<10}")
    print("-" * 80)
    
    for (ell, delta), r in sorted(results.items()):
        non_zero = sum(1 for c in r.coefficients if c != 0)
        print(f"({ell},{delta}){'':<5} {r.order:<10} {non_zero}/{r.order:<10} {r.computation_time:<12.3f} {'✓' if r.verified else '✗':<10}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast recurrence discovery")
    parser.add_argument('--ell', type=int, default=8, help='Window length')
    parser.add_argument('--delta', type=int, default=1, help='Deviation')
    parser.add_argument('--batch', action='store_true', help='Run batch analysis')
    parser.add_argument('--max-ell', type=int, default=10, help='Max ℓ for batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        print("=" * 80)
        print("BATCH ANALYSIS - Fast Recurrence Discovery")
        print("=" * 80)
        
        params = []
        for ell in range(4, args.max_ell + 1, 2):
            for delta in [1, 2]:
                if delta <= ell // 2:
                    params.append((ell, delta))
        
        print(f"Parameters to analyze: {params}\n")
        
        results = analyze_parameters_batch(params)
        print_comparison_table(results)
        
    else:
        print("=" * 80)
        print(f"FAST RECURRENCE DISCOVERY: (ℓ={args.ell}, δ={args.delta})")
        print("=" * 80)
        
        result = discover_recurrence_fast(args.ell, args.delta, verbose=True)
        
        print("\n" + "-" * 80)
        print("RESULT:")
        print("-" * 80)
        print(f"  Recurrence order: {result.order}")
        print(f"  Formula: {result.recurrence_formula()}")
        print(f"  Non-zero coefficients: {sum(1 for c in result.coefficients if c != 0)}/{result.order}")
        print(f"  Computation time: {result.computation_time:.3f}s")
        print(f"  Verified: {'✓' if result.verified else '✗'}")
        
        # Show some coefficient statistics
        coeffs = result.coefficients
        print(f"\n  First 10 coefficients: {coeffs[:10]}")
        print(f"  Last 10 coefficients: {coeffs[-10:]}")
        print(f"  Max coefficient: {max(abs(c) for c in coeffs)}")
        print(f"  Sum of coefficients: {sum(coeffs)}")
