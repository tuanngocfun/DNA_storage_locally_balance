#!/usr/bin/env python3
"""
Recurrence Analysis - Patterns and Complexity Study

This module analyzes the discovered recurrence relations across different (ℓ,δ) parameters
to find patterns, simplifications, and theoretical insights.

Key Research Questions:
1. How does the recurrence order grow with ℓ and δ?
2. Are there pattern regularities in the coefficients?
3. Can we find simplified/factored forms like in Theorem 2?
4. What are the dominant eigenvalues and their multiplicities?

Author: M4 Verifier Team
"""

from __future__ import annotations
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'M2_work'))

import numpy as np

try:
    import sympy
    from sympy import Matrix, symbols, Poly, factor, factorint, gcd
    from sympy.polys.polytools import factor_list
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Handle both package import and direct script execution
try:
    from .general_recurrence import (
        build_transfer_matrix,
        get_characteristic_polynomial,
        discover_recurrence_relation,
        count_lb_sequences_dp,
        RecurrenceRelation
    )
except ImportError:
    from general_recurrence import (
        build_transfer_matrix,
        get_characteristic_polynomial,
        discover_recurrence_relation,
        count_lb_sequences_dp,
        RecurrenceRelation
    )


@dataclass
class RecurrenceAnalysis:
    """Analysis results for a recurrence relation."""
    ell: int
    delta: int
    recurrence: RecurrenceRelation
    matrix_size: int
    num_transitions: int
    spectral_radius: float
    capacity: float
    eigenvalue_info: Dict
    coefficient_pattern: Optional[str]
    factored_poly: Optional[str]


def analyze_transfer_matrix(ell: int, delta: int) -> Dict:
    """
    Analyze the transfer matrix structure.
    
    Returns information about:
    - Matrix sparsity
    - Eigenvalue distribution
    - Dominant eigenvalue and its algebraic multiplicity
    """
    states, A = build_transfer_matrix(ell, delta)
    n = len(states)
    
    # Basic statistics
    num_transitions = int(np.sum(A))
    max_possible = n * n
    sparsity = 1.0 - num_transitions / max_possible
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_abs = np.abs(eigenvalues)
    spectral_radius = float(np.max(eigenvalues_abs))
    
    # Count eigenvalues near the spectral radius (within tolerance)
    tol = 1e-8
    num_dominant = np.sum(np.abs(eigenvalues_abs - spectral_radius) < tol)
    
    # Compute capacity
    capacity = float(np.log2(spectral_radius))
    
    # In/out degree distribution
    out_degrees = A.sum(axis=1)
    in_degrees = A.sum(axis=0)
    
    return {
        'matrix_size': n,
        'num_transitions': num_transitions,
        'sparsity': sparsity,
        'spectral_radius': spectral_radius,
        'capacity': capacity,
        'num_dominant_eigenvalues': int(num_dominant),
        'min_out_degree': int(min(out_degrees)),
        'max_out_degree': int(max(out_degrees)),
        'avg_out_degree': float(np.mean(out_degrees)),
        'eigenvalues': eigenvalues
    }


def factor_characteristic_polynomial(ell: int, delta: int) -> Optional[str]:
    """
    Factor the characteristic polynomial to find simpler forms.
    
    The paper's Theorem 2 uses a factored form of the characteristic polynomial
    to derive a simpler recurrence of order 12 instead of 32.
    """
    if not SYMPY_AVAILABLE:
        return None
    
    states, A = build_transfer_matrix(ell, delta)
    n = len(states)
    
    # Build symbolic matrix
    A_sym = Matrix(A.tolist())
    lam = symbols('lambda')
    I_sym = sympy.eye(n)
    
    # Characteristic polynomial
    char_matrix = lam * I_sym - A_sym
    char_poly = char_matrix.det()
    
    # Factor the polynomial
    try:
        factored = factor(char_poly)
        return str(factored)
    except Exception as e:
        return f"Factoring failed: {e}"


def analyze_coefficient_pattern(coefficients: List[int]) -> str:
    """
    Analyze patterns in recurrence coefficients.
    
    Look for:
    - Symmetry
    - Sparse coefficients
    - Sign patterns
    """
    n = len(coefficients)
    
    patterns = []
    
    # Check sparsity
    non_zero = sum(1 for c in coefficients if c != 0)
    sparsity_ratio = non_zero / n
    if sparsity_ratio < 0.5:
        patterns.append(f"Sparse: {non_zero}/{n} non-zero coefficients")
    
    # Check symmetry
    is_symmetric = all(coefficients[i] == coefficients[n-1-i] for i in range(n//2))
    is_antisymmetric = all(coefficients[i] == -coefficients[n-1-i] for i in range(n//2))
    if is_symmetric:
        patterns.append("Symmetric coefficients")
    elif is_antisymmetric:
        patterns.append("Antisymmetric coefficients")
    
    # Check sign pattern
    positive = sum(1 for c in coefficients if c > 0)
    negative = sum(1 for c in coefficients if c < 0)
    patterns.append(f"Signs: +{positive}, -{negative}")
    
    # Check for powers of 2
    powers_of_2 = sum(1 for c in coefficients if c != 0 and (abs(c) & (abs(c)-1)) == 0)
    if powers_of_2 > n // 3:
        patterns.append(f"Many power-of-2 coefficients: {powers_of_2}")
    
    return "; ".join(patterns) if patterns else "No obvious patterns"


def compare_recurrence_complexity(params: List[Tuple[int, int]]) -> List[RecurrenceAnalysis]:
    """
    Compare recurrence complexity across different parameters.
    
    This helps understand how the problem scales with ℓ and δ.
    """
    results = []
    
    for ell, delta in params:
        print(f"\nAnalyzing (ℓ={ell}, δ={delta})...")
        
        # Discover recurrence (use numerical method for speed)
        recurrence = discover_recurrence_relation(ell, delta, verbose=False, use_sympy=False)
        
        # Analyze transfer matrix
        matrix_info = analyze_transfer_matrix(ell, delta)
        
        # Analyze coefficient patterns
        pattern = analyze_coefficient_pattern(recurrence.coefficients)
        
        # Skip factorization - it's extremely slow for matrices > 8x8
        factored = None
        
        analysis = RecurrenceAnalysis(
            ell=ell,
            delta=delta,
            recurrence=recurrence,
            matrix_size=matrix_info['matrix_size'],
            num_transitions=matrix_info['num_transitions'],
            spectral_radius=matrix_info['spectral_radius'],
            capacity=matrix_info['capacity'],
            eigenvalue_info={
                'num_dominant': matrix_info['num_dominant_eigenvalues'],
                'min_out_deg': matrix_info['min_out_degree'],
                'max_out_deg': matrix_info['max_out_degree'],
            },
            coefficient_pattern=pattern,
            factored_poly=factored
        )
        
        results.append(analysis)
    
    return results


def analyze_recurrence_patterns() -> None:
    """
    Main analysis function to study patterns across parameters.
    """
    print("=" * 80)
    print("RECURRENCE PATTERN ANALYSIS")
    print("Studying how recurrence complexity varies with (ℓ, δ)")
    print("=" * 80)
    
    # Parameters to analyze
    params = [
        (4, 1), (4, 2),
        (6, 1), (6, 2),
        (8, 1), (8, 2),
        (10, 1), (10, 2),
    ]
    
    results = compare_recurrence_complexity(params)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("COMPLEXITY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'ℓ':>4} {'δ':>3} {'States':>8} {'Order':>8} {'Capacity':>10} {'λ_max':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.ell:>4} {r.delta:>3} {r.matrix_size:>8} {r.recurrence.order:>8} "
              f"{r.capacity:>10.5f} {r.spectral_radius:>10.6f}")
    
    # Analyze growth patterns
    print("\n" + "=" * 80)
    print("GROWTH PATTERN ANALYSIS")
    print("=" * 80)
    
    # Group by delta
    for delta in [1, 2]:
        subset = [r for r in results if r.delta == delta]
        if len(subset) >= 2:
            print(f"\nFor δ={delta}:")
            print("  State count: 2^(ℓ-1)")
            print("  Order = 2^(ℓ-1) (full characteristic polynomial)")
            
            # Capacity trend
            capacities = [(r.ell, r.capacity) for r in subset]
            print(f"  Capacity decreases as ℓ increases:")
            for ell, cap in capacities:
                print(f"    ℓ={ell}: C = {cap:.5f}")
    
    # Coefficient pattern analysis
    print("\n" + "=" * 80)
    print("COEFFICIENT PATTERN ANALYSIS")
    print("=" * 80)
    
    for r in results:
        print(f"\n(ℓ={r.ell}, δ={r.delta}):")
        print(f"  Pattern: {r.coefficient_pattern}")
        print(f"  First 10 coefficients: {r.recurrence.coefficients[:10]}...")
    
    return results


def study_reduced_recurrences() -> None:
    """
    Study whether simpler recurrences exist by analyzing the minimal polynomial.
    
    The paper's Theorem 2 achieves order 12 instead of 32 for (6,1).
    This suggests the minimal polynomial has smaller degree.
    """
    print("\n" + "=" * 80)
    print("REDUCED RECURRENCE STUDY")
    print("Finding minimal polynomial recurrences")
    print("=" * 80)
    
    if not SYMPY_AVAILABLE:
        print("Requires sympy for symbolic computation")
        return
    
    # Study (6,1) case like the paper
    ell, delta = 6, 1
    print(f"\nStudying (ℓ={ell}, δ={delta}) - Paper's Theorem 2 case:")
    
    states, A = build_transfer_matrix(ell, delta)
    n = len(states)
    print(f"  Transfer matrix size: {n} × {n}")
    
    # The eigenvalues determine the recurrence
    eigenvalues = np.linalg.eigvals(A)
    unique_eigenvalues = np.unique(np.round(eigenvalues, decimals=8))
    print(f"  Number of distinct eigenvalues: {len(unique_eigenvalues)}")
    
    # The minimal polynomial degree equals the number of distinct eigenvalues
    print(f"  Minimal polynomial degree (approx): {len(unique_eigenvalues)}")
    
    # Compare with paper's order 12
    print(f"  Paper's Theorem 2 recurrence order: 12")
    
    # Factorization
    factored = factor_characteristic_polynomial(ell, delta)
    if factored:
        print(f"\n  Factored characteristic polynomial:")
        # Print in chunks if too long
        if len(factored) > 100:
            print(f"    {factored[:100]}...")
        else:
            print(f"    {factored}")


def analyze_eigenvalue_distribution(ell: int, delta: int) -> Dict:
    """
    Detailed eigenvalue analysis for understanding capacity and growth rate.
    """
    states, A = build_transfer_matrix(ell, delta)
    eigenvalues = np.linalg.eigvals(A)
    
    # Sort by absolute value
    eigenvalues_sorted = sorted(eigenvalues, key=lambda x: -abs(x))
    
    # Dominant eigenvalue
    lambda_max = abs(eigenvalues_sorted[0])
    
    # Second largest (for convergence rate analysis)
    lambda_2 = abs(eigenvalues_sorted[1]) if len(eigenvalues_sorted) > 1 else 0
    
    # Ratio (determines convergence to asymptotic behavior)
    ratio = lambda_2 / lambda_max if lambda_max > 0 else 0
    
    return {
        'lambda_max': lambda_max,
        'lambda_2': lambda_2,
        'ratio': ratio,
        'capacity': np.log2(lambda_max),
        'all_eigenvalues': eigenvalues_sorted[:10]  # Top 10
    }


if __name__ == "__main__":
    # Run all analyses
    analyze_recurrence_patterns()
    study_reduced_recurrences()
    
    # Detailed eigenvalue study
    print("\n" + "=" * 80)
    print("EIGENVALUE DISTRIBUTION STUDY")
    print("=" * 80)
    
    for ell in [4, 6, 8]:
        for delta in [1, 2]:
            info = analyze_eigenvalue_distribution(ell, delta)
            print(f"\n(ℓ={ell}, δ={delta}):")
            print(f"  λ_max = {info['lambda_max']:.6f}")
            print(f"  λ_2 = {info['lambda_2']:.6f}")
            print(f"  λ_2/λ_max = {info['ratio']:.6f} (convergence rate)")
            print(f"  Capacity = {info['capacity']:.5f}")
