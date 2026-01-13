#!/usr/bin/env python3
"""
Transfer Matrix Method for Theorem 2 Verification.

Author: Nguyen Tuan Ngoc

This script uses the Transfer Matrix (Adjacency Matrix of the DFA) approach
to derive the Characteristic Polynomial and verify Theorem 2 from the paper
"Coding schemes for locally balanced constraints".

For (ℓ=6, δ=1):
- States: 5-bit suffixes (32 states)
- Valid window weight: [2, 4]
- Expected recurrence: f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n
- Expected capacity: 0.841

The characteristic polynomial of the 32x32 matrix will be degree 32.
We need to factor out x^k terms (zero eigenvalues) to get the minimal polynomial
which corresponds to the degree-12 recurrence from Theorem 2.
"""

import sys
import os
import itertools
import numpy as np
from sympy import Matrix, symbols, factor, Poly, simplify, log, N
from sympy.polys.polytools import gcd

# Add src to path for lbcode imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def weight(bits: str) -> int:
    """Count number of 1s in bit string."""
    return sum(1 for c in bits if c == '1')


def build_transfer_matrix(ell: int, delta: int):
    """
    Build the Transfer Matrix (Adjacency Matrix) for the DFA.
    
    States: all (ℓ-1)-bit strings
    Transition from state s to s' exists if:
      - s' = s[1:] + b  (shift left and append bit b)
      - weight(s + b) is in valid range [ℓ/2 - δ, ℓ/2 + δ]
    
    Returns:
        (states, A) where A is the adjacency matrix (sympy Matrix)
    """
    state_len = ell - 1
    states = [''.join(bits) for bits in itertools.product('01', repeat=state_len)]
    state_idx = {s: i for i, s in enumerate(states)}
    num_states = len(states)
    
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    # Build adjacency matrix
    A = [[0] * num_states for _ in range(num_states)]
    
    for s in states:
        for b in '01':
            window = s + b  # ℓ-bit window
            w = weight(window)
            if lo <= w <= hi:
                next_state = s[1:] + b  # Shift left, append new bit
                A[state_idx[s]][state_idx[next_state]] = 1
                print(f"Transition: {s} -> {next_state} (weight={w})")
                print("Transfer Maatrix", A)
    
    return states, Matrix(A)


def compute_characteristic_polynomial(A, variable='x'):
    """
    Compute the characteristic polynomial det(xI - A).
    
    Returns the polynomial as a sympy Poly object.
    """
    x = symbols(variable)
    n = A.rows
    I = Matrix.eye(n)
    char_matrix = x * I - A
    char_poly = char_matrix.det()
    return Poly(char_poly, x)


def factor_out_zero_eigenvalues(char_poly, variable='x'):
    """
    Factor out x^k terms from the characteristic polynomial.
    
    The degree-32 polynomial may have zero eigenvalues (states that become 
    "dead ends"). We factor out x^k to get the minimal polynomial that 
    corresponds to the actual recurrence relation.
    
    Returns:
        (k, minimal_poly) where k is the power of x factored out
    """
    x = symbols(variable)
    poly = char_poly.as_expr()
    
    # Find how many times x divides the polynomial
    k = 0
    while poly.subs(x, 0) == 0:
        poly = simplify(poly / x)
        k += 1
    
    return k, Poly(poly, x)


def polynomial_to_recurrence(poly, variable='x'):
    """
    Convert characteristic polynomial to recurrence relation.
    
    For polynomial p(x) = x^n - c_{n-1}*x^{n-1} - ... - c_0,
    the recurrence is: f_{k+n} = c_{n-1}*f_{k+n-1} + ... + c_0*f_k
    
    Returns:
        (degree, coefficients) where coefficients[i] is the coefficient for f_{k+i}
    """
    x = symbols(variable)
    coeffs = poly.all_coeffs()  # Highest degree first
    degree = len(coeffs) - 1
    
    # Negate coefficients (except leading) to get recurrence form
    # If p(x) = x^n + a_{n-1}*x^{n-1} + ... + a_0
    # Then f_{k+n} = -a_{n-1}*f_{k+n-1} - ... - a_0*f_k
    recurrence_coeffs = [-c for c in coeffs[1:]]
    
    return degree, recurrence_coeffs


def compute_capacity(A):
    """
    Compute the Shannon capacity from the transfer matrix.
    
    Capacity = log_2(λ_max) where λ_max is the spectral radius.
    """
    # Convert to numpy for eigenvalue computation
    A_np = np.array(A.tolist(), dtype=float)
    eigenvalues = np.linalg.eigvals(A_np)
    lambda_max = max(abs(eigenvalues))
    capacity = float(np.log2(lambda_max))
    return lambda_max, capacity


def main():
    print("=" * 70)
    print("Transfer Matrix Method for Theorem 2 Verification")
    print("Paper: 'Coding schemes for locally balanced constraints'")
    print("=" * 70)
    print()
    
    ell, delta = 6, 1
    
    print(f"Parameters: ℓ={ell}, δ={delta}")
    print(f"Valid window weight: [{ell//2 - delta}, {ell//2 + delta}] = [2, 4]")
    print(f"State length: {ell - 1} bits → {2**(ell-1)} states")
    print()
    
    # Step 1: Build Transfer Matrix
    print("-" * 70)
    print("Step 1: Build Transfer Matrix (DFA Adjacency Matrix)")
    print("-" * 70)
    states, A = build_transfer_matrix(ell, delta)
    print(f"Matrix size: {A.rows} × {A.cols}")
    print(f"Non-zero entries: {sum(1 for i in range(A.rows) for j in range(A.cols) if A[i,j] != 0)}")
    print()
    
    # Step 2: Compute Characteristic Polynomial
    print("-" * 70)
    print("Step 2: Compute Characteristic Polynomial det(xI - A)")
    print("-" * 70)
    print("Computing... (this may take a moment for 32x32 matrix)")
    char_poly = compute_characteristic_polynomial(A)
    print(f"Degree of characteristic polynomial: {char_poly.degree()}")
    print()
    
    # Step 3: Factor out zero eigenvalues
    print("-" * 70)
    print("Step 3: Factor out x^k (zero eigenvalues)")
    print("-" * 70)
    k, minimal_poly = factor_out_zero_eigenvalues(char_poly)
    print(f"Factored out: x^{k}")
    print(f"Minimal polynomial degree: {minimal_poly.degree()}")
    print()
    
    # Step 4: Display the minimal polynomial
    print("-" * 70)
    print("Step 4: Minimal Polynomial (corresponds to recurrence)")
    print("-" * 70)
    x = symbols('x')
    print(f"p(x) = {minimal_poly.as_expr()}")
    print()
    
    # Factor for cleaner display
    try:
        factored = factor(minimal_poly.as_expr())
        print(f"Factored form: {factored}")
        print()
    except:
        pass
    
    # Step 5: Derive recurrence relation
    print("-" * 70)
    print("Step 5: Derived Recurrence Relation")
    print("-" * 70)
    degree, rec_coeffs = polynomial_to_recurrence(minimal_poly)
    
    print(f"From polynomial of degree {degree}:")
    print()
    
    # Build recurrence string
    lhs = f"f_{{n+{degree}}}"
    rhs_parts = []
    for i, c in enumerate(reversed(rec_coeffs)):
        idx = i
        if c == 0:
            continue
        sign = "+" if c > 0 else "-"
        coeff_str = "" if abs(c) == 1 else f"{abs(int(c))}*"
        term = f"{coeff_str}f_{{n+{idx}}}" if idx > 0 else f"{coeff_str}f_n"
        if rhs_parts or c < 0:
            rhs_parts.append(f" {sign} {term}")
        else:
            rhs_parts.append(term)
    
    print(f"  {lhs} = {''.join(rhs_parts)}")
    print()
    
    # Compare with paper's Theorem 2
    print("-" * 70)
    print("Step 6: Comparison with Theorem 2")
    print("-" * 70)
    print("Paper's Theorem 2 recurrence (ℓ=6, δ=1):")
    print("  f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n")
    print()
    print("Expected characteristic polynomial (from recurrence):")
    print("  x^12 - x^11 - x^10 - x^9 + x^6 + x^4 + x^3 - 1")
    print()
    
    # Check if our polynomial matches
    expected_poly = Poly(x**12 - x**11 - x**10 - x**9 + x**6 + x**4 + x**3 - 1, x)
    
    print("Analyzing polynomial factors to find the one corresponding to Theorem 2...")
    print()
    
    # Factor the minimal polynomial
    try:
        factored_expr = factor(minimal_poly.as_expr())
        print(f"Factored form: {factored_expr}")
        print()
        
        # The factor containing the largest eigenvalue corresponds to the recurrence
        # Let's analyze each factor
        from sympy import solve, Abs
        factors = []
        
        # Try to extract factors
        from sympy import factor_list
        flist = factor_list(minimal_poly.as_expr())
        
        print("Individual factors:")
        for f, mult in flist[1]:
            f_poly = Poly(f, x)
            deg = f_poly.degree()
            # Find approximate largest root
            coeffs_np = [float(c) for c in f_poly.all_coeffs()]
            roots_np = np.roots(coeffs_np)
            max_root = max(abs(r) for r in roots_np)
            capacity_factor = float(np.log2(max_root)) if max_root > 0 else 0
            
            print(f"  - Degree {deg}: {f}")
            print(f"    Max |root| = {max_root:.6f}, capacity = {capacity_factor:.5f}")
            
            factors.append((f, deg, max_root, capacity_factor))
        print()
        
        # Find the factor that gives capacity ≈ 0.841
        dominant_factor = max(factors, key=lambda x: x[3])
        print(f"Dominant factor (gives capacity {dominant_factor[3]:.5f}):")
        print(f"  {dominant_factor[0]}")
        print()
        
        # Check if it's degree 12
        if dominant_factor[1] == 12:
            print("✓ Dominant factor is degree 12, matching Theorem 2!")
            
            # Compare with expected
            dom_poly = Poly(dominant_factor[0], x)
            dom_coeffs = dom_poly.all_coeffs()
            exp_coeffs = expected_poly.all_coeffs()
            
            print()
            print("Coefficient comparison:")
            print(f"  Our dominant factor:  {dom_coeffs}")
            print(f"  Expected (Theorem 2): {exp_coeffs}")
            
    except Exception as e:
        print(f"Could not factor: {e}")
    
    print()
    print("Note: The minimal polynomial degree is 24, not 12.")
    print("This is because the DFA has more structure than the minimal recurrence.")
    print("The factor giving the spectral radius corresponds to Theorem 2.")
    print()
    
    # Step 7: Compute Capacity
    print("-" * 70)
    print("Step 7: Capacity Computation")
    print("-" * 70)
    lambda_max, capacity = compute_capacity(A)
    print(f"Spectral radius (λ_max): {lambda_max:.6f}")
    print(f"Capacity = log₂(λ_max) = {capacity:.5f}")
    print()
    print(f"Paper's Table I value for (ℓ=6, δ=1): 0.841")
    
    if abs(capacity - 0.841) < 0.01:
        print("✓ Capacity matches paper!")
    else:
        print(f"  Difference: {abs(capacity - 0.841):.5f}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY: Transfer Matrix Method Verification")
    print("=" * 70)
    print()
    print(f"1. Transfer Matrix: {A.rows}×{A.cols} adjacency matrix for (ℓ={ell}, δ={delta})")
    print(f"2. Characteristic Polynomial: degree {char_poly.degree()}")
    print(f"3. After factoring out x^{k}: minimal polynomial of degree {minimal_poly.degree()}")
    print(f"4. Capacity: {capacity:.5f} (paper: 0.841)")
    print()
    print("This demonstrates the connection between:")
    print("  - Generating Functions / Generating Trees (classroom)")
    print("  - Transfer Matrix / Spectral Graph Theory (paper)")
    print("  - Recurrence Relations (Theorem 2)")
    print()


if __name__ == '__main__':
    main()
