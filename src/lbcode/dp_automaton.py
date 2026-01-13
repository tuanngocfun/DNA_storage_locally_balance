"""
DP Automaton - Count locally balanced sequences using Dynamic Programming.
Used to verify Theorem 2 (recurrence relations) from Section III.

Author: Nguyen Tuan Ngoc

The automaton has states = all (ℓ-1)-bit strings.
Transitions: state s + bit b is valid if the new ℓ-bit window satisfies LB constraint.

f_n = |Σ^n(ℓ,δ)| = number of n-bit (ℓ,δ)-locally balanced strings
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import itertools


def _weight(bits: str) -> int:
    """Count number of 1s in bit string."""
    return sum(1 for c in bits if c == '1')


def build_transition_matrix(ell: int, delta: int) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build the automaton transition structure.
    
    States: all (ℓ-1)-bit strings
    Transitions: from state s, can go to s' = s[1:] + b if weight(s + b) is in valid range
    
    Returns:
        (states, transitions) where transitions[s] = list of reachable states
    """
    state_len = ell - 1
    states = [''.join(bits) for bits in itertools.product('01', repeat=state_len)]
    
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    transitions: Dict[str, List[str]] = {s: [] for s in states}
    
    for s in states:
        for b in '01':
            window = s + b  # ℓ-bit window
            w = _weight(window)
            if lo <= w <= hi:
                next_state = s[1:] + b  # Shift left, append new bit
                transitions[s].append(next_state)
    
    return states, transitions


def count_lb_sequences_dp(n: int, ell: int, delta: int) -> int:
    """
    Count the number of n-bit (ℓ,δ)-locally balanced strings using DP.
    
    f_n = |Σ^n(ℓ,δ)|
    
    For n < ℓ, all 2^n strings are valid (no complete window exists).
    For n ≥ ℓ, we use the automaton.
    """
    if n < ell:
        return 1 << n  # 2^n, no windows to check
    
    states, transitions = build_transition_matrix(ell, delta)
    state_idx = {s: i for i, s in enumerate(states)}
    num_states = len(states)
    
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    # dp[s] = number of ways to reach state s
    # Start: for first (ℓ-1) bits, check if all partial windows are valid
    dp = [0] * num_states
    
    # Initialize: count valid (ℓ-1) prefixes
    for s in states:
        # For a prefix of length ℓ-1, we only need to ensure
        # the final state is reachable (i.e., at least one outgoing edge exists)
        # Actually, for exact counting, we need to track all reachable prefixes
        dp[state_idx[s]] = 1  # All (ℓ-1)-bit strings are initially possible
    
    # But wait - for n=ℓ-1, we return 2^(ℓ-1), which is all states
    # For n ≥ ℓ, we need (n - ℓ + 1) windows to all be valid
    
    # Actually, let's reconsider:
    # After reading k bits (k ≥ ℓ-1), state = last (ℓ-1) bits
    # The number of valid n-bit strings = sum over all states of paths of length (n - ℓ + 1) from any start state
    
    # Simpler approach: 
    # dp[k][s] = number of k-bit strings ending in state s, all windows valid
    # For k = ℓ-1: dp[s] = 1 for all s (no windows yet)
    # For k ≥ ℓ: dp[next_s] += dp[s] for each valid transition s + b → next_s
    
    # Current DP represents strings of length (ℓ-1)
    current_len = ell - 1
    
    while current_len < n:
        new_dp = [0] * num_states
        for s in states:
            if dp[state_idx[s]] > 0:
                for next_s in transitions[s]:
                    new_dp[state_idx[next_s]] += dp[state_idx[s]]
        dp = new_dp
        current_len += 1
    
    return sum(dp)


def count_lb_sequences_exact(n: int, ell: int, delta: int) -> int:
    """
    Alternative exact counting by enumerating all strings (slow but correct).
    Only use for small n to verify DP.
    """
    from .verifier import is_locally_balanced
    
    count = 0
    for v in range(1 << n):
        bits = format(v, f'0{n}b')
        if is_locally_balanced(bits, ell, delta):
            count += 1
    return count


def verify_recurrence_l6_d1(n_max: int = 40, verbose: bool = True) -> List[int]:
    """
    Verify Theorem 2 recurrence for (ℓ=6, δ=1):
    
    f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n
    
    This should hold for n ≥ 6 (when constraint is active).
    
    Returns:
        List of n values where mismatch occurred (should be empty)
    """
    ell, delta = 6, 1
    
    # Compute f_n for n = 0 to n_max + 12
    f = []
    for n in range(n_max + 13):
        f.append(count_lb_sequences_dp(n, ell, delta))
    
    if verbose:
        print(f"Verifying Theorem 2 recurrence for (ℓ={ell}, δ={delta})")
        print(f"f_n values (first 15): {f[:15]}")
        print()
    
    mismatches = []
    
    # Check recurrence for n >= 6
    for n in range(6, n_max + 1):
        # LHS
        lhs = f[n + 12]
        
        # RHS: f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n
        rhs = f[n+11] + f[n+10] + f[n+9] - f[n+6] - f[n+4] - f[n+3] + f[n]
        
        if lhs != rhs:
            mismatches.append(n)
            if verbose:
                print(f"MISMATCH at n={n}: f[{n+12}]={lhs} vs RHS={rhs}")
    
    if verbose:
        if not mismatches:
            print(f"✓ Recurrence verified for n ∈ [6, {n_max}]: 0 mismatches")
        else:
            print(f"✗ Mismatches at: {mismatches}")
    
    return mismatches


def compute_capacity(ell: int, delta: int) -> float:
    """
    Compute Shannon capacity using the largest eigenvalue of the transfer matrix.
    
    Capacity = log2(λ_max) where λ_max is the spectral radius of the adjacency matrix.
    """
    import numpy as np
    
    states, transitions = build_transition_matrix(ell, delta)
    state_idx = {s: i for i, s in enumerate(states)}
    num_states = len(states)
    
    # Build adjacency matrix
    A = np.zeros((num_states, num_states))
    for s in states:
        for next_s in transitions[s]:
            A[state_idx[s], state_idx[next_s]] = 1
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = max(abs(eigenvalues))
    
    capacity = np.log2(lambda_max)
    return float(capacity)


if __name__ == "__main__":
    # Test 1: Verify f_n computation for small n
    print("Test 1: f_n counts for (ℓ=4, δ=1)")
    for n in range(1, 10):
        dp_count = count_lb_sequences_dp(n, 4, 1)
        exact_count = count_lb_sequences_exact(n, 4, 1)
        match = "✓" if dp_count == exact_count else "✗"
        print(f"  n={n}: DP={dp_count}, Exact={exact_count} {match}")
    print()
    
    # Test 2: Verify recurrence for (ℓ=6, δ=1)
    print("Test 2: Theorem 2 recurrence verification")
    verify_recurrence_l6_d1(n_max=30)
    print()
    
    # Test 3: Compute capacity
    print("Test 3: Capacity computation")
    for ell, delta, expected in [(4, 1, 0.9468), (6, 1, 0.841), (8, 1, 0.824)]:
        cap = compute_capacity(ell, delta)
        print(f"  Capacity(ℓ={ell}, δ={delta}) = {cap:.4f} (paper: ~{expected})")
