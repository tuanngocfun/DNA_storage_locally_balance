#!/usr/bin/env python3
"""
Verify Theorem 2 recurrence from Section III.
For (ℓ=6, δ=1): f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lbcode.dp_automaton import verify_recurrence_l6_d1, compute_capacity, count_lb_sequences_dp


def main():
    parser = argparse.ArgumentParser(description='Verify Theorem 2 recurrence')
    parser.add_argument('--ell', type=int, default=6, help='Window length ℓ (default: 6)')
    parser.add_argument('--delta', type=int, default=1, help='Deviation δ (default: 1)')
    parser.add_argument('--n_max', type=int, default=40, help='Maximum n to check')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Theorem 2 Verification: Recurrence Relation")
    print("=" * 60)
    print()
    
    if args.ell == 6 and args.delta == 1:
        print("For (ℓ=6, δ=1):")
        print("  f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n")
        print()
        
        mismatches = verify_recurrence_l6_d1(n_max=args.n_max, verbose=True)
        
        print()
        if not mismatches:
            print("✓ Theorem 2 verified successfully!")
        else:
            print("✗ Verification failed!")
            sys.exit(1)
    else:
        print(f"Computing f_n values for (ℓ={args.ell}, δ={args.delta}):")
        for n in range(1, min(args.n_max, 20) + 1):
            f_n = count_lb_sequences_dp(n, args.ell, args.delta)
            print(f"  f_{n} = {f_n}")
    
    print()
    print("=" * 60)
    print("Capacity Computation")
    print("=" * 60)
    print()
    
    # Compute capacity
    cap = compute_capacity(args.ell, args.delta)
    print(f"Capacity(ℓ={args.ell}, δ={args.delta}) = {cap:.5f}")
    
    # Compare with paper values
    paper_caps = {
        (4, 1): 0.9468,
        (6, 1): 0.841,
        (8, 1): 0.824,
    }
    
    if (args.ell, args.delta) in paper_caps:
        paper_val = paper_caps[(args.ell, args.delta)]
        print(f"Paper Table I value: ~{paper_val}")
        if abs(cap - paper_val) < 0.01:
            print("✓ Matches paper!")
        else:
            print(f"  Difference: {abs(cap - paper_val):.5f}")


if __name__ == '__main__':
    main()
