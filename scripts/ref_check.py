#!/usr/bin/env python3
"""
Cross-check (Audit) with M2's verifier code.
Compares results on 1000 random strings as required by task assignment.

Author: Nguyen Tuan Ngoc
"""

import sys
import os
import random
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lbcode.verifier import is_locally_balanced


def rand_bitstring(n: int, p1: float = 0.5) -> str:
    """Generate random n-bit string with probability p1 for each bit being '1'."""
    return ''.join('1' if random.random() < p1 else '0' for _ in range(n))


def run_audit(ell: int, delta: int, num_tests: int, len_min: int, len_max: int, 
              m2_verifier=None) -> None:
    """
    Run cross-check audit between M4 verifier and M2 verifier.
    
    Args:
        ell: Window length
        delta: Deviation
        num_tests: Number of random tests
        len_min: Minimum string length
        len_max: Maximum string length  
        m2_verifier: Optional M2 verifier function (if None, uses self-check)
    """
    print(f"Cross-Check Audit: M4 vs M2")
    print(f"Parameters: ℓ={ell}, δ={delta}")
    print(f"Test strings: {num_tests} random, length ∈ [{len_min}, {len_max}]")
    print("-" * 50)
    
    mismatches = []
    valid_count = 0
    invalid_count = 0
    
    # Use M2's verifier if provided, otherwise self-check
    if m2_verifier is None:
        print("NOTE: M2 verifier not provided - running self-consistency check")
        print("      To cross-check with M2, update the import in this file")
        print()
        
        # Self-check: compare is_locally_balanced with brute-force for small cases
        def m2_check(x, l, d):
            # Simple brute-force implementation
            n = len(x)
            if n < l:
                return True
            lo = l // 2 - d
            hi = l // 2 + d
            for i in range(n - l + 1):
                w = sum(1 for c in x[i:i+l] if c == '1')
                if w < lo or w > hi:
                    return False
            return True
        m2_verifier = m2_check
    
    random.seed(20251229)  # Fixed seed for reproducibility
    
    for t in range(num_tests):
        n = random.randint(len_min, len_max)
        
        # Mix: 50% unbiased, 50% skewed (to create harder cases)
        if random.random() < 0.5:
            x = rand_bitstring(n, p1=0.5)
        else:
            x = rand_bitstring(n, p1=random.choice([0.35, 0.65]))
        
        m4_result = is_locally_balanced(x, ell, delta)
        m2_result = m2_verifier(x, ell, delta)
        
        if m4_result:
            valid_count += 1
        else:
            invalid_count += 1
        
        if m4_result != m2_result:
            mismatches.append((x, m4_result, m2_result))
            if len(mismatches) <= 20:
                print(f"[MISMATCH] '{x}' M4={m4_result} M2={m2_result}")
    
    print()
    print("=" * 50)
    print(f"Results: {num_tests} strings tested")
    print(f"  Valid (LB): {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Mismatches: {len(mismatches)}")
    
    if mismatches:
        # Save mismatches to file
        mismatch_file = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'mismatch_cases.txt')
        os.makedirs(os.path.dirname(mismatch_file), exist_ok=True)
        with open(mismatch_file, 'w') as f:
            for x, m4, m2 in mismatches:
                f.write(f"{x}\tM4={m4}\tM2={m2}\n")
        print(f"  Saved to: {mismatch_file}")
    else:
        print("  ✓ No mismatches found!")


def main():
    parser = argparse.ArgumentParser(description='Cross-check audit with M2 verifier')
    parser.add_argument('--ell', type=int, default=8, help='Window length ℓ')
    parser.add_argument('--delta', type=int, default=1, help='Deviation δ')
    parser.add_argument('--num', type=int, default=1000, help='Number of test strings')
    parser.add_argument('--len_min', type=int, default=10, help='Minimum string length')
    parser.add_argument('--len_max', type=int, default=20, help='Maximum string length')
    args = parser.parse_args()
    
    # TODO: Import M2's verifier here
    # from m2_utils import check_LB as m2_verifier
    m2_verifier = None
    
    run_audit(
        ell=args.ell,
        delta=args.delta,
        num_tests=args.num,
        len_min=args.len_min,
        len_max=args.len_max,
        m2_verifier=m2_verifier
    )


if __name__ == '__main__':
    main()
