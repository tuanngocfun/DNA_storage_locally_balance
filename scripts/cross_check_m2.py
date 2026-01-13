#!/usr/bin/env python3
"""
Cross-check M4 verifier with M2's definitions_lib.py
This script compares results from both implementations on random test strings.

Author: Nguyen Tuan Ngoc
"""

import sys
import os
import random

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'M2_work'))

from lbcode.verifier import is_locally_balanced as m4_is_locally_balanced

# Import M2's checker
try:
    from definitions_lib import DNAStorageCodeChecker
    M2_AVAILABLE = True
except ImportError:
    M2_AVAILABLE = False
    print("WARNING: M2's definitions_lib.py not found in M2_work/")


def m2_is_locally_balanced(x: str, ell: int, delta: int) -> bool:
    """Wrapper for M2's implementation"""
    is_balanced, violations = DNAStorageCodeChecker.is_locally_balanced(x, ell, delta)
    return is_balanced


def generate_random_string(length: int, bias: float = 0.5) -> str:
    """Generate random binary string with optional bias"""
    return ''.join('1' if random.random() < bias else '0' for _ in range(length))


def run_cross_check(num_tests: int = 1000, ell: int = 8, delta: int = 1,
                    min_len: int = 10, max_len: int = 20, verbose: bool = True):
    """
    Cross-check M4 verifier with M2's definitions_lib.py
    
    Returns:
        (total_tested, valid_count, invalid_count, mismatches)
    """
    if not M2_AVAILABLE:
        print("ERROR: Cannot run cross-check without M2's definitions_lib.py")
        return 0, 0, 0, []
    
    if verbose:
        print("=" * 60)
        print("Cross-Check: M4 Verifier vs M2 definitions_lib.py")
        print("=" * 60)
        print(f"Parameters: ℓ={ell}, δ={delta}")
        print(f"Test strings: {num_tests} random, length ∈ [{min_len}, {max_len}]")
        print()
    
    mismatches = []
    valid_count = 0
    invalid_count = 0
    
    for i in range(num_tests):
        length = random.randint(min_len, max_len)
        
        # Mix of biased and unbiased strings
        if i % 3 == 0:
            bias = random.uniform(0.2, 0.8)
        else:
            bias = 0.5
        
        test_str = generate_random_string(length, bias)
        
        m4_result = m4_is_locally_balanced(test_str, ell, delta)
        m2_result = m2_is_locally_balanced(test_str, ell, delta)
        
        if m4_result:
            valid_count += 1
        else:
            invalid_count += 1
        
        if m4_result != m2_result:
            mismatches.append({
                'string': test_str,
                'length': length,
                'm4_result': m4_result,
                'm2_result': m2_result
            })
    
    if verbose:
        print(f"Results: {num_tests} strings tested")
        print(f"  Valid (LB): {valid_count}")
        print(f"  Invalid: {invalid_count}")
        print(f"  Mismatches: {len(mismatches)}")
        
        if mismatches:
            print("\n  ⚠️ MISMATCH DETAILS:")
            for m in mismatches[:5]:  # Show first 5
                print(f"    String: {m['string']}")
                print(f"    M4={m['m4_result']}, M2={m['m2_result']}")
        else:
            print("  ✓ No mismatches found! M4 and M2 agree on all test cases.")
    
    return num_tests, valid_count, invalid_count, mismatches


def run_golden_test_comparison(verbose: bool = True):
    """
    Compare M4 and M2 on the golden test cases
    """
    if not M2_AVAILABLE:
        return []
    
    import json
    
    test_file = os.path.join(os.path.dirname(__file__), '..', 'test_data', 'golden_test_cases.json')
    
    if not os.path.exists(test_file):
        print(f"ERROR: Golden test file not found: {test_file}")
        return []
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Golden Test Cases: M4 vs M2 Comparison")
        print("=" * 60)
    
    ell = data.get('ell', 4)
    delta = data.get('delta', 1)
    mismatches = []
    total = 0
    
    for suite in data.get('test_suites', []):
        for case in suite.get('cases', []):
            total += 1
            test_str = case.get('input', '')
            expected = case.get('expected_lb', None)
            
            if expected is None:
                continue
            
            m4_result = m4_is_locally_balanced(test_str, ell, delta)
            m2_result = m2_is_locally_balanced(test_str, ell, delta)
            
            if m4_result != m2_result:
                mismatches.append({
                    'string': test_str,
                    'expected': expected,
                    'm4': m4_result,
                    'm2': m2_result
                })
    
    if verbose:
        print(f"Total golden cases: {total}")
        print(f"Mismatches between M4 and M2: {len(mismatches)}")
        if not mismatches:
            print("✓ M4 and M2 agree on all golden test cases!")
    
    return mismatches


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-check M4 vs M2 verifiers')
    parser.add_argument('--ell', type=int, default=8, help='Window length ℓ')
    parser.add_argument('--delta', type=int, default=1, help='Deviation δ')
    parser.add_argument('--num', type=int, default=1000, help='Number of random tests')
    args = parser.parse_args()
    
    # Run golden test comparison first
    run_golden_test_comparison()
    
    print()
    
    # Run random cross-check
    total, valid, invalid, mismatches = run_cross_check(
        num_tests=args.num,
        ell=args.ell,
        delta=args.delta
    )
    
    if mismatches:
        print("\n⚠️ CROSS-CHECK FAILED: Found disagreements between M4 and M2")
        sys.exit(1)
    else:
        print("\n✓ CROSS-CHECK PASSED: M4 and M2 implementations agree!")
        sys.exit(0)
