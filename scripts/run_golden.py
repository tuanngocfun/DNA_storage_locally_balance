#!/usr/bin/env python3
"""
Run golden test cases against the verifier.
Validates that the verifier implementation matches expected results.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lbcode.verifier import is_locally_balanced, is_rll


def run_golden_tests(json_path: str) -> bool:
    """
    Run all test cases from a golden test JSON file.
    
    Returns:
        True if all tests pass
    """
    print(f"Loading: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check for global parameters
    global_l = None
    global_delta = None
    if 'meta' in data and 'global_parameters' in data['meta']:
        global_l = data['meta']['global_parameters'].get('l')
        global_delta = data['meta']['global_parameters'].get('delta')
    
    if global_l and global_delta:
        lo = global_l // 2 - global_delta
        hi = global_l // 2 + global_delta
        print(f"Global parameters: ℓ={global_l}, δ={global_delta} => valid weight: [{lo},{hi}]")
    print()
    
    total_checks = 0
    passed = 0
    failed = 0
    
    for suite in data['test_suites']:
        suite_name = suite.get('suite_name', 'Unknown')
        
        # Get parameters (suite-level or global)
        params = suite.get('parameters', {})
        l = params.get('l', global_l)
        delta = params.get('delta', global_delta)
        k_rll = params.get('k_rll')
        
        print(f"--- Suite: {suite_name} (ℓ={l}, δ={delta}) ---")
        
        suite_pass = 0
        suite_fail = 0
        
        for case in suite['cases']:
            case_id = case['id']
            x = case['input']
            
            # Get expected result
            expect_lb = case.get('expect', case.get('expect_LB'))
            expect_rll = case.get('expect_RLL')
            
            # Test LB constraint
            if expect_lb is not None and l is not None and delta is not None:
                result = is_locally_balanced(x, l, delta)
                total_checks += 1
                if result == expect_lb:
                    passed += 1
                    suite_pass += 1
                    status = "PASS"
                else:
                    failed += 1
                    suite_fail += 1
                    status = "FAIL"
                    print(f"  [{status}] {case_id}: LB('{x}') = {result}, expected {expect_lb}")
            
            # Test RLL constraint
            if expect_rll is not None and k_rll is not None:
                result = is_rll(x, k_rll)
                total_checks += 1
                if result == expect_rll:
                    passed += 1
                    suite_pass += 1
                else:
                    failed += 1
                    suite_fail += 1
                    print(f"  [FAIL] {case_id}: RLL('{x}', k={k_rll}) = {result}, expected {expect_rll}")
        
        if suite_fail == 0:
            print(f"  ✓ {suite_pass}/{suite_pass} checks passed")
        else:
            print(f"  ✗ {suite_pass}/{suite_pass + suite_fail} checks passed, {suite_fail} failed")
        print()
    
    print("=" * 50)
    print(f"TOTAL: {passed}/{total_checks} checks passed")
    if failed > 0:
        print(f"FAILED: {failed} checks")
    print(f"ALL_OK = {failed == 0}")
    
    return failed == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run golden test cases')
    parser.add_argument('--json', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 
                                             '..', 'test_data', 'golden_test_cases.json'),
                        help='Path to golden test JSON file')
    parser.add_argument('--v2', action='store_true',
                        help='Use golden_test_cases_v2.json')
    args = parser.parse_args()
    
    if args.v2:
        json_path = os.path.join(os.path.dirname(__file__), 
                                 '..', 'test_data', 'golden_test_cases_v2.json')
    else:
        json_path = args.json
    
    success = run_golden_tests(json_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
