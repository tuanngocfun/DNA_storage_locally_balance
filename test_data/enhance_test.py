"""
Enhanced Test Suite Generator

Author: Nguyen Tuan Ngoc
"""

import json
import random

def is_locally_balanced(s, l, delta):
    """
    Checks if a binary string s satisfies the (l, delta)-locally balanced constraint.
    """
    bits = [int(c) for c in s]
    n = len(bits)
    min_w = l/2 - delta
    max_w = l/2 + delta
    
    for i in range(n - l + 1):
        window = bits[i : i+l]
        w = sum(window)
        if not (min_w <= w <= max_w):
            return False
    return True

def generate_enhanced_test_suite(num_cases=20, min_len=10, max_len=20):
    suite = {
        "meta": {
            "description": "Enhanced Test Suite with longer random sequences",
            "generated_by": "Assistant_M4_Helper"
        },
        "test_suites": []
    }

    # Configuration 1: l=4, delta=1 (Standard Paper Example)
    config_1 = {
        "suite_name": "Random_Long_Sequences_l4_d1",
        "parameters": {"l": 4, "delta": 1},
        "cases": []
    }
    
    # Generate a mix of Valid and Invalid
    # We want roughly 50% valid, 50% invalid if possible, but random strings are mostly invalid for tight constraints.
    # So we will generate until we find enough valid ones, and keep some invalid ones.
    
    target_valid = num_cases // 2
    target_invalid = num_cases // 2
    
    valid_count = 0
    invalid_count = 0
    
    attempts = 0
    max_attempts = 10000 
    
    l = 4
    delta = 1
    
    while (valid_count < target_valid or invalid_count < target_invalid) and attempts < max_attempts:
        attempts += 1
        length = random.randint(min_len, max_len)
        s = "".join(random.choice(['0', '1']) for _ in range(length))
        
        is_valid = is_locally_balanced(s, l, delta)
        
        if is_valid and valid_count < target_valid:
            config_1["cases"].append({
                "id": f"R_L4_{valid_count + invalid_count + 1}",
                "input": s,
                "expect": True,
                "note": f"Random valid length {length}"
            })
            valid_count += 1
        elif not is_valid and invalid_count < target_invalid:
             config_1["cases"].append({
                "id": f"R_L4_{valid_count + invalid_count + 1}",
                "input": s,
                "expect": False,
                "note": f"Random invalid length {length}"
            })
             invalid_count += 1
             
    suite["test_suites"].append(config_1)

    # Configuration 2: l=8, delta=1 (The Challenge Case)
    config_2 = {
        "suite_name": "Random_Long_Sequences_l8_d1",
        "parameters": {"l": 8, "delta": 1},
        "cases": []
    }
    
    l = 8
    delta = 1
    valid_count = 0
    invalid_count = 0
    attempts = 0
    
    while (valid_count < target_valid or invalid_count < target_invalid) and attempts < max_attempts:
        attempts += 1
        length = random.randint(min_len, max_len) # Length 10-20 might be short for l=8 but still valid
        s = "".join(random.choice(['0', '1']) for _ in range(length))
        
        is_valid = is_locally_balanced(s, l, delta)
        
        if is_valid and valid_count < target_valid:
            config_2["cases"].append({
                "id": f"R_L8_{valid_count + invalid_count + 1}",
                "input": s,
                "expect": True,
                "note": f"Random valid length {length}"
            })
            valid_count += 1
        elif not is_valid and invalid_count < target_invalid:
             config_2["cases"].append({
                "id": f"R_L8_{valid_count + invalid_count + 1}",
                "input": s,
                "expect": False,
                "note": f"Random invalid length {length}"
            })
             invalid_count += 1
             
    suite["test_suites"].append(config_2)
    
    return suite

enhanced_json = generate_enhanced_test_suite()
print(json.dumps(enhanced_json, indent=2))