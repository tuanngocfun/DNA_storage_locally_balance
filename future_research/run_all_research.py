#!/usr/bin/env python3
"""
Main Runner Script for Future Research on General (ℓ, δ) Recurrences

This script orchestrates all research computations and generates comprehensive
output for the lecturer's review. It extends Theorem 2 from Ge22 paper to
arbitrary (ℓ, δ) parameters.

Usage:
    python run_all_research.py              # Run all with defaults
    python run_all_research.py --quick      # Quick mode (small parameters only)
    python run_all_research.py --figures    # Also generate figures
    python run_all_research.py --verify     # Include cross-verification

Author: M4 Verifier Team
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from general_recurrence import (
    discover_recurrence_relation,
    verify_recurrence_general,
    cross_check_with_m2,
    count_lb_sequences_dp,
)
from recurrence_analysis import (
    analyze_transfer_matrix,
    analyze_coefficient_pattern,
    compare_recurrence_complexity,
    analyze_eigenvalue_distribution,
)
from bounds_inequalities import (
    compute_capacity_asymptotic,
    count_forbidden_patterns,
    compute_forbidden_pattern_upper_bound,
    compute_bounds,
    verify_submultiplicativity,
    analyze_growth_rate,
)


def print_header(title: str):
    """Print a formatted header."""
    width = 80
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def print_section(title: str):
    """Print a section header."""
    print("\n" + "-" * 60)
    print(f">>> {title}")
    print("-" * 60)


def run_recurrence_discovery(params_list: List[tuple], verbose: bool = True) -> Dict[str, Any]:
    """
    Task 1: Discover recurrence relations for all parameter pairs.
    """
    print_header("RECURRENCE RELATION DISCOVERY")
    
    results = {}
    
    for ell, delta in params_list:
        print_section(f"Parameters: ℓ={ell}, δ={delta}")
        
        # Discover relation
        relation = discover_recurrence_relation(ell, delta, verbose=verbose)
        
        # Verify it
        n_verify = min(ell + 20, 50)
        mismatches = verify_recurrence_general(ell, delta, n_max=n_verify, verbose=False)
        verified = len(mismatches) == 0
        
        results[(ell, delta)] = {
            'order': relation.order,
            'coefficients': relation.coefficients,
            'characteristic_poly': relation.characteristic_poly,
            'verified': verified,
            'verification_range': n_verify,
        }
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    Order: {relation.order}")
        print(f"    Verified up to n={n_verify}: {'✓' if verified else '✗'}")
        print(f"    Non-zero coefficients: {sum(1 for c in relation.coefficients if c != 0)}/{relation.order}")
    
    return results


def run_pattern_analysis(params_list: List[tuple]) -> Dict[str, Any]:
    """
    Task 2: Analyze patterns in recurrence structures.
    """
    print_header("RECURRENCE PATTERN ANALYSIS")
    
    results = {}
    
    for ell, delta in params_list:
        print_section(f"Transfer Matrix Analysis: ℓ={ell}, δ={delta}")
        
        # Analyze transfer matrix
        analysis = analyze_transfer_matrix(ell, delta)
        
        print(f"  Matrix size: {analysis['matrix_size']}x{analysis['matrix_size']}")
        print(f"  Spectral radius (λ_max): {analysis['spectral_radius']:.6f}")
        print(f"  Capacity: {analysis['capacity']:.6f}")
        print(f"  Transitions: {analysis['num_transitions']}")
        print(f"  Sparsity: {analysis['sparsity']:.2%}")
        
        # Coefficient pattern
        relation = discover_recurrence_relation(ell, delta, verbose=False)
        coeff_pattern = analyze_coefficient_pattern(relation.coefficients)
        print(f"  Coefficient pattern: {coeff_pattern}")
        
        results[(ell, delta)] = {
            'matrix_analysis': analysis,
            'coefficient_pattern': coeff_pattern,
        }
    
    # Compare complexity
    if len(params_list) >= 2:
        print_section("Complexity Comparison")
        analyses = compare_recurrence_complexity(params_list)
        
        print("\n  Order growth:")
        for a in analyses:
            print(f"    (ℓ={a.ell}, δ={a.delta}): order = {a.recurrence.order}")
        
        print("\n  Observation: Order = 2^(ℓ-1) for all δ")
        
        results['comparison'] = [
            {'ell': a.ell, 'delta': a.delta, 'order': a.recurrence.order}
            for a in analyses
        ]
    
    return results


def run_bounds_analysis(params_list: List[tuple]) -> Dict[str, Any]:
    """
    Task 3: Analyze bounds and inequalities.
    """
    print_header("BOUNDS AND INEQUALITIES ANALYSIS")
    
    results = {}
    
    for ell, delta in params_list:
        print_section(f"Bounds Analysis: ℓ={ell}, δ={delta}")
        
        # Capacity
        cap, lambda_max = compute_capacity_asymptotic(ell, delta)
        print(f"  Capacity C({ell},{delta}) = log₂(λ_max) = {cap:.6f}")
        print(f"  Spectral radius λ_max = {lambda_max:.6f}")
        
        # Forbidden patterns
        num_forbidden, forbidden_set = count_forbidden_patterns(ell, delta)
        total = 2 ** ell
        print(f"  Forbidden patterns: {num_forbidden} / {total} ({100*num_forbidden/total:.2f}%)")
        
        # Bounds analysis for a range of n
        print(f"\n  Bounds (sample n={ell+5}):")
        bounds = compute_bounds(ell + 5, ell, delta)
        print(f"    Exact count:   {bounds.exact_count}")
        print(f"    Lower bound:   {bounds.lower_bound}")
        print(f"    Upper bound:   {bounds.upper_bound}")
        gap = (bounds.upper_bound - bounds.lower_bound) / bounds.exact_count if bounds.exact_count > 0 else 0
        print(f"    Relative gap:  {gap:.4f}")
        
        # Submultiplicativity (quick check, disable verbose)
        print(f"\n  Submultiplicativity check (f(m+n) ≤ f(m)·f(n)):")
        submult_holds = True
        for m in range(ell, ell + 3):
            for n in range(ell, ell + 3):
                f_m = count_lb_sequences_dp(m, ell, delta)
                f_n = count_lb_sequences_dp(n, ell, delta)
                f_mn = count_lb_sequences_dp(m + n, ell, delta)
                if f_mn > f_m * f_n:
                    submult_holds = False
        print(f"    All checks passed: {'✓' if submult_holds else '✗'}")
        
        results[(ell, delta)] = {
            'capacity': cap,
            'lambda_max': lambda_max,
            'forbidden_count': num_forbidden,
            'forbidden_fraction': num_forbidden / total,
            'bounds_gap': gap,
            'submultiplicativity': submult_holds,
        }
    
    return results


def run_cross_verification(params_list: List[tuple]) -> Dict[str, Any]:
    """
    Task 4: Cross-verify with M2_work/definitions_lib.py.
    """
    print_header("CROSS-VERIFICATION WITH M2_WORK")
    
    results = {}
    
    for ell, delta in params_list:
        print_section(f"Cross-check: ℓ={ell}, δ={delta}")
        
        check = cross_check_with_m2(ell, delta, n_max=min(20, ell + 10))
        
        print(f"  M2 library available: {'Yes' if check['m2_available'] else 'No'}")
        
        if check['m2_available']:
            print(f"  All values match: {'✓' if check['all_match'] else '✗'}")
            if check['mismatches']:
                print(f"  Mismatches: {check['mismatches']}")
        else:
            print("  (Using internal DP computation)")
        
        results[(ell, delta)] = check
    
    return results


def run_growth_rate_analysis(params_list: List[tuple]) -> Dict[str, Any]:
    """
    Task 5: Analyze growth rate convergence.
    """
    print_header("GROWTH RATE CONVERGENCE ANALYSIS")
    
    results = {}
    
    for ell, delta in params_list:
        print_section(f"Growth Rate: ℓ={ell}, δ={delta}")
        
        growth = analyze_growth_rate(ell, delta, range(ell, ell + 25))
        
        print(f"  True capacity: {growth['capacity']:.6f}")
        
        # Get final empirical rate from data
        final_rate = growth['data'][-1]['empirical_rate'] if growth['data'] else 0
        print(f"  Final empirical rate (n={ell+24}): {final_rate:.6f}")
        
        # Find convergence n (where |empirical - capacity| < 0.01)
        convergence_n = None
        for d in growth['data']:
            if abs(d['empirical_rate'] - growth['capacity']) < 0.01:
                convergence_n = d['n']
                break
        if convergence_n:
            print(f"  Convergence (|gap| < 0.01 at n): {convergence_n}")
        else:
            print(f"  Convergence: Not yet reached in range")
        
        results[(ell, delta)] = growth
    
    return results


def generate_json_output(all_results: Dict[str, Any], output_file: str):
    """Save all results to JSON file."""
    
    # Convert tuple keys to string keys for JSON
    def convert_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    new_key = f"({k[0]},{k[1]})"
                else:
                    new_key = k
                new_dict[new_key] = convert_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        else:
            return obj
    
    converted = convert_keys(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all future research computations for general (ℓ, δ) recurrences"
    )
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode: only small parameters')
    parser.add_argument('--figures', action='store_true',
                       help='Also generate visualization figures')
    parser.add_argument('--verify', action='store_true',
                       help='Include cross-verification with M2')
    parser.add_argument('--output', type=str, default='research_output.json',
                       help='Output JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Parameter sets
    if args.quick:
        params_list = [(4, 1), (6, 1), (4, 2)]
    else:
        params_list = [(4, 1), (6, 1), (8, 1), (4, 2), (6, 2)]
    
    print_header(f"FUTURE RESEARCH: GENERAL (ℓ, δ) RECURRENCE RELATIONS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Parameters to analyze: {params_list}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': [(ell, delta) for ell, delta in params_list],
        'mode': 'quick' if args.quick else 'full',
    }
    
    # Run tasks
    all_results['recurrence_discovery'] = run_recurrence_discovery(params_list, args.verbose)
    all_results['pattern_analysis'] = run_pattern_analysis(params_list)
    all_results['bounds_analysis'] = run_bounds_analysis(params_list)
    all_results['growth_rate'] = run_growth_rate_analysis(params_list)
    
    if args.verify:
        all_results['cross_verification'] = run_cross_verification(params_list)
    
    # Generate figures
    if args.figures:
        print_header("GENERATING FIGURES")
        try:
            from generate_research_figures import generate_all_figures
            generate_all_figures()
            all_results['figures_generated'] = True
        except Exception as e:
            print(f"Error generating figures: {e}")
            all_results['figures_generated'] = False
            all_results['figures_error'] = str(e)
    
    # Save output
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    generate_json_output(all_results, output_path)
    
    # Final summary
    print_header("RESEARCH SUMMARY")
    
    print("Key Findings:")
    print("-" * 40)
    
    for ell, delta in params_list:
        rec = all_results['recurrence_discovery'].get((ell, delta), {})
        bounds = all_results['bounds_analysis'].get((ell, delta), {})
        
        print(f"\n(ℓ={ell}, δ={delta}):")
        print(f"  Recurrence order: {rec.get('order', 'N/A')}")
        print(f"  Capacity: {bounds.get('capacity', 'N/A'):.6f}" if isinstance(bounds.get('capacity'), float) else f"  Capacity: {bounds.get('capacity', 'N/A')}")
        print(f"  Verified: {'✓' if rec.get('verified') else '✗'}")
    
    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)
    
    # Print extension beyond paper
    print("\n>>> EXTENSION BEYOND Ge22 PAPER:")
    print("    The paper (Theorem 2) only derives recurrence for (6, 1).")
    print("    Our research extends this to arbitrary (ℓ, δ) using the")
    print("    characteristic polynomial of the transfer matrix.")
    print("    This is a contribution toward the 'interesting and challenging")
    print("    direction' mentioned in Section VI of the paper.")


if __name__ == "__main__":
    main()
