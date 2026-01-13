#!/usr/bin/env python3
"""
Search for the best rate s/m using Algorithm 1.
Reproduces Construction 3 results from the paper.

Author: Nguyen Tuan Ngoc
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lbcode.graph_alg1 import search_best_rate


def main():
    parser = argparse.ArgumentParser(description='Search for best rate using Algorithm 1')
    parser.add_argument('--ell', type=int, default=8, help='Window length ℓ (default: 8)')
    parser.add_argument('--delta', type=int, default=1, help='Deviation δ (default: 1)')
    parser.add_argument('--m_min', type=int, default=7, help='Minimum block length m')
    parser.add_argument('--m_max', type=int, default=14, help='Maximum block length m')
    args = parser.parse_args()
    
    print(f"Algorithm 1: Searching for best rate for (ℓ={args.ell}, δ={args.delta})")
    print(f"Paper benchmark for (8,1): Construction 3 achieves 10/13 = 0.76923")
    print(f"Paper capacity (Table I) for (8,1): ~0.824")
    print()
    
    best = search_best_rate(
        ell=args.ell, 
        delta=args.delta, 
        m_min=args.m_min, 
        m_max=args.m_max,
        verbose=True
    )
    
    if best:
        print()
        print(f"Rate achieved: {best.rate:.5f}")
        if args.ell == 8 and args.delta == 1:
            if best.m == 13 and best.s == 10:
                print("✓ Matches paper's Construction 3 result!")
            elif best.rate > 10/13:
                print("✓ BETTER than paper's Construction 3!")
            else:
                print("Rate is lower than paper - check implementation")


if __name__ == '__main__':
    main()
