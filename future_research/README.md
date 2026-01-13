# Future Research: Fast Recurrence Discovery for Locally Balanced Sequences

This folder contains research code that extends **Theorem 2** from the Ge22 paper to discover minimal linear recurrences for locally balanced sequences with **arbitrary (ℓ, δ) parameters**.

## Overview

The code implements a highly optimized algorithm to find minimal linear recurrence relations for the sequence `f_n(ℓ, δ)`, which counts the number of (ℓ, δ)-locally balanced binary strings of length `n`. This extends the theoretical work to handle cases far beyond what was previously computationally feasible.

### Key Innovation

Instead of computing the characteristic polynomial of massive transfer matrices (which becomes intractable for large ℓ), we:
1. **Compute sequence values** directly using state compression dynamic programming
2. **Apply Berlekamp-Massey algorithm** with Chinese Remainder Theorem (CRT) optimization
3. **Use majority voting** across multiple large primes to ensure correctness

This approach successfully handles transfer matrices up to **8192×8192** (ℓ=14) and beyond.

## Files

- **`fast_recurrence.py`** - Main implementation with CLI interface

## Algorithm Details

### 1. Sequence Computation (State Compression DP)

Computes `f_n` values using dynamic programming with arbitrary precision integers:
- **Time complexity**: `O(n_max × d)` where `d = 2^ℓ` is the number of states
- **Space complexity**: `O(d)` with state compression
- **Uses Python's arbitrary precision** to avoid integer overflow

### 2. Berlekamp-Massey with CRT Optimization

The core innovation that makes large-scale computation feasible:

```
For each of 12 large primes p near 10^18:
  1. Compute sequence modulo p
  2. Run Berlekamp-Massey algorithm mod p → find recurrence order k_p
  
Use majority voting to determine the true order k
  
For all primes p where k_p == k:
  1. Extract recurrence coefficients mod p
  2. Use CRT to reconstruct integer coefficients
```

**Why this works:**
- Modular arithmetic is **much faster** than rational arithmetic
- Combined modulus of 12 primes ≈ 10^216, sufficient for coefficients up to ~10^54
- Majority voting handles edge cases where a prime finds a smaller "coincidental" recurrence

**Primes used:**
```python
[10^18 + 9, 10^18 + 7, 10^18 + 3, 10^18 + 31, 10^18 + 51, 
 10^18 + 81, 10^18 + 111, 10^18 + 117, 10^18 + 127, 
 10^18 + 177, 10^18 + 207, 10^18 + 213]
```

### 3. Verification

After finding the recurrence, we verify it against additional sequence terms to ensure correctness.

## Results

| ℓ | Matrix Size | Recurrence Order | Computation Time | Status |
|---|-------------|------------------|------------------|--------|
| 8 | 128×128 | 53 | 0.05s | ✓ Verified |
| 10 | 512×512 | 184 | 0.2s | ✓ Verified |
| 12 | 2048×2048 | 677 | 3.8s | ✓ Verified |
| 14 | 8192×8192 | 2554 | 291.6s | ✓ Verified |

### Key Observations

1. **Minimal order << Matrix dimension**: For ℓ=14, the minimal recurrence order (2554) is much smaller than the transfer matrix size (8192×8192). This suggests significant factorization in the characteristic polynomial.

2. **Coefficient growth**: Maximum coefficient for ℓ=14 is approximately **5.6×10^54**, requiring high-precision arithmetic or our CRT approach.

3. **Computational feasibility**: The CRT-based approach is **orders of magnitude faster** than direct matrix methods or naive Berlekamp-Massey with rational arithmetic.

## Usage

### Basic Usage

```bash
python fast_recurrence.py --ell 8 --delta 1
```

### Command-Line Options

```
--ell, -l        Parameter ℓ (required)
--delta, -d      Parameter δ (default: 1)
--batch, -b      Run batch analysis for multiple ℓ values
--max-ell        Maximum ℓ for batch mode (default: 10)
--output, -o     Output format: text, json, or latex (default: text)
```

### Examples

**Single test:**
```bash
python fast_recurrence.py --ell 12 --delta 1
```

**Batch analysis:**
```bash
python fast_recurrence.py --batch --max-ell 14
```

**Export to JSON:**
```bash
python fast_recurrence.py --ell 10 --delta 1 --output json > results_10_1.json
```

## Output Format

The program outputs:
- **Recurrence order** (k)
- **Recurrence formula** in readable form
- **Computation time breakdown** (sequence computation, BM algorithm, verification)
- **Verification status** (✓ or ✗)
- **Coefficient statistics** (max coefficient, sum, non-zero count)

Example output:
```
================================================================================
FAST RECURRENCE DISCOVERY: (ℓ=8, δ=1)
================================================================================
Fast recurrence discovery for (ℓ=8, δ=1)...
  Expected order: 128
  Computing 286 terms of f_n sequence...
  Sequence computed in 0.005s
  f_1..f_10: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  Berlekamp-Massey found order 53 in 0.046s
  Verified: ✓
  Total time: 0.052s

--------------------------------------------------------------------------------
RESULT:
--------------------------------------------------------------------------------
  Recurrence order: 53
  Formula: f_{n+53} = ... (recurrence coefficients shown here)
  Non-zero coefficients: 51/53
  Max coefficient: 67108864
  Sum of coefficients: 0
```

## Performance Characteristics

### Time Complexity
- **Sequence computation**: `O(n_max × 2^ℓ)` where `n_max = 2 × (2^ℓ) + 30`
- **Berlekamp-Massey (per prime)**: `O(n_max^2)`
- **Overall**: `O(2^ℓ × n_max + 12 × n_max^2)` ≈ `O(2^ℓ × 2^(2ℓ))`

### Space Complexity
- **Sequence values**: `O(n_max)` arbitrary precision integers
- **DP states**: `O(2^ℓ)` arbitrary precision integers
- **Overall**: `O(n_max + 2^ℓ)`

### Practical Limits
- **ℓ ≤ 14**: Fast (< 5 minutes)
- **ℓ = 16**: Feasible but slow (estimated hours)
- **ℓ ≥ 18**: Challenging (sequence computation becomes bottleneck)

## Technical Details

### Why CRT is Essential

For ℓ=14, the maximum coefficient is approximately 5.6×10^54. Computing this directly requires:
- **Rational arithmetic** with Fractions: Each GCD operation on 5000-digit numbers is extremely slow
- **CRT approach**: All arithmetic stays in Z/(10^18)Z, using only 64-bit integer operations (fast!)

The combined modulus of 12 primes is:
```
∏(p_i) ≈ (10^18)^12 = 10^216
```

This is sufficient for coefficients up to ~10^54.

### Majority Voting Strategy

Different primes may find different recurrence orders due to:
1. **True minimal recurrence** works for all primes
2. **Spurious smaller recurrence** works mod p but not over integers

By using 12 primes and taking the most common order, we ensure we find the true minimal recurrence with high confidence.

### Fallback Mechanism

If majority voting fails or verification doesn't pass, the code automatically falls back to exact rational arithmetic using Python's `Fractions` class (slower but guaranteed correct).

## Connection to Ge22 Paper

This work extends **Theorem 2** which states that the number of (ℓ, δ)-locally balanced binary strings satisfies a linear recurrence whose order is at most the dimension of the transfer matrix.

**Key extensions:**
1. **Computational implementation** for arbitrary (ℓ, δ)
2. **Efficient algorithm** that avoids computing the full characteristic polynomial
3. **Empirical evidence** that the minimal order is often much smaller than the theoretical upper bound

## Future Directions

1. **Theoretical analysis**: Why is the minimal order so much smaller than 2^ℓ?
2. **Optimization**: Can we predict the recurrence order without computing it?
3. **Larger parameters**: Push to ℓ=16, 18, 20 with further optimizations
4. **Pattern analysis**: Study how the recurrence coefficients relate to the combinatorial structure
5. **Generalizations**: Extend to non-binary alphabets or other locality constraints

## References

- **Ge22 paper**: [Insert full citation to the paper about locally balanced sequences]
- **Berlekamp-Massey algorithm**: Efficient algorithm for finding minimal linear recurrence
- **Chinese Remainder Theorem**: Number-theoretic technique for modular arithmetic
- **Transfer matrix method**: Standard technique in combinatorics on words

## Author & Context

This code was developed as part of a Master's program research project on coding theory and locally balanced sequences. The implementation prioritizes:
- **Correctness**: Extensive verification and fallback mechanisms
- **Performance**: CRT optimization for practical computation
- **Extensibility**: Clean API for future research directions

## Dependencies

- **Python 3.8+**: Required for dataclasses and type hints
- **Standard library only**: No external dependencies
  - `math`, `fractions`, `time`, `argparse`, `json`, `dataclasses`

## Testing

The code includes built-in verification that checks the discovered recurrence against additional sequence terms. All results reported above passed verification.

---

**Last updated**: January 2026  
**Status**: Active research, ready for publication
