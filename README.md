# M4 Verifier - Locally Balanced Constraints

> **Role**: M4 - Verifier (Math & Graph)  
> **Author**: Nguyễn Tuấn Ngọc  
> **Paper**: "Coding for Locally Balanced Constraints" (Ge22)

A comprehensive implementation and verification suite for locally balanced binary string constraints based on the Ge22 paper. This project validates Sections IV & V of the paper through automated testing, graph algorithms, and mathematical recurrence verification.

---

## 📋 Overview

This project implements the **M4 Verifier** role for a group coding theory project, providing:

- **Definition Verification**: (ℓ,δ)-locally balanced constraint checker
- **Algorithm 1**: Graph-based optimal rate search (Construction 3)
- **Theorem 2**: Recurrence relation verification via dynamic programming
- **Capacity Analysis**: Shannon capacity computation using spectral radius
- **Cross-validation**: Audit against M2's implementation
- **Visualization**: Interactive React-based demos

---

## 🚀 Quick Start

```bash
# 1. Clone and navigate to the project
cd codes

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run all verification scripts
python scripts/run_golden.py                      # Golden tests (27/27 pass)
python scripts/search_best_m.py --ell 8           # Algorithm 1 rate search
python scripts/verify_recurrence.py               # Theorem 2 verification
python scripts/cross_check_m2.py                  # M2 cross-check (0 mismatches)
```

---

## 📁 Project Structure

```
codes/
├── src/lbcode/                    # Core Python library
│   ├── __init__.py                # Package exports
│   ├── verifier.py                # Definition 1: (ℓ,δ)-locally balanced check
│   ├── graph_alg1.py              # Algorithm 1: graph building & pruning
│   └── dp_automaton.py            # DP counting, recurrence, capacity
│
├── scripts/                       # Executable verification scripts
│   ├── run_golden.py              # Run golden test suites (v1 + v2)
│   ├── search_best_m.py           # Find best rate (s,m) for Algorithm 1
│   ├── verify_recurrence.py       # Verify Theorem 2 recurrence
│   ├── ref_check.py               # Self-check with brute-force baseline
│   ├── cross_check_m2.py          # Cross-check against M2's code
│   ├── transfer_matrix_theorem2.py # Transfer matrix method
│   └── generate_slide_figures.py  # Generate visualization plots
│
├── test_data/                     # Test datasets (JSON)
│   ├── golden_test_cases.json     # 13 basic test cases
│   ├── golden_test_cases_v2.json  # 14 extended test cases
│   └── enhance_test.py            # Enhanced test generator
│
├── M2_work/                       # M2's code (for cross-check)
│   └── definitions_lib.py         # DNAStorageCodeChecker implementation
│
├── M3_work/                       # M3's FSM construction
│   └── construction2_fsm.py       # Paper Construction 2 implementation
│
├── slides/                        # Presentation materials
│   ├── figures/                   # Generated plots (PNG)
│   │   ├── rate_curve.png         # Rate vs m plot
│   │   ├── fn_sequence.png        # f_n growth plot
│   │   └── combined_rate_comparison.png
│   ├── kimi2.md                   # Slide outline
│   ├── lecturer_notes.md          # Teaching notes
│   └── tables_and_plots.md        # Data tables & figures
│
├── visualization/                 # Interactive React demos
│   ├── main.jsx                   # Navigation hub
│   ├── demo.jsx                   # DP Automaton diagram
│   ├── golden_test_locally_balanced_bin_str.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── README.md                  # Visualization guide
│
├── requirements.txt               # Python dependencies (numpy, sympy, pytest)
├── task-assigment.md              # Group role assignments
└── README.md                      # This file
```

---

## 🧪 Core Features & Usage

### 1. Golden Test Cases

Validates implementation against standardized test cases shared across all team members.

```bash
python scripts/run_golden.py
```

**Expected Output:**
```
Loading: test_data/golden_test_cases.json
Global parameters: ℓ=4, δ=1 => valid weight: [1,3]

--- Suite: Basic Tests (ℓ=4, δ=1) ---
  ✓ 13/13 checks passed

TOTAL: 13/13 checks passed
ALL_OK = True
```

Run v2 test suite:
```bash
python scripts/run_golden.py --v2
```

---

### 2. Algorithm 1 - Rate Search (Construction 3)

Finds optimal encoding rate s/m by building graph G_m and iteratively pruning vertices.

```bash
# For (ℓ=8, δ=1) - Paper's main example
python scripts/search_best_m.py --ell 8 --delta 1 --m_min 7 --m_max 14

# For other parameters
python scripts/search_best_m.py --ell 4 --delta 1 --m_min 10 --m_max 15
python scripts/search_best_m.py --ell 6 --delta 1 --m_min 10 --m_max 16
```

**Expected Output (ℓ=8):**
```
Algorithm 1: Searching for best rate for (ℓ=8, δ=1)
Paper benchmark for (8,1): Construction 3 achieves 10/13 = 0.76923
--------------------------------------------------
m= 7: s= 4, rate=0.57143, |V|=32
m= 8: s= 5, rate=0.62500, |V|=64
m= 9: s= 6, rate=0.66667, |V|=114
m=10: s= 7, rate=0.70000, |V|=196
m=11: s= 8, rate=0.72727, |V|=332
m=12: s= 9, rate=0.75000, |V|=544
m=13: s=10, rate=0.76923, |V|=880
m=14: s=10, rate=0.71429, |V|=1404
--------------------------------------------------
BEST: m=13, s=10, rate=0.76923

✓ Matches paper's Construction 3 result!
```

**What it does:**
- Builds graph G_m where vertices are valid m-bit strings
- Edge x→y exists if concatenation xy satisfies locally balanced constraint
- Prunes vertices iteratively to find largest s with minimum out-degree ≥ 2^s
- Rate = s/m represents information bits per codeword bit

---

### 3. Theorem 2 - Recurrence Verification

Verifies the recurrence relation from Section III using dynamic programming.

```bash
python scripts/verify_recurrence.py --ell 6 --delta 1 --n_max 30
```

**Expected Output:**
```
==============================================================
Theorem 2 Verification: Recurrence Relation
==============================================================

For (ℓ=6, δ=1):
  f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n

Verifying Theorem 2 recurrence for (ℓ=6, δ=1)
f_n values (first 15): [1, 2, 4, 8, 16, 32, 50, 90, 162, 290, 518, 926, 1662, 2978, 5334]

✓ Recurrence verified for n ∈ [6, 30]: 0 mismatches

==============================================================
Capacity Computation
==============================================================

Capacity(ℓ=6, δ=1) = 0.84083
Paper Table I value: ~0.841
✓ Matches paper!
```

**What it does:**
- Counts f_n = number of n-bit locally balanced strings using DP automaton
- Verifies recurrence relation holds for specified range of n
- Computes Shannon capacity as log₂(λ_max) where λ_max is spectral radius

---

### 4. Cross-Check with M2's Implementation

Validates consistency between M4 verifier and M2's `definitions_lib.py`.

```bash
python scripts/cross_check_m2.py --num 1000
```

**Expected Output:**
```
============================================================
Golden Test Cases: M4 vs M2 Comparison
============================================================
Total golden cases: 13
Mismatches between M4 and M2: 0
✓ M4 and M2 agree on all golden test cases!

============================================================
Cross-Check: M4 Verifier vs M2 definitions_lib.py
============================================================
Parameters: ℓ=8, δ=1
Test strings: 1000 random, length ∈ [10, 20]

Results: 1000 strings tested
  Valid (LB): 287
  Invalid: 713
  Mismatches: 0
  ✓ No mismatches found! M4 and M2 agree on all test cases.

✓ CROSS-CHECK PASSED: M4 and M2 implementations agree!
```

---

### 5. Transfer Matrix Analysis

Alternative method for computing capacity using symbolic polynomial factorization.

```bash
python scripts/transfer_matrix_theorem2.py
```

**Expected Output:**
```
Transfer Matrix Analysis: (ℓ=6, δ=1)
Characteristic polynomial: degree 32
Dominant factor (degree-11): λ_max = 1.791081
Capacity = log₂(1.791081) = 0.84083 ✓
```

---

### 6. Generate Slide Figures

Creates publication-quality plots for presentations.

```bash
python scripts/generate_slide_figures.py
```

**Output Files:**
- `slides/figures/rate_curve.png` - Rate vs m plot
- `slides/figures/fn_sequence.png` - f_n growth plot
- `slides/figures/combined_rate_comparison.png` - Multi-parameter comparison

---

## ✅ Verification Summary

| Task | Result | Status |
|------|--------|--------|
| Golden Tests v1 | 13/13 PASS | ✅ |
| Golden Tests v2 | 14/14 PASS | ✅ |
| Algorithm 1 (ℓ=4, m=13) | rate = 11/13 = 0.846 | ✅ Matches paper |
| Algorithm 1 (ℓ=6, m=15) | rate = 12/15 = 0.800 | ✅ Matches paper |
| Algorithm 1 (ℓ=8, m=13) | rate = 10/13 = 0.769 | ✅ Matches paper |
| Theorem 2 Recurrence (ℓ=6) | 0 mismatches (n ∈ [6, 30]) | ✅ |
| Capacity (ℓ=4, δ=1) | 0.9468 | ✅ Matches paper |
| Capacity (ℓ=6, δ=1) | 0.84083 ≈ 0.841 | ✅ Matches paper |
| Capacity (ℓ=8, δ=1) | 0.82410 ≈ 0.824 | ✅ Matches paper |
| M2 Cross-check | 0 mismatches on 1000 strings | ✅ |

---

## 📊 Interactive Visualizations

Interactive React-based demonstrations help visualize the concepts:

### Available Demos

1. **DP Automaton Diagram** - State transition graph with adjustable parameters
2. **Locally Balanced Checker** - Animated window-by-window verification

### Running Locally

```bash
cd visualization
npm install
npm run dev
# Open http://localhost:5173
```

See `visualization/README.md` for detailed instructions.

---

## 📚 Key Concepts

### Definition 1: (ℓ, δ)-Locally Balanced

A binary string is **(ℓ, δ)-locally balanced** if every window of length ℓ has Hamming weight in `[ℓ/2 - δ, ℓ/2 + δ]`.

| Parameters | Valid Weight Range | Forbidden Patterns | Example Valid | Example Invalid |
|------------|-------------------|-------------------|---------------|-----------------|
| (ℓ=4, δ=1) | [1, 3] | `0000`, `1111` | `0110` | `0000` |
| (ℓ=6, δ=1) | [2, 4] | `000000`, `111111` | `011010` | `000011` |
| (ℓ=8, δ=1) | [3, 5] | Long runs | `01101001` | `00001111` |

### Algorithm 1: Graph Pruning

**Procedure:**
1. Build graph G_m with all valid m-bit strings as vertices
2. Add edge x→y if concatenation xy satisfies locally balanced constraint
3. Iteratively prune vertices with out-degree < 2^s
4. Find largest s where non-empty subgraph survives
5. Rate = s/m

**Complexity:**
- Vertices: O(2^m) in worst case, typically much smaller
- Edges: O(|V|²) to construct
- Pruning: O(|V| + |E|) per iteration

### Theorem 2: Recurrence Relation

For (ℓ=6, δ=1), the number of valid n-bit strings f_n satisfies:

```
f_{n+12} = f_{n+11} + f_{n+10} + f_{n+9} - f_{n+6} - f_{n+4} - f_{n+3} + f_n
```

Verified computationally using dynamic programming on the state automaton.

### Capacity

Shannon capacity C = log₂(λ_max) where λ_max is the spectral radius (largest eigenvalue) of the transfer matrix.

**Computed values:**
- C(4,1) = 0.9468
- C(6,1) = 0.8408
- C(8,1) = 0.8241

---

## 🔧 Development

### Running Tests

```bash
# All golden tests
python scripts/run_golden.py
python scripts/run_golden.py --v2

# Self-check against brute force
python scripts/ref_check.py

# Enhanced test generation
python test_data/enhance_test.py
```

### Adding New Test Cases

Edit `test_data/golden_test_cases.json`:

```json
{
  "meta": {
    "global_parameters": {"l": 4, "delta": 1}
  },
  "test_suites": [
    {
      "suite_name": "Basic Tests",
      "parameters": {"l": 4, "delta": 1},
      "cases": [
        {
          "id": "TC001",
          "input": "01010",
          "expect": true
        }
      ]
    }
  ]
}
```

### Library Usage

```python
from lbcode import is_locally_balanced, algorithm1_find_core

# Check if string is locally balanced
result = is_locally_balanced("01101001", ell=8, delta=1)
print(f"Valid: {result}")  # Valid: True

# Run Algorithm 1
core = algorithm1_find_core(m=13, ell=8, delta=1, verbose=True)
print(f"Best rate: {core.rate:.5f}")  # Best rate: 0.76923
```

---

## 📦 Dependencies

- **numpy** >= 1.20.0 - Numerical computations, eigenvalue calculations
- **sympy** >= 1.9 - Symbolic math for polynomial factorization
- **pytest** >= 7.0.0 - Testing framework

Install with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Team Members

This project is part of a group assignment with the following roles:

- **M1** (Lâm Xuân Bách) - Constructor A: Greedy algorithm
- **M2** (Mai Thị Hằng Thư) - Simulator & Tools: Utilities and testing
- **M3** (Dương Tất Hùng) - Constructor B: Paper FSM implementation
- **M4** (Nguyễn Tuấn Ngọc) - Verifier: Math & graph validation (this repo)

---

## 📄 License

This is an academic project for educational purposes. Code based on:

**Paper**: "Coding for Locally Balanced Constraints" by Ge et al. (2022)

---

## 📞 Support & Contact

For questions about this implementation:
- See `task-assigment.md` for project requirements
- Check `slides/lecturer_notes.md` for conceptual explanations
- Review code documentation in `src/lbcode/`

---

**Last Updated**: January 2026
