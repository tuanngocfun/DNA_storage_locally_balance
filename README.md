# M4 Verifier - Locally Balanced Constraints

> **Role**: M4 - Verifier (Math & Graph)  
> **Author**: Nguyễn Tuấn Ngọc  
> **Paper**: "Coding for Locally Balanced Constraints" (Ge22)

---

## 📋 Overview

This project implements the **Verifier** role for validating Sections IV & V of the paper:
- **Algorithm 1**: Graph-based optimal rate search (Construction 3)
- **Theorem 2**: Recurrence relation verification
- **Capacity**: Shannon capacity via spectral radius
- **Cross-check**: Audit against M2's implementation

---

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all verification scripts
python scripts/run_golden.py                      # Golden tests (27/27 pass)
python scripts/search_best_m.py --ell 8           # Algorithm 1 rate search
python scripts/verify_recurrence.py               # Theorem 2 verification
python scripts/cross_check_m2.py                  # M2 cross-check (0 mismatches)
python scripts/transfer_matrix_theorem2.py        # Transfer matrix analysis
python scripts/generate_slide_figures.py          # Generate plots
```

---

## 📁 Project Structure

```
codes/
├── src/lbcode/                    # Core Python library
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
│   └── golden_test_cases_v2.json  # 14 extended test cases
│
├── slides/                        # Presentation materials
│   ├── figures/                   # Generated plots (PNG)
│   │   ├── rate_curve.png         # Rate vs m plot
│   │   ├── fn_sequence.png        # f_n growth plot
│   │   └── combined_rate_comparison.png
│   ├── kimi2.md                   # Slide outline
│   └── tables_and_plots.md        # Data tables & figures
│
├── visualization/                 # Interactive React demos
│   └── (see visualization/README.md)
│
├── reports_for_lecturer/          # Detailed reports (Vietnamese)
│   ├── 01_work_completed.md       # Work summary & results
│   ├── 02_questions_for_lecturer.md
│   └── 03_code_explanations.md    # Code deep-dive
│
├── M2_work/                       # M2's code (for cross-check)
│   └── definitions_lib.py
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧪 Running Verification Scripts

### 1. Golden Test Cases
```bash
python scripts/run_golden.py
```
**Expected Output:**
```
TOTAL: 13/13 checks passed (v1)
TOTAL: 14/14 checks passed (v2)
ALL_OK = True
```

### 2. Algorithm 1 - Rate Search (Construction 3)
```bash
# For (ℓ=8, δ=1) - Paper's main example
python scripts/search_best_m.py --ell 8 --delta 1 --m_min 7 --m_max 14

# For other parameters
python scripts/search_best_m.py --ell 4 --delta 1 --m_min 10 --m_max 15
python scripts/search_best_m.py --ell 6 --delta 1 --m_min 10 --m_max 16
```
**Expected Output (ℓ=8):**
```
BEST: m=13, s=10, rate=0.76923
✓ Matches paper's Construction 3 result!
```

### 3. Theorem 2 - Recurrence Verification
```bash
python scripts/verify_recurrence.py --ell 6 --delta 1 --n_max 30
```
**Expected Output:**
```
f_n values: [1, 2, 4, 8, 16, 32, 50, 90, 162, 290, 518, 926, 1662...]
✓ Recurrence verified for n ∈ [6, 30]: 0 mismatches
```

### 4. Capacity Computation
```bash
python scripts/verify_recurrence.py  # Capacity is printed at the end
```
**Expected Output:**
```
Capacity(ℓ=6, δ=1) = 0.84083 ≈ Paper 0.841 ✓
Capacity(ℓ=8, δ=1) = 0.82410 ≈ Paper 0.824 ✓
```

### 5. Cross-Check with M2
```bash
python scripts/cross_check_m2.py
```
**Expected Output:**
```
Golden cases: 11/11 pass
Random 1000 strings: 0 mismatches
✓ CROSS-CHECK PASSED!
```

### 6. Transfer Matrix Method
```bash
python scripts/transfer_matrix_theorem2.py
```
**Expected Output:**
```
Characteristic Polynomial: degree 32
Dominant factor (degree-11): λ_max = 1.791081
Capacity = log₂(1.791081) = 0.84083 ✓
```

### 7. Generate Slide Figures
```bash
python scripts/generate_slide_figures.py
```
**Output Files:**
- `slides/figures/rate_curve.png`
- `slides/figures/fn_sequence.png`
- `slides/figures/combined_rate_comparison.png`

---

## ✅ Verification Summary

| Task | Result | Status |
|------|--------|--------|
| Golden Tests v1 | 13/13 PASS | ✅ |
| Golden Tests v2 | 14/14 PASS | ✅ |
| Algorithm 1 (ℓ=4, m=13) | rate = 11/13 = 0.846 | ✅ Matches paper |
| Algorithm 1 (ℓ=6, m=15) | rate = 12/15 = 0.800 | ✅ Matches paper |
| Algorithm 1 (ℓ=8, m=13) | rate = 10/13 = 0.769 | ✅ Matches paper |
| Theorem 2 Recurrence | 0 mismatches (n ∈ [6, 30]) | ✅ |
| Capacity (ℓ=6, δ=1) | 0.84083 ≈ 0.841 | ✅ Matches paper |
| Capacity (ℓ=8, δ=1) | 0.82410 ≈ 0.824 | ✅ Matches paper |
| M2 Cross-check | 0 mismatches on 1000 strings | ✅ |

---

## 📊 Interactive Visualizations

See `visualization/README.md` for interactive React demos:
- **DP Automaton Diagram**: State transition graph visualization
- **Locally Balanced Checker**: Animated window-by-window verification

```bash
cd visualization
npm install
npm run dev
# Open http://localhost:5173
```

---

## 📚 Key Concepts

### Definition 1: (ℓ, δ)-Locally Balanced
A binary string is locally balanced if every window of length ℓ has Hamming weight in `[ℓ/2 - δ, ℓ/2 + δ]`.

| Parameters | Valid Weight Range | Forbidden Patterns |
|------------|-------------------|-------------------|
| (ℓ=4, δ=1) | [1, 3] | `0000`, `1111` |
| (ℓ=6, δ=1) | [2, 4] | `000000`, etc. |
| (ℓ=8, δ=1) | [3, 5] | Long runs |

### Algorithm 1: Graph Pruning
1. Build graph G_m with all valid m-bit strings as vertices
2. Edge x→y if concatenation xy is locally balanced
3. Iteratively prune vertices with out-degree < 2^s
4. Find largest s where non-empty subgraph survives

### Capacity
Shannon capacity = log₂(λ_max) where λ_max is spectral radius of transfer matrix.

---

## 📞 Support

For questions about this implementation, refer to:
- `reports_for_lecturer/01_work_completed.md` - Complete work summary
- `reports_for_lecturer/03_code_explanations.md` - Code deep-dive
