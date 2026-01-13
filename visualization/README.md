# M4 Verifier - Interactive Visualizations

> React-based interactive demos for understanding locally balanced constraints

---

## 🚀 Quick Start

```bash
# Navigate to visualization folder
cd visualization

# Install dependencies (already done if you see node_modules/)
npm install

# Start development server
npm run dev

# Open in browser
# → http://localhost:5173
```

---

## 📱 Available Demos

### 1. DP Automaton Diagram (`demo.jsx`)
Interactive state transition graph visualization:
- **States**: All (ℓ-1)-bit strings
- **Edges**: Valid transitions based on window weight
- **Hover**: Highlight outgoing edges from any state
- **Adjustable**: Change ℓ and δ parameters in real-time

### 2. Locally Balanced Checker (`golden_test_locally_balanced_bin_str.jsx`)
Animated window-by-window verification:
- **Input**: Type any binary string
- **Animation**: Watch each window being checked
- **Feedback**: See weight and validity for each window
- **Random**: Generate random test cases

---

## 📁 File Structure

```
visualization/
├── main.jsx           # Navigation hub (switch between demos)
├── demo.jsx           # DP Automaton diagram
├── golden_test_locally_balanced_bin_str.jsx  # LB checker
├── index.html         # HTML entry point
├── vite.config.js     # Vite configuration
├── package.json       # Dependencies & scripts
└── README.md          # This file
```

---

## 🔧 Manual Setup (if npm install fails)

```bash
# Initialize package.json
npm init -y

# Install React and dependencies
npm install react react-dom lucide-react

# Install Vite for development
npm install -D vite @vitejs/plugin-react

# Add "dev" script to package.json if missing
# "scripts": { "dev": "vite" }

# Start
npm run dev
```

---

## ✅ Logic Verification

Both demos have been verified against the Python implementation:

| Aspect | React | Python (`dp_automaton.py`) | Match |
|--------|-------|---------------------------|-------|
| State length | `ell - 1` bits | `ell - 1` bits | ✅ |
| Valid weight | `[ell/2 - delta, ell/2 + delta]` | `[ell//2 - delta, ell//2 + delta]` | ✅ |
| Transition | `s + bit → s.slice(1) + bit` | `s + b → s[1:] + b` | ✅ |
| Weight calc | `filter(b === '1').length` | `prefix sum or count` | ✅ |

---

## 🎮 Usage Tips

### Automaton Diagram (ℓ=4)
1. Set ℓ=4, δ=1 (default)
2. Observe 8 states (3-bit strings: 000, 001, ..., 111)
3. Hover over any node to see its valid transitions
4. Check the transition table below the graph

### Locally Balanced Checker
1. Enter a binary string like `01101001`
2. Click "Check Balance" to start animation
3. Watch each window highlight green (valid) or red (invalid)
4. Final result shows overall pass/fail

---

## 🌐 Alternative: Online Demo

If you can't run locally, use StackBlitz:

1. Go to https://stackblitz.com/fork/react
2. Copy the contents of `demo.jsx` into `App.jsx`
3. Add `lucide-react` to dependencies in `package.json`
4. The demo will run automatically!

---

## 📊 Screenshots

When running, you'll see:
- **Home Page**: Navigation cards for both demos
- **Automaton Diagram**: Circular graph with 8 nodes and edges
- **LB Checker**: Animated binary string with sliding window

---

## 🔗 Related Files

- **Python implementation**: `../src/lbcode/dp_automaton.py`
- **Golden test data**: `../test_data/golden_test_cases.json`
- **Verification report**: `../reports_for_lecturer/03_code_explanations.md`
