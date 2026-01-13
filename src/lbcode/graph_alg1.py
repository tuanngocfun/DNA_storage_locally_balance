"""
Graph Algorithm 1 - Section IV & V of Ge22 Paper
Build graph G_m and prune to find out-degree core for Construction 3.

Author: Nguyen Tuan Ngoc

G_m Definition:
- Vertices: All binary strings of length m that are internally (ℓ,δ)-locally balanced
- Edges: Edge x→y exists if concatenation xy is (ℓ,δ)-locally balanced
  (i.e., all windows crossing the boundary satisfy the constraint)

Algorithm 1:
- Start with G_m
- Find largest s such that a non-empty subgraph exists with min out-degree ≥ 2^s
- This gives rate = s/m for the block code
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import List, Set, Optional, Tuple

from .verifier import is_locally_balanced


@dataclass
class CoreResult:
    """Result from Algorithm 1."""
    m: int                      # Block length
    s: int                      # Number of bits encoded per block
    rate: float                 # Rate = s/m
    vertices_bits: List[str]    # Active vertices as bitstrings (length m)
    adj: List[List[int]]        # Adjacency list (indices within vertices_bits)
    num_vertices: int           # |V| of the core
    max_outdeg: int             # Maximum out-degree in core


def _int_to_bits(v: int, m: int) -> str:
    """Convert integer to m-bit binary string."""
    return format(v, f"0{m}b")


def build_valid_vertices(m: int, ell: int, delta: int) -> List[int]:
    """
    Build list of valid vertices for G_m.
    
    A vertex (m-bit string) is valid if it internally satisfies 
    the (ℓ,δ)-locally balanced constraint.
    
    Returns:
        List of integers representing valid m-bit strings
    """
    V = []
    for v in range(1 << m):  # 0 to 2^m - 1
        bits = _int_to_bits(v, m)
        if is_locally_balanced(bits, ell, delta):
            V.append(v)
    return V


def _precompute_weights(v: int, m: int, ell: int) -> Tuple[List[int], List[int]]:
    """
    Precompute prefix and suffix weights for efficient edge checking.
    
    For crossing windows, we need:
    - suffix_w[t] = weight of last t bits of v (for t in 1..ell-1)
    - prefix_w[t] = weight of first t bits of v (for t in 1..ell-1)
    
    A window crossing xy boundary uses:
    - suffix of x (length ell-t) + prefix of y (length t)
    """
    prefix_w = [0] * ell
    suffix_w = [0] * ell
    
    bits = _int_to_bits(v, m)
    
    for t in range(1, ell):
        if t <= m:
            prefix_w[t] = sum(1 for c in bits[:t] if c == '1')
            suffix_w[t] = sum(1 for c in bits[-t:] if c == '1')
    
    return prefix_w, suffix_w


def build_edges(valid_vertices: List[int], m: int, ell: int, delta: int) -> List[List[int]]:
    """
    Build adjacency lists for G_m.
    
    Edge x→y exists iff all ℓ-windows crossing the boundary of xy satisfy
    the weight constraint. Internal windows are already satisfied by 
    vertex filtering.
    
    Returns:
        adj[i] = list of indices j such that edge i→j exists
    """
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    # Precompute prefix/suffix weights for all vertices
    pref = []
    suff = []
    for v in valid_vertices:
        p, s = _precompute_weights(v, m, ell)
        pref.append(p)
        suff.append(s)
    
    n_v = len(valid_vertices)
    adj = [[] for _ in range(n_v)]
    
    # Check each potential edge
    for i in range(n_v):
        for j in range(n_v):
            # Check all crossing windows
            # Window uses (ell-t) bits from suffix of x and t bits from prefix of y
            ok = True
            for t in range(1, min(ell, m + 1)):
                suffix_bits = min(ell - t, m)
                prefix_bits = min(t, m)
                if suffix_bits + prefix_bits == ell:
                    w = suff[i][suffix_bits] + pref[j][prefix_bits]
                    if w < lo or w > hi:
                        ok = False
                        break
            
            if ok:
                adj[i].append(j)
    
    return adj


def outcore_vertices(adj: List[List[int]], min_outdeg: int) -> Set[int]:
    """
    Find the directed out-degree core: vertices that can maintain 
    out-degree ≥ min_outdeg in the induced subgraph.
    
    Uses iterative pruning:
    1. Start with all vertices active
    2. Remove any vertex with out-degree < min_outdeg
    3. Repeat until stable
    """
    n = len(adj)
    
    # Track out-degree within active set
    outdeg = [len(adj[i]) for i in range(n)]
    
    # Build reverse adjacency for efficient updates
    rev = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            rev[v].append(u)
    
    active: Set[int] = set(range(n))
    
    # Queue vertices to remove
    q = deque([i for i in range(n) if outdeg[i] < min_outdeg])
    
    while q:
        v = q.popleft()
        if v not in active:
            continue
        active.remove(v)
        
        # Removing v reduces out-degree of predecessors
        for u in rev[v]:
            if u in active:
                outdeg[u] -= 1
                if outdeg[u] < min_outdeg:
                    q.append(u)
    
    return active


def algorithm1_find_core(m: int, ell: int, delta: int, verbose: bool = False) -> Optional[CoreResult]:
    """
    Algorithm 1 from Section IV: Find the largest s such that G_m has
    a subgraph with minimum out-degree ≥ 2^s.
    
    Args:
        m: Block length (state length)
        ell: Window length for locally balanced constraint
        delta: Allowed weight deviation
        verbose: Print progress info
        
    Returns:
        CoreResult with the best (s, core subgraph), or None if no core exists
    """
    import math
    
    if verbose:
        print(f"Building G_{m} for (ℓ={ell}, δ={delta})...")
    
    # Step 1: Build valid vertices
    V = build_valid_vertices(m, ell, delta)
    if not V:
        if verbose:
            print(f"  No valid vertices found!")
        return None
    
    if verbose:
        print(f"  Valid vertices: {len(V)}")
    
    # Step 2: Build edges
    adj = build_edges(V, m, ell, delta)
    
    # Find max out-degree
    max_deg = max(len(adj[i]) for i in range(len(adj))) if adj else 0
    if max_deg == 0:
        if verbose:
            print(f"  No edges found!")
        return None
    
    if verbose:
        print(f"  Max out-degree (Δ): {max_deg}")
    
    # Step 3: Find largest s with non-empty 2^s-core
    s_max = int(math.floor(math.log2(max_deg)))
    
    for s in range(s_max, -1, -1):
        threshold = 1 << s  # 2^s
        active = outcore_vertices(adj, threshold)
        
        if active:
            # Build induced subgraph
            active_list = sorted(active)
            old_to_new = {old: new for new, old in enumerate(active_list)}
            
            new_adj = []
            vertices_bits = []
            
            for old_u in active_list:
                vertices_bits.append(_int_to_bits(V[old_u], m))
                new_neighbors = [old_to_new[old_v] for old_v in adj[old_u] if old_v in active]
                new_adj.append(new_neighbors)
            
            core_max_deg = max(len(new_adj[i]) for i in range(len(new_adj))) if new_adj else 0
            
            if verbose:
                rate = s / m
                print(f"  Found s={s}-core: |V|={len(vertices_bits)}, rate={rate:.5f}")
            
            return CoreResult(
                m=m,
                s=s,
                rate=s / m,
                vertices_bits=vertices_bits,
                adj=new_adj,
                num_vertices=len(vertices_bits),
                max_outdeg=core_max_deg
            )
    
    return None


def search_best_rate(ell: int, delta: int, m_min: int, m_max: int, 
                     verbose: bool = True) -> Optional[CoreResult]:
    """
    Search for the best rate s/m over a range of m values.
    
    Args:
        ell: Window length
        delta: Allowed deviation
        m_min: Minimum block length to try
        m_max: Maximum block length to try
        verbose: Print progress
        
    Returns:
        Best CoreResult found
    """
    if verbose:
        print(f"Searching for best rate for (ℓ={ell}, δ={delta})")
        print(f"m range: [{m_min}, {m_max}]")
        print("-" * 50)
    
    best = None
    
    for m in range(m_min, m_max + 1):
        core = algorithm1_find_core(m, ell, delta, verbose=False)
        
        if core is None:
            if verbose:
                print(f"m={m:2d}: no core found")
            continue
        
        if verbose:
            print(f"m={m:2d}: s={core.s:2d}, rate={core.rate:.5f}, |V|={core.num_vertices}")
        
        if best is None or core.rate > best.rate:
            best = core
    
    if best and verbose:
        print("-" * 50)
        print(f"BEST: m={best.m}, s={best.s}, rate={best.rate:.5f}")
    
    return best


if __name__ == "__main__":
    # Test: Reproduce paper results for (ℓ=8, δ=1)
    # Expected: m=13 gives s=10, rate=10/13≈0.769
    print("Testing Algorithm 1 for (ℓ=8, δ=1):")
    print("Paper benchmark: Construction 3 achieves 10/13 = 0.76923")
    print()
    
    search_best_rate(ell=8, delta=1, m_min=7, m_max=14)
