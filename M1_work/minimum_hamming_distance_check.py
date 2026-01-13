
from typing import List, Tuple, Dict

def hamming_distance(a: str, b: str) -> int:

    # Compute Hamming distance between two equal-length binary strings.

    if len(a) != len(b):
        raise ValueError("Codewords must have the same length.")
    return sum(1 for x, y in zip(a, b) if x != y)

def min_hamming_distance(codebook: List[str]) -> Tuple[int, Tuple[int, int]]:

    # Compute the minimum pairwise Hamming distance in the codebook.
    # Returns:
    #     (min_distance, (i, j)) where i, j are indices of the pair achieving min distance.
    #     If codebook has < 2 codewords, returns (0, (-1, -1)).

    n = len(codebook)
    if n < 2:
        return 0, (-1, -1)
    min_d = None
    min_pair = (-1, -1)
    # Check consistent lengths
    lengths = {len(cw) for cw in codebook}
    if len(lengths) != 1:
        raise ValueError("All codewords must have the same length.")
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(codebook[i], codebook[j])
            if min_d is None or d < min_d:
                min_d = d
                min_pair = (i, j)
                if min_d == 0:
                    # can't be smaller than 0, early exit
                    return 0, (i, j)
    return min_d if min_d is not None else 0, min_pair

def codebook_satisfies_distance(codebook: List[str], d_min: int) -> bool:

    # Returns True if every pair of distinct codewords has Hamming distance >= d_min.

    md, _ = min_hamming_distance(codebook)
    return md >= d_min


def report(codebook: List[str], d_min: int) -> None:
    n = len(codebook)
    print(f"Target minimum distance: {d_min}")

    if n < 2:
        print("Actual minimum distance in codebook: 0")
        print("WARNING: Codebook has fewer than 2 codewords; minimum distance is undefined.")
        return

    # Verify consistent lengths
    lengths = {len(cw) for cw in codebook}
    if len(lengths) != 1:
        raise ValueError("All codewords must have the same length.")

    # Compute actual min distance
    md, _ = min_hamming_distance(codebook)
    print(f"Actual minimum distance in codebook: {md}")

    # Gather all offending pairs (distance < d_min)
    offending: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(codebook[i], codebook[j])
            if d < d_min:
                offending.append((i, j, d))

    if len(offending) == 0:
        print("PASS: The codebook satisfies the minimum Hamming distance requirement.")
    else:
        print("FAIL: The codebook does NOT satisfy the requirement.")
        print("-- All offending pairs (codewords) --")
        for i, j, d in offending:
            print(f"{codebook[i]} - {codebook[j]} - Hamming distance: {d}")

        
 # Frequency of appearances per offending codeword
        freq: Dict[int, int] = {}
        for i, j, _ in offending:
            freq[i] = freq.get(i, 0) + 1
            freq[j] = freq.get(j, 0) + 1

        print("\n-- Offending codeword frequency (number of pairs each appears in) --")
        # Sort by descending frequency, then by index for stability
        for idx, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"[{idx}] {codebook[idx]}  |  appears in {count} offending pair(s)")


# ===== Randomized maximum clique picker on the accepted-pairs graph =====
from typing import List, Tuple, Set, Dict, Optional
import random
from collections import defaultdict

def pairwise_distances(codebook: List[str]) -> List[Tuple[int, int, int]]:
    """
    Return a list of (i, j, d) for all i < j, where d is Hamming distance(codebook[i], codebook[j]).
    """
    n = len(codebook)
    lengths = {len(cw) for cw in codebook}
    if len(lengths) != 1:
        raise ValueError("All codewords must have the same length.")

    res: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(codebook[i], codebook[j])
            res.append((i, j, d))
    return res

def partition_pairs_by_threshold(pairs: List[Tuple[int, int, int]], d_min: int):
    """
    Split pairs into accepted (d >= d_min) and rejected (d < d_min).
    """
    accepted = [(i, j, d) for (i, j, d) in pairs if d >= d_min]
    rejected = [(i, j, d) for (i, j, d) in pairs if d < d_min]
    return accepted, rejected

def build_adj_from_pairs(n: int, pairs: List[Tuple[int, int, int]]) -> Dict[int, Set[int]]:
    """
    Build adjacency dict from list of (i, j, d) pairs (we ignore d after filtering).
    """
    adj: Dict[int, Set[int]] = {k: set() for k in range(n)}
    for i, j, _ in pairs:
        adj[i].add(j)
        adj[j].add(i)
    return adj

def maximum_clique_random(adj: Dict[int, Set[int]], seed: Optional[int] = None) -> Set[int]:
    """
    Enumerate all maximum cliques via Bron–Kerbosch (pivoting), then
    return one **random** maximum clique using the provided seed.
    """
    rng = random.Random(seed)
    V = set(adj.keys())
    best_size = 0
    best_cliques: List[Set[int]] = []

    def bron_kerbosch(R: Set[int], P: Set[int], X: Set[int]):
        nonlocal best_size, best_cliques
        if not P and not X:
            s = len(R)
            if s > best_size:
                best_size = s
                best_cliques = [set(R)]
            elif s == best_size:
                best_cliques.append(set(R))
            return
        # Randomized pivot selection from P ∪ X
        union = list(P | X)
        if union:
            # Prefer a high-degree pivot but randomize among equals
            degrees = [(v, len(adj[v] & P)) for v in union]
            max_deg = max(d for _, d in degrees)
            candidates = [v for v, d in degrees if d == max_deg]
            u = rng.choice(candidates)
            # Randomize iteration order of candidates P \ N(u)
            iter_candidates = list(P - (adj[u] & P))
            rng.shuffle(iter_candidates)
        else:
            iter_candidates = list(P)
            rng.shuffle(iter_candidates)

        for v in iter_candidates:
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}
            X = X | {v}

    # Randomize initial order of P for variety
    P0 = set(V)
    X0 = set()
    bron_kerbosch(set(), P0, X0)

    if not best_cliques:
        return set()
    return rng.choice(best_cliques)

def minimal_removal_via_pairs(codebook: List[str], d_min: int,
                              randomize: bool = True,
                              seed: Optional[int] = None):

    # Using the pairwise distance array:
    #   - Partition pairs into accepted/rejected by threshold.
    #   - Build the accepted-pairs graph and find a maximum clique.
    #   - If randomize=True, pick a random one among all maximum cliques (seeded).
    # Returns: (keep_indices_sorted, remove_indices_sorted, accepted_pairs, rejected_pairs).

    pairs = pairwise_distances(codebook)
    accepted, rejected = partition_pairs_by_threshold(pairs, d_min)
    n = len(codebook)
    acc_adj = build_adj_from_pairs(n, accepted)

    # Edge case: no accepted edges at all -> keep any single vertex (if exists)
    if all(len(neigh) == 0 for neigh in acc_adj.values()):
        keep = [0] if n > 0 else []
        remove = sorted(set(range(n)) - set(keep))
        return keep, remove, accepted, rejected

    keep_set = (maximum_clique_random(acc_adj, seed) if randomize
                else maximum_clique_random(acc_adj, seed=0))  # deterministic with seed=0
    keep = sorted(keep_set)
    remove = sorted(set(range(n)) - keep_set)
    return keep, remove, accepted, rejected

# if __name__ == "__main__":
#     # --- Example usage ---
#     # You can replace this list with your own codebook, e.g., read from a file.
#     codebook = [
#         "111000",
#         "110100",
#         "101010",
#         "011001",
#         "001111",
#         "000111",
#     ]

#     try:
#         d_min = int(input("Enter the required minimum Hamming distance (e.g., 3): ").strip())
#     except ValueError:
#         print("Invalid input. Using default d_min = 3.")
#         d_min = 3

#     report(codebook, d_min)
#     # Or programmatically:
#     # ok = codebook_satisfies_distance(codebook, d_min)
#     # print("OK:", ok)
