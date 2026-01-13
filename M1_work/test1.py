
from typing import List
from generate_constant_weight_codewords import generate_constant_weight_codewords
from lb_check import is_locally_balanced
from minimum_hamming_distance_check import (
    report,
    min_hamming_distance,
    minimal_removal_via_pairs,  # <-- randomized maximum clique helper (ensure it's implemented)
    codebook_satisfies_distance,
)
from rate_calculate import calculate_rate

def _prompt_int(prompt: str, validator, error_msg: str) -> int:
    """Prompt repeatedly until a valid integer satisfying validator(x) is given."""
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print("Error: please enter an integer.")
            continue
        if validator(val):
            return val
        print(error_msg)

def step3_distance_and_optional_removal(lb_codewords: List[str], n: int, d_min: int) -> List[str]:
    """
    Step 3 logic encapsulated to avoid scope issues.
    - Shows report.
    - If requirement passes, skip removal prompt and show rate.
    - If requirement fails, optionally remove minimum codewords via maximum clique (randomized).
    Returns: final codebook (kept set).
    """
    # Always show the LB set
    print("-- Final codewords to be tested (locally-balanced set of codewords) --")
    if len(lb_codewords) == 0:
        print("(none)")
    else:
        for i, cw in enumerate(lb_codewords, 1):
            print(f"{i}: {cw}")

    # Show offending pairs & frequency summary
    report(lb_codewords, d_min)

    # Current min distance
    md, _ = min_hamming_distance(lb_codewords)

    # If already satisfies d_min OR empty, skip removal prompt
    if md >= d_min or len(lb_codewords) == 0:
        if len(lb_codewords) == 0:
            print("\nNo locally balanced codewords to evaluate.")
            return lb_codewords[:]  # nothing to do
        M = len(lb_codewords)
        rate = calculate_rate(n, M)
        print(f"Rate = log2(M)/n with M={M}, n={n}: {rate:.6f}")
        return lb_codewords[:]  # keep as is

    # Otherwise, prompt for removal to meet d_min
    do_fix = input(
        "Do you want to remove the MINIMUM number of codewords to meet the required distance? [y/N]: "
    ).strip().lower()

    if do_fix not in ("y", "yes"):
        print("\nNOTE: Codebook does not meet the required distance, and you chose not to remove codewords.")
        M = len(lb_codewords)
        rate = calculate_rate(n, M)
        print(f"Current (failing) rate with M={M}, n={n}: {rate:.6f}")
        return lb_codewords[:]  # unchanged

    # Randomization controls
    use_random = input("Randomize selection among equally-good solutions? [y/N]: ").strip().lower() in ("y", "yes")
    raw_seed = input("Seed (press Enter for none/true random): ").strip()
    seed = None if raw_seed == "" else int(raw_seed)

    # Compute minimal removal via pairwise distances + maximum clique on accepted graph
    keep_idx, remove_idx, acc_pairs, rej_pairs = minimal_removal_via_pairs(
        lb_codewords, d_min, randomize=use_random, seed=seed
    )

    # Build final kept set
    if len(remove_idx) == 0:
        print("\nAlready satisfies the requirement; nothing to remove.")
        final_codebook = lb_codewords[:]  # copy
    else:
        print("\n-- Codewords to be REMOVED (minimal set) --")
        for idx in remove_idx:
            print(f"[{idx+1}] {lb_codewords[idx]}")  # 1-based display

        final_codebook = [lb_codewords[i] for i in keep_idx]
        print("\n-- Codewords KEPT (maximum clique) --")
        for i, cw in enumerate(final_codebook, 1):
            print(f"{i}: {cw}")

        print("\n-- Accepted pairs among KEPT (distance >= d_min) --")
        kept_set = set(keep_idx)
        for i, j, d in acc_pairs:
            if i in kept_set and j in kept_set:
                print(f"[{i+1}] {lb_codewords[i]}  <->  [{j+1}] {lb_codewords[j]}  | d={d}")

        print("\n-- Rejected pairs (distance < d_min) excluding kept-kept pairs --")
        for i, j, d in rej_pairs:
            if not (i in kept_set and j in kept_set):
                print(f"[{i+1}] {lb_codewords[i]}  <->  [{j+1}] {lb_codewords[j]}  | d={d}")

    # Recompute stats & rate on the final set
    ok = codebook_satisfies_distance(final_codebook, d_min)
    md2, _ = min_hamming_distance(final_codebook)
    print(f"\nAfter removal: min distance = {md2} | satisfies requirement: {ok}")

    if len(final_codebook) > 0:
        M2 = len(final_codebook)
        rate2 = calculate_rate(n, M2)
        print(f"New rate with M={M2}, n={n}: {rate2:.6f}")

    return final_codebook

def main() -> None:
    print("=== Step 1: Generate constant-weight codebook ===")
    n = _prompt_int(
        "Enter an even integer n (length of codewords): ",
        lambda x: x > 0 and x % 2 == 0,
        "n must be a positive even integer.",
    )

    # Generate all codewords of length n with weight n//2
    codebook: List[str] = generate_constant_weight_codewords(n)
    print(f"Generated {len(codebook)} codewords of length {n} with weight {n//2}.")
    print("-- All codewords --")
    for i, cw in enumerate(codebook, 1):
        print(f"{i}: {cw}")

    print("\n=== Step 2: Local balance check ===")
    l = _prompt_int(
        "Enter window size l (must be even): ",
        lambda x: x > 0 and x % 2 == 0,
        "l must be a positive even integer.",
    )
    delta = _prompt_int(
        "Enter delta (>= 0): ",
        lambda x: x >= 0,
        "delta must be a non-negative integer.",
    )

    lb_codewords: List[str] = []
    not_lb_codewords: List[str] = []
    for cw in codebook:
        ok = is_locally_balanced(cw, l=l, delta=delta)
        (lb_codewords if ok else not_lb_codewords).append(cw)

    print(
        f"Local balance summary: PASS={len(lb_codewords)}, FAIL={len(not_lb_codewords)} (total={len(codebook)})"
    )
    if len(not_lb_codewords) > 0:
        show_non = input("Do you want to list NOT locally balanced codewords? [y/N]: ").strip().lower()
        if show_non in ("y", "yes"):
            print("-- Not locally balanced codewords --")
            for i, cw in enumerate(not_lb_codewords, 1):
                print(f"{i}: {cw}")
    else:
        print("No codewords failed the local balance check.")

    print("-- Locally balanced codewords --")
    for i, cw in enumerate(lb_codewords, 1):
        print(f"{i}: {cw}")

    print("\n=== Step 3: Minimum Hamming distance check ===")
    d_min = _prompt_int(
        "Enter required minimum Hamming distance d_min (>= 0): ",
        lambda x: x >= 0,
        "d_min must be a non-negative integer.",
    )

    # Run the step 3 logic (encapsulated) so variables are passed explicitly and scoped correctly
    _ = step3_distance_and_optional_removal(lb_codewords, n, d_min)

if __name__ == "__main__":
    main()