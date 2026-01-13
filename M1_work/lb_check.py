

def is_locally_balanced(x: str, l: int, delta: int) -> bool:

    # Check the (l, delta)-locally balanced constraint for a binary string x.
    # Requires every consecutive substring of length l to have weight in
    # [l/2 - delta, l/2 + delta]. (l must be even)
 
    if l % 2 != 0:
        raise ValueError("Window length l must be even.")
    if any(c not in '01' for c in x):
        return False

    n = len(x)
    low = l // 2 - delta
    high = l // 2 + delta

    if n < l:
        wt = x.count('1')
        return low <= wt <= high

    # Prefix sums for O(1) window weights
    pref = [0] * (n + 1)
    for i, ch in enumerate(x, 1):
        pref[i] = pref[i - 1] + (1 if ch == '1' else 0)

    for i in range(0, n - l + 1):
        wt = pref[i + l] - pref[i]
        if not (low <= wt <= high):
            return False
    return True

# Manual input
# if __name__ == "__main__":
#     x = input("Enter the binary string: ").strip()
#     l = int(input("Enter window size l (must be even): ").strip())
#     delta = int(input("Enter delta: ").strip())

#     result = is_locally_balanced(x, l=l, delta=delta)
#     print(f"Result for string '{x}' with l={l}, delta={delta}: {result}")

# # Test the function
# # "parameters": { "l": 4, "delta": 1 }
# print(is_locally_balanced("01010", 4, 1))  # Expected True
# print(is_locally_balanced("00110", 4, 1))  # Expected True
# print(is_locally_balanced("00000", 4, 1))  # Expected False
# print(is_locally_balanced("11111", 4, 1))  # Expected False
# print(is_locally_balanced("10001", 4, 1))  # Expected True

# print(is_locally_balanced("01001", 4, 1))  # Expected True
# print(is_locally_balanced("01000", 4, 1))  # Expected True
# print(is_locally_balanced("00101", 4, 1))  # Expected True
# print(is_locally_balanced("00011", 4, 1))  # Expected True

# print(is_locally_balanced("010001", 4, 1))  # Expected True
# print(is_locally_balanced("0100001", 4, 1))  # Expected False

# print(is_locally_balanced("11100101000", 4, 1))  # Expected True
# print(is_locally_balanced("000110111101001", 4, 1))  # Expected False
# print(is_locally_balanced("00001100001011", 4, 1))  # Expected False
# print(is_locally_balanced("01010010101001110101", 4, 1))  # Expected True
# print(is_locally_balanced("011100011000011111", 4, 1))  # Expected False
# print(is_locally_balanced("10101010001010", 4, 1))  # Expected True
# print(is_locally_balanced("11010010100", 4, 1))  # Expected True
# print(is_locally_balanced("00100110110100010100", 4, 1))  # Expected True

# #"parameters": { "l": 8, "delta": 1 }
# print(is_locally_balanced("001000101011011110", 8, 1))  # Expected False
# print(is_locally_balanced("10001110010011", 8, 1))  # Expected True
# print(is_locally_balanced("110011000110", 8, 1))  # Expected True
# print(is_locally_balanced("1010011011000011011", 8, 1))  # Expected True
# print(is_locally_balanced("1111000111100000001", 8, 1))  # Expected False
# print(is_locally_balanced("001001011001010", 8, 1))  # Expected True






