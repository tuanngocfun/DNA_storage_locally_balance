
import math

def calculate_rate(n, M):
    if n <= 0 or M <= 0:
        raise ValueError("Both n and M must be positive integers.")
    
    # Rate formula: log2(M) / n
    rate = math.log2(M) / n
    return rate

# # Example usage:
# n = 10      # codeword length
# M = 128     # number of codewords
# print(f"Rate: {calculate_rate(n, M):.4f}")
