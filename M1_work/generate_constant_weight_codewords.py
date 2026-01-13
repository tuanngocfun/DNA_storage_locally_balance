
import itertools

def generate_constant_weight_codewords(n: int):

    # Generate all binary strings of length n with Hamming weight = n/2.
    # Assumes n is even.
    # Returns a list of strings like '010101...'.

    if n % 2 != 0:
        raise ValueError("n must be even for weight = n/2.")
    
    weight = n // 2
    codewords = []
    
    for positions in itertools.combinations(range(n), weight):
        bits = ['0'] * n
        for pos in positions:
            bits[pos] = '1'
        codewords.append(''.join(bits))
    
    return codewords

# if __name__ == "__main__":
#     try:
#         n = int(input("Enter an even integer n (length of codewords): "))
#         if n % 2 != 0:
#             print("Error: n must be even.")
#         else:
#             codewords = generate_constant_weight_codewords(n)
#             print(f"Total codewords for n={n}, weight={n//2} which maximum numbers of codeword is {len(codewords)}")
            
#             # Ask user how many examples to show
#             try:
#                 k = int(input(f"How many codewords do you want to display? (max {len(codewords)}): "))
#                 if k < 1 or k > len(codewords):
#                     print(f"Invalid number. Showing all {len(codewords)} codewords.")
#                     k = len(codewords)
#             except ValueError:
#                 print("Invalid input. Showing all codewords.")
#                 k = len(codewords)
            
#             # Display requested number
#             print(f"Displaying {k} codewords:")
#             for cw in codewords[:k]:
#                 print(cw)
#     except ValueError:
#         print("Invalid input. Please enter an integer.")