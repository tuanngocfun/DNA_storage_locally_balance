"""
DNA-Based Storage Code Definition Checkers
Implements all definitions from the paper for verifying code properties
Compatible with Python 2.7+ and Python 3.x
"""

class DNAStorageCodeChecker:
    """Checker for DNA-based storage code definitions"""
    
    @staticmethod
    def hamming_weight(x):
        """
        Calculate Hamming weight (number of ones in binary sequence)
        
        Args:
            x: Binary string
        Returns:
            Number of ones in x
        """
        return x.count('1')
    
    @staticmethod
    def consecutive_subword(x, i, l):
        """
        Extract consecutive subword x[i:i+l]
        
        Args:
            x: Binary string
            i: Starting position (0-indexed)
            l: Length of subword
        Returns:
            Substring of length l starting at position i
        """
        if i + l > len(x):
            raise ValueError("Subword extends beyond string length: {0}+{1} > {2}".format(i, l, len(x)))
        return x[i:i+l]
    
    @staticmethod
    def is_locally_balanced(x, l, delta):
        """
        Definition 1: Check if word is (l,delta)-locally balanced
        
        A binary word is (l,delta)-locally balanced if the weight of every 
        consecutive substring of length l stays within [l/2-delta, l/2+delta]
        
        Args:
            x: Binary string
            l: Window length
            delta: Balance tolerance
        Returns:
            (is_balanced, violations) tuple
        """
        n = len(x)
        violations = []
        lower_bound = l // 2 - delta
        upper_bound = l // 2 + delta
        
        for i in range(n - l + 1):
            subword = x[i:i+l]
            weight = DNAStorageCodeChecker.hamming_weight(subword)
            
            if not (lower_bound <= weight <= upper_bound):
                violations.append(
                    "Position {0}: weight={1}, subword='{2}', bounds=[{3},{4}]".format(
                        i, weight, subword, lower_bound, upper_bound
                    )
                )
        
        return len(violations) == 0, violations
    
    @staticmethod
    def is_strongly_locally_balanced(x, l, delta):
        """
        Definition 2: Check if word is strongly (l,delta)-locally balanced
        
        Satisfies (l',delta)-locally balanced for all even l' >= l
        
        Args:
            x: Binary string
            l: Minimum window length (must be even)
            delta: Balance tolerance
        Returns:
            (is_balanced, results_dict) tuple
        """
        if l % 2 != 0:
            raise ValueError("l must be even, got {0}".format(l))
        
        n = len(x)
        results = {}
        
        # Check for all even l' from l up to n
        for l_prime in range(l, n + 1, 2):
            is_bal, viols = DNAStorageCodeChecker.is_locally_balanced(x, l_prime, delta)
            results[l_prime] = {
                'balanced': is_bal,
                'violations': viols
            }
            
            if not is_bal:
                return False, results
        
        return True, results
    
    @staticmethod
    def running_digital_sum(x):
        """
        Definition 4: Calculate Running Digital Sum (RDS)
        
        RDS sequence where s_0 = 0 and s_i = (# of 1s) - (# of 0s) in first i bits
        
        Args:
            x: Binary string
        Returns:
            List of RDS values [s_0, s_1, ..., s_n]
        """
        rds = [0]
        cumsum = 0
        
        for bit in x:
            if bit == '1':
                cumsum += 1
            elif bit == '0':
                cumsum -= 1
            else:
                raise ValueError("Invalid bit: {0}".format(bit))
            rds.append(cumsum)
        
        return rds
    
    @staticmethod
    def distance(x):
        """
        Calculate distance (gap between max and min RDS values)
        
        Args:
            x: Binary string
        Returns:
            dis(x) = max(RDS) - min(RDS)
        """
        rds = DNAStorageCodeChecker.running_digital_sum(x)
        return max(rds) - min(rds)
    
    @staticmethod
    def is_delta_rds_word(x, delta):
        """
        Definition 5: Check if word is delta-RDS word
        
        A word is delta-RDS if dis(x) <= delta
        
        Args:
            x: Binary string
            delta: Maximum allowed distance
        Returns:
            (is_valid, info_dict) tuple
        """
        rds = DNAStorageCodeChecker.running_digital_sum(x)
        dist = max(rds) - min(rds)
        
        info = {
            'rds': rds,
            'distance': dist,
            'max_rds': max(rds),
            'min_rds': min(rds),
            'threshold': delta
        }
        
        return dist <= delta, info
    
    @staticmethod
    def check_weight_restriction(x, s, t):
        """
        Check if first s bits have weight t (for X_n(l,delta;s,t) sets)
        
        Args:
            x: Binary string
            s: Prefix length
            t: Required weight of prefix
        Returns:
            True if wt(x[0:s]) == t
        """
        if s > len(x):
            raise ValueError("Prefix length {0} exceeds string length {1}".format(s, len(x)))
        
        prefix = x[:s]
        return DNAStorageCodeChecker.hamming_weight(prefix) == t
    
    @staticmethod
    def verify_all_properties(x, l=6, delta=1):
        """
        Comprehensive verification of all code properties
        
        Args:
            x: Binary string to check
            l: Window length for local balance (default 6)
            delta: Tolerance parameter (default 1)
        Returns:
            Dictionary with all verification results
        """
        results = {
            'word': x,
            'length': len(x),
            'hamming_weight': DNAStorageCodeChecker.hamming_weight(x),
            'parameters': {'l': l, 'delta': delta}
        }
        
        # Check locally balanced
        is_lb, lb_viols = DNAStorageCodeChecker.is_locally_balanced(x, l, delta)
        results['locally_balanced'] = {
            'valid': is_lb,
            'violations': lb_viols
        }
        
        # Check strongly locally balanced (only if l is even and length permits)
        if l % 2 == 0:
            is_slb, slb_results = DNAStorageCodeChecker.is_strongly_locally_balanced(x, l, delta)
            results['strongly_locally_balanced'] = {
                'valid': is_slb,
                'details': slb_results
            }
        
        # Check RDS properties
        is_rds, rds_info = DNAStorageCodeChecker.is_delta_rds_word(x, delta)
        results['delta_rds_word'] = {
            'valid': is_rds,
            'info': rds_info
        }
        
        return results


# Example usage and test cases
if __name__ == "__main__":
    checker = DNAStorageCodeChecker()
    
    # Test cases
    test_words = [
        "101010",      # Balanced
        "111000",      # Unbalanced
        "10101010",    # Long balanced
        "11110000",    # Long unbalanced
        "0110"         # Short word
    ]
    
    print("DNA Storage Code Verification Tests")
    print("=" * 60)
    
    for word in test_words:
        print("\nWord: {0}".format(word))
        print("-" * 60)
        
        results = checker.verify_all_properties(word, l=6, delta=1)
        
        print("Length: {0}".format(results['length']))
        print("Hamming Weight: {0}".format(results['hamming_weight']))
        print("Locally Balanced (6,1): {0}".format(results['locally_balanced']['valid']))
        
        if not results['locally_balanced']['valid']:
            print("  Violations: {0}".format(len(results['locally_balanced']['violations'])))
            for v in results['locally_balanced']['violations'][:3]:
                print("    {0}".format(v))
        
        print("delta-RDS Word (delta=1): {0}".format(results['delta_rds_word']['valid']))
        print("  Distance: {0}".format(results['delta_rds_word']['info']['distance']))
        print("  RDS: {0}".format(results['delta_rds_word']['info']['rds']))
    
    # Detailed example
    print("\n" + "=" * 60)
    print("Detailed Analysis Example")
    print("=" * 60)
    
    example = "01101001"
    print("\nAnalyzing: {0}".format(example))
    
    print("\nHamming Weight: {0}".format(checker.hamming_weight(example)))
    print("Running Digital Sum: {0}".format(checker.running_digital_sum(example)))
    print("Distance: {0}".format(checker.distance(example)))
    
    is_lb, viols = checker.is_locally_balanced(example, 4, 1)
    print("\n(4,1)-Locally Balanced: {0}".format(is_lb))
    if not is_lb:
        for v in viols:
            print("  {0}".format(v))
    
    is_rds, info = checker.is_delta_rds_word(example, 2)
    print("\ndelta-RDS (delta=2): {0}".format(is_rds))
    print("  Max RDS: {0}, Min RDS: {1}".format(info['max_rds'], info['min_rds']))