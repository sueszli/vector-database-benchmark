def modular_exponential(base, exponent, mod):
    if False:
        return 10
    'Computes (base ^ exponent) % mod.\n    Time complexity - O(log n)\n    Use similar to Python in-built function pow.'
    if exponent < 0:
        raise ValueError('Exponent must be positive.')
    base %= mod
    result = 1
    while exponent > 0:
        if exponent & 1:
            result = result * base % mod
        exponent = exponent >> 1
        base = base * base % mod
    return result