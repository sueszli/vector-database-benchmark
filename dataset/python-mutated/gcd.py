"""
Functions for calculating the greatest common divisor of two integers or
their least common multiple.
"""

def gcd(a, b):
    if False:
        for i in range(10):
            print('nop')
    "Computes the greatest common divisor of integers a and b using\n    Euclid's Algorithm.\n    gcd{𝑎,𝑏}=gcd{−𝑎,𝑏}=gcd{𝑎,−𝑏}=gcd{−𝑎,−𝑏}\n    See proof: https://proofwiki.org/wiki/GCD_for_Negative_Integers\n    "
    a_int = isinstance(a, int)
    b_int = isinstance(b, int)
    a = abs(a)
    b = abs(b)
    if not (a_int or b_int):
        raise ValueError('Input arguments are not integers')
    if a == 0 or b == 0:
        raise ValueError('One or more input arguments equals zero')
    while b != 0:
        (a, b) = (b, a % b)
    return a

def lcm(a, b):
    if False:
        return 10
    'Computes the lowest common multiple of integers a and b.'
    return abs(a) * abs(b) / gcd(a, b)
'\nGiven a positive integer x, computes the number of trailing zero of x.\nExample\nInput : 34(100010)\n           ~~~~~^\nOutput : 1\n\nInput : 40(101000)\n           ~~~^^^\nOutput : 3\n'

def trailing_zero(x):
    if False:
        for i in range(10):
            print('nop')
    count = 0
    while x and (not x & 1):
        count += 1
        x >>= 1
    return count
'\nGiven two non-negative integer a and b,\ncomputes the greatest common divisor of a and b using bitwise operator.\n'

def gcd_bit(a, b):
    if False:
        for i in range(10):
            print('nop')
    ' Similar to gcd but uses bitwise operators and less error handling.'
    tza = trailing_zero(a)
    tzb = trailing_zero(b)
    a >>= tza
    b >>= tzb
    while b:
        if a < b:
            (a, b) = (b, a)
        a -= b
        a >>= trailing_zero(a)
    return a << min(tza, tzb)