def extended_gcd(a: int, b: int) -> [int, int, int]:
    if False:
        print('Hello World!')
    'Extended GCD algorithm.\n    Return s, t, g\n    such that a * s + b * t = GCD(a, b)\n    and s and t are co-prime.\n    '
    (old_s, s) = (1, 0)
    (old_t, t) = (0, 1)
    (old_r, r) = (a, b)
    while r != 0:
        quotient = old_r // r
        (old_r, r) = (r, old_r - quotient * r)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)
    return (old_s, old_t, old_r)

def modular_inverse(a: int, m: int) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Returns x such that a * x = 1 (mod m)\n    a and m must be coprime\n    '
    (s, _, g) = extended_gcd(a, m)
    if g != 1:
        raise ValueError('a and m must be coprime')
    return s % m