"""
Function to find the primitive root of a number.
"""
import math
'\nFor positive integer n and given integer a that satisfies gcd(a, n) = 1,\nthe order of a modulo n is the smallest positive integer k that satisfies\npow (a, k) % n = 1. In other words, (a^k) ≡ 1 (mod n).\nOrder of certain number may or may not be exist. If so, return -1.\n'

def find_order(a, n):
    if False:
        i = 10
        return i + 15
    '\n    Find order for positive integer n and given integer a that satisfies gcd(a, n) = 1.\n    Time complexity O(nlog(n))\n    '
    if (a == 1) & (n == 1):
        return 1
    if math.gcd(a, n) != 1:
        print('a and n should be relative prime!')
        return -1
    for i in range(1, n):
        if pow(a, i) % n == 1:
            return i
    return -1
"\nEuler's totient function, also known as phi-function ϕ(n),\ncounts the number of integers between 1 and n inclusive,\nwhich are coprime to n.\n(Two numbers are coprime if their greatest common divisor (GCD) equals 1).\nCode from /algorithms/maths/euler_totient.py, written by 'goswami-rahul'\n"

def euler_totient(n):
    if False:
        print('Hello World!')
    "Euler's totient function or Phi function.\n    Time Complexity: O(sqrt(n))."
    result = n
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            while n % i == 0:
                n //= i
            result -= result // i
    if n > 1:
        result -= result // n
    return result
"\nFor positive integer n and given integer a that satisfies gcd(a, n) = 1,\na is the primitive root of n, if a's order k for n satisfies k = ϕ(n).\nPrimitive roots of certain number may or may not exist.\nIf so, return empty list.\n"

def find_primitive_root(n):
    if False:
        for i in range(10):
            print('nop')
    if n == 1:
        return [0]
    phi = euler_totient(n)
    p_root_list = []
    ' It will return every primitive roots of n. '
    for i in range(1, n):
        if math.gcd(i, n) == 1:
            order = find_order(i, n)
            if order == phi:
                p_root_list.append(i)
    return p_root_list