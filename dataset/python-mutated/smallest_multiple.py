"""
Smallest multiple

2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
What is the smallest positive number that is evenly divisible by all of the numbers from A to B?

=========================================
The solution is the least common multiple for more than 2 numbers (in this case all numbers from "start" to "end")
    Time Complexity:    O(N)    , N = start - end, GCD complexity is O(Log min(a, b))
    Space Complexity:   O(1)
"""

def smallest_multiple(start, end):
    if False:
        while True:
            i = 10
    result = 1
    for k in range(start, end + 1):
        result = lcm(max(result, k), min(result, k))
    return result

def lcm(a, b):
    if False:
        while True:
            i = 10
    return a * b // gcd(a, b)

def gcd(a, b):
    if False:
        return 10
    while b != 0:
        (a, b) = (b, a % b)
    return a
print(smallest_multiple(1, 10))
print(smallest_multiple(1, 20))