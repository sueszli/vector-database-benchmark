"""
Return list of all primes less than n,
Using sieve of Eratosthenes.

Modification:
We don't need to check all even numbers, we can make the sieve excluding even
numbers and adding 2 to the primes list by default.

We are going to make an array of: x / 2 - 1 if number is even, else x / 2
(The -1 with even number it's to exclude the number itself)
Because we just need numbers [from 3..x if x is odd]

# We can get value represented at index i with (i*2 + 3)

For example, for x = 10, we start with an array of x / 2 - 1 = 4
[1, 1, 1, 1]
 3  5  7  9

For x = 11:
[1, 1, 1, 1, 1]
 3  5  7  9  11  # 11 is odd, it's included in the list

With this, we have reduced the array size to a half,
and complexity it's also a half now.
"""

def get_primes(n):
    if False:
        return 10
    'Return list of all primes less than n,\n    Using sieve of Eratosthenes.\n    '
    if n <= 0:
        raise ValueError("'n' must be a positive integer.")
    sieve_size = n // 2 - 1 if n % 2 == 0 else n // 2
    sieve = [True for _ in range(sieve_size)]
    primes = []
    if n >= 2:
        primes.append(2)
    for i in range(sieve_size):
        if sieve[i]:
            value_at_i = i * 2 + 3
            primes.append(value_at_i)
            for j in range(i, sieve_size, value_at_i):
                sieve[j] = False
    return primes