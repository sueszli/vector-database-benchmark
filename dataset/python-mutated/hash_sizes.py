"""
Compute hashtable sizes with nices properties
- prime sizes (for small to medium sizes)
- 2 prime-factor sizes (for big sizes)
- fast growth for small sizes
- slow growth for big sizes

Note:
     this is just a tool for developers.
     within borgbackup, it is just used to generate hash_sizes definition for _hashindex.c.
"""
from collections import namedtuple
(K, M, G) = (2 ** 10, 2 ** 20, 2 ** 30)
(start, end_p1, end_p2) = (1 * K, 127 * M, 2 * G - 10 * M)
Policy = namedtuple('Policy', 'upto grow')
policies = [Policy(256 * K, 2.0), Policy(2 * M, 1.7), Policy(16 * M, 1.4), Policy(128 * M, 1.2), Policy(2 * G - 1, 1.1)]

def eratosthenes():
    if False:
        while True:
            i = 10
    'Yields the sequence of prime numbers via the Sieve of Eratosthenes.'
    D = {}
    q = 2
    while True:
        p = D.pop(q, None)
        if p is None:
            yield q
            D[q * q] = q
        else:
            x = p + q
            while x in D:
                x += p
            D[x] = p
        q += 1

def two_prime_factors(pfix=65537):
    if False:
        i = 10
        return i + 15
    'Yields numbers with 2 prime factors pfix and p.'
    for p in eratosthenes():
        yield (pfix * p)

def get_grow_factor(size):
    if False:
        print('Hello World!')
    for p in policies:
        if size < p.upto:
            return p.grow

def find_bigger_prime(gen, i):
    if False:
        i = 10
        return i + 15
    while True:
        p = next(gen)
        if p >= i:
            return p

def main():
    if False:
        return 10
    sizes = []
    i = start
    gen = eratosthenes()
    while i < end_p1:
        grow_factor = get_grow_factor(i)
        p = find_bigger_prime(gen, i)
        sizes.append(p)
        i = int(i * grow_factor)
    gen = two_prime_factors()
    while i < end_p2:
        grow_factor = get_grow_factor(i)
        p = find_bigger_prime(gen, i)
        sizes.append(p)
        i = int(i * grow_factor)
    print('static int hash_sizes[] = {\n    %s\n};\n' % ', '.join((str(size) for size in sizes)))
if __name__ == '__main__':
    main()