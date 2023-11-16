import cython
from cython.cimports.libcpp.vector import vector

def primes(nb_primes: cython.uint):
    if False:
        print('Hello World!')
    i: cython.int
    p: vector[cython.int]
    p.reserve(nb_primes)
    n: cython.int = 2
    while p.size() < nb_primes:
        for i in p:
            if n % i == 0:
                break
        else:
            p.push_back(n)
        n += 1
    return p