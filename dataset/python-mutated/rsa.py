"""
RSA encryption algorithm
a method for encrypting a number that uses seperate encryption and decryption keys
this file only implements the key generation algorithm

there are three important numbers in RSA called n, e, and d
e is called the encryption exponent
d is called the decryption exponent
n is called the modulus

these three numbers satisfy
((x ** e) ** d) % n == x % n

to use this system for encryption, n and e are made publicly available, and d is kept secret
a number x can be encrypted by computing (x ** e) % n
the original number can then be recovered by computing (E ** d) % n, where E is
the encrypted number

fortunately, python provides a three argument version of pow() that can compute powers modulo
a number very quickly:
(a ** b) % c == pow(a,b,c)
"""
import random

def generate_key(k, seed=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    the RSA key generating algorithm\n    k is the number of bits in n\n    '

    def modinv(a, m):
        if False:
            print('Hello World!')
        'calculate the inverse of a mod m\n        that is, find b such that (a * b) % m == 1'
        b = 1
        while not a * b % m == 1:
            b += 1
        return b

    def gen_prime(k, seed=None):
        if False:
            while True:
                i = 10
        'generate a prime with k bits'

        def is_prime(num):
            if False:
                i = 10
                return i + 15
            if num == 2:
                return True
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        random.seed(seed)
        while True:
            key = random.randrange(int(2 ** (k - 1)), int(2 ** k))
            if is_prime(key):
                return key
    p_size = k / 2
    q_size = k - p_size
    e = gen_prime(k, seed)
    while True:
        p = gen_prime(p_size, seed)
        if p % e != 1:
            break
    while True:
        q = gen_prime(q_size, seed)
        if q % e != 1:
            break
    n = p * q
    l = (p - 1) * (q - 1)
    d = modinv(e, l)
    return (int(n), int(e), int(d))

def encrypt(data, e, n):
    if False:
        while True:
            i = 10
    return pow(int(data), int(e), int(n))

def decrypt(data, d, n):
    if False:
        while True:
            i = 10
    return pow(int(data), int(d), int(n))