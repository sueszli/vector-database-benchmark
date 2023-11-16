"""
Elias γ code or Elias gamma code is a universal code
encoding positive integers.
It is used most commonly when coding integers whose
upper-bound cannot be determined beforehand.
Elias δ code or Elias delta code is a universal code
 encoding the positive integers,
that includes Elias γ code when calculating.
Both were developed by Peter Elias.

"""
from math import log
log2 = lambda x: log(x, 2)

def binary(x, l=1):
    if False:
        print('Hello World!')
    fmt = '{0:0%db}' % l
    return fmt.format(x)

def unary(x):
    if False:
        return 10
    return (x - 1) * '1' + '0'

def elias_generic(lencoding, x):
    if False:
        print('Hello World!')
    '\n\tThe compressed data is calculated in two parts.\n\tThe first part is the unary number of 1 + ⌊log2(x)⌋.\n\tThe second part is the binary number of x - 2^(⌊log2(x)⌋).\n\tFor the final result we add these two parts.\n\t'
    if x == 0:
        return '0'
    first_part = 1 + int(log2(x))
    a = x - 2 ** int(log2(x))
    k = int(log2(x))
    return lencoding(first_part) + binary(a, k)

def elias_gamma(x):
    if False:
        print('Hello World!')
    '\n\tFor the first part we put the unary number of x.\n\t'
    return elias_generic(unary, x)

def elias_delta(x):
    if False:
        while True:
            i = 10
    '\n\tFor the first part we put the elias_g of the number.\n\t'
    return elias_generic(elias_gamma, x)