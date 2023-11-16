import numpy
from .constellation_map_generator import constellation_map_generator
"\nNote on the naming scheme. Each constellation is named using a prefix\nfor the type of constellation, the order of the constellation, and a\ndistinguishing feature, which comes in three modes:\n\n- No extra feature: the basic Gray-coded constellation map; others\n  will be derived from this type.\n- A single number: an indexed number to uniquely identify different\n  constellation maps.\n- 0xN_x0_x1..._xM: A permutation of the base constellation, explained\n  below.\n\nFor rectangular constellations (BPSK, QPSK, QAM), we can define a\nhyperspace and look for all symmetries. This is also known as the\nautomorphism group of the hypercube, aka the hyperoctahedral\ngroup. What this means is that we can easily define all possible\nrotations in terms of the first base mapping by creating the mapping:\n\n  f(x) = k XOR pi(x)\n\nThe x is the bit string for the symbol we are altering. Then k is a\nbit string of n bits where n is the number of bits per symbol in the\nconstellation (e.g., 2 for QPSK or 6 for QAM64). The pi is a\npermutation function specified as pi_0, pi_1..., pi_n-1. This permutes\nthe bits from the base constellation symbol to a new code, which is\nthen xor'd by k.\n\nThe value of k is from 0 to 2^n-1 and pi is a list of all bit\npositions.\n\nThe total number of Gray coded modulations is (2^n)*(n!).\n\nWe create aliases for all possible naming schemes for the\nconstellations. So if a hyperoctahedral group is defined, we also set\nthis function equal to a function name using a unique ID number, and\nwe always select one rotation as our basic rotation that the other\nrotations are based off of.\n"

def psk_2_0x0():
    if False:
        print('Hello World!')
    '\n    0 | 1\n    '
    const_points = [-1, 1]
    symbols = [0, 1]
    return (const_points, symbols)
psk_2 = psk_2_0x0
psk_2_0 = psk_2

def psk_2_0x1():
    if False:
        return 10
    '\n    1 | 0\n    '
    const_points = [-1, 1]
    symbols = [1, 0]
    return (const_points, symbols)
psk_2_1 = psk_2_0x1

def sd_psk_2_0x0(x, Es=1):
    if False:
        return 10
    '\n    0 | 1\n    '
    x_re = x.real
    dist = Es * numpy.sqrt(2)
    return [dist * x_re]
sd_psk_2 = sd_psk_2_0x0
sd_psk_2_0 = sd_psk_2

def sd_psk_2_0x1(x, Es=1):
    if False:
        print('Hello World!')
    '\n    1 | 0\n    '
    x_re = [x.real]
    dist = Es * numpy.sqrt(2)
    return -dist * x_re
sd_psk_2_1 = sd_psk_2_0x1

def psk_4_0x0_0_1():
    if False:
        return 10
    '\n    | 10 | 11\n    | -------\n    | 00 | 01\n    '
    const_points = [-1 - 1j, 1 - 1j, -1 + 1j, 1 + 1j]
    symbols = [0, 1, 2, 3]
    return (const_points, symbols)
psk_4 = psk_4_0x0_0_1
psk_4_0 = psk_4

def psk_4_0x1_0_1():
    if False:
        while True:
            i = 10
    '\n    | 11 | 10\n    | -------\n    | 01 | 00\n    '
    k = 1
    pi = [0, 1]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_1 = psk_4_0x1_0_1

def psk_4_0x2_0_1():
    if False:
        for i in range(10):
            print('nop')
    '\n    | 00 | 01\n    | -------\n    | 10 | 11\n    '
    k = 2
    pi = [0, 1]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_2 = psk_4_0x2_0_1

def psk_4_0x3_0_1():
    if False:
        i = 10
        return i + 15
    '\n    | 01 | 00\n    | -------\n    | 11 | 10\n    '
    k = 3
    pi = [0, 1]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_3 = psk_4_0x3_0_1

def psk_4_0x0_1_0():
    if False:
        print('Hello World!')
    '\n    | 01 | 11\n    | -------\n    | 00 | 10\n    '
    k = 0
    pi = [1, 0]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_4 = psk_4_0x0_1_0

def psk_4_0x1_1_0():
    if False:
        print('Hello World!')
    '\n    | 00 | 10\n    | -------\n    | 01 | 11\n    '
    k = 1
    pi = [1, 0]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_5 = psk_4_0x1_1_0

def psk_4_0x2_1_0():
    if False:
        print('Hello World!')
    '\n    | 11 | 01\n    | -------\n    | 10 | 00\n    '
    k = 2
    pi = [1, 0]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_6 = psk_4_0x2_1_0

def psk_4_0x3_1_0():
    if False:
        return 10
    '\n    | 10 | 00\n    | -------\n    | 11 | 01\n    '
    k = 3
    pi = [1, 0]
    return constellation_map_generator(psk_4()[0], psk_4()[1], k, pi)
psk_4_7 = psk_4_0x3_1_0

def sd_psk_4_0x0_0_1(x, Es=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    | 10 | 11\n    | -------\n    | 00 | 01\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [dist * x_im, dist * x_re]
sd_psk_4 = sd_psk_4_0x0_0_1
sd_psk_4_0 = sd_psk_4

def sd_psk_4_0x1_0_1(x, Es=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    | 11 | 10\n    | -------\n    | 01 | 00\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [dist * x_im, -dist * x_re]
sd_psk_4_1 = sd_psk_4_0x1_0_1

def sd_psk_4_0x2_0_1(x, Es=1):
    if False:
        i = 10
        return i + 15
    '\n    | 00 | 01\n    | -------\n    | 10 | 11\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [-dist * x_im, dist * x_re]
sd_psk_4_2 = sd_psk_4_0x2_0_1

def sd_psk_4_0x3_0_1(x, Es=1):
    if False:
        i = 10
        return i + 15
    '\n    | 01 | 00\n    | -------\n    | 11 | 10\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [-dist * x_im, -dist * x_re]
sd_psk_4_3 = sd_psk_4_0x3_0_1

def sd_psk_4_0x0_1_0(x, Es=1):
    if False:
        return 10
    '\n    | 01 | 11\n    | -------\n    | 00 | 10\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [dist * x_re, dist * x_im]
sd_psk_4_4 = sd_psk_4_0x0_1_0

def sd_psk_4_0x1_1_0(x, Es=1):
    if False:
        return 10
    '\n    | 00 | 10\n    | -------\n    | 01 | 11\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [dist * x_re, -dist * x_im]
sd_psk_4_5 = sd_psk_4_0x1_1_0

def sd_psk_4_0x2_1_0(x, Es=1):
    if False:
        i = 10
        return i + 15
    '\n    | 11 | 01\n    | -------\n    | 10 | 00\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [-dist * x_re, dist * x_im]
sd_psk_4_6 = sd_psk_4_0x2_1_0

def sd_psk_4_0x3_1_0(x, Es=1):
    if False:
        return 10
    '\n    | 10 | 00\n    | -------\n    | 11 | 01\n    '
    x_re = x.real
    x_im = x.imag
    dist = Es * numpy.sqrt(2)
    return [-dist * x_re, -dist * x_im]
sd_psk_4_7 = sd_psk_4_0x3_1_0