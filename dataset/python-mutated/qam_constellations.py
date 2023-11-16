import numpy
from .constellation_map_generator import constellation_map_generator
"\nNote on the naming scheme. Each constellation is named using a prefix\nfor the type of constellation, the order of the constellation, and a\ndistinguishing feature, which comes in three modes:\n\n- No extra feature: the basic Gray-coded constellation map; others\n  will be derived from this type.\n- A single number: an indexed number to uniquely identify different\n  constellation maps.\n- 0xN_x0_x1..._xM: A permutation of the base constellation, explained\n  below.\n\nFor rectangular constellations (BPSK, QPSK, QAM), we can define a\nhyperspace and look for all symmetries. This is also known as the\nautomorphism group of the hypercube, aka the hyperoctahedral\ngroup. What this means is that we can easily define all possible\nrotations in terms of the first base mapping by creating the mapping:\n\n  f(x) = k XOR pi(x)\n\nThe x is the bit string for the symbol we are altering. Then k is a\nbit string of n bits where n is the number of bits per symbol in the\nconstellation (e.g., 2 for QPSK or 6 for QAM64). The pi is a\npermutation function specified as pi_0, pi_1..., pi_n-1. This permutes\nthe bits from the base constellation symbol to a new code, which is\nthen xor'd by k.\n\nThe value of k is from 0 to 2^n-1 and pi is a list of all bit\npositions.\n\nThe permutation are given for b0_b1_b2_... for the total number of\nbits. In the constellation diagrams shown in the comments, the bit\nordering is shown as [b3b2b1b0]. Bit values returned from the soft bit\nLUTs are in the order [b3, b2, b1, b0].\n\n\nThe total number of Gray coded modulations is (2^n)*(n!).\n\nWe create aliases for all possible naming schemes for the\nconstellations. So if a hyperoctahedral group is defined, we also set\nthis function equal to a function name using a unique ID number, and\nwe always select one rotation as our basic rotation that the other\nrotations are based off of.\n\nFor 16QAM:\n - n = 4\n - (2^n)*(n!) = 384\n - k \\in [0x0, 0xF]\n - pi = 0, 1, 2, 3\n        0, 1, 3, 2\n        0, 2, 1, 3\n        0, 2, 3, 1\n        0, 3, 1, 2\n        0, 3, 2, 1\n        1, 0, 2, 3\n        1, 0, 3, 2\n        1, 2, 0, 3\n        1, 2, 3, 0\n        1, 3, 0, 2\n        1, 3, 2, 0\n        2, 0, 1, 3\n        2, 0, 3, 1\n        2, 1, 0, 3\n        2, 1, 3, 0\n        2, 3, 0, 1\n        2, 3, 1, 0\n        3, 0, 1, 2\n        3, 0, 2, 1\n        3, 1, 0, 2\n        3, 1, 2, 0\n        3, 2, 0, 1\n        3, 2, 1, 0\n"

def qam_16_0x0_0_1_2_3():
    if False:
        return 10
    '\n    | 0010  0110 | 1110  1010\n    |\n    | 0011  0111 | 1111  1011\n    | -----------------------\n    | 0001  0101 | 1101  1001\n    |\n    | 0000  0100 | 1100  1000\n    '
    const_points = [-3 - 3j, -1 - 3j, 1 - 3j, 3 - 3j, -3 - 1j, -1 - 1j, 1 - 1j, 3 - 1j, -3 + 1j, -1 + 1j, 1 + 1j, 3 + 1j, -3 + 3j, -1 + 3j, 1 + 3j, 3 + 3j]
    symbols = [0, 4, 12, 8, 1, 5, 13, 9, 3, 7, 15, 11, 2, 6, 14, 10]
    return (const_points, symbols)
qam_16 = qam_16_0x0_0_1_2_3
qam_16_0 = qam_16

def qam_16_0x1_0_1_2_3():
    if False:
        print('Hello World!')
    '\n    | 0011  0111 | 1111  1011\n    |\n    | 0010  0110 | 1110  1010\n    | -----------------------\n    | 0000  0100 | 1100  1000\n    |\n    | 0001  0101 | 1101  1001\n    '
    k = 1
    pi = [0, 1, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_1 = qam_16_0x1_0_1_2_3

def qam_16_0x2_0_1_2_3():
    if False:
        i = 10
        return i + 15
    '\n    | 0000  0100 | 1100  1000\n    |\n    | 0001  0101 | 1101  1001\n    | -----------------------\n    | 0011  0111 | 1111  1011\n    |\n    | 0010  0110 | 1110  1010\n    '
    k = 2
    pi = [0, 1, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_2 = qam_16_0x2_0_1_2_3

def qam_16_0x3_0_1_2_3():
    if False:
        print('Hello World!')
    '\n    | 0001  0101 | 1101  1001\n    |\n    | 0000  0100 | 1100  1000\n    | -----------------------\n    | 0010  0110 | 1110  1010\n    |\n    | 0011  0111 | 1111  1011\n    '
    k = 3
    pi = [0, 1, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_3 = qam_16_0x3_0_1_2_3

def qam_16_0x0_1_0_2_3():
    if False:
        i = 10
        return i + 15
    '\n    | 0001  0101 | 1101  1001\n    |\n    | 0011  0111 | 1111  1011\n    | -----------------------\n    | 0010  0110 | 1110  1010\n    |\n    | 0000  0100 | 1100  1000\n    '
    k = 0
    pi = [1, 0, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_4 = qam_16_0x0_1_0_2_3

def qam_16_0x1_1_0_2_3():
    if False:
        return 10
    '\n    | 0000  0100 | 1100  1000\n    |\n    | 0010  0110 | 1110  1010\n    | -----------------------\n    | 0011  0111 | 1111  1011\n    |\n    | 0001  0101 | 1101  1001\n    '
    k = 1
    pi = [1, 0, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_5 = qam_16_0x1_1_0_2_3

def qam_16_0x2_1_0_2_3():
    if False:
        print('Hello World!')
    '\n    | 0011  0111 | 1111  1011\n    |\n    | 0001  0101 | 1101  1001\n    | -----------------------\n    | 0000  0100 | 1100  1000\n    |\n    | 0010  0110 | 1110  1010\n    '
    k = 2
    pi = [1, 0, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_6 = qam_16_0x2_1_0_2_3

def qam_16_0x3_1_0_2_3():
    if False:
        i = 10
        return i + 15
    '\n    | 0010  0110 | 1110  1010\n    |\n    | 0000  0100 | 1100  1000\n    | -----------------------\n    | 0001  0101 | 1101  1001\n    |\n    | 0011  0111 | 1111  1011\n    '
    k = 3
    pi = [1, 0, 2, 3]
    return constellation_map_generator(qam_16()[0], qam_16()[1], k, pi)
qam_16_7 = qam_16_0x3_1_0_2_3

def sd_qam_16_0x0_0_1_2_3(x, Es=1):
    if False:
        i = 10
        return i + 15
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0010  0110 | 1110  1010\n    |\n    | 0011  0111 | 1111  1011\n    | -----------------------\n    | 0001  0101 | 1101  1001\n    |\n    | 0000  0100 | 1100  1000\n    '
    dist = Es * numpy.sqrt(2)
    boundary = dist / 3.0
    dist0 = dist / 6.0
    x_re = x.real
    x_im = x.imag
    if x_re < -boundary:
        b3 = boundary * (x_re + dist0)
    elif x_re < boundary:
        b3 = x_re
    else:
        b3 = boundary * (x_re - dist0)
    if x_im < -boundary:
        b1 = boundary * (x_im + dist0)
    elif x_im < boundary:
        b1 = x_im
    else:
        b1 = boundary * (x_im - dist0)
    b2 = -abs(x_re) + boundary
    b0 = -abs(x_im) + boundary
    return [Es / 2.0 * b3, Es / 2.0 * b2, Es / 2.0 * b1, Es / 2.0 * b0]
sd_qam_16 = sd_qam_16_0x0_0_1_2_3
sd_qam_16_0 = sd_qam_16

def sd_qam_16_0x1_0_1_2_3(x, Es=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0011  0111 | 1111  1011\n    |\n    | 0010  0110 | 1110  1010\n    | -----------------------\n    | 0000  0100 | 1100  1000\n    |\n    | 0001  0101 | 1101  1001\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b1 = 2 * (x_im + 1)
    elif x_im < 2:
        b1 = x_im
    else:
        b1 = 2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b0 = +abs(x_im) - 2
    return [b3, b2, b1, b0]
sd_qam_16_1 = sd_qam_16_0x1_0_1_2_3

def sd_qam_16_0x2_0_1_2_3(x, Es=1):
    if False:
        while True:
            i = 10
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0000  0100 | 1100  1000\n    |\n    | 0001  0101 | 1101  1001\n    | -----------------------\n    | 0011  0111 | 1111  1011\n    |\n    | 0010  0110 | 1110  1010\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b1 = -2 * (x_im + 1)
    elif x_im < 2:
        b1 = -x_im
    else:
        b1 = -2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b0 = -abs(x_im) + 2
    return [b3, b2, b1, b0]
sd_qam_16_2 = sd_qam_16_0x2_0_1_2_3

def sd_qam_16_0x3_0_1_2_3(x, Es=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0001  0101 | 1101  1001\n    |\n    | 0000  0100 | 1100  1000\n    | -----------------------\n    | 0010  0110 | 1110  1010\n    |\n    | 0011  0111 | 1111  1011\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b1 = -2 * (x_im + 1)
    elif x_im < 2:
        b1 = -x_im
    else:
        b1 = -2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b0 = +abs(x_im) - 2
    return [b3, b2, b1, b0]
sd_qam_16_3 = sd_qam_16_0x3_0_1_2_3

def sd_qam_16_0x0_1_0_2_3(x, Es=1):
    if False:
        return 10
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0001  0101 | 1101  1001\n    |\n    | 0011  0111 | 1111  1011\n    | -----------------------\n    | 0010  0110 | 1110  1010\n    |\n    | 0000  0100 | 1100  1000\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b0 = 2 * (x_im + 1)
    elif x_im < 2:
        b0 = x_im
    else:
        b0 = 2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b1 = -abs(x_im) + 2
    return [b3, b2, b1, b0]
sd_qam_16_4 = sd_qam_16_0x0_1_0_2_3

def sd_qam_16_0x1_1_0_2_3(x, Es=1):
    if False:
        print('Hello World!')
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0000  0100 | 1100  1000\n    |\n    | 0010  0110 | 1110  1010\n    | -----------------------\n    | 0011  0111 | 1111  1011\n    |\n    | 0001  0101 | 1101  1001\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b0 = -2 * (x_im + 1)
    elif x_im < 2:
        b0 = -x_im
    else:
        b0 = -2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b1 = -abs(x_im) + 2
    return [b3, b2, b1, b0]
sd_qam_16_5 = sd_qam_16_0x1_1_0_2_3

def sd_qam_16_0x2_1_0_2_3(x, Es=1):
    if False:
        return 10
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0011  0111 | 1111  1011\n    |\n    | 0001  0101 | 1101  1001\n    | -----------------------\n    | 0000  0100 | 1100  1000\n    |\n    | 0010  0110 | 1110  1010\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b0 = 2 * (x_im + 1)
    elif x_im < 2:
        b0 = x_im
    else:
        b0 = 2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b1 = +abs(x_im) - 2
    return [b3, b2, b1, b0]
sd_qam_16_6 = sd_qam_16_0x2_1_0_2_3

def sd_qam_16_0x3_1_0_2_3(x, Es=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    | Soft bit LUT generator for constellation:\n    |\n    | 0010  0110 | 1110  1010\n    |\n    | 0000  0100 | 1100  1000\n    | -----------------------\n    | 0001  0101 | 1101  1001\n    |\n    | 0011  0111 | 1111  1011\n    '
    x_re = 3 * x.real
    x_im = 3 * x.imag
    if x_re < -2:
        b3 = 2 * (x_re + 1)
    elif x_re < 2:
        b3 = x_re
    else:
        b3 = 2 * (x_re - 1)
    if x_im < -2:
        b0 = -2 * (x_im + 1)
    elif x_im < 2:
        b0 = -x_im
    else:
        b0 = -2 * (x_im - 1)
    b2 = -abs(x_re) + 2
    b1 = +abs(x_im) - 2
    return [b3, b2, b1, b0]
sd_qam_16_7 = sd_qam_16_0x3_1_0_2_3