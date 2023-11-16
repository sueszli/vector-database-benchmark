def lfsr_args(seed, *exp):
    if False:
        return 10
    '\n    Produce arguments to create scrambler objects from exponent polynomial expressions.\n     seed: start-value of register\n    *exp: exponents of desired polynomial.\n     Example:\n    >>> l = digital.lfsr(*lfrs_args(0b11001,7,1,0))\n    Creates an lfsr object with seed 0b11001, mask 0b1000011, K=6\n    '
    from functools import reduce
    return (reduce(int.__xor__, map(lambda x: 2 ** x, exp)), seed, max(exp) - 1)