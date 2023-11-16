try:
    from gmpy2 import mpz as mpz
except ImportError:
    try:
        from gmpy import mpz as mpz
    except ImportError:

        def mpz(x):
            if False:
                i = 10
                return i + 15
            return x
        pass
__all__ = ['powMod']

def powMod(x, y, mod):
    if False:
        i = 10
        return i + 15
    "\n    (Efficiently) Calculate and return `x' to the power of `y' mod `mod'.\n\n    If possible, the three numbers are converted to GMPY's bignum\n    representation which speeds up exponentiation.  If GMPY is not installed,\n    built-in exponentiation is used.\n    "
    x = mpz(x)
    y = mpz(y)
    mod = mpz(mod)
    return pow(x, y, mod)