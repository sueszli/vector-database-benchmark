import numpy as np
try:
    import mpmath
except ImportError:
    pass

def mpmath_wrightomega(x):
    if False:
        while True:
            i = 10
    return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))

def wrightomega_series_error(x):
    if False:
        i = 10
        return i + 15
    series = x
    desired = mpmath_wrightomega(x)
    return abs(series - desired) / desired

def wrightomega_exp_error(x):
    if False:
        print('Hello World!')
    exponential_approx = mpmath.exp(x)
    desired = mpmath_wrightomega(x)
    return abs(exponential_approx - desired) / desired

def main():
    if False:
        i = 10
        return i + 15
    desired_error = 2 * np.finfo(float).eps
    print('Series Error')
    for x in [100000.0, 10000000000.0, 1000000000000000.0, 1e+20]:
        with mpmath.workdps(100):
            error = wrightomega_series_error(x)
        print(x, error, error < desired_error)
    print('Exp error')
    for x in [-10, -25, -50, -100, -200, -400, -700, -740]:
        with mpmath.workdps(100):
            error = wrightomega_exp_error(x)
        print(x, error, error < desired_error)
if __name__ == '__main__':
    main()