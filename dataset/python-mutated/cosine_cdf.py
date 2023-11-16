import mpmath

def f(x):
    if False:
        return 10
    return (mpmath.pi + x + mpmath.sin(x)) / (2 * mpmath.pi)
mpmath.mp.dps = 40
ts = mpmath.taylor(f, -mpmath.pi, 20)
(p, q) = mpmath.pade(ts, 9, 10)
p = [float(c) for c in p]
q = [float(c) for c in q]
print('p =', p)
print('q =', q)