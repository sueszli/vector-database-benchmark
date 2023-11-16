"""
This example calculates the Ricci tensor from the metric and does this
on the example of Schwarzschild solution.

If you want to derive this by hand, follow the wiki page here:

https://en.wikipedia.org/wiki/Deriving_the_Schwarzschild_solution

Also read the above wiki and follow the references from there if
something is not clear, like what the Ricci tensor is, etc.

"""
from sympy import exp, Symbol, sin, dsolve, Function, Matrix, Eq, pprint, solve

def grad(f, X):
    if False:
        while True:
            i = 10
    a = []
    for x in X:
        a.append(f.diff(x))
    return a

def d(m, x):
    if False:
        print('Hello World!')
    return grad(m[0, 0], x)

class MT:

    def __init__(self, m):
        if False:
            for i in range(10):
                print('nop')
        self.gdd = m
        self.guu = m.inv()

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'g_dd =\n' + str(self.gdd)

    def dd(self, i, j):
        if False:
            print('Hello World!')
        return self.gdd[i, j]

    def uu(self, i, j):
        if False:
            i = 10
            return i + 15
        return self.guu[i, j]

class G:

    def __init__(self, g, x):
        if False:
            while True:
                i = 10
        self.g = g
        self.x = x

    def udd(self, i, k, l):
        if False:
            return 10
        g = self.g
        x = self.x
        r = 0
        for m in [0, 1, 2, 3]:
            r += g.uu(i, m) / 2 * (g.dd(m, k).diff(x[l]) + g.dd(m, l).diff(x[k]) - g.dd(k, l).diff(x[m]))
        return r

class Riemann:

    def __init__(self, G, x):
        if False:
            for i in range(10):
                print('nop')
        self.G = G
        self.x = x

    def uddd(self, rho, sigma, mu, nu):
        if False:
            i = 10
            return i + 15
        G = self.G
        x = self.x
        r = G.udd(rho, nu, sigma).diff(x[mu]) - G.udd(rho, mu, sigma).diff(x[nu])
        for lam in [0, 1, 2, 3]:
            r += G.udd(rho, mu, lam) * G.udd(lam, nu, sigma) - G.udd(rho, nu, lam) * G.udd(lam, mu, sigma)
        return r

class Ricci:

    def __init__(self, R, x):
        if False:
            for i in range(10):
                print('nop')
        self.R = R
        self.x = x
        self.g = R.G.g

    def dd(self, mu, nu):
        if False:
            while True:
                i = 10
        R = self.R
        x = self.x
        r = 0
        for lam in [0, 1, 2, 3]:
            r += R.uddd(lam, mu, lam, nu)
        return r

    def ud(self, mu, nu):
        if False:
            return 10
        r = 0
        for lam in [0, 1, 2, 3]:
            r += self.g.uu(mu, lam) * self.dd(lam, nu)
        return r.expand()

def curvature(Rmn):
    if False:
        while True:
            i = 10
    return Rmn.ud(0, 0) + Rmn.ud(1, 1) + Rmn.ud(2, 2) + Rmn.ud(3, 3)
nu = Function('nu')
lam = Function('lambda')
t = Symbol('t')
r = Symbol('r')
theta = Symbol('theta')
phi = Symbol('phi')
gdd = Matrix(((-exp(nu(r)), 0, 0, 0), (0, exp(lam(r)), 0, 0), (0, 0, r ** 2, 0), (0, 0, 0, r ** 2 * sin(theta) ** 2)))
g = MT(gdd)
X = (t, r, theta, phi)
Gamma = G(g, X)
Rmn = Ricci(Riemann(Gamma, X), X)

def pprint_Gamma_udd(i, k, l):
    if False:
        print('Hello World!')
    pprint(Eq(Symbol('Gamma^%i_%i%i' % (i, k, l)), Gamma.udd(i, k, l)))

def pprint_Rmn_dd(i, j):
    if False:
        i = 10
        return i + 15
    pprint(Eq(Symbol('R_%i%i' % (i, j)), Rmn.dd(i, j)))

def eq1():
    if False:
        for i in range(10):
            print('nop')
    r = Symbol('r')
    e = Rmn.dd(0, 0)
    e = e.subs(nu(r), -lam(r))
    pprint(dsolve(e, lam(r)))

def eq2():
    if False:
        for i in range(10):
            print('nop')
    r = Symbol('r')
    e = Rmn.dd(1, 1)
    C = Symbol('CC')
    e = e.subs(nu(r), -lam(r))
    pprint(dsolve(e, lam(r)))

def eq3():
    if False:
        i = 10
        return i + 15
    r = Symbol('r')
    e = Rmn.dd(2, 2)
    e = e.subs(nu(r), -lam(r))
    pprint(dsolve(e, lam(r)))

def eq4():
    if False:
        print('Hello World!')
    r = Symbol('r')
    e = Rmn.dd(3, 3)
    e = e.subs(nu(r), -lam(r))
    pprint(dsolve(e, lam(r)))
    pprint(dsolve(e, lam(r), 'best'))

def main():
    if False:
        while True:
            i = 10
    print('Initial metric:')
    pprint(gdd)
    print('-' * 40)
    print('Christoffel symbols:')
    pprint_Gamma_udd(0, 1, 0)
    pprint_Gamma_udd(0, 0, 1)
    print()
    pprint_Gamma_udd(1, 0, 0)
    pprint_Gamma_udd(1, 1, 1)
    pprint_Gamma_udd(1, 2, 2)
    pprint_Gamma_udd(1, 3, 3)
    print()
    pprint_Gamma_udd(2, 2, 1)
    pprint_Gamma_udd(2, 1, 2)
    pprint_Gamma_udd(2, 3, 3)
    print()
    pprint_Gamma_udd(3, 2, 3)
    pprint_Gamma_udd(3, 3, 2)
    pprint_Gamma_udd(3, 1, 3)
    pprint_Gamma_udd(3, 3, 1)
    print('-' * 40)
    print('Ricci tensor:')
    pprint_Rmn_dd(0, 0)
    e = Rmn.dd(1, 1)
    pprint_Rmn_dd(1, 1)
    pprint_Rmn_dd(2, 2)
    pprint_Rmn_dd(3, 3)
    print('-' * 40)
    print("Solve Einstein's equations:")
    e = e.subs(nu(r), -lam(r)).doit()
    l = dsolve(e, lam(r))
    pprint(l)
    lamsol = solve(l, lam(r))[0]
    metric = gdd.subs(lam(r), lamsol).subs(nu(r), -lamsol)
    print('metric:')
    pprint(metric)
if __name__ == '__main__':
    main()