"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
REG_EPS = 1e-09

def setup_ldl_factor(c, G, h, dims, A, b):
    if False:
        return 10
    '\n    The meanings of arguments in this function are identical to those of the\n    function cvxopt.solvers.conelp. Refer to CVXOPT documentation\n\n        https://cvxopt.org/userguide/coneprog.html#linear-cone-programs\n\n    for more information.\n\n    Note: CVXOPT allows G and A to be passed as dense matrix objects. However,\n    this function will only ever be called with spmatrix objects. If creating\n    a custom kktsolver of your own, you need to conform to this sparse matrix\n    assumption.\n    '
    factor = kkt_ldl(G, dims, A)
    return factor

def kkt_ldl(G, dims, A):
    if False:
        while True:
            i = 10
    '\n    Returns a function handle "factor", which conforms to the CVXOPT\n    custom KKT solver specifications:\n\n        https://cvxopt.org/userguide/coneprog.html#exploiting-structure.\n\n    For convenience, we provide a short outline for how this function works.\n\n    First, we allocate workspace for use in "factor". The factor function is\n    called with data (H, W). Once called, the factor function computes an LDL\n    factorization of the 3 x 3 system:\n\n        [ H           A\'   G\'*W^{-1}  ]\n        [ A           0    0          ].\n        [ W^{-T}*G    0   -I          ]\n\n    Once that LDL factorization is computed, "factor" constructs another\n    inner function, called "solve". The solve function uses the newly\n    constructed LDL factorization to compute solutions to linear systems of\n    the form\n\n        [ H     A\'   G\'    ]   [ ux ]   [ bx ]\n        [ A     0    0     ] * [ uy ] = [ by ].\n        [ G     0   -W\'*W  ]   [ uz ]   [ bz ]\n\n    The factor function concludes by returning a reference to the solve function.\n\n    Notes: In the 3 x 3 system, H is n x n, A is p x n, and G is N x n, where\n    N = dims[\'l\'] + sum(dims[\'q\']) + sum( k**2 for k in dims[\'s\'] ). For cone\n    programs, H is the zero matrix.\n    '
    from cvxopt import blas, lapack
    from cvxopt.base import matrix
    from cvxopt.misc import pack, scale, unpack
    (p, n) = A.size
    ldK = n + p + dims['l'] + sum(dims['q']) + sum([int(k * (k + 1) / 2) for k in dims['s']])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (G.size[0], 1))

    def factor(W, H=None):
        if False:
            return 10
        blas.scal(0.0, K)
        if H is not None:
            K[:n, :n] = H
        K[n:n + p, :n] = A
        for k in range(n):
            g[:] = G[:, k]
            scale(g, W, trans='T', inverse='I')
            pack(g, K, dims, 0, offsety=k * ldK + n + p)
        K[(ldK + 1) * (p + n)::ldK + 1] = -1.0
        K[0:(ldK + 1) * n:ldK + 1] += REG_EPS
        K[(ldK + 1) * n::ldK + 1] += -REG_EPS
        lapack.sytrf(K, ipiv)

        def solve(x, y, z):
            if False:
                return 10
            blas.copy(x, u)
            blas.copy(y, u, offsety=n)
            scale(z, W, trans='T', inverse='I')
            pack(z, u, dims, 0, offsety=n + p)
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n=n)
            blas.copy(u, y, offsetx=n, n=p)
            unpack(u, z, dims, 0, offsetx=n + p)
        return solve
    return factor