"""Functions returning normal forms of matrices"""
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_form as _snf, invariant_factors as _invf, hermite_normal_form as _hnf

def _to_domain(m, domain=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert Matrix to DomainMatrix'
    ring = getattr(m, 'ring', None)
    m = m.applyfunc(lambda e: e.as_expr() if isinstance(e, Poly) else e)
    dM = DomainMatrix.from_Matrix(m)
    domain = domain or ring
    if domain is not None:
        dM = dM.convert_to(domain)
    return dM

def smith_normal_form(m, domain=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the Smith Normal Form of a matrix `m` over the ring `domain`.\n    This will only work if the ring is a principal ideal domain.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, ZZ\n    >>> from sympy.matrices.normalforms import smith_normal_form\n    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])\n    >>> print(smith_normal_form(m, domain=ZZ))\n    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, -30]])\n\n    '
    dM = _to_domain(m, domain)
    return _snf(dM).to_Matrix()

def invariant_factors(m, domain=None):
    if False:
        return 10
    '\n    Return the tuple of abelian invariants for a matrix `m`\n    (as in the Smith-Normal form)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm\n    .. [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf\n\n    '
    dM = _to_domain(m, domain)
    factors = _invf(dM)
    factors = tuple((dM.domain.to_sympy(f) for f in factors))
    if hasattr(m, 'ring'):
        if m.ring.is_PolynomialRing:
            K = m.ring
            to_poly = lambda f: Poly(f, K.symbols, domain=K.domain)
            factors = tuple((to_poly(f) for f in factors))
    return factors

def hermite_normal_form(A, *, D=None, check_rank=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the Hermite Normal Form of a Matrix *A* of integers.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.matrices.normalforms import hermite_normal_form\n    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])\n    >>> print(hermite_normal_form(m))\n    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])\n\n    Parameters\n    ==========\n\n    A : $m \\times n$ ``Matrix`` of integers.\n\n    D : int, optional\n        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*\n        being any multiple of $\\det(W)$ may be provided. In this case, if *A*\n        also has rank $m$, then we may use an alternative algorithm that works\n        mod *D* in order to prevent coefficient explosion.\n\n    check_rank : boolean, optional (default=False)\n        The basic assumption is that, if you pass a value for *D*, then\n        you already believe that *A* has rank $m$, so we do not waste time\n        checking it for you. If you do want this to be checked (and the\n        ordinary, non-modulo *D* algorithm to be used if the check fails), then\n        set *check_rank* to ``True``.\n\n    Returns\n    =======\n\n    ``Matrix``\n        The HNF of matrix *A*.\n\n    Raises\n    ======\n\n    DMDomainError\n        If the domain of the matrix is not :ref:`ZZ`.\n\n    DMShapeError\n        If the mod *D* algorithm is used but the matrix has more rows than\n        columns.\n\n    References\n    ==========\n\n    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*\n       (See Algorithms 2.4.5 and 2.4.8.)\n\n    '
    if D is not None and (not ZZ.of_type(D)):
        D = ZZ(int(D))
    return _hnf(A._rep, D=D, check_rank=check_rank).to_Matrix()