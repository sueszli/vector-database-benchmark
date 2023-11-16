"""
This module contains functions for the computation
of Euclidean, (generalized) Sturmian, (modified) subresultant
polynomial remainder sequences (prs's) of two polynomials;
included are also three functions for the computation of the
resultant of two polynomials.

Except for the function res_z(), which computes the resultant
of two polynomials, the pseudo-remainder function prem()
of sympy is _not_ used by any of the functions in the module.

Instead of prem() we use the function

rem_z().

Included is also the function quo_z().

An explanation of why we avoid prem() can be found in the
references stated in the docstring of rem_z().

1. Theoretical background:
==========================
Consider the polynomials f, g in Z[x] of degrees deg(f) = n and
deg(g) = m with n >= m.

Definition 1:
=============
The sign sequence of a polynomial remainder sequence (prs) is the
sequence of signs of the leading coefficients of its polynomials.

Sign sequences can be computed with the function:

sign_seq(poly_seq, x)

Definition 2:
=============
A polynomial remainder sequence (prs) is called complete if the
degree difference between any two consecutive polynomials is 1;
otherwise, it called incomplete.

It is understood that f, g belong to the sequences mentioned in
the two definitions above.

1A. Euclidean and subresultant prs's:
=====================================
The subresultant prs of f, g is a sequence of polynomials in Z[x]
analogous to the Euclidean prs, the sequence obtained by applying
on f, g Euclid's algorithm for polynomial greatest common divisors
(gcd) in Q[x].

The subresultant prs differs from the Euclidean prs in that the
coefficients of each polynomial in the former sequence are determinants
--- also referred to as subresultants --- of appropriately selected
sub-matrices of sylvester1(f, g, x), Sylvester's matrix of 1840 of
dimensions (n + m) * (n + m).

Recall that the determinant of sylvester1(f, g, x) itself is
called the resultant of f, g and serves as a criterion of whether
the two polynomials have common roots or not.

In SymPy the resultant is computed with the function
resultant(f, g, x). This function does _not_ evaluate the
determinant of sylvester(f, g, x, 1); instead, it returns
the last member of the subresultant prs of f, g, multiplied
(if needed) by an appropriate power of -1; see the caveat below.

In this module we use three functions to compute the
resultant of f, g:
a) res(f, g, x) computes the resultant by evaluating
the determinant of sylvester(f, g, x, 1);
b) res_q(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Q[x] with the function rem();
c) res_z(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Z[x] with the function prem().

Caveat: If Df = degree(f, x) and Dg = degree(g, x), then:

resultant(f, g, x) = (-1)**(Df*Dg) * resultant(g, f, x).

For complete prs's the sign sequence of the Euclidean prs of f, g
is identical to the sign sequence of the subresultant prs of f, g
and the coefficients of one sequence  are easily computed from the
coefficients of the  other.

For incomplete prs's the polynomials in the subresultant prs, generally
differ in sign from those of the Euclidean prs, and --- unlike the
case of complete prs's --- it is not at all obvious how to compute
the coefficients of one sequence from the coefficients of the  other.

1B. Sturmian and modified subresultant prs's:
=============================================
For the same polynomials f, g in Z[x] mentioned above, their ``modified''
subresultant prs is a sequence of polynomials similar to the Sturmian
prs, the sequence obtained by applying in Q[x] Sturm's algorithm on f, g.

The two sequences differ in that the coefficients of each polynomial
in the modified subresultant prs are the determinants --- also referred
to as modified subresultants --- of appropriately selected  sub-matrices
of sylvester2(f, g, x), Sylvester's matrix of 1853 of dimensions 2n x 2n.

The determinant of sylvester2 itself is called the modified resultant
of f, g and it also can serve as a criterion of whether the two
polynomials have common roots or not.

For complete prs's the  sign sequence of the Sturmian prs of f, g is
identical to the sign sequence of the modified subresultant prs of
f, g and the coefficients of one sequence  are easily computed from
the coefficients of the other.

For incomplete prs's the polynomials in the modified subresultant prs,
generally differ in sign from those of the Sturmian prs, and --- unlike
the case of complete prs's --- it is not at all obvious how to compute
the coefficients of one sequence from the coefficients of the  other.

As Sylvester pointed out, the coefficients of the polynomial remainders
obtained as (modified) subresultants are the smallest possible without
introducing rationals and without computing (integer) greatest common
divisors.

1C. On terminology:
===================
Whence the terminology? Well generalized Sturmian prs's are
``modifications'' of Euclidean prs's; the hint came from the title
of the Pell-Gordon paper of 1917.

In the literature one also encounters the name ``non signed'' and
``signed'' prs for Euclidean and Sturmian prs respectively.

Likewise ``non signed'' and ``signed'' subresultant prs for
subresultant and modified subresultant prs respectively.

2. Functions in the module:
===========================
No function utilizes SymPy's function prem().

2A. Matrices:
=============
The functions sylvester(f, g, x, method=1) and
sylvester(f, g, x, method=2) compute either Sylvester matrix.
They can be used to compute (modified) subresultant prs's by
direct determinant evaluation.

The function bezout(f, g, x, method='prs') provides a matrix of
smaller dimensions than either Sylvester matrix. It is the function
of choice for computing (modified) subresultant prs's by direct
determinant evaluation.

sylvester(f, g, x, method=1)
sylvester(f, g, x, method=2)
bezout(f, g, x, method='prs')

The following identity holds:

bezout(f, g, x, method='prs') =
backward_eye(deg(f))*bezout(f, g, x, method='bz')*backward_eye(deg(f))

2B. Subresultant and modified subresultant prs's by
===================================================
determinant evaluations:
=======================
We use the Sylvester matrices of 1840 and 1853 to
compute, respectively, subresultant and modified
subresultant polynomial remainder sequences. However,
for large matrices this approach takes a lot of time.

Instead of utilizing the Sylvester matrices, we can
employ the Bezout matrix which is of smaller dimensions.

subresultants_sylv(f, g, x)
modified_subresultants_sylv(f, g, x)
subresultants_bezout(f, g, x)
modified_subresultants_bezout(f, g, x)

2C. Subresultant prs's by ONE determinant evaluation:
=====================================================
All three functions in this section evaluate one determinant
per remainder polynomial; this is the determinant of an
appropriately selected sub-matrix of sylvester1(f, g, x),
Sylvester's matrix of 1840.

To compute the remainder polynomials the function
subresultants_rem(f, g, x) employs rem(f, g, x).
By contrast, the other two functions implement Van Vleck's ideas
of 1900 and compute the remainder polynomials by trinagularizing
sylvester2(f, g, x), Sylvester's matrix of 1853.


subresultants_rem(f, g, x)
subresultants_vv(f, g, x)
subresultants_vv_2(f, g, x).

2E. Euclidean, Sturmian prs's in Q[x]:
======================================
euclid_q(f, g, x)
sturm_q(f, g, x)

2F. Euclidean, Sturmian and (modified) subresultant prs's P-G:
==============================================================
All functions in this section are based on the Pell-Gordon (P-G)
theorem of 1917.
Computations are done in Q[x], employing the function rem(f, g, x)
for the computation of the remainder polynomials.

euclid_pg(f, g, x)
sturm pg(f, g, x)
subresultants_pg(f, g, x)
modified_subresultants_pg(f, g, x)

2G. Euclidean, Sturmian and (modified) subresultant prs's A-M-V:
================================================================
All functions in this section are based on the Akritas-Malaschonok-
Vigklas (A-M-V) theorem of 2015.
Computations are done in Z[x], employing the function rem_z(f, g, x)
for the computation of the remainder polynomials.

euclid_amv(f, g, x)
sturm_amv(f, g, x)
subresultants_amv(f, g, x)
modified_subresultants_amv(f, g, x)

2Ga. Exception:
===============
subresultants_amv_q(f, g, x)

This function employs rem(f, g, x) for the computation of
the remainder polynomials, despite the fact that it implements
the A-M-V Theorem.

It is included in our module in order to show that theorems P-G
and A-M-V can be implemented utilizing either the function
rem(f, g, x) or the function rem_z(f, g, x).

For clearly historical reasons --- since the Collins-Brown-Traub
coefficients-reduction factor beta_i was not available in 1917 ---
we have implemented the Pell-Gordon theorem with the function
rem(f, g, x) and the A-M-V Theorem  with the function rem_z(f, g, x).

2H. Resultants:
===============
res(f, g, x)
res_q(f, g, x)
res_z(f, g, x)
"""
from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError

def sylvester(f, g, x, method=1):
    if False:
        print('Hello World!')
    "\n      The input polynomials f, g are in Z[x] or in Q[x]. Let m = degree(f, x),\n      n = degree(g, x) and mx = max(m, n).\n\n      a. If method = 1 (default), computes sylvester1, Sylvester's matrix of 1840\n          of dimension (m + n) x (m + n). The determinants of properly chosen\n          submatrices of this matrix (a.k.a. subresultants) can be\n          used to compute the coefficients of the Euclidean PRS of f, g.\n\n      b. If method = 2, computes sylvester2, Sylvester's matrix of 1853\n          of dimension (2*mx) x (2*mx). The determinants of properly chosen\n          submatrices of this matrix (a.k.a. ``modified'' subresultants) can be\n          used to compute the coefficients of the Sturmian PRS of f, g.\n\n      Applications of these Matrices can be found in the references below.\n      Especially, for applications of sylvester2, see the first reference!!\n\n      References\n      ==========\n      1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem\n      by Van Vleck Regarding Sturm Sequences. Serdica Journal of Computing,\n      Vol. 7, No 4, 101-134, 2013.\n\n      2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n      and Modified Subresultant Polynomial Remainder Sequences.''\n      Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    "
    (m, n) = (degree(Poly(f, x), x), degree(Poly(g, x), x))
    if m == n and n < 0:
        return Matrix([])
    if m == n and n == 0:
        return Matrix([])
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])
    fp = Poly(f, x).all_coeffs()
    gp = Poly(g, x).all_coeffs()
    if method <= 1:
        M = zeros(m + n)
        k = 0
        for i in range(n):
            j = k
            for coeff in fp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        k = 0
        for i in range(n, m + n):
            j = k
            for coeff in gp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        return M
    if method >= 2:
        if len(fp) < len(gp):
            h = []
            for i in range(len(gp) - len(fp)):
                h.append(0)
            fp[:0] = h
        else:
            h = []
            for i in range(len(fp) - len(gp)):
                h.append(0)
            gp[:0] = h
        mx = max(m, n)
        dim = 2 * mx
        M = zeros(dim)
        k = 0
        for i in range(mx):
            j = k
            for coeff in fp:
                M[2 * i, j] = coeff
                j = j + 1
            j = k
            for coeff in gp:
                M[2 * i + 1, j] = coeff
                j = j + 1
            k = k + 1
        return M

def process_matrix_output(poly_seq, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    poly_seq is a polynomial remainder sequence computed either by\n    (modified_)subresultants_bezout or by (modified_)subresultants_sylv.\n\n    This function removes from poly_seq all zero polynomials as well\n    as all those whose degree is equal to the degree of a preceding\n    polynomial in poly_seq, as we scan it from left to right.\n\n    '
    L = poly_seq[:]
    d = degree(L[1], x)
    i = 2
    while i < len(L):
        d_i = degree(L[i], x)
        if d_i < 0:
            L.remove(L[i])
            i = i - 1
        if d == d_i:
            L.remove(L[i])
            i = i - 1
        if d_i >= 0:
            d = d_i
        i = i + 1
    return L

def subresultants_sylv(f, g, x):
    if False:
        return 10
    '\n    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed\n    that deg(f) >= deg(g).\n\n    Computes the subresultant polynomial remainder sequence (prs)\n    of f, g by evaluating determinants of appropriately selected\n    submatrices of sylvester(f, g, x, 1). The dimensions of the\n    latter are (deg(f) + deg(g)) x (deg(f) + deg(g)).\n\n    Each coefficient is computed by evaluating the determinant of the\n    corresponding submatrix of sylvester(f, g, x, 1).\n\n    If the subresultant prs is complete, then the output coincides\n    with the Euclidean sequence of the polynomials f, g.\n\n    References:\n    ===========\n    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants\n    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,\n    Vol. 15, 233-266, 2004.\n\n    '
    if f == 0 or g == 0:
        return [f, g]
    n = degF = degree(f, x)
    m = degG = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, degF, degG, f, g) = (m, n, degG, degF, g, f)
    if n > 0 and m == 0:
        return [f, g]
    SR_L = [f, g]
    S = sylvester(f, g, x, 1)
    j = m - 1
    while j > 0:
        Sp = S[:, :]
        for ind in range(m + n - j, m + n):
            Sp.row_del(m + n - j)
        for ind in range(m - j, m):
            Sp.row_del(m - j)
        (coeff_L, k, l) = ([], Sp.rows, 0)
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1
    SR_L.append(S.det())
    return process_matrix_output(SR_L, x)

def modified_subresultants_sylv(f, g, x):
    if False:
        print('Hello World!')
    '\n    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed\n    that deg(f) >= deg(g).\n\n    Computes the modified subresultant polynomial remainder sequence (prs)\n    of f, g by evaluating determinants of appropriately selected\n    submatrices of sylvester(f, g, x, 2). The dimensions of the\n    latter are (2*deg(f)) x (2*deg(f)).\n\n    Each coefficient is computed by evaluating the determinant of the\n    corresponding submatrix of sylvester(f, g, x, 2).\n\n    If the modified subresultant prs is complete, then the output coincides\n    with the Sturmian sequence of the polynomials f, g.\n\n    References:\n    ===========\n    1. A. G. Akritas,G.I. Malaschonok and P.S. Vigklas:\n    Sturm Sequences and Modified Subresultant Polynomial Remainder\n    Sequences. Serdica Journal of Computing, Vol. 8, No 1, 29--46, 2014.\n\n    '
    if f == 0 or g == 0:
        return [f, g]
    n = degF = degree(f, x)
    m = degG = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, degF, degG, f, g) = (m, n, degG, degF, g, f)
    if n > 0 and m == 0:
        return [f, g]
    SR_L = [f, g]
    S = sylvester(f, g, x, 2)
    j = m - 1
    while j > 0:
        Sp = S[0:2 * n - 2 * j, :]
        (coeff_L, k, l) = ([], Sp.rows, 0)
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1
    SR_L.append(S.det())
    return process_matrix_output(SR_L, x)

def res(f, g, x):
    if False:
        while True:
            i = 10
    '\n    The input polynomials f, g are in Z[x] or in Q[x].\n\n    The output is the resultant of f, g computed by evaluating\n    the determinant of the matrix sylvester(f, g, x, 1).\n\n    References:\n    ===========\n    1. J. S. Cohen: Computer Algebra and Symbolic Computation\n     - Mathematical Methods. A. K. Peters, 2003.\n\n    '
    if f == 0 or g == 0:
        raise PolynomialError('The resultant of %s and %s is not defined' % (f, g))
    else:
        return sylvester(f, g, x, 1).det()

def res_q(f, g, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    The input polynomials f, g are in Z[x] or in Q[x].\n\n    The output is the resultant of f, g computed recursively\n    by polynomial divisions in Q[x], using the function rem.\n    See Cohen's book p. 281.\n\n    References:\n    ===========\n    1. J. S. Cohen: Computer Algebra and Symbolic Computation\n     - Mathematical Methods. A. K. Peters, 2003.\n    "
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1) ** (m * n) * res_q(g, f, x)
    elif n == 0:
        return g ** m
    else:
        r = rem(f, g, x)
        if r == 0:
            return 0
        else:
            s = degree(r, x)
            l = LC(g, x)
            return (-1) ** (m * n) * l ** (m - s) * res_q(g, r, x)

def res_z(f, g, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    The input polynomials f, g are in Z[x] or in Q[x].\n\n    The output is the resultant of f, g computed recursively\n    by polynomial divisions in Z[x], using the function prem().\n    See Cohen's book p. 283.\n\n    References:\n    ===========\n    1. J. S. Cohen: Computer Algebra and Symbolic Computation\n     - Mathematical Methods. A. K. Peters, 2003.\n    "
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1) ** (m * n) * res_z(g, f, x)
    elif n == 0:
        return g ** m
    else:
        r = prem(f, g, x)
        if r == 0:
            return 0
        else:
            delta = m - n + 1
            w = (-1) ** (m * n) * res_z(g, r, x)
            s = degree(r, x)
            l = LC(g, x)
            k = delta * n - m + s
            return quo(w, l ** k, x)

def sign_seq(poly_seq, x):
    if False:
        i = 10
        return i + 15
    '\n    Given a sequence of polynomials poly_seq, it returns\n    the sequence of signs of the leading coefficients of\n    the polynomials in poly_seq.\n\n    '
    return [sign(LC(poly_seq[i], x)) for i in range(len(poly_seq))]

def bezout(p, q, x, method='bz'):
    if False:
        i = 10
        return i + 15
    "\n    The input polynomials p, q are in Z[x] or in Q[x]. Let\n    mx = max(degree(p, x), degree(q, x)).\n\n    The default option bezout(p, q, x, method='bz') returns Bezout's\n    symmetric matrix of p and q, of dimensions (mx) x (mx). The\n    determinant of this matrix is equal to the determinant of sylvester2,\n    Sylvester's matrix of 1853, whose dimensions are (2*mx) x (2*mx);\n    however the subresultants of these two matrices may differ.\n\n    The other option, bezout(p, q, x, 'prs'), is of interest to us\n    in this module because it returns a matrix equivalent to sylvester2.\n    In this case all subresultants of the two matrices are identical.\n\n    Both the subresultant polynomial remainder sequence (prs) and\n    the modified subresultant prs of p and q can be computed by\n    evaluating determinants of appropriately selected submatrices of\n    bezout(p, q, x, 'prs') --- one determinant per coefficient of the\n    remainder polynomials.\n\n    The matrices bezout(p, q, x, 'bz') and bezout(p, q, x, 'prs')\n    are related by the formula\n\n    bezout(p, q, x, 'prs') =\n    backward_eye(deg(p)) * bezout(p, q, x, 'bz') * backward_eye(deg(p)),\n\n    where backward_eye() is the backward identity function.\n\n    References\n    ==========\n    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants\n    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,\n    Vol. 15, 233-266, 2004.\n\n    "
    (m, n) = (degree(Poly(p, x), x), degree(Poly(q, x), x))
    if m == n and n < 0:
        return Matrix([])
    if m == n and n == 0:
        return Matrix([])
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])
    y = var('y')
    expr = p * q.subs({x: y}) - p.subs({x: y}) * q
    poly = Poly(quo(expr, x - y), x, y)
    mx = max(m, n)
    B = zeros(mx)
    for i in range(mx):
        for j in range(mx):
            if method == 'prs':
                B[mx - 1 - i, mx - 1 - j] = poly.nth(i, j)
            else:
                B[i, j] = poly.nth(i, j)
    return B

def backward_eye(n):
    if False:
        print('Hello World!')
    '\n    Returns the backward identity matrix of dimensions n x n.\n\n    Needed to "turn" the Bezout matrices\n    so that the leading coefficients are first.\n    See docstring of the function bezout(p, q, x, method=\'bz\').\n    '
    M = eye(n)
    for i in range(int(M.rows / 2)):
        M.row_swap(0 + i, M.rows - 1 - i)
    return M

def subresultants_bezout(p, q, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the subresultant polynomial remainder sequence\n    of p, q by evaluating determinants of appropriately selected\n    submatrices of bezout(p, q, x, 'prs'). The dimensions of the\n    latter are deg(p) x deg(p).\n\n    Each coefficient is computed by evaluating the determinant of the\n    corresponding submatrix of bezout(p, q, x, 'prs').\n\n    bezout(p, q, x, 'prs) is used instead of sylvester(p, q, x, 1),\n    Sylvester's matrix of 1840, because the dimensions of the latter\n    are (deg(p) + deg(q)) x (deg(p) + deg(q)).\n\n    If the subresultant prs is complete, then the output coincides\n    with the Euclidean sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants\n    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,\n    Vol. 15, 233-266, 2004.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    (f, g) = (p, q)
    n = degF = degree(f, x)
    m = degG = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, degF, degG, f, g) = (m, n, degG, degF, g, f)
    if n > 0 and m == 0:
        return [f, g]
    SR_L = [f, g]
    F = LC(f, x) ** (degF - degG)
    B = bezout(f, g, x, 'prs')
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        (k, coeff_L) = (j - 1, [])
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1
        SR_L.append(int((-1) ** (j * (j - 1) / 2)) * (Poly(coeff_L, x) / F).as_expr())
        j = j + 1
    return process_matrix_output(SR_L, x)

def modified_subresultants_bezout(p, q, x):
    if False:
        i = 10
        return i + 15
    "\n    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the modified subresultant polynomial remainder sequence\n    of p, q by evaluating determinants of appropriately selected\n    submatrices of bezout(p, q, x, 'prs'). The dimensions of the\n    latter are deg(p) x deg(p).\n\n    Each coefficient is computed by evaluating the determinant of the\n    corresponding submatrix of bezout(p, q, x, 'prs').\n\n    bezout(p, q, x, 'prs') is used instead of sylvester(p, q, x, 2),\n    Sylvester's matrix of 1853, because the dimensions of the latter\n    are 2*deg(p) x 2*deg(p).\n\n    If the modified subresultant prs is complete, and LC( p ) > 0, the output\n    coincides with the (generalized) Sturm's sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n    and Modified Subresultant Polynomial Remainder Sequences.''\n    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    2. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants\n    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,\n    Vol. 15, 233-266, 2004.\n\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    (f, g) = (p, q)
    n = degF = degree(f, x)
    m = degG = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, degF, degG, f, g) = (m, n, degG, degF, g, f)
    if n > 0 and m == 0:
        return [f, g]
    SR_L = [f, g]
    B = bezout(f, g, x, 'prs')
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        (k, coeff_L) = (j - 1, [])
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1
        SR_L.append(Poly(coeff_L, x).as_expr())
        j = j + 1
    return process_matrix_output(SR_L, x)

def sturm_pg(p, q, x, method=0):
    if False:
        print('Hello World!')
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].\n    If q = diff(p, x, 1) it is the usual Sturm sequence.\n\n    A. If method == 0, default, the remainder coefficients of the sequence\n       are (in absolute value) ``modified'' subresultants, which for non-monic\n       polynomials are greater than the coefficients of the corresponding\n       subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).\n\n    B. If method == 1, the remainder coefficients of the sequence are (in\n       absolute value) subresultants, which for non-monic polynomials are\n       smaller than the coefficients of the corresponding ``modified''\n       subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).\n\n    If the Sturm sequence is complete, method=0 and LC( p ) > 0, the coefficients\n    of the polynomials in the sequence are ``modified'' subresultants.\n    That is, they are  determinants of appropriately selected submatrices of\n    sylvester2, Sylvester's matrix of 1853. In this case the Sturm sequence\n    coincides with the ``modified'' subresultant prs, of the polynomials\n    p, q.\n\n    If the Sturm sequence is incomplete and method=0 then the signs of the\n    coefficients of the polynomials in the sequence may differ from the signs\n    of the coefficients of the corresponding polynomials in the ``modified''\n    subresultant prs; however, the absolute values are the same.\n\n    To compute the coefficients, no determinant evaluation takes place. Instead,\n    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);\n    the coefficients of the remainders computed this way become (``modified'')\n    subresultants with the help of the Pell-Gordon Theorem of 1917.\n    See also the function euclid_pg(p, q, x).\n\n    References\n    ==========\n    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding\n    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,\n    Second Series, 18 (1917), No. 4, 188-193.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n    and Modified Subresultant Polynomial Remainder Sequences.''\n    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (p, q) = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    flag = 0
    if LC(p, x) < 0:
        flag = 1
        p = -p
        q = -q
    lcf = LC(p, x) ** (d0 - d1)
    (a0, a1) = (p, q)
    sturm_seq = [a0, a1]
    del0 = d0 - d1
    rho1 = LC(a1, x)
    exp_deg = d1 - 1
    a2 = -rem(a0, a1, domain=QQ)
    rho2 = LC(a2, x)
    d2 = degree(a2, x)
    deg_diff_new = exp_deg - d2
    del1 = d1 - d2
    mul_fac_old = rho1 ** (del0 + del1 - deg_diff_new)
    if method == 0:
        sturm_seq.append(simplify(lcf * a2 * Abs(mul_fac_old)))
    else:
        sturm_seq.append(simplify(a2 * Abs(mul_fac_old)))
    deg_diff_old = deg_diff_new
    while d2 > 0:
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        del0 = del1
        exp_deg = d1 - 1
        a2 = -rem(a0, a1, domain=QQ)
        rho3 = LC(a2, x)
        d2 = degree(a2, x)
        deg_diff_new = exp_deg - d2
        del1 = d1 - d2
        expo_old = deg_diff_old
        expo_new = del0 + del1 - deg_diff_new
        mul_fac_new = rho2 ** expo_new * rho1 ** expo_old * mul_fac_old
        (deg_diff_old, mul_fac_old) = (deg_diff_new, mul_fac_new)
        (rho1, rho2) = (rho2, rho3)
        if method == 0:
            sturm_seq.append(simplify(lcf * a2 * Abs(mul_fac_old)))
        else:
            sturm_seq.append(simplify(a2 * Abs(mul_fac_old)))
    if flag:
        sturm_seq = [-i for i in sturm_seq]
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)
    return sturm_seq

def sturm_q(p, q, x):
    if False:
        while True:
            i = 10
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the (generalized) Sturm sequence of p and q in Q[x].\n    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).\n\n    The coefficients of the polynomials in the Sturm sequence can be uniquely\n    determined from the corresponding coefficients of the polynomials found\n    either in:\n\n        (a) the ``modified'' subresultant prs, (references 1, 2)\n\n    or in\n\n        (b) the subresultant prs (reference 3).\n\n    References\n    ==========\n    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding\n    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,\n    Second Series, 18 (1917), No. 4, 188-193.\n\n    2 Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n    and Modified Subresultant Polynomial Remainder Sequences.''\n    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (p, q) = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    flag = 0
    if LC(p, x) < 0:
        flag = 1
        p = -p
        q = -q
    (a0, a1) = (p, q)
    sturm_seq = [a0, a1]
    a2 = -rem(a0, a1, domain=QQ)
    d2 = degree(a2, x)
    sturm_seq.append(a2)
    while d2 > 0:
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        a2 = -rem(a0, a1, domain=QQ)
        d2 = degree(a2, x)
        sturm_seq.append(a2)
    if flag:
        sturm_seq = [-i for i in sturm_seq]
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)
    return sturm_seq

def sturm_amv(p, q, x, method=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].\n    If q = diff(p, x, 1) it is the usual Sturm sequence.\n\n    A. If method == 0, default, the remainder coefficients of the\n       sequence are (in absolute value) ``modified\'\' subresultants, which\n       for non-monic polynomials are greater than the coefficients of the\n       corresponding subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).\n\n    B. If method == 1, the remainder coefficients of the sequence are (in\n       absolute value) subresultants, which for non-monic polynomials are\n       smaller than the coefficients of the corresponding ``modified\'\'\n       subresultants by the factor Abs( LC(p)**( deg(p)- deg(q)) ).\n\n    If the Sturm sequence is complete, method=0 and LC( p ) > 0, then the\n    coefficients of the polynomials in the sequence are ``modified\'\' subresultants.\n    That is, they are  determinants of appropriately selected submatrices of\n    sylvester2, Sylvester\'s matrix of 1853. In this case the Sturm sequence\n    coincides with the ``modified\'\' subresultant prs, of the polynomials\n    p, q.\n\n    If the Sturm sequence is incomplete and method=0 then the signs of the\n    coefficients of the polynomials in the sequence may differ from the signs\n    of the coefficients of the corresponding polynomials in the ``modified\'\'\n    subresultant prs; however, the absolute values are the same.\n\n    To compute the coefficients, no determinant evaluation takes place.\n    Instead, we first compute the euclidean sequence  of p and q using\n    euclid_amv(p, q, x) and then: (a) change the signs of the remainders in the\n    Euclidean sequence according to the pattern "-, -, +, +, -, -, +, +,..."\n    (see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference)\n    and (b) if method=0, assuming deg(p) > deg(q), we multiply the remainder\n    coefficients of the Euclidean sequence times the factor\n    Abs( LC(p)**( deg(p)- deg(q)) ) to make them modified subresultants.\n    See also the function sturm_pg(p, q, x).\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.\'\' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders\n    Obtained in Finding the Greatest Common Divisor of Two Polynomials.\'\' Serdica\n    Journal of Computing 9(2) (2015), 123-138.\n\n    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial\n    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].\'\'\n    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.\n\n    '
    prs = euclid_amv(p, q, x)
    if prs == [] or len(prs) == 2:
        return prs
    lcf = Abs(LC(prs[0]) ** (degree(prs[0], x) - degree(prs[1], x)))
    sturm_seq = [prs[0], prs[1]]
    flag = 0
    m = len(prs)
    i = 2
    while i <= m - 1:
        if flag == 0:
            sturm_seq.append(-prs[i])
            i = i + 1
            if i == m:
                break
            sturm_seq.append(-prs[i])
            i = i + 1
            flag = 1
        elif flag == 1:
            sturm_seq.append(prs[i])
            i = i + 1
            if i == m:
                break
            sturm_seq.append(prs[i])
            i = i + 1
            flag = 0
    if method == 0 and lcf > 1:
        aux_seq = [sturm_seq[0], sturm_seq[1]]
        for i in range(2, m):
            aux_seq.append(simplify(sturm_seq[i] * lcf))
        sturm_seq = aux_seq
    return sturm_seq

def euclid_pg(p, q, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the Euclidean sequence of p and q in Z[x] or Q[x].\n\n    If the Euclidean sequence is complete the coefficients of the polynomials\n    in the sequence are subresultants. That is, they are  determinants of\n    appropriately selected submatrices of sylvester1, Sylvester\'s matrix of 1840.\n    In this case the Euclidean sequence coincides with the subresultant prs\n    of the polynomials p, q.\n\n    If the Euclidean sequence is incomplete the signs of the coefficients of the\n    polynomials in the sequence may differ from the signs of the coefficients of\n    the corresponding polynomials in the subresultant prs; however, the absolute\n    values are the same.\n\n    To compute the Euclidean sequence, no determinant evaluation takes place.\n    We first compute the (generalized) Sturm sequence  of p and q using\n    sturm_pg(p, q, x, 1), in which case the coefficients are (in absolute value)\n    equal to subresultants. Then we change the signs of the remainders in the\n    Sturm sequence according to the pattern "-, -, +, +, -, -, +, +,..." ;\n    see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference as well as\n    the function sturm_pg(p, q, x).\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.\'\' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders\n    Obtained in Finding the Greatest Common Divisor of Two Polynomials.\'\' Serdica\n    Journal of Computing 9(2) (2015), 123-138.\n\n    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial\n    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].\'\'\n    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.\n    '
    prs = sturm_pg(p, q, x, 1)
    if prs == [] or len(prs) == 2:
        return prs
    euclid_seq = [prs[0], prs[1]]
    flag = 0
    m = len(prs)
    i = 2
    while i <= m - 1:
        if flag == 0:
            euclid_seq.append(-prs[i])
            i = i + 1
            if i == m:
                break
            euclid_seq.append(-prs[i])
            i = i + 1
            flag = 1
        elif flag == 1:
            euclid_seq.append(prs[i])
            i = i + 1
            if i == m:
                break
            euclid_seq.append(prs[i])
            i = i + 1
            flag = 0
    return euclid_seq

def euclid_q(p, q, x):
    if False:
        return 10
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the Euclidean sequence of p and q in Q[x].\n    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).\n\n    The coefficients of the polynomials in the Euclidean sequence can be uniquely\n    determined from the corresponding coefficients of the polynomials found\n    either in:\n\n        (a) the ``modified'' subresultant polynomial remainder sequence,\n    (references 1, 2)\n\n    or in\n\n        (b) the subresultant polynomial remainder sequence (references 3).\n\n    References\n    ==========\n    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding\n    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,\n    Second Series, 18 (1917), No. 4, 188-193.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n    and Modified Subresultant Polynomial Remainder Sequences.''\n    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (p, q) = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    flag = 0
    if LC(p, x) < 0:
        flag = 1
        p = -p
        q = -q
    (a0, a1) = (p, q)
    euclid_seq = [a0, a1]
    a2 = rem(a0, a1, domain=QQ)
    d2 = degree(a2, x)
    euclid_seq.append(a2)
    while d2 > 0:
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        a2 = rem(a0, a1, domain=QQ)
        d2 = degree(a2, x)
        euclid_seq.append(a2)
    if flag:
        euclid_seq = [-i for i in euclid_seq]
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)
    return euclid_seq

def euclid_amv(f, g, x):
    if False:
        i = 10
        return i + 15
    "\n    f, g are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(f, x) >= degree(g, x).\n\n    Computes the Euclidean sequence of p and q in Z[x] or Q[x].\n\n    If the Euclidean sequence is complete the coefficients of the polynomials\n    in the sequence are subresultants. That is, they are  determinants of\n    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.\n    In this case the Euclidean sequence coincides with the subresultant prs,\n    of the polynomials p, q.\n\n    If the Euclidean sequence is incomplete the signs of the coefficients of the\n    polynomials in the sequence may differ from the signs of the coefficients of\n    the corresponding polynomials in the subresultant prs; however, the absolute\n    values are the same.\n\n    To compute the coefficients, no determinant evaluation takes place.\n    Instead, polynomial divisions in Z[x] or Q[x] are performed, using\n    the function rem_z(f, g, x);  the coefficients of the remainders\n    computed this way become subresultants with the help of the\n    Collins-Brown-Traub formula for coefficient reduction.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial\n    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''\n    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.\n\n    "
    if f == 0 or g == 0:
        return [f, g]
    d0 = degree(f, x)
    d1 = degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (f, g) = (g, f)
    if d0 > 0 and d1 == 0:
        return [f, g]
    a0 = f
    a1 = g
    euclid_seq = [a0, a1]
    (deg_dif_p1, c) = (degree(a0, x) - degree(a1, x) + 1, -1)
    i = 1
    a2 = rem_z(a0, a1, x) / Abs((-1) ** deg_dif_p1)
    euclid_seq.append(a2)
    d2 = degree(a2, x)
    while d2 >= 1:
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        i += 1
        sigma0 = -LC(a0)
        c = sigma0 ** (deg_dif_p1 - 1) / c ** (deg_dif_p1 - 2)
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs(c ** (deg_dif_p1 - 1) * sigma0)
        euclid_seq.append(a2)
        d2 = degree(a2, x)
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)
    return euclid_seq

def modified_subresultants_pg(p, q, x):
    if False:
        i = 10
        return i + 15
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the ``modified'' subresultant prs of p and q in Z[x] or Q[x];\n    the coefficients of the polynomials in the sequence are\n    ``modified'' subresultants. That is, they are  determinants of appropriately\n    selected submatrices of sylvester2, Sylvester's matrix of 1853.\n\n    To compute the coefficients, no determinant evaluation takes place. Instead,\n    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);\n    the coefficients of the remainders computed this way become ``modified''\n    subresultants with the help of the Pell-Gordon Theorem of 1917.\n\n    If the ``modified'' subresultant prs is complete, and LC( p ) > 0, it coincides\n    with the (generalized) Sturm sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding\n    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,\n    Second Series, 18 (1917), No. 4, 188-193.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n    and Modified Subresultant Polynomial Remainder Sequences.''\n    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (p, q) = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    k = var('k')
    u_list = []
    subres_l = [p, q]
    (a0, a1) = (p, q)
    del0 = d0 - d1
    degdif = del0
    rho_1 = LC(a0)
    rho_list_minus_1 = sign(LC(a0, x))
    rho1 = LC(a1, x)
    rho_list = [sign(rho1)]
    p_list = [del0]
    u = summation(k, (k, 1, p_list[0]))
    u_list.append(u)
    v = sum(p_list)
    exp_deg = d1 - 1
    a2 = -rem(a0, a1, domain=QQ)
    rho2 = LC(a2, x)
    d2 = degree(a2, x)
    deg_diff_new = exp_deg - d2
    del1 = d1 - d2
    mul_fac_old = rho1 ** (del0 + del1 - deg_diff_new)
    p_list.append(1 + deg_diff_new)
    num = 1
    for u in u_list:
        num *= (-1) ** u
    num = num * (-1) ** v
    if deg_diff_new == 0:
        den = 1
        for k in range(len(rho_list)):
            den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
    else:
        den = 1
        for k in range(len(rho_list) - 1):
            den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
        expo = p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new
        den = den * rho_list[len(rho_list) - 1] ** expo
    if sign(num / den) > 0:
        subres_l.append(simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
    else:
        subres_l.append(-simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
    k = var('k')
    rho_list.append(sign(rho2))
    u = summation(k, (k, 1, p_list[len(p_list) - 1]))
    u_list.append(u)
    v = sum(p_list)
    deg_diff_old = deg_diff_new
    while d2 > 0:
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        del0 = del1
        exp_deg = d1 - 1
        a2 = -rem(a0, a1, domain=QQ)
        rho3 = LC(a2, x)
        d2 = degree(a2, x)
        deg_diff_new = exp_deg - d2
        del1 = d1 - d2
        expo_old = deg_diff_old
        expo_new = del0 + del1 - deg_diff_new
        mul_fac_new = rho2 ** expo_new * rho1 ** expo_old * mul_fac_old
        (deg_diff_old, mul_fac_old) = (deg_diff_new, mul_fac_new)
        (rho1, rho2) = (rho2, rho3)
        p_list.append(1 + deg_diff_new)
        num = 1
        for u in u_list:
            num *= (-1) ** u
        num = num * (-1) ** v
        if deg_diff_new == 0:
            den = 1
            for k in range(len(rho_list)):
                den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
        else:
            den = 1
            for k in range(len(rho_list) - 1):
                den *= rho_list[k] ** (p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
            expo = p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new
            den = den * rho_list[len(rho_list) - 1] ** expo
        if sign(num / den) > 0:
            subres_l.append(simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
        else:
            subres_l.append(-simplify(rho_1 ** degdif * a2 * Abs(mul_fac_old)))
        k = var('k')
        rho_list.append(sign(rho2))
        u = summation(k, (k, 1, p_list[len(p_list) - 1]))
        u_list.append(u)
        v = sum(p_list)
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)
    m = len(subres_l)
    if LC(p) < 0:
        aux_seq = [subres_l[0], subres_l[1]]
        for i in range(2, m):
            aux_seq.append(simplify(subres_l[i] * -1))
        subres_l = aux_seq
    return subres_l

def subresultants_pg(p, q, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the subresultant prs of p and q in Z[x] or Q[x], from\n    the modified subresultant prs of p and q.\n\n    The coefficients of the polynomials in these two sequences differ only\n    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in\n    Theorem 2 of the reference.\n\n    The coefficients of the polynomials in the output sequence are\n    subresultants. That is, they are  determinants of appropriately\n    selected submatrices of sylvester1, Sylvester\'s matrix of 1840.\n\n    If the subresultant prs is complete, then it coincides with the\n    Euclidean sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders\n    Obtained in Finding the Greatest Common Divisor of Two Polynomials."\n    Serdica Journal of Computing 9(2) (2015), 123-138.\n\n    '
    lst = modified_subresultants_pg(p, q, x)
    if lst == [] or len(lst) == 2:
        return lst
    lcf = LC(lst[0]) ** (degree(lst[0], x) - degree(lst[1], x))
    subr_seq = [lst[0], lst[1]]
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m - 1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]
    fact = [(-1) ** (j * (j - 1) / S(2)) for j in j_seq]
    lst_s = lst[2:]
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append(-lst_s[k] / lcf)
        else:
            subr_seq.append(lst_s[k] / lcf)
    return subr_seq

def subresultants_amv_q(p, q, x):
    if False:
        return 10
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the subresultant prs of p and q in Q[x];\n    the coefficients of the polynomials in the sequence are\n    subresultants. That is, they are  determinants of appropriately\n    selected submatrices of sylvester1, Sylvester's matrix of 1840.\n\n    To compute the coefficients, no determinant evaluation takes place.\n    Instead, polynomial divisions in Q[x] are performed, using the\n    function rem(p, q, x);  the coefficients of the remainders\n    computed this way become subresultants with the help of the\n    Akritas-Malaschonok-Vigklas Theorem of 2015.\n\n    If the subresultant prs is complete, then it coincides with the\n    Euclidean sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial\n    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''\n    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (p, q) = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    (i, s) = (0, 0)
    p_odd_index_sum = 0
    subres_l = [p, q]
    (a0, a1) = (p, q)
    sigma1 = LC(a1, x)
    p0 = d0 - d1
    if p0 % 2 == 1:
        s += 1
    phi = floor((s + 1) / 2)
    mul_fac = 1
    d2 = d1
    while d2 > 0:
        i += 1
        a2 = rem(a0, a1, domain=QQ)
        if i == 1:
            sigma2 = LC(a2, x)
        else:
            sigma3 = LC(a2, x)
            (sigma1, sigma2) = (sigma2, sigma3)
        d2 = degree(a2, x)
        p1 = d1 - d2
        psi = i + phi + p_odd_index_sum
        mul_fac = sigma1 ** (p0 + 1) * mul_fac
        num = (-1) ** psi
        den = sign(mul_fac)
        if sign(num / den) > 0:
            subres_l.append(simplify(expand(a2 * Abs(mul_fac))))
        else:
            subres_l.append(-simplify(expand(a2 * Abs(mul_fac))))
        if p1 - 1 > 0:
            mul_fac = mul_fac * sigma1 ** (p1 - 1)
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        p0 = p1
        if p0 % 2 == 1:
            s += 1
        phi = floor((s + 1) / 2)
        if i % 2 == 1:
            p_odd_index_sum += p0
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)
    return subres_l

def compute_sign(base, expo):
    if False:
        print('Hello World!')
    '\n    base != 0 and expo >= 0 are integers;\n\n    returns the sign of base**expo without\n    evaluating the power itself!\n    '
    sb = sign(base)
    if sb == 1:
        return 1
    pe = expo % 2
    if pe == 0:
        return -sb
    else:
        return sb

def rem_z(p, q, x):
    if False:
        i = 10
        return i + 15
    "\n    Intended mainly for p, q polynomials in Z[x] so that,\n    on dividing p by q, the remainder will also be in Z[x]. (However,\n    it also works fine for polynomials in Q[x].) It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    It premultiplies p by the _absolute_ value of the leading coefficient\n    of q, raised to the power deg(p) - deg(q) + 1 and then performs\n    polynomial division in Q[x], using the function rem(p, q, x).\n\n    By contrast the function prem(p, q, x) does _not_ use the absolute\n    value of the leading coefficient of q.\n    This results not only in ``messing up the signs'' of the Euclidean and\n    Sturmian prs's as mentioned in the second reference,\n    but also in violation of the main results of the first and third\n    references --- Theorem 4 and Theorem 1 respectively. Theorems 4 and 1\n    establish a one-to-one correspondence between the Euclidean and the\n    Sturmian prs of p, q, on one hand, and the subresultant prs of p, q,\n    on the other.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders\n    Obtained in Finding the Greatest Common Divisor of Two Polynomials.''\n    Serdica Journal of Computing, 9(2) (2015), 123-138.\n\n    2. https://planetMath.org/sturmstheorem\n\n    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result on\n    the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    "
    if p.as_poly().is_univariate and q.as_poly().is_univariate and (p.as_poly().gens == q.as_poly().gens):
        delta = degree(p, x) - degree(q, x) + 1
        return rem(Abs(LC(q, x)) ** delta * p, q, x)
    else:
        return prem(p, q, x)

def quo_z(p, q, x):
    if False:
        return 10
    '\n    Intended mainly for p, q polynomials in Z[x] so that,\n    on dividing p by q, the quotient will also be in Z[x]. (However,\n    it also works fine for polynomials in Q[x].) It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    It premultiplies p by the _absolute_ value of the leading coefficient\n    of q, raised to the power deg(p) - deg(q) + 1 and then performs\n    polynomial division in Q[x], using the function quo(p, q, x).\n\n    By contrast the function pquo(p, q, x) does _not_ use the absolute\n    value of the leading coefficient of q.\n\n    See also function rem_z(p, q, x) for additional comments and references.\n\n    '
    if p.as_poly().is_univariate and q.as_poly().is_univariate and (p.as_poly().gens == q.as_poly().gens):
        delta = degree(p, x) - degree(q, x) + 1
        return quo(Abs(LC(q, x)) ** delta * p, q, x)
    else:
        return pquo(p, q, x)

def subresultants_amv(f, g, x):
    if False:
        return 10
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(f, x) >= degree(g, x).\n\n    Computes the subresultant prs of p and q in Z[x] or Q[x];\n    the coefficients of the polynomials in the sequence are\n    subresultants. That is, they are  determinants of appropriately\n    selected submatrices of sylvester1, Sylvester's matrix of 1840.\n\n    To compute the coefficients, no determinant evaluation takes place.\n    Instead, polynomial divisions in Z[x] or Q[x] are performed, using\n    the function rem_z(p, q, x);  the coefficients of the remainders\n    computed this way become subresultants with the help of the\n    Akritas-Malaschonok-Vigklas Theorem of 2015 and the Collins-Brown-\n    Traub formula for coefficient reduction.\n\n    If the subresultant prs is complete, then it coincides with the\n    Euclidean sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result\n    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial\n    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''\n    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.\n\n    "
    if f == 0 or g == 0:
        return [f, g]
    d0 = degree(f, x)
    d1 = degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        (d0, d1) = (d1, d0)
        (f, g) = (g, f)
    if d0 > 0 and d1 == 0:
        return [f, g]
    a0 = f
    a1 = g
    subres_l = [a0, a1]
    (deg_dif_p1, c) = (degree(a0, x) - degree(a1, x) + 1, -1)
    sigma1 = LC(a1, x)
    (i, s) = (0, 0)
    p_odd_index_sum = 0
    p0 = deg_dif_p1 - 1
    if p0 % 2 == 1:
        s += 1
    phi = floor((s + 1) / 2)
    i += 1
    a2 = rem_z(a0, a1, x) / Abs((-1) ** deg_dif_p1)
    sigma2 = LC(a2, x)
    d2 = degree(a2, x)
    p1 = d1 - d2
    sgn_den = compute_sign(sigma1, p0 + 1)
    psi = i + phi + p_odd_index_sum
    num = (-1) ** psi
    den = sgn_den
    if sign(num / den) > 0:
        subres_l.append(a2)
    else:
        subres_l.append(-a2)
    if p1 % 2 == 1:
        s += 1
    if p1 - 1 > 0:
        sgn_den = sgn_den * compute_sign(sigma1, p1 - 1)
    while d2 >= 1:
        phi = floor((s + 1) / 2)
        if i % 2 == 1:
            p_odd_index_sum += p1
        (a0, a1, d0, d1) = (a1, a2, d1, d2)
        p0 = p1
        i += 1
        sigma0 = -LC(a0)
        c = sigma0 ** (deg_dif_p1 - 1) / c ** (deg_dif_p1 - 2)
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs(c ** (deg_dif_p1 - 1) * sigma0)
        sigma3 = LC(a2, x)
        d2 = degree(a2, x)
        p1 = d1 - d2
        psi = i + phi + p_odd_index_sum
        (sigma1, sigma2) = (sigma2, sigma3)
        sgn_den = compute_sign(sigma1, p0 + 1) * sgn_den
        num = (-1) ** psi
        den = sgn_den
        if sign(num / den) > 0:
            subres_l.append(a2)
        else:
            subres_l.append(-a2)
        if p1 % 2 == 1:
            s += 1
        if p1 - 1 > 0:
            sgn_den = sgn_den * compute_sign(sigma1, p1 - 1)
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)
    return subres_l

def modified_subresultants_amv(p, q, x):
    if False:
        return 10
    '\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the modified subresultant prs of p and q in Z[x] or Q[x],\n    from the subresultant prs of p and q.\n    The coefficients of the polynomials in the two sequences differ only\n    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in\n    Theorem 2 of the reference.\n\n    The coefficients of the polynomials in the output sequence are\n    modified subresultants. That is, they are  determinants of appropriately\n    selected submatrices of sylvester2, Sylvester\'s matrix of 1853.\n\n    If the modified subresultant prs is complete, and LC( p ) > 0, it coincides\n    with the (generalized) Sturm\'s sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders\n    Obtained in Finding the Greatest Common Divisor of Two Polynomials."\n    Serdica Journal of Computing, Serdica Journal of Computing, 9(2) (2015), 123-138.\n\n    '
    lst = subresultants_amv(p, q, x)
    if lst == [] or len(lst) == 2:
        return lst
    lcf = LC(lst[0]) ** (degree(lst[0], x) - degree(lst[1], x))
    subr_seq = [lst[0], lst[1]]
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m - 1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]
    fact = [(-1) ** (j * (j - 1) / S(2)) for j in j_seq]
    lst_s = lst[2:]
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append(simplify(-lst_s[k] * lcf))
        else:
            subr_seq.append(simplify(lst_s[k] * lcf))
    return subr_seq

def correct_sign(deg_f, deg_g, s1, rdel, cdel):
    if False:
        i = 10
        return i + 15
    "\n    Used in various subresultant prs algorithms.\n\n    Evaluates the determinant, (a.k.a. subresultant) of a properly selected\n    submatrix of s1, Sylvester's matrix of 1840, to get the correct sign\n    and value of the leading coefficient of a given polynomial remainder.\n\n    deg_f, deg_g are the degrees of the original polynomials p, q for which the\n    matrix s1 = sylvester(p, q, x, 1) was constructed.\n\n    rdel denotes the expected degree of the remainder; it is the number of\n    rows to be deleted from each group of rows in s1 as described in the\n    reference below.\n\n    cdel denotes the expected degree minus the actual degree of the remainder;\n    it is the number of columns to be deleted --- starting with the last column\n    forming the square matrix --- from the matrix resulting after the row deletions.\n\n    References\n    ==========\n    Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences\n    and Modified Subresultant Polynomial Remainder Sequences.''\n    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.\n\n    "
    M = s1[:, :]
    for i in range(M.rows - deg_f - 1, M.rows - deg_f - rdel - 1, -1):
        M.row_del(i)
    for i in range(M.rows - 1, M.rows - rdel - 1, -1):
        M.row_del(i)
    for i in range(cdel):
        M.col_del(M.rows - 1)
    Md = M[:, 0:M.rows]
    return Md.det()

def subresultants_rem(p, q, x):
    if False:
        return 10
    "\n    p, q are polynomials in Z[x] or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the subresultant prs of p and q in Z[x] or Q[x];\n    the coefficients of the polynomials in the sequence are\n    subresultants. That is, they are  determinants of appropriately\n    selected submatrices of sylvester1, Sylvester's matrix of 1840.\n\n    To compute the coefficients polynomial divisions in Q[x] are\n    performed, using the function rem(p, q, x). The coefficients\n    of the remainders computed this way become subresultants by evaluating\n    one subresultant per remainder --- that of the leading coefficient.\n    This way we obtain the correct sign and value of the leading coefficient\n    of the remainder and we easily ``force'' the rest of the coefficients\n    to become subresultants.\n\n    If the subresultant prs is complete, then it coincides with the\n    Euclidean sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G.:``Three New Methods for Computing Subresultant\n    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    (f, g) = (p, q)
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, deg_f, deg_g, f, g) = (m, n, deg_g, deg_f, g, f)
    if n > 0 and m == 0:
        return [f, g]
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]
    while deg_g > 0:
        r = rem(p, q, x)
        d = degree(r, x)
        if d < 0:
            return sr_list
        exp_deg = deg_g - 1
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        r = simplify(r / LC(r, x) * sign_value)
        sr_list.append(r)
        (deg_f, deg_g) = (deg_g, d)
        (p, q) = (q, r)
    m = len(sr_list)
    if sr_list[m - 1] == nan or sr_list[m - 1] == 0:
        sr_list.pop(m - 1)
    return sr_list

def pivot(M, i, j):
    if False:
        while True:
            i = 10
    "\n    M is a matrix, and M[i, j] specifies the pivot element.\n\n    All elements below M[i, j], in the j-th column, will\n    be zeroed, if they are not already 0, according to\n    Dodgson-Bareiss' integer preserving transformations.\n\n    References\n    ==========\n    1. Akritas, A. G.: ``A new method for computing polynomial greatest\n    common divisors and polynomial remainder sequences.''\n    Numerische MatheMatik 52, 119-127, 1988.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem\n    by Van Vleck Regarding Sturm Sequences.''\n    Serdica Journal of Computing, 7, No 4, 101-134, 2013.\n\n    "
    ma = M[:, :]
    rs = ma.rows
    cs = ma.cols
    for r in range(i + 1, rs):
        if ma[r, j] != 0:
            for c in range(j + 1, cs):
                ma[r, c] = ma[i, j] * ma[r, c] - ma[i, c] * ma[r, j]
            ma[r, j] = 0
    return ma

def rotate_r(L, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Rotates right by k. L is a row of a matrix or a list.\n\n    '
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(len(ll) - 1)
        ll.insert(0, el)
    return ll if isinstance(L, list) else Matrix([ll])

def rotate_l(L, k):
    if False:
        i = 10
        return i + 15
    '\n    Rotates left by k. L is a row of a matrix or a list.\n\n    '
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(0)
        ll.insert(len(ll) - 1, el)
    return ll if isinstance(L, list) else Matrix([ll])

def row2poly(row, deg, x):
    if False:
        i = 10
        return i + 15
    '\n    Converts the row of a matrix to a poly of degree deg and variable x.\n    Some entries at the beginning and/or at the end of the row may be zero.\n\n    '
    k = 0
    poly = []
    leng = len(row)
    while row[k] == 0:
        k = k + 1
    for j in range(deg + 1):
        if k + j <= leng:
            poly.append(row[k + j])
    return Poly(poly, x)

def create_ma(deg_f, deg_g, row1, row2, col_num):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a ``small'' matrix M to be triangularized.\n\n    deg_f, deg_g are the degrees of the divident and of the\n    divisor polynomials respectively, deg_g > deg_f.\n\n    The coefficients of the divident poly are the elements\n    in row2 and those of the divisor poly are the elements\n    in row1.\n\n    col_num defines the number of columns of the matrix M.\n\n    "
    if deg_g - deg_f >= 1:
        print('Reverse degrees')
        return
    m = zeros(deg_f - deg_g + 2, col_num)
    for i in range(deg_f - deg_g + 1):
        m[i, :] = rotate_r(row1, i)
    m[deg_f - deg_g + 1, :] = row2
    return m

def find_degree(M, deg_f):
    if False:
        for i in range(10):
            print('nop')
    "\n    Finds the degree of the poly corresponding (after triangularization)\n    to the _last_ row of the ``small'' matrix M, created by create_ma().\n\n    deg_f is the degree of the divident poly.\n    If _last_ row is all 0's returns None.\n\n    "
    j = deg_f
    for i in range(0, M.cols):
        if M[M.rows - 1, i] == 0:
            j = j - 1
        else:
            return j if j >= 0 else 0

def final_touches(s2, r, deg_g):
    if False:
        i = 10
        return i + 15
    "\n    s2 is sylvester2, r is the row pointer in s2,\n    deg_g is the degree of the poly last inserted in s2.\n\n    After a gcd of degree > 0 has been found with Van Vleck's\n    method, and was inserted into s2, if its last term is not\n    in the last column of s2, then it is inserted as many\n    times as needed, rotated right by one each time, until\n    the condition is met.\n\n    "
    R = s2.row(r - 1)
    for i in range(s2.cols):
        if R[0, i] == 0:
            continue
        else:
            break
    mr = s2.cols - (i + deg_g + 1)
    i = 0
    while mr != 0 and r + i < s2.rows:
        s2[r + i, :] = rotate_r(R, i + 1)
        i += 1
        mr -= 1
    return s2

def subresultants_vv(p, q, x, method=0):
    if False:
        print('Hello World!')
    "\n    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the subresultant prs of p, q by triangularizing,\n    in Z[x] or in Q[x], all the smaller matrices encountered in the\n    process of triangularizing sylvester2, Sylvester's matrix of 1853;\n    see references 1 and 2 for Van Vleck's method. With each remainder,\n    sylvester2 gets updated and is prepared to be printed if requested.\n\n    If sylvester2 has small dimensions and you want to see the final,\n    triangularized matrix use this version with method=1; otherwise,\n    use either this version with method=0 (default) or the faster version,\n    subresultants_vv_2(p, q, x), where sylvester2 is used implicitly.\n\n    Sylvester's matrix sylvester1  is also used to compute one\n    subresultant per remainder; namely, that of the leading\n    coefficient, in order to obtain the correct sign and to\n    force the remainder coefficients to become subresultants.\n\n    If the subresultant prs is complete, then it coincides with the\n    Euclidean sequence of the polynomials p, q.\n\n    If the final, triangularized matrix s2 is printed, then:\n        (a) if deg(p) - deg(q) > 1 or deg( gcd(p, q) ) > 0, several\n            of the last rows in s2 will remain unprocessed;\n        (b) if deg(p) - deg(q) == 0, p will not appear in the final matrix.\n\n    References\n    ==========\n    1. Akritas, A. G.: ``A new method for computing polynomial greatest\n    common divisors and polynomial remainder sequences.''\n    Numerische MatheMatik 52, 119-127, 1988.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem\n    by Van Vleck Regarding Sturm Sequences.''\n    Serdica Journal of Computing, 7, No 4, 101-134, 2013.\n\n    3. Akritas, A. G.:``Three New Methods for Computing Subresultant\n    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    (f, g) = (p, q)
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, deg_f, deg_g, f, g) = (m, n, deg_g, deg_f, g, f)
    if n > 0 and m == 0:
        return [f, g]
    s1 = sylvester(f, g, x, 1)
    s2 = sylvester(f, g, x, 2)
    sr_list = [f, g]
    col_num = 2 * n
    row0 = Poly(f, x, domain=QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g, x, domain=QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])
    r = 2
    if deg_f - deg_g > 1:
        r = 1
        for i in range(deg_f - deg_g - 1):
            s2[r + i, :] = rotate_r(row0, i + 1)
        r = r + deg_f - deg_g - 1
        for i in range(deg_f - deg_g):
            s2[r + i, :] = rotate_r(row1, r + i)
        r = r + deg_f - deg_g
    if deg_f - deg_g == 0:
        r = 0
    while deg_g > 0:
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]
        d = find_degree(M, deg_f)
        if d is None:
            break
        exp_deg = deg_g - 1
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        temp2 = LC(poly, x)
        poly = simplify(poly / temp2 * sign_value)
        row0 = M[0, :]
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row0, r + i)
        r = r + deg_g - d
        row1 = rotate_l(M[M.rows - 1, :], deg_f - d)
        row1 = row1 / temp2 * sign_value
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row1, r + i)
        r = r + deg_g - d
        (deg_f, deg_g) = (deg_g, d)
        sr_list.append(poly)
    if method != 0 and s2.rows > 2:
        s2 = final_touches(s2, r, deg_g)
        pprint(s2)
    elif method != 0 and s2.rows == 2:
        s2[1, :] = rotate_r(s2.row(1), 1)
        pprint(s2)
    return sr_list

def subresultants_vv_2(p, q, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed\n    that degree(p, x) >= degree(q, x).\n\n    Computes the subresultant prs of p, q by triangularizing,\n    in Z[x] or in Q[x], all the smaller matrices encountered in the\n    process of triangularizing sylvester2, Sylvester's matrix of 1853;\n    see references 1 and 2 for Van Vleck's method.\n\n    If the sylvester2 matrix has big dimensions use this version,\n    where sylvester2 is used implicitly. If you want to see the final,\n    triangularized matrix sylvester2, then use the first version,\n    subresultants_vv(p, q, x, 1).\n\n    sylvester1, Sylvester's matrix of 1840, is also used to compute\n    one subresultant per remainder; namely, that of the leading\n    coefficient, in order to obtain the correct sign and to\n    ``force'' the remainder coefficients to become subresultants.\n\n    If the subresultant prs is complete, then it coincides with the\n    Euclidean sequence of the polynomials p, q.\n\n    References\n    ==========\n    1. Akritas, A. G.: ``A new method for computing polynomial greatest\n    common divisors and polynomial remainder sequences.''\n    Numerische MatheMatik 52, 119-127, 1988.\n\n    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem\n    by Van Vleck Regarding Sturm Sequences.''\n    Serdica Journal of Computing, 7, No 4, 101-134, 2013.\n\n    3. Akritas, A. G.:``Three New Methods for Computing Subresultant\n    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.\n\n    "
    if p == 0 or q == 0:
        return [p, q]
    (f, g) = (p, q)
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        (n, m, deg_f, deg_g, f, g) = (m, n, deg_g, deg_f, g, f)
    if n > 0 and m == 0:
        return [f, g]
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]
    col_num = 2 * n
    row0 = Poly(f, x, domain=QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g, x, domain=QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])
    while deg_g > 0:
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]
        d = find_degree(M, deg_f)
        if d is None:
            return sr_list
        exp_deg = deg_g - 1
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        poly = simplify(poly / LC(poly, x) * sign_value)
        sr_list.append(poly)
        (deg_f, deg_g) = (deg_g, d)
        row0 = row1
        row1 = Poly(poly, x, domain=QQ).all_coeffs()
        leng1 = len(row1)
        for i in range(col_num - leng1):
            row1.append(0)
        row1 = Matrix([row1])
    return sr_list