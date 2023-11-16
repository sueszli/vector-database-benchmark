from sympy.combinatorics.permutations import Permutation
from sympy.core.symbol import symbols
from sympy.matrices import Matrix
from sympy.utilities.iterables import variations, rotate_left

def symmetric(n):
    if False:
        while True:
            i = 10
    '\n    Generates the symmetric group of order n, Sn.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.generators import symmetric\n    >>> list(symmetric(3))\n    [(2), (1 2), (2)(0 1), (0 1 2), (0 2 1), (0 2)]\n    '
    for perm in variations(range(n), n):
        yield Permutation(perm)

def cyclic(n):
    if False:
        i = 10
        return i + 15
    '\n    Generates the cyclic group of order n, Cn.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.generators import cyclic\n    >>> list(cyclic(5))\n    [(4), (0 1 2 3 4), (0 2 4 1 3),\n     (0 3 1 4 2), (0 4 3 2 1)]\n\n    See Also\n    ========\n\n    dihedral\n    '
    gen = list(range(n))
    for i in range(n):
        yield Permutation(gen)
        gen = rotate_left(gen, 1)

def alternating(n):
    if False:
        while True:
            i = 10
    '\n    Generates the alternating group of order n, An.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.generators import alternating\n    >>> list(alternating(3))\n    [(2), (0 1 2), (0 2 1)]\n    '
    for perm in variations(range(n), n):
        p = Permutation(perm)
        if p.is_even:
            yield p

def dihedral(n):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generates the dihedral group of order 2n, Dn.\n\n    The result is given as a subgroup of Sn, except for the special cases n=1\n    (the group S2) and n=2 (the Klein 4-group) where that's not possible\n    and embeddings in S2 and S4 respectively are given.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.generators import dihedral\n    >>> list(dihedral(3))\n    [(2), (0 2), (0 1 2), (1 2), (0 2 1), (2)(0 1)]\n\n    See Also\n    ========\n\n    cyclic\n    "
    if n == 1:
        yield Permutation([0, 1])
        yield Permutation([1, 0])
    elif n == 2:
        yield Permutation([0, 1, 2, 3])
        yield Permutation([1, 0, 3, 2])
        yield Permutation([2, 3, 0, 1])
        yield Permutation([3, 2, 1, 0])
    else:
        gen = list(range(n))
        for i in range(n):
            yield Permutation(gen)
            yield Permutation(gen[::-1])
            gen = rotate_left(gen, 1)

def rubik_cube_generators():
    if False:
        return 10
    "Return the permutations of the 3x3 Rubik's cube, see\n    https://www.gap-system.org/Doc/Examples/rubik.html\n    "
    a = [[(1, 3, 8, 6), (2, 5, 7, 4), (9, 33, 25, 17), (10, 34, 26, 18), (11, 35, 27, 19)], [(9, 11, 16, 14), (10, 13, 15, 12), (1, 17, 41, 40), (4, 20, 44, 37), (6, 22, 46, 35)], [(17, 19, 24, 22), (18, 21, 23, 20), (6, 25, 43, 16), (7, 28, 42, 13), (8, 30, 41, 11)], [(25, 27, 32, 30), (26, 29, 31, 28), (3, 38, 43, 19), (5, 36, 45, 21), (8, 33, 48, 24)], [(33, 35, 40, 38), (34, 37, 39, 36), (3, 9, 46, 32), (2, 12, 47, 29), (1, 14, 48, 27)], [(41, 43, 48, 46), (42, 45, 47, 44), (14, 22, 30, 38), (15, 23, 31, 39), (16, 24, 32, 40)]]
    return [Permutation([[i - 1 for i in xi] for xi in x], size=48) for x in a]

def rubik(n):
    if False:
        while True:
            i = 10
    "Return permutations for an nxn Rubik's cube.\n\n    Permutations returned are for rotation of each of the slice\n    from the face up to the last face for each of the 3 sides (in this order):\n    front, right and bottom. Hence, the first n - 1 permutations are for the\n    slices from the front.\n    "
    if n < 2:
        raise ValueError('dimension of cube must be > 1')

    def getr(f, i):
        if False:
            while True:
                i = 10
        return faces[f].col(n - i)

    def getl(f, i):
        if False:
            i = 10
            return i + 15
        return faces[f].col(i - 1)

    def getu(f, i):
        if False:
            i = 10
            return i + 15
        return faces[f].row(i - 1)

    def getd(f, i):
        if False:
            for i in range(10):
                print('nop')
        return faces[f].row(n - i)

    def setr(f, i, s):
        if False:
            i = 10
            return i + 15
        faces[f][:, n - i] = Matrix(n, 1, s)

    def setl(f, i, s):
        if False:
            for i in range(10):
                print('nop')
        faces[f][:, i - 1] = Matrix(n, 1, s)

    def setu(f, i, s):
        if False:
            while True:
                i = 10
        faces[f][i - 1, :] = Matrix(1, n, s)

    def setd(f, i, s):
        if False:
            for i in range(10):
                print('nop')
        faces[f][n - i, :] = Matrix(1, n, s)

    def cw(F, r=1):
        if False:
            return 10
        for _ in range(r):
            face = faces[F]
            rv = []
            for c in range(n):
                for r in range(n - 1, -1, -1):
                    rv.append(face[r, c])
            faces[F] = Matrix(n, n, rv)

    def ccw(F):
        if False:
            return 10
        cw(F, 3)

    def fcw(i, r=1):
        if False:
            while True:
                i = 10
        for _ in range(r):
            if i == 0:
                cw(F)
            i += 1
            temp = getr(L, i)
            setr(L, i, list(getu(D, i)))
            setu(D, i, list(reversed(getl(R, i))))
            setl(R, i, list(getd(U, i)))
            setd(U, i, list(reversed(temp)))
            i -= 1

    def fccw(i):
        if False:
            print('Hello World!')
        fcw(i, 3)

    def FCW(r=1):
        if False:
            return 10
        for _ in range(r):
            cw(F)
            ccw(B)
            cw(U)
            t = faces[U]
            cw(L)
            faces[U] = faces[L]
            cw(D)
            faces[L] = faces[D]
            cw(R)
            faces[D] = faces[R]
            faces[R] = t

    def FCCW():
        if False:
            print('Hello World!')
        FCW(3)

    def UCW(r=1):
        if False:
            i = 10
            return i + 15
        for _ in range(r):
            cw(U)
            ccw(D)
            t = faces[F]
            faces[F] = faces[R]
            faces[R] = faces[B]
            faces[B] = faces[L]
            faces[L] = t

    def UCCW():
        if False:
            while True:
                i = 10
        UCW(3)
    (U, F, R, B, L, D) = names = symbols('U, F, R, B, L, D')
    faces = {}
    count = 0
    for fi in range(6):
        f = []
        for a in range(n ** 2):
            f.append(count)
            count += 1
        faces[names[fi]] = Matrix(n, n, f)

    def perm(show=0):
        if False:
            print('Hello World!')
        p = []
        for f in names:
            p.extend(faces[f])
        if show:
            return p
        g.append(Permutation(p))
    g = []
    I = list(range(6 * n ** 2))
    for i in range(n - 1):
        fcw(i)
        perm()
        fccw(i)
    assert perm(1) == I
    UCW()
    for i in range(n - 1):
        fcw(i)
        UCCW()
        perm()
        UCW()
        fccw(i)
    UCCW()
    assert perm(1) == I
    FCW()
    UCCW()
    FCCW()
    for i in range(n - 1):
        fcw(i)
        FCW()
        UCW()
        FCCW()
        perm()
        FCW()
        UCCW()
        FCCW()
        fccw(i)
    FCW()
    UCW()
    FCCW()
    assert perm(1) == I
    return g