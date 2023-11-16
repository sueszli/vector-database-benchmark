"""

Module for the DDM class.

The DDM class is an internal representation used by DomainMatrix. The letters
DDM stand for Dense Domain Matrix. A DDM instance represents a matrix using
elements from a polynomial Domain (e.g. ZZ, QQ, ...) in a dense-matrix
representation.

Basic usage:

    >>> from sympy import ZZ, QQ
    >>> from sympy.polys.matrices.ddm import DDM
    >>> A = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> A.shape
    (2, 2)
    >>> A
    [[0, 1], [-1, 0]]
    >>> type(A)
    <class 'sympy.polys.matrices.ddm.DDM'>
    >>> A @ A
    [[-1, 0], [0, -1]]

The ddm_* functions are designed to operate on DDM as well as on an ordinary
list of lists:

    >>> from sympy.polys.matrices.dense import ddm_idet
    >>> ddm_idet(A, QQ)
    1
    >>> ddm_idet([[0, 1], [-1, 0]], QQ)
    1
    >>> A
    [[-1, 0], [0, -1]]

Note that ddm_idet modifies the input matrix in-place. It is recommended to
use the DDM.det method as a friendlier interface to this instead which takes
care of copying the matrix:

    >>> B = DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)
    >>> B.det()
    1

Normally DDM would not be used directly and is just part of the internal
representation of DomainMatrix which adds further functionality including e.g.
unifying domains.

The dense format used by DDM is a list of lists of elements e.g. the 2x2
identity matrix is like [[1, 0], [0, 1]]. The DDM class itself is a subclass
of list and its list items are plain lists. Elements are accessed as e.g.
ddm[i][j] where ddm[i] gives the ith row and ddm[i][j] gets the element in the
jth column of that row. Subclassing list makes e.g. iteration and indexing
very efficient. We do not override __getitem__ because it would lose that
benefit.

The core routines are implemented by the ddm_* functions defined in dense.py.
Those functions are intended to be able to operate on a raw list-of-lists
representation of matrices with most functions operating in-place. The DDM
class takes care of copying etc and also stores a Domain object associated
with its elements. This makes it possible to implement things like A + B with
domain checking and also shape checking so that the list of lists
representation is friendlier.

"""
from itertools import chain
from sympy.utilities.decorator import doctest_depends_on
from .exceptions import DMBadInputError, DMDomainError, DMNonSquareMatrixError, DMShapeError
from sympy.polys.domains import QQ
from .dense import ddm_transpose, ddm_iadd, ddm_isub, ddm_ineg, ddm_imul, ddm_irmul, ddm_imatmul, ddm_irref, ddm_irref_den, ddm_idet, ddm_iinv, ddm_ilu_split, ddm_ilu_solve, ddm_berk
from .lll import ddm_lll, ddm_lll_transform

class DDM(list):
    """Dense matrix based on polys domain elements

    This is a list subclass and is a wrapper for a list of lists that supports
    basic matrix arithmetic +, -, *, **.
    """
    fmt = 'dense'
    is_DFM = False
    is_DDM = True

    def __init__(self, rowslist, shape, domain):
        if False:
            print('Hello World!')
        if not (isinstance(rowslist, list) and all((type(row) is list for row in rowslist))):
            raise DMBadInputError('rowslist must be a list of lists')
        (m, n) = shape
        if len(rowslist) != m or any((len(row) != n for row in rowslist)):
            raise DMBadInputError('Inconsistent row-list/shape')
        super().__init__(rowslist)
        self.shape = (m, n)
        self.rows = m
        self.cols = n
        self.domain = domain

    def getitem(self, i, j):
        if False:
            while True:
                i = 10
        return self[i][j]

    def setitem(self, i, j, value):
        if False:
            for i in range(10):
                print('nop')
        self[i][j] = value

    def extract_slice(self, slice1, slice2):
        if False:
            i = 10
            return i + 15
        ddm = [row[slice2] for row in self[slice1]]
        rows = len(ddm)
        cols = len(ddm[0]) if ddm else len(range(self.shape[1])[slice2])
        return DDM(ddm, (rows, cols), self.domain)

    def extract(self, rows, cols):
        if False:
            print('Hello World!')
        ddm = []
        for i in rows:
            rowi = self[i]
            ddm.append([rowi[j] for j in cols])
        return DDM(ddm, (len(rows), len(cols)), self.domain)

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        if False:
            return 10
        '\n        Create a :class:`DDM` from a list of lists.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> A = DDM.from_list([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)\n        >>> A\n        [[0, 1], [-1, 0]]\n        >>> A == DDM([[ZZ(0), ZZ(1)], [ZZ(-1), ZZ(0)]], (2, 2), ZZ)\n        True\n\n        See Also\n        ========\n\n        from_list_flat\n        '
        return cls(rowslist, shape, domain)

    @classmethod
    def from_ddm(cls, other):
        if False:
            while True:
                i = 10
        return other.copy()

    def to_list(self):
        if False:
            return 10
        '\n        Convert to a list of lists.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> A.to_list()\n        [[1, 2], [3, 4]]\n\n        See Also\n        ========\n\n        to_list_flat\n        '
        return list(self)

    def to_list_flat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert to a flat list of elements.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> A.to_list_flat()\n        [1, 2, 3, 4]\n        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        from_list_flat\n        '
        flat = []
        for row in self:
            flat.extend(row)
        return flat

    @classmethod
    def from_list_flat(cls, flat, shape, domain):
        if False:
            while True:
                i = 10
        '\n        Create a :class:`DDM` from a flat list of elements.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> A = DDM.from_list_flat([1, 2, 3, 4], (2, 2), QQ)\n        >>> A\n        [[1, 2], [3, 4]]\n        >>> A == DDM.from_list_flat(A.to_list_flat(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        to_list_flat\n        '
        assert type(flat) is list
        (rows, cols) = shape
        if not len(flat) == rows * cols:
            raise DMBadInputError('Inconsistent flat-list shape')
        lol = [flat[i * cols:(i + 1) * cols] for i in range(rows)]
        return cls(lol, shape, domain)

    def flatiter(self):
        if False:
            for i in range(10):
                print('nop')
        return chain.from_iterable(self)

    def flat(self):
        if False:
            while True:
                i = 10
        items = []
        for row in self:
            items.extend(row)
        return items

    def to_flat_nz(self):
        if False:
            return 10
        '\n        Convert to a flat list of nonzero elements and data.\n\n        Explanation\n        ===========\n\n        This is used to operate on a list of the elements of a matrix and then\n        reconstruct a matrix using :meth:`from_flat_nz`. Zero elements are\n        included in the list but that may change in the future.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> elements, data = A.to_flat_nz()\n        >>> elements\n        [1, 2, 3, 4]\n        >>> A == DDM.from_flat_nz(elements, data, A.domain)\n        True\n\n        See Also\n        ========\n\n        from_flat_nz\n        sympy.polys.matrices.sdm.SDM.to_flat_nz\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz\n        '
        return self.to_sdm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        if False:
            return 10
        '\n        Reconstruct a :class:`DDM` after calling :meth:`to_flat_nz`.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> elements, data = A.to_flat_nz()\n        >>> elements\n        [1, 2, 3, 4]\n        >>> A == DDM.from_flat_nz(elements, data, A.domain)\n        True\n\n        See Also\n        ========\n\n        to_flat_nz\n        sympy.polys.matrices.sdm.SDM.from_flat_nz\n        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz\n        '
        return SDM.from_flat_nz(elements, data, domain).to_ddm()

    def to_dok(self):
        if False:
            print('Hello World!')
        '\n        Convert :class:`DDM` to dictionary of keys (dok) format.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> A.to_dok()\n        {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}\n\n        See Also\n        ========\n\n        from_dok\n        sympy.polys.matrices.sdm.SDM.to_dok\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dok\n        '
        dok = {}
        for (i, row) in enumerate(self):
            for (j, element) in enumerate(row):
                if element:
                    dok[i, j] = element
        return dok

    @classmethod
    def from_dok(cls, dok, shape, domain):
        if False:
            print('Hello World!')
        '\n        Create a :class:`DDM` from a dictionary of keys (dok) format.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> dok = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}\n        >>> A = DDM.from_dok(dok, (2, 2), QQ)\n        >>> A\n        [[1, 2], [3, 4]]\n\n        See Also\n        ========\n\n        to_dok\n        sympy.polys.matrices.sdm.SDM.from_dok\n        sympy.polys.matrices.domainmatrix.DomainMatrix.from_dok\n        '
        (rows, cols) = shape
        lol = [[domain.zero] * cols for _ in range(rows)]
        for ((i, j), element) in dok.items():
            lol[i][j] = element
        return DDM(lol, shape, domain)

    def to_ddm(self):
        if False:
            return 10
        '\n        Convert to a :class:`DDM`.\n\n        This just returns ``self`` but exists to parallel the corresponding\n        method in other matrix types like :class:`~.SDM`.\n\n        See Also\n        ========\n\n        to_sdm\n        to_dfm\n        to_dfm_or_ddm\n        sympy.polys.matrices.sdm.SDM.to_ddm\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_ddm\n        '
        return self

    def to_sdm(self):
        if False:
            i = 10
            return i + 15
        "\n        Convert to a :class:`~.SDM`.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> A.to_sdm()\n        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}\n        >>> type(A.to_sdm())\n        <class 'sympy.polys.matrices.sdm.SDM'>\n\n        See Also\n        ========\n\n        SDM\n        sympy.polys.matrices.sdm.SDM.to_ddm\n        "
        return SDM.from_list(self, self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        if False:
            return 10
        "\n        Convert to :class:`~.DDM` to :class:`~.DFM`.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> A.to_dfm()\n        [[1, 2], [3, 4]]\n        >>> type(A.to_dfm())\n        <class 'sympy.polys.matrices._dfm.DFM'>\n\n        See Also\n        ========\n\n        DFM\n        sympy.polys.matrices._dfm.DFM.to_ddm\n        "
        return DFM(list(self), self.shape, self.domain)

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        if False:
            print('Hello World!')
        "\n        Convert to :class:`~.DFM` if possible or otherwise return self.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy import QQ\n        >>> A = DDM([[1, 2], [3, 4]], (2, 2), QQ)\n        >>> A.to_dfm_or_ddm()\n        [[1, 2], [3, 4]]\n        >>> type(A.to_dfm_or_ddm())\n        <class 'sympy.polys.matrices._dfm.DFM'>\n\n        See Also\n        ========\n\n        to_dfm\n        to_ddm\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm\n        "
        if DFM._supports_domain(self.domain):
            return self.to_dfm()
        return self

    def convert_to(self, K):
        if False:
            return 10
        Kold = self.domain
        if K == Kold:
            return self.copy()
        rows = [[K.convert_from(e, Kold) for e in row] for row in self]
        return DDM(rows, self.shape, K)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        rowsstr = ['[%s]' % ', '.join(map(str, row)) for row in self]
        return '[%s]' % ', '.join(rowsstr)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        cls = type(self).__name__
        rows = list.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, DDM):
            return False
        return super().__eq__(other) and self.domain == other.domain

    def __ne__(self, other):
        if False:
            return 10
        return not self.__eq__(other)

    @classmethod
    def zeros(cls, shape, domain):
        if False:
            for i in range(10):
                print('nop')
        z = domain.zero
        (m, n) = shape
        rowslist = [[z] * n for _ in range(m)]
        return DDM(rowslist, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        if False:
            for i in range(10):
                print('nop')
        one = domain.one
        (m, n) = shape
        rowlist = [[one] * n for _ in range(m)]
        return DDM(rowlist, shape, domain)

    @classmethod
    def eye(cls, size, domain):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(size, tuple):
            (m, n) = size
        elif isinstance(size, int):
            m = n = size
        one = domain.one
        ddm = cls.zeros((m, n), domain)
        for i in range(min(m, n)):
            ddm[i][i] = one
        return ddm

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        copyrows = [row[:] for row in self]
        return DDM(copyrows, self.shape, self.domain)

    def transpose(self):
        if False:
            i = 10
            return i + 15
        (rows, cols) = self.shape
        if rows:
            ddmT = ddm_transpose(self)
        else:
            ddmT = [[]] * cols
        return DDM(ddmT, (cols, rows), self.domain)

    def __add__(a, b):
        if False:
            i = 10
            return i + 15
        if not isinstance(b, DDM):
            return NotImplemented
        return a.add(b)

    def __sub__(a, b):
        if False:
            while True:
                i = 10
        if not isinstance(b, DDM):
            return NotImplemented
        return a.sub(b)

    def __neg__(a):
        if False:
            while True:
                i = 10
        return a.neg()

    def __mul__(a, b):
        if False:
            while True:
                i = 10
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if False:
            while True:
                i = 10
        if b in a.domain:
            return a.mul(b)
        else:
            return NotImplemented

    def __matmul__(a, b):
        if False:
            i = 10
            return i + 15
        if isinstance(b, DDM):
            return a.matmul(b)
        else:
            return NotImplemented

    @classmethod
    def _check(cls, a, op, b, ashape, bshape):
        if False:
            return 10
        if a.domain != b.domain:
            msg = 'Domain mismatch: %s %s %s' % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = 'Shape mismatch: %s %s %s' % (a.shape, op, b.shape)
            raise DMShapeError(msg)

    def add(a, b):
        if False:
            i = 10
            return i + 15
        'a + b'
        a._check(a, '+', b, a.shape, b.shape)
        c = a.copy()
        ddm_iadd(c, b)
        return c

    def sub(a, b):
        if False:
            return 10
        'a - b'
        a._check(a, '-', b, a.shape, b.shape)
        c = a.copy()
        ddm_isub(c, b)
        return c

    def neg(a):
        if False:
            for i in range(10):
                print('nop')
        '-a'
        b = a.copy()
        ddm_ineg(b)
        return b

    def mul(a, b):
        if False:
            return 10
        c = a.copy()
        ddm_imul(c, b)
        return c

    def rmul(a, b):
        if False:
            i = 10
            return i + 15
        c = a.copy()
        ddm_irmul(c, b)
        return c

    def matmul(a, b):
        if False:
            while True:
                i = 10
        'a @ b (matrix product)'
        (m, o) = a.shape
        (o2, n) = b.shape
        a._check(a, '*', b, o, o2)
        c = a.zeros((m, n), a.domain)
        ddm_imatmul(c, a, b)
        return c

    def mul_elementwise(a, b):
        if False:
            i = 10
            return i + 15
        assert a.shape == b.shape
        assert a.domain == b.domain
        c = [[aij * bij for (aij, bij) in zip(ai, bi)] for (ai, bi) in zip(a, b)]
        return DDM(c, a.shape, a.domain)

    def hstack(A, *B):
        if False:
            return 10
        'Horizontally stacks :py:class:`~.DDM` matrices.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import DDM\n\n        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)\n        >>> A.hstack(B)\n        [[1, 2, 5, 6], [3, 4, 7, 8]]\n\n        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)\n        >>> A.hstack(B, C)\n        [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]\n        '
        Anew = list(A.copy())
        (rows, cols) = A.shape
        domain = A.domain
        for Bk in B:
            (Bkrows, Bkcols) = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain
            cols += Bkcols
            for (i, Bki) in enumerate(Bk):
                Anew[i].extend(Bki)
        return DDM(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        if False:
            return 10
        'Vertically stacks :py:class:`~.DDM` matrices.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import DDM\n\n        >>> A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DDM([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)\n        >>> A.vstack(B)\n        [[1, 2], [3, 4], [5, 6], [7, 8]]\n\n        >>> C = DDM([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)\n        >>> A.vstack(B, C)\n        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]\n        '
        Anew = list(A.copy())
        (rows, cols) = A.shape
        domain = A.domain
        for Bk in B:
            (Bkrows, Bkcols) = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain
            rows += Bkrows
            Anew.extend(Bk.copy())
        return DDM(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        if False:
            while True:
                i = 10
        elements = [list(map(func, row)) for row in self]
        return DDM(elements, self.shape, domain)

    def nnz(a):
        if False:
            for i in range(10):
                print('nop')
        'Number of non-zero entries in :py:class:`~.DDM` matrix.\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz\n        '
        return sum((sum(map(bool, row)) for row in a))

    def scc(a):
        if False:
            i = 10
            return i + 15
        'Strongly connected components of a square matrix *a*.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import DDM\n        >>> A = DDM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], (2, 2), ZZ)\n        >>> A.scc()\n        [[0], [1]]\n\n        See also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.scc\n\n        '
        return a.to_sdm().scc()

    @classmethod
    def diag(cls, values, domain):
        if False:
            print('Hello World!')
        'Returns a square diagonal matrix with *values* on the diagonal.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import DDM\n        >>> DDM.diag([ZZ(1), ZZ(2), ZZ(3)], ZZ)\n        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]\n\n        See also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.diag\n        '
        return SDM.diag(values, domain).to_ddm()

    def rref(a):
        if False:
            print('Hello World!')
        'Reduced-row echelon form of a and list of pivots.\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.rref\n            Higher level interface to this function.\n        sympy.polys.matrices.dense.ddm_irref\n            The underlying algorithm.\n        '
        b = a.copy()
        K = a.domain
        partial_pivot = K.is_RealField or K.is_ComplexField
        pivots = ddm_irref(b, _partial_pivot=partial_pivot)
        return (b, pivots)

    def rref_den(a):
        if False:
            print('Hello World!')
        'Reduced-row echelon form of a with denominator and list of pivots\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den\n            Higher level interface to this function.\n        sympy.polys.matrices.dense.ddm_irref_den\n            The underlying algorithm.\n        '
        b = a.copy()
        K = a.domain
        (denom, pivots) = ddm_irref_den(b, K)
        return (b, denom, pivots)

    def nullspace(a):
        if False:
            i = 10
            return i + 15
        'Returns a basis for the nullspace of a.\n\n        The domain of the matrix must be a field.\n\n        See Also\n        ========\n\n        rref\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace\n        '
        (rref, pivots) = a.rref()
        return rref.nullspace_from_rref(pivots)

    def nullspace_from_rref(a, pivots=None):
        if False:
            while True:
                i = 10
        'Compute the nullspace of a matrix from its rref.\n\n        The domain of the matrix can be any domain.\n\n        Returns a tuple (basis, nonpivots).\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace\n            The higher level interface to this function.\n        '
        (m, n) = a.shape
        K = a.domain
        if pivots is None:
            pivots = []
            last_pivot = -1
            for i in range(m):
                ai = a[i]
                for j in range(last_pivot + 1, n):
                    if ai[j]:
                        last_pivot = j
                        pivots.append(j)
                        break
        if not pivots:
            return (a.eye(n, K), list(range(n)))
        pivot_val = a[0][pivots[0]]
        basis = []
        nonpivots = []
        for i in range(n):
            if i in pivots:
                continue
            nonpivots.append(i)
            vec = [pivot_val if i == j else K.zero for j in range(n)]
            for (ii, jj) in enumerate(pivots):
                vec[jj] -= a[ii][i]
            basis.append(vec)
        basis_ddm = DDM(basis, (len(basis), n), K)
        return (basis_ddm, nonpivots)

    def particular(a):
        if False:
            for i in range(10):
                print('nop')
        return a.to_sdm().particular().to_ddm()

    def det(a):
        if False:
            while True:
                i = 10
        'Determinant of a'
        (m, n) = a.shape
        if m != n:
            raise DMNonSquareMatrixError('Determinant of non-square matrix')
        b = a.copy()
        K = b.domain
        deta = ddm_idet(b, K)
        return deta

    def inv(a):
        if False:
            while True:
                i = 10
        'Inverse of a'
        (m, n) = a.shape
        if m != n:
            raise DMNonSquareMatrixError('Determinant of non-square matrix')
        ainv = a.copy()
        K = a.domain
        ddm_iinv(ainv, a, K)
        return ainv

    def lu(a):
        if False:
            for i in range(10):
                print('nop')
        'L, U decomposition of a'
        (m, n) = a.shape
        K = a.domain
        U = a.copy()
        L = a.eye(m, K)
        swaps = ddm_ilu_split(L, U, K)
        return (L, U, swaps)

    def lu_solve(a, b):
        if False:
            print('Hello World!')
        'x where a*x = b'
        (m, n) = a.shape
        (m2, o) = b.shape
        a._check(a, 'lu_solve', b, m, m2)
        if not a.domain.is_Field:
            raise DMDomainError('lu_solve requires a field')
        (L, U, swaps) = a.lu()
        x = a.zeros((n, o), a.domain)
        ddm_ilu_solve(x, L, U, swaps, b)
        return x

    def charpoly(a):
        if False:
            i = 10
            return i + 15
        'Coefficients of characteristic polynomial of a'
        K = a.domain
        (m, n) = a.shape
        if m != n:
            raise DMNonSquareMatrixError('Charpoly of non-square matrix')
        vec = ddm_berk(a, K)
        coeffs = [vec[i][0] for i in range(n + 1)]
        return coeffs

    def is_zero_matrix(self):
        if False:
            return 10
        '\n        Says whether this matrix has all zero entries.\n        '
        zero = self.domain.zero
        return all((Mij == zero for Mij in self.flatiter()))

    def is_upper(self):
        if False:
            i = 10
            return i + 15
        '\n        Says whether this matrix is upper-triangular. True can be returned\n        even if the matrix is not square.\n        '
        zero = self.domain.zero
        return all((Mij == zero for (i, Mi) in enumerate(self) for Mij in Mi[:i]))

    def is_lower(self):
        if False:
            print('Hello World!')
        '\n        Says whether this matrix is lower-triangular. True can be returned\n        even if the matrix is not square.\n        '
        zero = self.domain.zero
        return all((Mij == zero for (i, Mi) in enumerate(self) for Mij in Mi[i + 1:]))

    def is_diagonal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Says whether this matrix is diagonal. True can be returned even if\n        the matrix is not square.\n        '
        return self.is_upper() and self.is_lower()

    def diagonal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of the elements from the diagonal of the matrix.\n        '
        (m, n) = self.shape
        return [self[i][i] for i in range(min(m, n))]

    def lll(A, delta=QQ(3, 4)):
        if False:
            i = 10
            return i + 15
        return ddm_lll(A, delta=delta)

    def lll_transform(A, delta=QQ(3, 4)):
        if False:
            return 10
        return ddm_lll_transform(A, delta=delta)
from .sdm import SDM
from .dfm import DFM