"""

Module for the SDM class.

"""
from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from sympy.polys.domains import QQ
from .ddm import DDM

class SDM(dict):
    """Sparse matrix based on polys domain elements

    This is a dict subclass and is a wrapper for a dict of dicts that supports
    basic matrix arithmetic +, -, *, **.


    In order to create a new :py:class:`~.SDM`, a dict
    of dicts mapping non-zero elements to their
    corresponding row and column in the matrix is needed.

    We also need to specify the shape and :py:class:`~.Domain`
    of our :py:class:`~.SDM` object.

    We declare a 2x2 :py:class:`~.SDM` matrix belonging
    to QQ domain as shown below.
    The 2x2 Matrix in the example is

    .. math::
           A = \\left[\\begin{array}{ccc}
                0 & \\frac{1}{2} \\\\
                0 & 0 \\end{array} \\right]


    >>> from sympy.polys.matrices.sdm import SDM
    >>> from sympy import QQ
    >>> elemsdict = {0:{1:QQ(1, 2)}}
    >>> A = SDM(elemsdict, (2, 2), QQ)
    >>> A
    {0: {1: 1/2}}

    We can manipulate :py:class:`~.SDM` the same way
    as a Matrix class

    >>> from sympy import ZZ
    >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)
    >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)
    >>> A + B
    {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}

    Multiplication

    >>> A*B
    {0: {1: 8}, 1: {0: 3}}
    >>> A*ZZ(2)
    {0: {1: 4}, 1: {0: 2}}

    """
    fmt = 'sparse'
    is_DFM = False
    is_DDM = False

    def __init__(self, elemsdict, shape, domain):
        if False:
            while True:
                i = 10
        super().__init__(elemsdict)
        self.shape = (self.rows, self.cols) = (m, n) = shape
        self.domain = domain
        if not all((0 <= r < m for r in self)):
            raise DMBadInputError('Row out of range')
        if not all((0 <= c < n for row in self.values() for c in row)):
            raise DMBadInputError('Column out of range')

    def getitem(self, i, j):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self[i][j]
        except KeyError:
            (m, n) = self.shape
            if -m <= i < m and -n <= j < n:
                try:
                    return self[i % m][j % n]
                except KeyError:
                    return self.domain.zero
            else:
                raise IndexError('index out of range')

    def setitem(self, i, j, value):
        if False:
            while True:
                i = 10
        (m, n) = self.shape
        if not (-m <= i < m and -n <= j < n):
            raise IndexError('index out of range')
        (i, j) = (i % m, j % n)
        if value:
            try:
                self[i][j] = value
            except KeyError:
                self[i] = {j: value}
        else:
            rowi = self.get(i, None)
            if rowi is not None:
                try:
                    del rowi[j]
                except KeyError:
                    pass
                else:
                    if not rowi:
                        del self[i]

    def extract_slice(self, slice1, slice2):
        if False:
            return 10
        (m, n) = self.shape
        ri = range(m)[slice1]
        ci = range(n)[slice2]
        sdm = {}
        for (i, row) in self.items():
            if i in ri:
                row = {ci.index(j): e for (j, e) in row.items() if j in ci}
                if row:
                    sdm[ri.index(i)] = row
        return self.new(sdm, (len(ri), len(ci)), self.domain)

    def extract(self, rows, cols):
        if False:
            i = 10
            return i + 15
        if not (self and rows and cols):
            return self.zeros((len(rows), len(cols)), self.domain)
        (m, n) = self.shape
        if not -m <= min(rows) <= max(rows) < m:
            raise IndexError('Row index out of range')
        if not -n <= min(cols) <= max(cols) < n:
            raise IndexError('Column index out of range')
        rowmap = defaultdict(list)
        colmap = defaultdict(list)
        for (i2, i1) in enumerate(rows):
            rowmap[i1 % m].append(i2)
        for (j2, j1) in enumerate(cols):
            colmap[j1 % n].append(j2)
        rowset = set(rowmap)
        colset = set(colmap)
        sdm1 = self
        sdm2 = {}
        for i1 in rowset & sdm1.keys():
            row1 = sdm1[i1]
            row2 = {}
            for j1 in colset & row1.keys():
                row1_j1 = row1[j1]
                for j2 in colmap[j1]:
                    row2[j2] = row1_j1
            if row2:
                for i2 in rowmap[i1]:
                    sdm2[i2] = row2.copy()
        return self.new(sdm2, (len(rows), len(cols)), self.domain)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        rowsstr = []
        for (i, row) in self.items():
            elemsstr = ', '.join(('%s: %s' % (j, elem) for (j, elem) in row.items()))
            rowsstr.append('%s: {%s}' % (i, elemsstr))
        return '{%s}' % ', '.join(rowsstr)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        cls = type(self).__name__
        rows = dict.__repr__(self)
        return '%s(%s, %s, %s)' % (cls, rows, self.shape, self.domain)

    @classmethod
    def new(cls, sdm, shape, domain):
        if False:
            return 10
        '\n\n        Parameters\n        ==========\n\n        sdm: A dict of dicts for non-zero elements in SDM\n        shape: tuple representing dimension of SDM\n        domain: Represents :py:class:`~.Domain` of SDM\n\n        Returns\n        =======\n\n        An :py:class:`~.SDM` object\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> elemsdict = {0:{1: QQ(2)}}\n        >>> A = SDM.new(elemsdict, (2, 2), QQ)\n        >>> A\n        {0: {1: 2}}\n\n        '
        return cls(sdm, shape, domain)

    def copy(A):
        if False:
            while True:
                i = 10
        '\n        Returns the copy of a :py:class:`~.SDM` object\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}\n        >>> A = SDM(elemsdict, (2, 2), QQ)\n        >>> B = A.copy()\n        >>> B\n        {0: {1: 2}, 1: {}}\n\n        '
        Ac = {i: Ai.copy() for (i, Ai) in A.items()}
        return A.new(Ac, A.shape, A.domain)

    @classmethod
    def from_list(cls, ddm, shape, domain):
        if False:
            i = 10
            return i + 15
        '\n        Create :py:class:`~.SDM` object from a list of lists.\n\n        Parameters\n        ==========\n\n        ddm:\n            list of lists containing domain elements\n        shape:\n            Dimensions of :py:class:`~.SDM` matrix\n        domain:\n            Represents :py:class:`~.Domain` of :py:class:`~.SDM` object\n\n        Returns\n        =======\n\n        :py:class:`~.SDM` containing elements of ddm\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> ddm = [[QQ(1, 2), QQ(0)], [QQ(0), QQ(3, 4)]]\n        >>> A = SDM.from_list(ddm, (2, 2), QQ)\n        >>> A\n        {0: {0: 1/2}, 1: {1: 3/4}}\n\n        See Also\n        ========\n\n        to_list\n        from_list_flat\n        from_dok\n        from_ddm\n        '
        (m, n) = shape
        if not (len(ddm) == m and all((len(row) == n for row in ddm))):
            raise DMBadInputError('Inconsistent row-list/shape')
        getrow = lambda i: {j: ddm[i][j] for j in range(n) if ddm[i][j]}
        irows = ((i, getrow(i)) for i in range(m))
        sdm = {i: row for (i, row) in irows if row}
        return cls(sdm, shape, domain)

    @classmethod
    def from_ddm(cls, ddm):
        if False:
            print('Hello World!')
        '\n        Create :py:class:`~.SDM` from a :py:class:`~.DDM`.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)\n        >>> A = SDM.from_ddm(ddm)\n        >>> A\n        {0: {0: 1/2}, 1: {1: 3/4}}\n        >>> SDM.from_ddm(ddm).to_ddm() == ddm\n        True\n\n        See Also\n        ========\n\n        to_ddm\n        from_list\n        from_list_flat\n        from_dok\n        '
        return cls.from_list(ddm, ddm.shape, ddm.domain)

    def to_list(M):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert a :py:class:`~.SDM` object to a list of lists.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> elemsdict = {0:{1:QQ(2)}, 1:{}}\n        >>> A = SDM(elemsdict, (2, 2), QQ)\n        >>> A.to_list()\n        [[0, 2], [0, 0]]\n\n\n        '
        (m, n) = M.shape
        zero = M.domain.zero
        ddm = [[zero] * n for _ in range(m)]
        for (i, row) in M.items():
            for (j, e) in row.items():
                ddm[i][j] = e
        return ddm

    def to_list_flat(M):
        if False:
            return 10
        '\n        Convert :py:class:`~.SDM` to a flat list.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)\n        >>> A.to_list_flat()\n        [0, 2, 3, 0]\n        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        from_list_flat\n        to_list\n        to_dok\n        to_ddm\n        '
        (m, n) = M.shape
        zero = M.domain.zero
        flat = [zero] * (m * n)
        for (i, row) in M.items():
            for (j, e) in row.items():
                flat[i * n + j] = e
        return flat

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        if False:
            return 10
        '\n        Create :py:class:`~.SDM` from a flat list of elements.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM.from_list_flat([QQ(0), QQ(2), QQ(0), QQ(0)], (2, 2), QQ)\n        >>> A\n        {0: {1: 2}}\n        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        to_list_flat\n        from_list\n        from_dok\n        from_ddm\n        '
        (m, n) = shape
        if len(elements) != m * n:
            raise DMBadInputError('Inconsistent flat-list shape')
        sdm = defaultdict(dict)
        for (inj, element) in enumerate(elements):
            if element:
                (i, j) = divmod(inj, n)
                sdm[i][j] = element
        return cls(sdm, shape, domain)

    def to_flat_nz(M):
        if False:
            return 10
        '\n        Convert :class:`SDM` to a flat list of nonzero elements and data.\n\n        Explanation\n        ===========\n\n        This is used to operate on a list of the elements of a matrix and then\n        reconstruct a modified matrix with elements in the same positions using\n        :meth:`from_flat_nz`. Zero elements are omitted from the list.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0:{1:QQ(2)}, 1:{0: QQ(3)}}, (2, 2), QQ)\n        >>> elements, data = A.to_flat_nz()\n        >>> elements\n        [2, 3]\n        >>> A == A.from_flat_nz(elements, data, A.domain)\n        True\n\n        See Also\n        ========\n\n        from_flat_nz\n        to_list_flat\n        sympy.polys.matrices.ddm.DDM.to_flat_nz\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_flat_nz\n        '
        dok = M.to_dok()
        indices = tuple(dok)
        elements = list(dok.values())
        data = (indices, M.shape)
        return (elements, data)

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        if False:
            while True:
                i = 10
        '\n        Reconstruct a :class:`~.SDM` after calling :meth:`to_flat_nz`.\n\n        See :meth:`to_flat_nz` for explanation.\n\n        See Also\n        ========\n\n        to_flat_nz\n        from_list_flat\n        sympy.polys.matrices.ddm.DDM.from_flat_nz\n        sympy.polys.matrices.domainmatrix.DomainMatrix.from_flat_nz\n        '
        (indices, shape) = data
        dok = dict(zip(indices, elements))
        return cls.from_dok(dok, shape, domain)

    def to_dok(M):
        if False:
            print('Hello World!')
        '\n        Convert to dictionary of keys (dok) format.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3)}}, (2, 2), QQ)\n        >>> A.to_dok()\n        {(0, 1): 2, (1, 0): 3}\n\n        See Also\n        ========\n\n        from_dok\n        to_list\n        to_list_flat\n        to_ddm\n        '
        return {(i, j): e for (i, row) in M.items() for (j, e) in row.items()}

    @classmethod
    def from_dok(cls, dok, shape, domain):
        if False:
            print('Hello World!')
        '\n        Create :py:class:`~.SDM` from dictionary of keys (dok) format.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> dok = {(0, 1): QQ(2), (1, 0): QQ(3)}\n        >>> A = SDM.from_dok(dok, (2, 2), QQ)\n        >>> A\n        {0: {1: 2}, 1: {0: 3}}\n        >>> A == SDM.from_dok(A.to_dok(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        to_dok\n        from_list\n        from_list_flat\n        from_ddm\n        '
        sdm = defaultdict(dict)
        for ((i, j), e) in dok.items():
            if e:
                sdm[i][j] = e
        return cls(sdm, shape, domain)

    def to_ddm(M):
        if False:
            while True:
                i = 10
        '\n        Convert a :py:class:`~.SDM` object to a :py:class:`~.DDM` object\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)\n        >>> A.to_ddm()\n        [[0, 2], [0, 0]]\n\n        '
        return DDM(M.to_list(), M.shape, M.domain)

    def to_sdm(M):
        if False:
            i = 10
            return i + 15
        '\n        Convert to :py:class:`~.SDM` format (returns self).\n        '
        return M

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(M):
        if False:
            i = 10
            return i + 15
        '\n        Convert a :py:class:`~.SDM` object to a :py:class:`~.DFM` object\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)\n        >>> A.to_dfm()\n        [[0, 2], [0, 0]]\n\n        See Also\n        ========\n\n        to_ddm\n        to_dfm_or_ddm\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm\n        '
        return M.to_ddm().to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(M):
        if False:
            print('Hello World!')
        "\n        Convert to :py:class:`~.DFM` if possible, else :py:class:`~.DDM`.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)\n        >>> A.to_dfm_or_ddm()\n        [[0, 2], [0, 0]]\n        >>> type(A.to_dfm_or_ddm())  # depends on the ground types\n        <class 'sympy.polys.matrices._dfm.DFM'>\n\n        See Also\n        ========\n\n        to_ddm\n        to_dfm\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm\n        "
        return M.to_ddm().to_dfm_or_ddm()

    @classmethod
    def zeros(cls, shape, domain):
        if False:
            print('Hello World!')
        '\n\n        Returns a :py:class:`~.SDM` of size shape,\n        belonging to the specified domain\n\n        In the example below we declare a matrix A where,\n\n        .. math::\n            A := \\left[\\begin{array}{ccc}\n            0 & 0 & 0 \\\\\n            0 & 0 & 0 \\end{array} \\right]\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM.zeros((2, 3), QQ)\n        >>> A\n        {}\n\n        '
        return cls({}, shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        if False:
            return 10
        one = domain.one
        (m, n) = shape
        row = dict(zip(range(n), [one] * n))
        sdm = {i: row.copy() for i in range(m)}
        return cls(sdm, shape, domain)

    @classmethod
    def eye(cls, shape, domain):
        if False:
            return 10
        '\n\n        Returns a identity :py:class:`~.SDM` matrix of dimensions\n        size x size, belonging to the specified domain\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> I = SDM.eye((2, 2), QQ)\n        >>> I\n        {0: {0: 1}, 1: {1: 1}}\n\n        '
        if isinstance(shape, int):
            (rows, cols) = (shape, shape)
        else:
            (rows, cols) = shape
        one = domain.one
        sdm = {i: {i: one} for i in range(min(rows, cols))}
        return cls(sdm, (rows, cols), domain)

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        if False:
            print('Hello World!')
        if shape is None:
            shape = (len(diagonal), len(diagonal))
        sdm = {i: {i: v} for (i, v) in enumerate(diagonal) if v}
        return cls(sdm, shape, domain)

    def transpose(M):
        if False:
            while True:
                i = 10
        '\n\n        Returns the transpose of a :py:class:`~.SDM` matrix\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import QQ\n        >>> A = SDM({0:{1:QQ(2)}, 1:{}}, (2, 2), QQ)\n        >>> A.transpose()\n        {1: {0: 2}}\n\n        '
        MT = sdm_transpose(M)
        return M.new(MT, M.shape[::-1], M.domain)

    def __add__(A, B):
        if False:
            i = 10
            return i + 15
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError('Matrix size mismatch: %s + %s' % (A.shape, B.shape))
        return A.add(B)

    def __sub__(A, B):
        if False:
            while True:
                i = 10
        if not isinstance(B, SDM):
            return NotImplemented
        elif A.shape != B.shape:
            raise DMShapeError('Matrix size mismatch: %s - %s' % (A.shape, B.shape))
        return A.sub(B)

    def __neg__(A):
        if False:
            return 10
        return A.neg()

    def __mul__(A, B):
        if False:
            return 10
        'A * B'
        if isinstance(B, SDM):
            return A.matmul(B)
        elif B in A.domain:
            return A.mul(B)
        else:
            return NotImplemented

    def __rmul__(a, b):
        if False:
            while True:
                i = 10
        if b in a.domain:
            return a.rmul(b)
        else:
            return NotImplemented

    def matmul(A, B):
        if False:
            while True:
                i = 10
        '\n        Performs matrix multiplication of two SDM matrices\n\n        Parameters\n        ==========\n\n        A, B: SDM to multiply\n\n        Returns\n        =======\n\n        SDM\n            SDM after multiplication\n\n        Raises\n        ======\n\n        DomainError\n            If domain of A does not match\n            with that of B\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> B = SDM({0:{0:ZZ(2), 1:ZZ(3)}, 1:{0:ZZ(4)}}, (2, 2), ZZ)\n        >>> A.matmul(B)\n        {0: {0: 8}, 1: {0: 2, 1: 3}}\n\n        '
        if A.domain != B.domain:
            raise DMDomainError
        (m, n) = A.shape
        (n2, o) = B.shape
        if n != n2:
            raise DMShapeError
        C = sdm_matmul(A, B, A.domain, m, o)
        return A.new(C, (m, o), A.domain)

    def mul(A, b):
        if False:
            while True:
                i = 10
        '\n        Multiplies each element of A with a scalar b\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> A.mul(ZZ(3))\n        {0: {1: 6}, 1: {0: 3}}\n\n        '
        Csdm = unop_dict(A, lambda aij: aij * b)
        return A.new(Csdm, A.shape, A.domain)

    def rmul(A, b):
        if False:
            print('Hello World!')
        Csdm = unop_dict(A, lambda aij: b * aij)
        return A.new(Csdm, A.shape, A.domain)

    def mul_elementwise(A, B):
        if False:
            while True:
                i = 10
        if A.domain != B.domain:
            raise DMDomainError
        if A.shape != B.shape:
            raise DMShapeError
        zero = A.domain.zero
        fzero = lambda e: zero
        Csdm = binop_dict(A, B, mul, fzero, fzero)
        return A.new(Csdm, A.shape, A.domain)

    def add(A, B):
        if False:
            i = 10
            return i + 15
        '\n\n        Adds two :py:class:`~.SDM` matrices\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> B = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)\n        >>> A.add(B)\n        {0: {0: 3, 1: 2}, 1: {0: 1, 1: 4}}\n\n        '
        Csdm = binop_dict(A, B, add, pos, pos)
        return A.new(Csdm, A.shape, A.domain)

    def sub(A, B):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Subtracts two :py:class:`~.SDM` matrices\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> B  = SDM({0:{0: ZZ(3)}, 1:{1:ZZ(4)}}, (2, 2), ZZ)\n        >>> A.sub(B)\n        {0: {0: -3, 1: 2}, 1: {0: 1, 1: -4}}\n\n        '
        Csdm = binop_dict(A, B, sub, pos, neg)
        return A.new(Csdm, A.shape, A.domain)

    def neg(A):
        if False:
            i = 10
            return i + 15
        '\n\n        Returns the negative of a :py:class:`~.SDM` matrix\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> A.neg()\n        {0: {1: -2}, 1: {0: -1}}\n\n        '
        Csdm = unop_dict(A, neg)
        return A.new(Csdm, A.shape, A.domain)

    def convert_to(A, K):
        if False:
            print('Hello World!')
        '\n        Converts the :py:class:`~.Domain` of a :py:class:`~.SDM` matrix to K\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ, QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> A.convert_to(QQ)\n        {0: {1: 2}, 1: {0: 1}}\n\n        '
        Kold = A.domain
        if K == Kold:
            return A.copy()
        Ak = unop_dict(A, lambda e: K.convert_from(e, Kold))
        return A.new(Ak, A.shape, K)

    def nnz(A):
        if False:
            while True:
                i = 10
        'Number of non-zero elements in the :py:class:`~.SDM` matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{1: ZZ(2)}, 1:{0:ZZ(1)}}, (2, 2), ZZ)\n        >>> A.nnz()\n        2\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nnz\n        '
        return sum(map(len, A.values()))

    def scc(A):
        if False:
            for i in range(10):
                print('nop')
        'Strongly connected components of a square matrix *A*.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0: ZZ(2)}, 1:{1:ZZ(1)}}, (2, 2), ZZ)\n        >>> A.scc()\n        [[0], [1]]\n\n        See also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.scc\n        '
        (rows, cols) = A.shape
        assert rows == cols
        V = range(rows)
        Emap = {v: list(A.get(v, [])) for v in V}
        return _strongly_connected_components(V, Emap)

    def rref(A):
        if False:
            while True:
                i = 10
        '\n\n        Returns reduced-row echelon form and list of pivots for the :py:class:`~.SDM`\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)\n        >>> A.rref()\n        ({0: {0: 1, 1: 2}}, [0])\n\n        '
        (B, pivots, _) = sdm_irref(A)
        return (A.new(B, A.shape, A.domain), pivots)

    def rref_den(A):
        if False:
            return 10
        '\n\n        Returns reduced-row echelon form (RREF) with denominator and pivots.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(2), 1:QQ(4)}}, (2, 2), QQ)\n        >>> A.rref_den()\n        ({0: {0: 1, 1: 2}}, 1, [0])\n\n        '
        K = A.domain
        (A_rref_sdm, denom, pivots) = sdm_rref_den(A, K)
        A_rref = A.new(A_rref_sdm, A.shape, A.domain)
        return (A_rref, denom, pivots)

    def inv(A):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Returns inverse of a matrix A\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)\n        >>> A.inv()\n        {0: {0: -2, 1: 1}, 1: {0: 3/2, 1: -1/2}}\n\n        '
        return A.to_dfm_or_ddm().inv().to_sdm()

    def det(A):
        if False:
            print('Hello World!')
        '\n        Returns determinant of A\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)\n        >>> A.det()\n        -2\n\n        '
        return A.to_dfm_or_ddm().det()

    def lu(A):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Returns LU decomposition for a matrix A\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)\n        >>> A.lu()\n        ({0: {0: 1}, 1: {0: 3, 1: 1}}, {0: {0: 1, 1: 2}, 1: {1: -2}}, [])\n\n        '
        (L, U, swaps) = A.to_ddm().lu()
        return (A.from_ddm(L), A.from_ddm(U), swaps)

    def lu_solve(A, b):
        if False:
            return 10
        '\n\n        Uses LU decomposition to solve Ax = b,\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)\n        >>> b = SDM({0:{0:QQ(1)}, 1:{0:QQ(2)}}, (2, 1), QQ)\n        >>> A.lu_solve(b)\n        {1: {0: 1/2}}\n\n        '
        return A.from_ddm(A.to_ddm().lu_solve(b.to_ddm()))

    def nullspace(A):
        if False:
            print('Hello World!')
        '\n        Nullspace of a :py:class:`~.SDM` matrix A.\n\n        The domain of the matrix must be a field.\n\n        It is better to use the :meth:`~.DomainMatrix.nullspace` method rather\n        than this method which is otherwise no longer used.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)\n        >>> A.nullspace()\n        ({0: {0: -2, 1: 1}}, [1])\n\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace\n            The preferred way to get the nullspace of a matrix.\n\n        '
        ncols = A.shape[1]
        one = A.domain.one
        (B, pivots, nzcols) = sdm_irref(A)
        (K, nonpivots) = sdm_nullspace_from_rref(B, one, ncols, pivots, nzcols)
        K = dict(enumerate(K))
        shape = (len(K), ncols)
        return (A.new(K, shape, A.domain), nonpivots)

    def nullspace_from_rref(A, pivots=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns nullspace for a :py:class:`~.SDM` matrix ``A`` in RREF.\n\n        The domain of the matrix can be any domain.\n\n        The matrix must already be in reduced row echelon form (RREF).\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0: QQ(2), 1: QQ(4)}}, (2, 2), QQ)\n        >>> A_rref, pivots = A.rref()\n        >>> A_null, nonpivots = A_rref.nullspace_from_rref(pivots)\n        >>> A_null\n        {0: {0: -2, 1: 1}}\n        >>> pivots\n        [0]\n        >>> nonpivots\n        [1]\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace\n            The higher-level function that would usually be called instead of\n            calling this one directly.\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.nullspace_from_rref\n            The higher-level direct equivalent of this function.\n\n        sympy.polys.matrices.ddm.DDM.nullspace_from_rref\n            The equivalent function for dense :py:class:`~.DDM` matrices.\n\n        '
        (m, n) = A.shape
        K = A.domain
        if pivots is None:
            pivots = sorted(map(min, A.values()))
        if not pivots:
            return (A.eye((n, n), K), list(range(n)))
        elif len(pivots) == n:
            return (A.zeros((0, n), K), [])
        pivot_val = A[0][pivots[0]]
        assert not K.is_zero(pivot_val)
        pivots_set = set(pivots)
        nonzero_cols = defaultdict(list)
        for (i, Ai) in A.items():
            for (j, Aij) in Ai.items():
                nonzero_cols[j].append((i, Aij))
        basis = []
        nonpivots = []
        for j in range(n):
            if j in pivots_set:
                continue
            nonpivots.append(j)
            vec = {j: pivot_val}
            for (ip, Aij) in nonzero_cols[j]:
                vec[pivots[ip]] = -Aij
            basis.append(vec)
        sdm = dict(enumerate(basis))
        A_null = A.new(sdm, (len(basis), n), K)
        return (A_null, nonpivots)

    def particular(A):
        if False:
            i = 10
            return i + 15
        ncols = A.shape[1]
        (B, pivots, nzcols) = sdm_irref(A)
        P = sdm_particular_from_rref(B, ncols, pivots)
        rep = {0: P} if P else {}
        return A.new(rep, (1, ncols - 1), A.domain)

    def hstack(A, *B):
        if False:
            while True:
                i = 10
        'Horizontally stacks :py:class:`~.SDM` matrices.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n\n        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)\n        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)\n        >>> A.hstack(B)\n        {0: {0: 1, 1: 2, 2: 5, 3: 6}, 1: {0: 3, 1: 4, 2: 7, 3: 8}}\n\n        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)\n        >>> A.hstack(B, C)\n        {0: {0: 1, 1: 2, 2: 5, 3: 6, 4: 9, 5: 10}, 1: {0: 3, 1: 4, 2: 7, 3: 8, 4: 11, 5: 12}}\n        '
        Anew = dict(A.copy())
        (rows, cols) = A.shape
        domain = A.domain
        for Bk in B:
            (Bkrows, Bkcols) = Bk.shape
            assert Bkrows == rows
            assert Bk.domain == domain
            for (i, Bki) in Bk.items():
                Ai = Anew.get(i, None)
                if Ai is None:
                    Anew[i] = Ai = {}
                for (j, Bkij) in Bki.items():
                    Ai[j + cols] = Bkij
            cols += Bkcols
        return A.new(Anew, (rows, cols), A.domain)

    def vstack(A, *B):
        if False:
            print('Hello World!')
        'Vertically stacks :py:class:`~.SDM` matrices.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices.sdm import SDM\n\n        >>> A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)\n        >>> B = SDM({0: {0: ZZ(5), 1: ZZ(6)}, 1: {0: ZZ(7), 1: ZZ(8)}}, (2, 2), ZZ)\n        >>> A.vstack(B)\n        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}}\n\n        >>> C = SDM({0: {0: ZZ(9), 1: ZZ(10)}, 1: {0: ZZ(11), 1: ZZ(12)}}, (2, 2), ZZ)\n        >>> A.vstack(B, C)\n        {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}, 2: {0: 5, 1: 6}, 3: {0: 7, 1: 8}, 4: {0: 9, 1: 10}, 5: {0: 11, 1: 12}}\n        '
        Anew = dict(A.copy())
        (rows, cols) = A.shape
        domain = A.domain
        for Bk in B:
            (Bkrows, Bkcols) = Bk.shape
            assert Bkcols == cols
            assert Bk.domain == domain
            for (i, Bki) in Bk.items():
                Anew[i + rows] = Bki
            rows += Bkrows
        return A.new(Anew, (rows, cols), A.domain)

    def applyfunc(self, func, domain):
        if False:
            i = 10
            return i + 15
        sdm = {i: {j: func(e) for (j, e) in row.items()} for (i, row) in self.items()}
        return self.new(sdm, self.shape, domain)

    def charpoly(A):
        if False:
            return 10
        "\n        Returns the coefficients of the characteristic polynomial\n        of the :py:class:`~.SDM` matrix. These elements will be domain elements.\n        The domain of the elements will be same as domain of the :py:class:`~.SDM`.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ, Symbol\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy.polys import Poly\n        >>> A = SDM({0:{0:QQ(1), 1:QQ(2)}, 1:{0:QQ(3), 1:QQ(4)}}, (2, 2), QQ)\n        >>> A.charpoly()\n        [1, -5, -2]\n\n        We can create a polynomial using the\n        coefficients using :py:class:`~.Poly`\n\n        >>> x = Symbol('x')\n        >>> p = Poly(A.charpoly(), x, domain=A.domain)\n        >>> p\n        Poly(x**2 - 5*x - 2, x, domain='QQ')\n\n        "
        K = A.domain
        (n, _) = A.shape
        pdict = sdm_berk(A, n, K)
        plist = [K.zero] * (n + 1)
        for (i, pi) in pdict.items():
            plist[i] = pi
        return plist

    def is_zero_matrix(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Says whether this matrix has all zero entries.\n        '
        return not self

    def is_upper(self):
        if False:
            i = 10
            return i + 15
        '\n        Says whether this matrix is upper-triangular. True can be returned\n        even if the matrix is not square.\n        '
        return all((i <= j for (i, row) in self.items() for j in row))

    def is_lower(self):
        if False:
            return 10
        '\n        Says whether this matrix is lower-triangular. True can be returned\n        even if the matrix is not square.\n        '
        return all((i >= j for (i, row) in self.items() for j in row))

    def is_diagonal(self):
        if False:
            return 10
        '\n        Says whether this matrix is diagonal. True can be returned\n        even if the matrix is not square.\n        '
        return all((i == j for (i, row) in self.items() for j in row))

    def diagonal(self):
        if False:
            return 10
        '\n        Returns the diagonal of the matrix as a list.\n        '
        (m, n) = self.shape
        zero = self.domain.zero
        return [row.get(i, zero) for (i, row) in self.items() if i < n]

    def lll(A, delta=QQ(3, 4)):
        if False:
            print('Hello World!')
        '\n        Returns the LLL-reduced basis for the :py:class:`~.SDM` matrix.\n        '
        return A.to_dfm_or_ddm().lll(delta=delta).to_sdm()

    def lll_transform(A, delta=QQ(3, 4)):
        if False:
            i = 10
            return i + 15
        '\n        Returns the LLL-reduced basis and transformation matrix.\n        '
        (reduced, transform) = A.to_dfm_or_ddm().lll_transform(delta=delta)
        return (reduced.to_sdm(), transform.to_sdm())

def binop_dict(A, B, fab, fa, fb):
    if False:
        i = 10
        return i + 15
    (Anz, Bnz) = (set(A), set(B))
    C = {}
    for i in Anz & Bnz:
        (Ai, Bi) = (A[i], B[i])
        Ci = {}
        (Anzi, Bnzi) = (set(Ai), set(Bi))
        for j in Anzi & Bnzi:
            Cij = fab(Ai[j], Bi[j])
            if Cij:
                Ci[j] = Cij
        for j in Anzi - Bnzi:
            Cij = fa(Ai[j])
            if Cij:
                Ci[j] = Cij
        for j in Bnzi - Anzi:
            Cij = fb(Bi[j])
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    for i in Anz - Bnz:
        Ai = A[i]
        Ci = {}
        for (j, Aij) in Ai.items():
            Cij = fa(Aij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    for i in Bnz - Anz:
        Bi = B[i]
        Ci = {}
        for (j, Bij) in Bi.items():
            Cij = fb(Bij)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    return C

def unop_dict(A, f):
    if False:
        i = 10
        return i + 15
    B = {}
    for (i, Ai) in A.items():
        Bi = {}
        for (j, Aij) in Ai.items():
            Bij = f(Aij)
            if Bij:
                Bi[j] = Bij
        if Bi:
            B[i] = Bi
    return B

def sdm_transpose(M):
    if False:
        for i in range(10):
            print('nop')
    MT = {}
    for (i, Mi) in M.items():
        for (j, Mij) in Mi.items():
            try:
                MT[j][i] = Mij
            except KeyError:
                MT[j] = {i: Mij}
    return MT

def sdm_dotvec(A, B, K):
    if False:
        for i in range(10):
            print('nop')
    return K.sum((A[j] * B[j] for j in A.keys() & B.keys()))

def sdm_matvecmul(A, B, K):
    if False:
        i = 10
        return i + 15
    C = {}
    for (i, Ai) in A.items():
        Ci = sdm_dotvec(Ai, B, K)
        if Ci:
            C[i] = Ci
    return C

def sdm_matmul(A, B, K, m, o):
    if False:
        while True:
            i = 10
    if K.is_EXRAW:
        return sdm_matmul_exraw(A, B, K, m, o)
    C = {}
    B_knz = set(B)
    for (i, Ai) in A.items():
        Ci = {}
        Ai_knz = set(Ai)
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            for (j, Bkj) in B[k].items():
                Cij = Ci.get(j, None)
                if Cij is not None:
                    Cij = Cij + Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
                    else:
                        Ci.pop(j)
                else:
                    Cij = Aik * Bkj
                    if Cij:
                        Ci[j] = Cij
        if Ci:
            C[i] = Ci
    return C

def sdm_matmul_exraw(A, B, K, m, o):
    if False:
        i = 10
        return i + 15
    zero = K.zero
    C = {}
    B_knz = set(B)
    for (i, Ai) in A.items():
        Ci_list = defaultdict(list)
        Ai_knz = set(Ai)
        for k in Ai_knz & B_knz:
            Aik = Ai[k]
            if zero * Aik == zero:
                for (j, Bkj) in B[k].items():
                    Ci_list[j].append(Aik * Bkj)
            else:
                for j in range(o):
                    Ci_list[j].append(Aik * B[k].get(j, zero))
        for k in Ai_knz - B_knz:
            zAik = zero * Ai[k]
            if zAik != zero:
                for j in range(o):
                    Ci_list[j].append(zAik)
        Ci = {}
        for (j, Cij_list) in Ci_list.items():
            Cij = K.sum(Cij_list)
            if Cij:
                Ci[j] = Cij
        if Ci:
            C[i] = Ci
    for (k, Bk) in B.items():
        for (j, Bkj) in Bk.items():
            if zero * Bkj != zero:
                for i in range(m):
                    Aik = A.get(i, {}).get(k, zero)
                    if Aik == zero:
                        Ci = C.get(i, {})
                        Cij = Ci.get(j, zero) + Aik * Bkj
                        if Cij != zero:
                            Ci[j] = Cij
                        else:
                            raise RuntimeError
                        C[i] = Ci
    return C

def sdm_irref(A):
    if False:
        while True:
            i = 10
    'RREF and pivots of a sparse matrix *A*.\n\n    Compute the reduced row echelon form (RREF) of the matrix *A* and return a\n    list of the pivot columns. This routine does not work in place and leaves\n    the original matrix *A* unmodified.\n\n    The domain of the matrix must be a field.\n\n    Examples\n    ========\n\n    This routine works with a dict of dicts sparse representation of a matrix:\n\n    >>> from sympy import QQ\n    >>> from sympy.polys.matrices.sdm import sdm_irref\n    >>> A = {0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}\n    >>> Arref, pivots, _ = sdm_irref(A)\n    >>> Arref\n    {0: {0: 1}, 1: {1: 1}}\n    >>> pivots\n    [0, 1]\n\n    The analogous calculation with :py:class:`~.MutableDenseMatrix` would be\n\n    >>> from sympy import Matrix\n    >>> M = Matrix([[1, 2], [3, 4]])\n    >>> Mrref, pivots = M.rref()\n    >>> Mrref\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> pivots\n    (0, 1)\n\n    Notes\n    =====\n\n    The cost of this algorithm is determined purely by the nonzero elements of\n    the matrix. No part of the cost of any step in this algorithm depends on\n    the number of rows or columns in the matrix. No step depends even on the\n    number of nonzero rows apart from the primary loop over those rows. The\n    implementation is much faster than ddm_rref for sparse matrices. In fact\n    at the time of writing it is also (slightly) faster than the dense\n    implementation even if the input is a fully dense matrix so it seems to be\n    faster in all cases.\n\n    The elements of the matrix should support exact division with ``/``. For\n    example elements of any domain that is a field (e.g. ``QQ``) should be\n    fine. No attempt is made to handle inexact arithmetic.\n\n    See Also\n    ========\n\n    sympy.polys.matrices.domainmatrix.DomainMatrix.rref\n        The higher-level function that would normally be used to call this\n        routine.\n    sympy.polys.matrices.dense.ddm_irref\n        The dense equivalent of this routine.\n    sdm_rref_den\n        Fraction-free version of this routine.\n    '
    Arows = sorted((Ai.copy() for Ai in A.values()), key=min)
    pivot_row_map = {}
    reduced_pivots = set()
    nonreduced_pivots = set()
    nonzero_columns = defaultdict(set)
    while Arows:
        Ai = Arows.pop()
        Ai = {j: Aij for (j, Aij) in Ai.items() if j not in reduced_pivots}
        for j in nonreduced_pivots & set(Ai):
            Aj = pivot_row_map[j]
            Aij = Ai[j]
            Ainz = set(Ai)
            Ajnz = set(Aj)
            for k in Ajnz - Ainz:
                Ai[k] = -Aij * Aj[k]
            Ai.pop(j)
            Ainz.remove(j)
            for k in Ajnz & Ainz:
                Aik = Ai[k] - Aij * Aj[k]
                if Aik:
                    Ai[k] = Aik
                else:
                    Ai.pop(k)
        if not Ai:
            continue
        j = min(Ai)
        Aij = Ai[j]
        pivot_row_map[j] = Ai
        Ainz = set(Ai)
        Aijinv = Aij ** (-1)
        for l in Ai:
            Ai[l] *= Aijinv
        for k in nonzero_columns.pop(j, ()):
            Ak = pivot_row_map[k]
            Akj = Ak[j]
            Aknz = set(Ak)
            for l in Ainz - Aknz:
                Ak[l] = -Akj * Ai[l]
                nonzero_columns[l].add(k)
            Ak.pop(j)
            Aknz.remove(j)
            for l in Ainz & Aknz:
                Akl = Ak[l] - Akj * Ai[l]
                if Akl:
                    Ak[l] = Akl
                else:
                    Ak.pop(l)
                    if l != j:
                        nonzero_columns[l].remove(k)
            if len(Ak) == 1:
                reduced_pivots.add(k)
                nonreduced_pivots.remove(k)
        if len(Ai) == 1:
            reduced_pivots.add(j)
        else:
            nonreduced_pivots.add(j)
            for l in Ai:
                if l != j:
                    nonzero_columns[l].add(j)
    pivots = sorted(reduced_pivots | nonreduced_pivots)
    pivot2row = {p: n for (n, p) in enumerate(pivots)}
    nonzero_columns = {c: {pivot2row[p] for p in s} for (c, s) in nonzero_columns.items()}
    rows = [pivot_row_map[i] for i in pivots]
    rref = dict(enumerate(rows))
    return (rref, pivots, nonzero_columns)

def sdm_rref_den(A, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the reduced row echelon form (RREF) of A with denominator.\n\n    The RREF is computed using fraction-free Gauss-Jordan elimination.\n\n    Explanation\n    ===========\n\n    The algorithm used is the fraction-free version of Gauss-Jordan elimination\n    described as FFGJ in [1]_. Here it is modified to handle zero or missing\n    pivots and to avoid redundant arithmetic. This implementation is also\n    optimized for sparse matrices.\n\n    The domain $K$ must support exact division (``K.exquo``) but does not need\n    to be a field. This method is suitable for most exact rings and fields like\n    :ref:`ZZ`, :ref:`QQ` and :ref:`QQ(a)`. In the case of :ref:`QQ` or\n    :ref:`K(x)` it might be more efficient to clear denominators and use\n    :ref:`ZZ` or :ref:`K[x]` instead.\n\n    For inexact domains like :ref:`RR` and :ref:`CC` use ``ddm_irref`` instead.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.matrices.sdm import sdm_rref_den\n    >>> from sympy.polys.domains import ZZ\n    >>> A = {0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}\n    >>> A_rref, den, pivots = sdm_rref_den(A, ZZ)\n    >>> A_rref\n    {0: {0: -2}, 1: {1: -2}}\n    >>> den\n    -2\n    >>> pivots\n    [0, 1]\n\n    See Also\n    ========\n\n    sympy.polys.matrices.domainmatrix.DomainMatrix.rref_den\n        Higher-level interface to ``sdm_rref_den`` that would usually be used\n        instead of calling this function directly.\n    sympy.polys.matrices.sdm.sdm_rref_den\n        The ``SDM`` method that uses this function.\n    sdm_irref\n        Computes RREF using field division.\n    ddm_irref_den\n        The dense version of this algorithm.\n\n    References\n    ==========\n\n    .. [1] Fraction-free algorithms for linear and polynomial equations.\n        George C. Nakos , Peter R. Turner , Robert M. Williams.\n        https://dl.acm.org/doi/10.1145/271130.271133\n    '
    if not A:
        return ({}, K.one, [])
    elif len(A) == 1:
        (Ai,) = A.values()
        j = min(Ai)
        Aij = Ai[j]
        return ({0: Ai.copy()}, Aij, [j])
    (_, rows_in_order) = zip(*sorted(A.items()))
    col_to_row_reduced = {}
    col_to_row_unreduced = {}
    reduced = col_to_row_reduced.keys()
    unreduced = col_to_row_unreduced.keys()
    A_rref_rows = []
    denom = None
    divisor = None
    A_rows = sorted(rows_in_order, key=min)
    for Ai in A_rows:
        Ai = {j: Aij for (j, Aij) in Ai.items() if j not in reduced}
        Ai_cancel = {}
        for j in unreduced & Ai.keys():
            Aij = Ai.pop(j)
            Aj = A_rref_rows[col_to_row_unreduced[j]]
            for (k, Ajk) in Aj.items():
                Aik_cancel = Ai_cancel.get(k)
                if Aik_cancel is None:
                    Ai_cancel[k] = Aij * Ajk
                else:
                    Aik_cancel = Aik_cancel + Aij * Ajk
                    if Aik_cancel:
                        Ai_cancel[k] = Aik_cancel
                    else:
                        Ai_cancel.pop(k)
        Ai_nz = set(Ai)
        Ai_cancel_nz = set(Ai_cancel)
        d = denom or K.one
        for k in Ai_cancel_nz - Ai_nz:
            Ai[k] = -Ai_cancel[k]
        for k in Ai_nz - Ai_cancel_nz:
            Ai[k] = Ai[k] * d
        for k in Ai_cancel_nz & Ai_nz:
            Aik = Ai[k] * d - Ai_cancel[k]
            if Aik:
                Ai[k] = Aik
            else:
                Ai.pop(k)
        if not Ai:
            continue
        j = min(Ai)
        Aij = Ai.pop(j)
        for (pk, k) in list(col_to_row_unreduced.items()):
            Ak = A_rref_rows[k]
            if j not in Ak:
                for (l, Akl) in Ak.items():
                    Akl = Akl * Aij
                    if divisor is not None:
                        Akl = K.exquo(Akl, divisor)
                    Ak[l] = Akl
                continue
            Akj = Ak.pop(j)
            Ai_nz = set(Ai)
            Ak_nz = set(Ak)
            for l in Ai_nz - Ak_nz:
                Ak[l] = -Akj * Ai[l]
                if divisor is not None:
                    Ak[l] = K.exquo(Ak[l], divisor)
            for l in Ak_nz - Ai_nz:
                Ak[l] = Aij * Ak[l]
                if divisor is not None:
                    Ak[l] = K.exquo(Ak[l], divisor)
            for l in Ai_nz & Ak_nz:
                Akl = Aij * Ak[l] - Akj * Ai[l]
                if Akl:
                    if divisor is not None:
                        Akl = K.exquo(Akl, divisor)
                    Ak[l] = Akl
                else:
                    Ak.pop(l)
            if not Ak:
                col_to_row_unreduced.pop(pk)
                col_to_row_reduced[pk] = k
        i = len(A_rref_rows)
        A_rref_rows.append(Ai)
        if Ai:
            col_to_row_unreduced[j] = i
        else:
            col_to_row_reduced[j] = i
        if not K.is_one(Aij):
            if denom is None:
                denom = Aij
            else:
                denom *= Aij
        if divisor is not None:
            denom = K.exquo(denom, divisor)
        divisor = denom
    if denom is None:
        denom = K.one
    col_to_row = {**col_to_row_reduced, **col_to_row_unreduced}
    row_to_col = {i: j for (j, i) in col_to_row.items()}
    A_rref_rows_col = [(row_to_col[i], Ai) for (i, Ai) in enumerate(A_rref_rows)]
    (pivots, A_rref) = zip(*sorted(A_rref_rows_col))
    pivots = list(pivots)
    for (i, Ai) in enumerate(A_rref):
        Ai[pivots[i]] = denom
    A_rref_sdm = dict(enumerate(A_rref))
    return (A_rref_sdm, denom, pivots)

def sdm_nullspace_from_rref(A, one, ncols, pivots, nonzero_cols):
    if False:
        print('Hello World!')
    'Get nullspace from A which is in RREF'
    nonpivots = sorted(set(range(ncols)) - set(pivots))
    K = []
    for j in nonpivots:
        Kj = {j: one}
        for i in nonzero_cols.get(j, ()):
            Kj[pivots[i]] = -A[i][j]
        K.append(Kj)
    return (K, nonpivots)

def sdm_particular_from_rref(A, ncols, pivots):
    if False:
        i = 10
        return i + 15
    'Get a particular solution from A which is in RREF'
    P = {}
    for (i, j) in enumerate(pivots):
        Ain = A[i].get(ncols - 1, None)
        if Ain is not None:
            P[j] = Ain / A[i][j]
    return P

def sdm_berk(M, n, K):
    if False:
        for i in range(10):
            print('nop')
    "\n    Berkowitz algorithm for computing the characteristic polynomial.\n\n    Explanation\n    ===========\n\n    The Berkowitz algorithm is a division-free algorithm for computing the\n    characteristic polynomial of a matrix over any commutative ring using only\n    arithmetic in the coefficient ring. This implementation is for sparse\n    matrices represented in a dict-of-dicts format (like :class:`SDM`).\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.polys.matrices.sdm import sdm_berk\n    >>> from sympy.polys.domains import ZZ\n    >>> M = {0: {0: ZZ(1), 1:ZZ(2)}, 1: {0:ZZ(3), 1:ZZ(4)}}\n    >>> sdm_berk(M, 2, ZZ)\n    {0: 1, 1: -5, 2: -2}\n    >>> Matrix([[1, 2], [3, 4]]).charpoly()\n    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')\n\n    See Also\n    ========\n\n    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly\n        The high-level interface to this function.\n    sympy.polys.matrices.dense.ddm_berk\n        The dense version of this function.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm\n    "
    zero = K.zero
    one = K.one
    if n == 0:
        return {0: one}
    elif n == 1:
        pdict = {0: one}
        if (M00 := M.get(0, {}).get(0, zero)):
            pdict[1] = -M00
    (a, R, C, A) = (K.zero, {}, {}, defaultdict(dict))
    for (i, Mi) in M.items():
        for (j, Mij) in Mi.items():
            if i and j:
                A[i - 1][j - 1] = Mij
            elif i:
                C[i - 1] = Mij
            elif j:
                R[j - 1] = Mij
            else:
                a = Mij
    AnC = C
    RC = sdm_dotvec(R, C, K)
    Tvals = [one, -a, -RC]
    for i in range(3, n + 1):
        AnC = sdm_matvecmul(A, AnC, K)
        if not AnC:
            break
        RAnC = sdm_dotvec(R, AnC, K)
        Tvals.append(-RAnC)
    while Tvals and (not Tvals[-1]):
        Tvals.pop()
    q = sdm_berk(A, n - 1, K)
    Tvals = Tvals[::-1]
    Tq = {}
    for i in range(min(q), min(max(q) + len(Tvals), n + 1)):
        Ti = dict(enumerate(Tvals, i - len(Tvals) + 1))
        if (Tqi := sdm_dotvec(Ti, q, K)):
            Tq[i] = Tqi
    return Tq