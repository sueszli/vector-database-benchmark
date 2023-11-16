"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
from collections import Counter
from functools import reduce
from typing import Union as tUnion, Tuple as tTuple
from sympy.utilities.decorator import doctest_depends_on
from sympy.core.sympify import _sympify
from ..domains import Domain
from ..constructor import construct_domain
from .exceptions import DMFormatError, DMBadInputError, DMShapeError, DMDomainError, DMNotAField, DMNonSquareMatrixError, DMNonInvertibleMatrixError
from .domainscalar import DomainScalar
from sympy.polys.domains import ZZ, EXRAW, QQ
from sympy.polys.densearith import dup_mul
from sympy.polys.densebasic import dup_convert
from sympy.polys.densetools import dup_mul_ground, dup_quo_ground, dup_content, dup_clear_denoms, dup_primitive, dup_transform
from sympy.polys.factortools import dup_factor_list
from sympy.polys.polyutils import _sort_factors
from .ddm import DDM
from .sdm import SDM
from .dfm import DFM
from .rref import _dm_rref, _dm_rref_den

def DM(rows, domain):
    if False:
        i = 10
        return i + 15
    'Convenient alias for DomainMatrix.from_list\n\n    Examples\n    ========\n\n    >>> from sympy import ZZ\n    >>> from sympy.polys.matrices import DM\n    >>> DM([[1, 2], [3, 4]], ZZ)\n    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)\n\n    See Also\n    ========\n\n    DomainMatrix.from_list\n    '
    return DomainMatrix.from_list(rows, domain)

class DomainMatrix:
    """
    Associate Matrix with :py:class:`~.Domain`

    Explanation
    ===========

    DomainMatrix uses :py:class:`~.Domain` for its internal representation
    which makes it faster than the SymPy Matrix class (currently) for many
    common operations, but this advantage makes it not entirely compatible
    with Matrix. DomainMatrix are analogous to numpy arrays with "dtype".
    In the DomainMatrix, each element has a domain such as :ref:`ZZ`
    or  :ref:`QQ(a)`.


    Examples
    ========

    Creating a DomainMatrix from the existing Matrix class:

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> A = DomainMatrix.from_Matrix(Matrix1)
    >>> A
    DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    Directly forming a DomainMatrix:

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> A = DomainMatrix([
    ...    [ZZ(1), ZZ(2)],
    ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> A
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DDM
    SDM
    Domain
    Poly

    """
    rep: tUnion[SDM, DDM, DFM]
    shape: tTuple[int, int]
    domain: Domain

    def __new__(cls, rows, shape, domain, *, fmt=None):
        if False:
            print('Hello World!')
        '\n        Creates a :py:class:`~.DomainMatrix`.\n\n        Parameters\n        ==========\n\n        rows : Represents elements of DomainMatrix as list of lists\n        shape : Represents dimension of DomainMatrix\n        domain : Represents :py:class:`~.Domain` of DomainMatrix\n\n        Raises\n        ======\n\n        TypeError\n            If any of rows, shape and domain are not provided\n\n        '
        if isinstance(rows, (DDM, SDM, DFM)):
            raise TypeError('Use from_rep to initialise from SDM/DDM')
        elif isinstance(rows, list):
            rep = DDM(rows, shape, domain)
        elif isinstance(rows, dict):
            rep = SDM(rows, shape, domain)
        else:
            msg = 'Input should be list-of-lists or dict-of-dicts'
            raise TypeError(msg)
        if fmt is not None:
            if fmt == 'sparse':
                rep = rep.to_sdm()
            elif fmt == 'dense':
                rep = rep.to_ddm()
            else:
                raise ValueError("fmt should be 'sparse' or 'dense'")
        if rep.fmt == 'dense' and DFM._supports_domain(domain):
            rep = rep.to_dfm()
        return cls.from_rep(rep)

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        rep = self.rep
        if rep.fmt == 'dense':
            arg = self.to_list()
        elif rep.fmt == 'sparse':
            arg = dict(rep)
        else:
            raise RuntimeError
        args = (arg, rep.shape, rep.domain)
        return (self.__class__, args)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        (i, j) = key
        (m, n) = self.shape
        if not (isinstance(i, slice) or isinstance(j, slice)):
            return DomainScalar(self.rep.getitem(i, j), self.domain)
        if not isinstance(i, slice):
            if not -m <= i < m:
                raise IndexError('Row index out of range')
            i = i % m
            i = slice(i, i + 1)
        if not isinstance(j, slice):
            if not -n <= j < n:
                raise IndexError('Column index out of range')
            j = j % n
            j = slice(j, j + 1)
        return self.from_rep(self.rep.extract_slice(i, j))

    def getitem_sympy(self, i, j):
        if False:
            while True:
                i = 10
        return self.domain.to_sympy(self.rep.getitem(i, j))

    def extract(self, rowslist, colslist):
        if False:
            print('Hello World!')
        return self.from_rep(self.rep.extract(rowslist, colslist))

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        (i, j) = key
        if not self.domain.of_type(value):
            raise TypeError
        if isinstance(i, int) and isinstance(j, int):
            self.rep.setitem(i, j, value)
        else:
            raise NotImplementedError

    @classmethod
    def from_rep(cls, rep):
        if False:
            print('Hello World!')
        'Create a new DomainMatrix efficiently from DDM/SDM.\n\n        Examples\n        ========\n\n        Create a :py:class:`~.DomainMatrix` with an dense internal\n        representation as :py:class:`~.DDM`:\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy.polys.matrices.ddm import DDM\n        >>> drep = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> dM = DomainMatrix.from_rep(drep)\n        >>> dM\n        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)\n\n        Create a :py:class:`~.DomainMatrix` with a sparse internal\n        representation as :py:class:`~.SDM`:\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy.polys.matrices.sdm import SDM\n        >>> from sympy import ZZ\n        >>> drep = SDM({0:{1:ZZ(1)},1:{0:ZZ(2)}}, (2, 2), ZZ)\n        >>> dM = DomainMatrix.from_rep(drep)\n        >>> dM\n        DomainMatrix({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)\n\n        Parameters\n        ==========\n\n        rep: SDM or DDM\n            The internal sparse or dense representation of the matrix.\n\n        Returns\n        =======\n\n        DomainMatrix\n            A :py:class:`~.DomainMatrix` wrapping *rep*.\n\n        Notes\n        =====\n\n        This takes ownership of rep as its internal representation. If rep is\n        being mutated elsewhere then a copy should be provided to\n        ``from_rep``. Only minimal verification or checking is done on *rep*\n        as this is supposed to be an efficient internal routine.\n\n        '
        if not (isinstance(rep, (DDM, SDM)) or (DFM is not None and isinstance(rep, DFM))):
            raise TypeError('rep should be of type DDM or SDM')
        self = super().__new__(cls)
        self.rep = rep
        self.shape = rep.shape
        self.domain = rep.domain
        return self

    @classmethod
    def from_list(cls, rows, domain):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert a list of lists into a DomainMatrix\n\n        Parameters\n        ==========\n\n        rows: list of lists\n            Each element of the inner lists should be either the single arg,\n            or tuple of args, that would be passed to the domain constructor\n            in order to form an element of the domain. See examples.\n\n        Returns\n        =======\n\n        DomainMatrix containing elements defined in rows\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import FF, QQ, ZZ\n        >>> A = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], ZZ)\n        >>> A\n        DomainMatrix([[1, 0, 1], [0, 0, 1]], (2, 3), ZZ)\n        >>> B = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], FF(7))\n        >>> B\n        DomainMatrix([[1 mod 7, 0 mod 7, 1 mod 7], [0 mod 7, 0 mod 7, 1 mod 7]], (2, 3), GF(7))\n        >>> C = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)\n        >>> C\n        DomainMatrix([[1/2, 3], [1/4, 5]], (2, 2), QQ)\n\n        See Also\n        ========\n\n        from_list_sympy\n\n        '
        nrows = len(rows)
        ncols = 0 if not nrows else len(rows[0])
        conv = lambda e: domain(*e) if isinstance(e, tuple) else domain(e)
        domain_rows = [[conv(e) for e in row] for row in rows]
        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert a list of lists of Expr into a DomainMatrix using construct_domain\n\n        Parameters\n        ==========\n\n        nrows: number of rows\n        ncols: number of columns\n        rows: list of lists\n\n        Returns\n        =======\n\n        DomainMatrix containing elements of rows\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy.abc import x, y, z\n        >>> A = DomainMatrix.from_list_sympy(1, 3, [[x, y, z]])\n        >>> A\n        DomainMatrix([[x, y, z]], (1, 3), ZZ[x,y,z])\n\n        See Also\n        ========\n\n        sympy.polys.constructor.construct_domain, from_dict_sympy\n\n        '
        assert len(rows) == nrows
        assert all((len(row) == ncols for row in rows))
        items_sympy = [_sympify(item) for row in rows for item in row]
        (domain, items_domain) = cls.get_domain(items_sympy, **kwargs)
        domain_rows = [[items_domain[ncols * r + c] for c in range(ncols)] for r in range(nrows)]
        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_dict_sympy(cls, nrows, ncols, elemsdict, **kwargs):
        if False:
            print('Hello World!')
        '\n\n        Parameters\n        ==========\n\n        nrows: number of rows\n        ncols: number of cols\n        elemsdict: dict of dicts containing non-zero elements of the DomainMatrix\n\n        Returns\n        =======\n\n        DomainMatrix containing elements of elemsdict\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy.abc import x,y,z\n        >>> elemsdict = {0: {0:x}, 1:{1: y}, 2: {2: z}}\n        >>> A = DomainMatrix.from_dict_sympy(3, 3, elemsdict)\n        >>> A\n        DomainMatrix({0: {0: x}, 1: {1: y}, 2: {2: z}}, (3, 3), ZZ[x,y,z])\n\n        See Also\n        ========\n\n        from_list_sympy\n\n        '
        if not all((0 <= r < nrows for r in elemsdict)):
            raise DMBadInputError('Row out of range')
        if not all((0 <= c < ncols for row in elemsdict.values() for c in row)):
            raise DMBadInputError('Column out of range')
        items_sympy = [_sympify(item) for row in elemsdict.values() for item in row.values()]
        (domain, items_domain) = cls.get_domain(items_sympy, **kwargs)
        idx = 0
        items_dict = {}
        for (i, row) in elemsdict.items():
            items_dict[i] = {}
            for j in row:
                items_dict[i][j] = items_domain[idx]
                idx += 1
        return DomainMatrix(items_dict, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, fmt='sparse', **kwargs):
        if False:
            print('Hello World!')
        "\n        Convert Matrix to DomainMatrix\n\n        Parameters\n        ==========\n\n        M: Matrix\n\n        Returns\n        =======\n\n        Returns DomainMatrix with identical elements as M\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> M = Matrix([\n        ...    [1.0, 3.4],\n        ...    [2.4, 1]])\n        >>> A = DomainMatrix.from_Matrix(M)\n        >>> A\n        DomainMatrix({0: {0: 1.0, 1: 3.4}, 1: {0: 2.4, 1: 1.0}}, (2, 2), RR)\n\n        We can keep internal representation as ddm using fmt='dense'\n        >>> from sympy import Matrix, QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')\n        >>> A.rep\n        [[1/2, 3/4], [0, 0]]\n\n        See Also\n        ========\n\n        Matrix\n\n        "
        if fmt == 'dense':
            return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)
        return cls.from_dict_sympy(*M.shape, M.todod(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):
        if False:
            while True:
                i = 10
        (K, items_K) = construct_domain(items_sympy, **kwargs)
        return (K, items_K)

    def choose_domain(self, **opts):
        if False:
            return 10
        'Convert to a domain found by :func:`~.construct_domain`.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> M = DM([[1, 2], [3, 4]], ZZ)\n        >>> M\n        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)\n        >>> M.choose_domain(field=True)\n        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)\n\n        >>> from sympy.abc import x\n        >>> M = DM([[1, x], [x**2, x**3]], ZZ[x])\n        >>> M.choose_domain(field=True).domain\n        ZZ(x)\n\n        Keyword arguments are passed to :func:`~.construct_domain`.\n\n        See Also\n        ========\n\n        construct_domain\n        convert_to\n        '
        (elements, data) = self.to_sympy().to_flat_nz()
        (dom, elements_dom) = construct_domain(elements, **opts)
        return self.from_flat_nz(elements_dom, data, dom)

    def copy(self):
        if False:
            return 10
        return self.from_rep(self.rep.copy())

    def convert_to(self, K):
        if False:
            while True:
                i = 10
        '\n        Change the domain of DomainMatrix to desired domain or field\n\n        Parameters\n        ==========\n\n        K : Represents the desired domain or field.\n            Alternatively, ``None`` may be passed, in which case this method\n            just returns a copy of this DomainMatrix.\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix with the desired domain or field\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ, ZZ_I\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n\n        >>> A.convert_to(ZZ_I)\n        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ_I)\n\n        '
        if K == self.domain:
            return self.copy()
        rep = self.rep
        if rep.is_DFM and (not DFM._supports_domain(K)):
            rep_K = rep.to_ddm().convert_to(K)
        elif rep.is_DDM and DFM._supports_domain(K):
            rep_K = rep.convert_to(K).to_dfm()
        else:
            rep_K = rep.convert_to(K)
        return self.from_rep(rep_K)

    def to_sympy(self):
        if False:
            return 10
        return self.convert_to(EXRAW)

    def to_field(self):
        if False:
            return 10
        '\n        Returns a DomainMatrix with the appropriate field\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix with the appropriate field\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n\n        >>> A.to_field()\n        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)\n\n        '
        K = self.domain.get_field()
        return self.convert_to(K)

    def to_sparse(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a sparse DomainMatrix representation of *self*.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)\n        >>> A.rep\n        [[1, 0], [0, 2]]\n        >>> B = A.to_sparse()\n        >>> B.rep\n        {0: {0: 1}, 1: {1: 2}}\n        '
        if self.rep.fmt == 'sparse':
            return self
        return self.from_rep(self.rep.to_sdm())

    def to_dense(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a dense DomainMatrix representation of *self*.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)\n        >>> A.rep\n        {0: {0: 1}, 1: {1: 2}}\n        >>> B = A.to_dense()\n        >>> B.rep\n        [[1, 0], [0, 2]]\n\n        '
        rep = self.rep
        if rep.fmt == 'dense':
            return self
        return self.from_rep(rep.to_dfm_or_ddm())

    def to_ddm(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a :class:`~.DDM` representation of *self*.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)\n        >>> ddm = A.to_ddm()\n        >>> ddm\n        [[1, 0], [0, 2]]\n        >>> type(ddm)\n        <class 'sympy.polys.matrices.ddm.DDM'>\n\n        See Also\n        ========\n\n        to_sdm\n        to_dense\n        sympy.polys.matrices.ddm.DDM.to_sdm\n        "
        return self.rep.to_ddm()

    def to_sdm(self):
        if False:
            while True:
                i = 10
        "\n        Return a :class:`~.SDM` representation of *self*.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)\n        >>> sdm = A.to_sdm()\n        >>> sdm\n        {0: {0: 1}, 1: {1: 2}}\n        >>> type(sdm)\n        <class 'sympy.polys.matrices.sdm.SDM'>\n\n        See Also\n        ========\n\n        to_ddm\n        to_sparse\n        sympy.polys.matrices.sdm.SDM.to_ddm\n        "
        return self.rep.to_sdm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        if False:
            while True:
                i = 10
        "\n        Return a :class:`~.DFM` representation of *self*.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)\n        >>> dfm = A.to_dfm()\n        >>> dfm\n        [[1, 0], [0, 2]]\n        >>> type(dfm)\n        <class 'sympy.polys.matrices._dfm.DFM'>\n\n        See Also\n        ========\n\n        to_ddm\n        to_dense\n        DFM\n        "
        return self.rep.to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        if False:
            return 10
        "\n        Return a :class:`~.DFM` or :class:`~.DDM` representation of *self*.\n\n        Explanation\n        ===========\n\n        The :class:`~.DFM` representation can only be used if the ground types\n        are ``flint`` and the ground domain is supported by ``python-flint``.\n        This method will return a :class:`~.DFM` representation if possible,\n        but will return a :class:`~.DDM` representation otherwise.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)\n        >>> dfm = A.to_dfm_or_ddm()\n        >>> dfm\n        [[1, 0], [0, 2]]\n        >>> type(dfm)  # Depends on the ground domain and ground types\n        <class 'sympy.polys.matrices._dfm.DFM'>\n\n        See Also\n        ========\n\n        to_ddm: Always return a :class:`~.DDM` representation.\n        to_dfm: Returns a :class:`~.DFM` representation or raise an error.\n        to_dense: Convert internally to a :class:`~.DFM` or :class:`~.DDM`\n        DFM: The :class:`~.DFM` dense FLINT matrix representation.\n        DDM: The Python :class:`~.DDM` dense domain matrix representation.\n        "
        return self.rep.to_dfm_or_ddm()

    @classmethod
    def _unify_domain(cls, *matrices):
        if False:
            i = 10
            return i + 15
        'Convert matrices to a common domain'
        domains = {matrix.domain for matrix in matrices}
        if len(domains) == 1:
            return matrices
        domain = reduce(lambda x, y: x.unify(y), domains)
        return tuple((matrix.convert_to(domain) for matrix in matrices))

    @classmethod
    def _unify_fmt(cls, *matrices, fmt=None):
        if False:
            while True:
                i = 10
        "Convert matrices to the same format.\n\n        If all matrices have the same format, then return unmodified.\n        Otherwise convert both to the preferred format given as *fmt* which\n        should be 'dense' or 'sparse'.\n        "
        formats = {matrix.rep.fmt for matrix in matrices}
        if len(formats) == 1:
            return matrices
        if fmt == 'sparse':
            return tuple((matrix.to_sparse() for matrix in matrices))
        elif fmt == 'dense':
            return tuple((matrix.to_dense() for matrix in matrices))
        else:
            raise ValueError("fmt should be 'sparse' or 'dense'")

    def unify(self, *others, fmt=None):
        if False:
            return 10
        "\n        Unifies the domains and the format of self and other\n        matrices.\n\n        Parameters\n        ==========\n\n        others : DomainMatrix\n\n        fmt: string 'dense', 'sparse' or `None` (default)\n            The preferred format to convert to if self and other are not\n            already in the same format. If `None` or not specified then no\n            conversion if performed.\n\n        Returns\n        =======\n\n        Tuple[DomainMatrix]\n            Matrices with unified domain and format\n\n        Examples\n        ========\n\n        Unify the domain of DomainMatrix that have different domains:\n\n        >>> from sympy import ZZ, QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)\n        >>> B = DomainMatrix([[QQ(1, 2), QQ(2)]], (1, 2), QQ)\n        >>> Aq, Bq = A.unify(B)\n        >>> Aq\n        DomainMatrix([[1, 2]], (1, 2), QQ)\n        >>> Bq\n        DomainMatrix([[1/2, 2]], (1, 2), QQ)\n\n        Unify the format (dense or sparse):\n\n        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)\n        >>> B = DomainMatrix({0:{0: ZZ(1)}}, (2, 2), ZZ)\n        >>> B.rep\n        {0: {0: 1}}\n\n        >>> A2, B2 = A.unify(B, fmt='dense')\n        >>> B2.rep\n        [[1, 0], [0, 0]]\n\n        See Also\n        ========\n\n        convert_to, to_dense, to_sparse\n\n        "
        matrices = (self,) + others
        matrices = DomainMatrix._unify_domain(*matrices)
        if fmt is not None:
            matrices = DomainMatrix._unify_fmt(*matrices, fmt=fmt)
        return matrices

    def to_Matrix(self):
        if False:
            i = 10
            return i + 15
        '\n        Convert DomainMatrix to Matrix\n\n        Returns\n        =======\n\n        Matrix\n            MutableDenseMatrix for the DomainMatrix\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n\n        >>> A.to_Matrix()\n        Matrix([\n            [1, 2],\n            [3, 4]])\n\n        See Also\n        ========\n\n        from_Matrix\n\n        '
        from sympy.matrices.dense import MutableDenseMatrix
        if self.domain in (ZZ, QQ, EXRAW):
            if self.rep.fmt == 'sparse':
                rep = self.copy()
            else:
                rep = self.to_sparse()
        else:
            rep = self.convert_to(EXRAW).to_sparse()
        return MutableDenseMatrix._fromrep(rep)

    def to_list(self):
        if False:
            while True:
                i = 10
        '\n        Convert :class:`DomainMatrix` to list of lists.\n\n        See Also\n        ========\n\n        from_list\n        to_list_flat\n        to_flat_nz\n        to_dok\n        '
        return self.rep.to_list()

    def to_list_flat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert :class:`DomainMatrix` to flat list.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> A.to_list_flat()\n        [1, 2, 3, 4]\n\n        See Also\n        ========\n\n        from_list_flat\n        to_list\n        to_flat_nz\n        to_dok\n        '
        return self.rep.to_list_flat()

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        if False:
            return 10
        '\n        Create :class:`DomainMatrix` from flat list.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> element_list = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]\n        >>> A = DomainMatrix.from_list_flat(element_list, (2, 2), ZZ)\n        >>> A\n        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)\n        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        to_list_flat\n        '
        ddm = DDM.from_list_flat(elements, shape, domain)
        return cls.from_rep(ddm.to_dfm_or_ddm())

    def to_flat_nz(self):
        if False:
            i = 10
            return i + 15
        '\n        Convert :class:`DomainMatrix` to list of nonzero elements and data.\n\n        Explanation\n        ===========\n\n        Returns a tuple ``(elements, data)`` where ``elements`` is a list of\n        elements of the matrix with zeros possibly excluded. The matrix can be\n        reconstructed by passing these to :meth:`from_flat_nz`. The idea is to\n        be able to modify a flat list of the elements and then create a new\n        matrix of the same shape with the modified elements in the same\n        positions.\n\n        The format of ``data`` differs depending on whether the underlying\n        representation is dense or sparse but either way it represents the\n        positions of the elements in the list in a way that\n        :meth:`from_flat_nz` can use to reconstruct the matrix. The\n        :meth:`from_flat_nz` method should be called on the same\n        :class:`DomainMatrix` that was used to call :meth:`to_flat_nz`.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> elements, data = A.to_flat_nz()\n        >>> elements\n        [1, 2, 3, 4]\n        >>> A == A.from_flat_nz(elements, data, A.domain)\n        True\n\n        Create a matrix with the elements doubled:\n\n        >>> elements_doubled = [2*x for x in elements]\n        >>> A2 = A.from_flat_nz(elements_doubled, data, A.domain)\n        >>> A2 == 2*A\n        True\n\n        See Also\n        ========\n\n        from_flat_nz\n        '
        return self.rep.to_flat_nz()

    def from_flat_nz(self, elements, data, domain):
        if False:
            i = 10
            return i + 15
        '\n        Reconstruct :class:`DomainMatrix` after calling :meth:`to_flat_nz`.\n\n        See :meth:`to_flat_nz` for explanation.\n\n        See Also\n        ========\n\n        to_flat_nz\n        '
        rep = self.rep.from_flat_nz(elements, data, domain)
        return self.from_rep(rep)

    def to_dok(self):
        if False:
            print('Hello World!')
        '\n        Convert :class:`DomainMatrix` to dictionary of keys (dok) format.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(0)],\n        ...    [ZZ(0), ZZ(4)]], (2, 2), ZZ)\n        >>> A.to_dok()\n        {(0, 0): 1, (1, 1): 4}\n\n        The matrix can be reconstructed by calling :meth:`from_dok` although\n        the reconstructed matrix will always be in sparse format:\n\n        >>> A.to_sparse() == A.from_dok(A.to_dok(), A.shape, A.domain)\n        True\n\n        See Also\n        ========\n\n        from_dok\n        to_list\n        to_list_flat\n        to_flat_nz\n        '
        return self.rep.to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        if False:
            print('Hello World!')
        '\n        Create :class:`DomainMatrix` from dictionary of keys (dok) format.\n\n        See :meth:`to_dok` for explanation.\n\n        See Also\n        ========\n\n        to_dok\n        '
        return cls.from_rep(SDM.from_dok(dok, shape, domain))

    def nnz(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of nonzero elements in the matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[1, 0], [0, 4]], ZZ)\n        >>> A.nnz()\n        2\n        '
        return self.rep.nnz()

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'DomainMatrix(%s, %r, %r)' % (str(self.rep), self.shape, self.domain)

    def transpose(self):
        if False:
            i = 10
            return i + 15
        'Matrix transpose of ``self``'
        return self.from_rep(self.rep.transpose())

    def flat(self):
        if False:
            print('Hello World!')
        (rows, cols) = self.shape
        return [self[i, j].element for i in range(rows) for j in range(cols)]

    @property
    def is_zero_matrix(self):
        if False:
            print('Hello World!')
        return self.rep.is_zero_matrix()

    @property
    def is_upper(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Says whether this matrix is upper-triangular. True can be returned\n        even if the matrix is not square.\n        '
        return self.rep.is_upper()

    @property
    def is_lower(self):
        if False:
            print('Hello World!')
        '\n        Says whether this matrix is lower-triangular. True can be returned\n        even if the matrix is not square.\n        '
        return self.rep.is_lower()

    @property
    def is_diagonal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        True if the matrix is diagonal.\n\n        Can return true for non-square matrices. A matrix is diagonal if\n        ``M[i,j] == 0`` whenever ``i != j``.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> M = DM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], ZZ)\n        >>> M.is_diagonal\n        True\n\n        See Also\n        ========\n\n        is_upper\n        is_lower\n        is_square\n        diagonal\n        '
        return self.rep.is_diagonal()

    def diagonal(self):
        if False:
            return 10
        '\n        Get the diagonal entries of the matrix as a list.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> M = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)\n        >>> M.diagonal()\n        [1, 4]\n\n        See Also\n        ========\n\n        is_diagonal\n        diag\n        '
        return self.rep.diagonal()

    @property
    def is_square(self):
        if False:
            i = 10
            return i + 15
        '\n        True if the matrix is square.\n        '
        return self.shape[0] == self.shape[1]

    def rank(self):
        if False:
            for i in range(10):
                print('nop')
        (rref, pivots) = self.rref()
        return len(pivots)

    def hstack(A, *B):
        if False:
            print('Hello World!')
        'Horizontally stack the given matrices.\n\n        Parameters\n        ==========\n\n        B: DomainMatrix\n            Matrices to stack horizontally.\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix by stacking horizontally.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n\n        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)\n        >>> A.hstack(B)\n        DomainMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], (2, 4), ZZ)\n\n        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)\n        >>> A.hstack(B, C)\n        DomainMatrix([[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]], (2, 6), ZZ)\n\n        See Also\n        ========\n\n        unify\n        '
        (A, *B) = A.unify(*B, fmt=A.rep.fmt)
        return DomainMatrix.from_rep(A.rep.hstack(*(Bk.rep for Bk in B)))

    def vstack(A, *B):
        if False:
            while True:
                i = 10
        'Vertically stack the given matrices.\n\n        Parameters\n        ==========\n\n        B: DomainMatrix\n            Matrices to stack vertically.\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix by stacking vertically.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n\n        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)\n        >>> A.vstack(B)\n        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], (4, 2), ZZ)\n\n        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)\n        >>> A.vstack(B, C)\n        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], (6, 2), ZZ)\n\n        See Also\n        ========\n\n        unify\n        '
        (A, *B) = A.unify(*B, fmt='dense')
        return DomainMatrix.from_rep(A.rep.vstack(*(Bk.rep for Bk in B)))

    def applyfunc(self, func, domain=None):
        if False:
            i = 10
            return i + 15
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.applyfunc(func, domain))

    def __add__(A, B):
        if False:
            return 10
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        (A, B) = A.unify(B, fmt='dense')
        return A.add(B)

    def __sub__(A, B):
        if False:
            return 10
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        (A, B) = A.unify(B, fmt='dense')
        return A.sub(B)

    def __neg__(A):
        if False:
            for i in range(10):
                print('nop')
        return A.neg()

    def __mul__(A, B):
        if False:
            return 10
        'A * B'
        if isinstance(B, DomainMatrix):
            (A, B) = A.unify(B, fmt='dense')
            return A.matmul(B)
        elif B in A.domain:
            return A.scalarmul(B)
        elif isinstance(B, DomainScalar):
            (A, B) = A.unify(B)
            return A.scalarmul(B.element)
        else:
            return NotImplemented

    def __rmul__(A, B):
        if False:
            i = 10
            return i + 15
        if B in A.domain:
            return A.rscalarmul(B)
        elif isinstance(B, DomainScalar):
            (A, B) = A.unify(B)
            return A.rscalarmul(B.element)
        else:
            return NotImplemented

    def __pow__(A, n):
        if False:
            i = 10
            return i + 15
        'A ** n'
        if not isinstance(n, int):
            return NotImplemented
        return A.pow(n)

    def _check(a, op, b, ashape, bshape):
        if False:
            print('Hello World!')
        if a.domain != b.domain:
            msg = 'Domain mismatch: %s %s %s' % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = 'Shape mismatch: %s %s %s' % (a.shape, op, b.shape)
            raise DMShapeError(msg)
        if a.rep.fmt != b.rep.fmt:
            msg = 'Format mismatch: %s %s %s' % (a.rep.fmt, op, b.rep.fmt)
            raise DMFormatError(msg)
        if type(a.rep) != type(b.rep):
            msg = 'Type mismatch: %s %s %s' % (type(a.rep), op, type(b.rep))
            raise DMFormatError(msg)

    def add(A, B):
        if False:
            print('Hello World!')
        '\n        Adds two DomainMatrix matrices of the same Domain\n\n        Parameters\n        ==========\n\n        A, B: DomainMatrix\n            matrices to add\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix after Addition\n\n        Raises\n        ======\n\n        DMShapeError\n            If the dimensions of the two DomainMatrix are not equal\n\n        ValueError\n            If the domain of the two DomainMatrix are not same\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DomainMatrix([\n        ...    [ZZ(4), ZZ(3)],\n        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)\n\n        >>> A.add(B)\n        DomainMatrix([[5, 5], [5, 5]], (2, 2), ZZ)\n\n        See Also\n        ========\n\n        sub, matmul\n\n        '
        A._check('+', B, A.shape, B.shape)
        return A.from_rep(A.rep.add(B.rep))

    def sub(A, B):
        if False:
            for i in range(10):
                print('nop')
        '\n        Subtracts two DomainMatrix matrices of the same Domain\n\n        Parameters\n        ==========\n\n        A, B: DomainMatrix\n            matrices to subtract\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix after Subtraction\n\n        Raises\n        ======\n\n        DMShapeError\n            If the dimensions of the two DomainMatrix are not equal\n\n        ValueError\n            If the domain of the two DomainMatrix are not same\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DomainMatrix([\n        ...    [ZZ(4), ZZ(3)],\n        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)\n\n        >>> A.sub(B)\n        DomainMatrix([[-3, -1], [1, 3]], (2, 2), ZZ)\n\n        See Also\n        ========\n\n        add, matmul\n\n        '
        A._check('-', B, A.shape, B.shape)
        return A.from_rep(A.rep.sub(B.rep))

    def neg(A):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the negative of DomainMatrix\n\n        Parameters\n        ==========\n\n        A : Represents a DomainMatrix\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix after Negation\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n\n        >>> A.neg()\n        DomainMatrix([[-1, -2], [-3, -4]], (2, 2), ZZ)\n\n        '
        return A.from_rep(A.rep.neg())

    def mul(A, b):
        if False:
            print('Hello World!')
        '\n        Performs term by term multiplication for the second DomainMatrix\n        w.r.t first DomainMatrix. Returns a DomainMatrix whose rows are\n        list of DomainMatrix matrices created after term by term multiplication.\n\n        Parameters\n        ==========\n\n        A, B: DomainMatrix\n            matrices to multiply term-wise\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix after term by term multiplication\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> b = ZZ(2)\n\n        >>> A.mul(b)\n        DomainMatrix([[2, 4], [6, 8]], (2, 2), ZZ)\n\n        See Also\n        ========\n\n        matmul\n\n        '
        return A.from_rep(A.rep.mul(b))

    def rmul(A, b):
        if False:
            print('Hello World!')
        return A.from_rep(A.rep.rmul(b))

    def matmul(A, B):
        if False:
            print('Hello World!')
        '\n        Performs matrix multiplication of two DomainMatrix matrices\n\n        Parameters\n        ==========\n\n        A, B: DomainMatrix\n            to multiply\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix after multiplication\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DomainMatrix([\n        ...    [ZZ(1), ZZ(1)],\n        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)\n\n        >>> A.matmul(B)\n        DomainMatrix([[1, 3], [3, 7]], (2, 2), ZZ)\n\n        See Also\n        ========\n\n        mul, pow, add, sub\n\n        '
        A._check('*', B, A.shape[1], B.shape[0])
        return A.from_rep(A.rep.matmul(B.rep))

    def _scalarmul(A, lamda, reverse):
        if False:
            i = 10
            return i + 15
        if lamda == A.domain.zero:
            return DomainMatrix.zeros(A.shape, A.domain)
        elif lamda == A.domain.one:
            return A.copy()
        elif reverse:
            return A.rmul(lamda)
        else:
            return A.mul(lamda)

    def scalarmul(A, lamda):
        if False:
            while True:
                i = 10
        return A._scalarmul(lamda, reverse=False)

    def rscalarmul(A, lamda):
        if False:
            i = 10
            return i + 15
        return A._scalarmul(lamda, reverse=True)

    def mul_elementwise(A, B):
        if False:
            for i in range(10):
                print('nop')
        assert A.domain == B.domain
        return A.from_rep(A.rep.mul_elementwise(B.rep))

    def __truediv__(A, lamda):
        if False:
            while True:
                i = 10
        ' Method for Scalar Division'
        if isinstance(lamda, int) or ZZ.of_type(lamda):
            lamda = DomainScalar(ZZ(lamda), ZZ)
        elif A.domain.is_Field and lamda in A.domain:
            K = A.domain
            lamda = DomainScalar(K.convert(lamda), K)
        if not isinstance(lamda, DomainScalar):
            return NotImplemented
        (A, lamda) = A.to_field().unify(lamda)
        if lamda.element == lamda.domain.zero:
            raise ZeroDivisionError
        if lamda.element == lamda.domain.one:
            return A
        return A.mul(1 / lamda.element)

    def pow(A, n):
        if False:
            print('Hello World!')
        '\n        Computes A**n\n\n        Parameters\n        ==========\n\n        A : DomainMatrix\n\n        n : exponent for A\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix on computing A**n\n\n        Raises\n        ======\n\n        NotImplementedError\n            if n is negative.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(1)],\n        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)\n\n        >>> A.pow(2)\n        DomainMatrix([[1, 2], [0, 1]], (2, 2), ZZ)\n\n        See Also\n        ========\n\n        matmul\n\n        '
        (nrows, ncols) = A.shape
        if nrows != ncols:
            raise DMNonSquareMatrixError('Power of a nonsquare matrix')
        if n < 0:
            raise NotImplementedError('Negative powers')
        elif n == 0:
            return A.eye(nrows, A.domain)
        elif n == 1:
            return A
        elif n % 2 == 1:
            return A * A ** (n - 1)
        else:
            sqrtAn = A ** (n // 2)
            return sqrtAn * sqrtAn

    def scc(self):
        if False:
            while True:
                i = 10
        'Compute the strongly connected components of a DomainMatrix\n\n        Explanation\n        ===========\n\n        A square matrix can be considered as the adjacency matrix for a\n        directed graph where the row and column indices are the vertices. In\n        this graph if there is an edge from vertex ``i`` to vertex ``j`` if\n        ``M[i, j]`` is nonzero. This routine computes the strongly connected\n        components of that graph which are subsets of the rows and columns that\n        are connected by some nonzero element of the matrix. The strongly\n        connected components are useful because many operations such as the\n        determinant can be computed by working with the submatrices\n        corresponding to each component.\n\n        Examples\n        ========\n\n        Find the strongly connected components of a matrix:\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> M = DomainMatrix([[ZZ(1), ZZ(0), ZZ(2)],\n        ...                   [ZZ(0), ZZ(3), ZZ(0)],\n        ...                   [ZZ(4), ZZ(6), ZZ(5)]], (3, 3), ZZ)\n        >>> M.scc()\n        [[1], [0, 2]]\n\n        Compute the determinant from the components:\n\n        >>> MM = M.to_Matrix()\n        >>> MM\n        Matrix([\n        [1, 0, 2],\n        [0, 3, 0],\n        [4, 6, 5]])\n        >>> MM[[1], [1]]\n        Matrix([[3]])\n        >>> MM[[0, 2], [0, 2]]\n        Matrix([\n        [1, 2],\n        [4, 5]])\n        >>> MM.det()\n        -9\n        >>> MM[[1], [1]].det() * MM[[0, 2], [0, 2]].det()\n        -9\n\n        The components are given in reverse topological order and represent a\n        permutation of the rows and columns that will bring the matrix into\n        block lower-triangular form:\n\n        >>> MM[[1, 0, 2], [1, 0, 2]]\n        Matrix([\n        [3, 0, 0],\n        [0, 1, 2],\n        [6, 4, 5]])\n\n        Returns\n        =======\n\n        List of lists of integers\n            Each list represents a strongly connected component.\n\n        See also\n        ========\n\n        sympy.matrices.matrices.MatrixBase.strongly_connected_components\n        sympy.utilities.iterables.strongly_connected_components\n\n        '
        if not self.is_square:
            raise DMNonSquareMatrixError('Matrix must be square for scc')
        return self.rep.scc()

    def clear_denoms(self, convert=False):
        if False:
            while True:
                i = 10
        '\n        Clear denominators, but keep the domain unchanged.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[(1,2), (1,3)], [(1,4), (1,5)]], QQ)\n        >>> den, Anum = A.clear_denoms()\n        >>> den.to_sympy()\n        60\n        >>> Anum.to_Matrix()\n        Matrix([\n        [30, 20],\n        [15, 12]])\n        >>> den * A == Anum\n        True\n\n        The numerator matrix will be in the same domain as the original matrix\n        unless ``convert`` is set to ``True``:\n\n        >>> A.clear_denoms()[1].domain\n        QQ\n        >>> A.clear_denoms(convert=True)[1].domain\n        ZZ\n\n        The denominator is always in the associated ring:\n\n        >>> A.clear_denoms()[0].domain\n        ZZ\n        >>> A.domain.get_ring()\n        ZZ\n\n        See Also\n        ========\n\n        sympy.polys.polytools.Poly.clear_denoms\n        '
        (elems0, data) = self.to_flat_nz()
        K0 = self.domain
        K1 = K0.get_ring() if K0.has_assoc_Ring else K0
        (den, elems1) = dup_clear_denoms(elems0, K0, K1, convert=convert)
        if convert:
            (Kden, Knum) = (K1, K1)
        else:
            (Kden, Knum) = (K1, K0)
        den = DomainScalar(den, Kden)
        num = self.from_flat_nz(elems1, data, Knum)
        return (den, num)

    def cancel_denom(self, denom):
        if False:
            while True:
                i = 10
        '\n        Cancel factors between a matrix and a denominator.\n\n        Returns a matrix and denominator on lowest terms.\n\n        Requires ``gcd`` in the ground domain.\n\n        Methods like :meth:`solve_den`, :meth:`inv_den` and :meth:`rref_den`\n        return a matrix and denominator but not necessarily on lowest terms.\n        Reduction to lowest terms without fractions can be performed with\n        :meth:`cancel_denom`.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import ZZ\n        >>> M = DM([[2, 2, 0],\n        ...         [0, 2, 2],\n        ...         [0, 0, 2]], ZZ)\n        >>> Minv, den = M.inv_den()\n        >>> Minv.to_Matrix()\n        Matrix([\n        [1, -1,  1],\n        [0,  1, -1],\n        [0,  0,  1]])\n        >>> den\n        2\n        >>> Minv_reduced, den_reduced = Minv.cancel_denom(den)\n        >>> Minv_reduced.to_Matrix()\n        Matrix([\n        [1, -1,  1],\n        [0,  1, -1],\n        [0,  0,  1]])\n        >>> den_reduced\n        2\n        >>> Minv_reduced.to_field() / den_reduced == Minv.to_field() / den\n        True\n\n        The denominator is made canonical with respect to units (e.g. a\n        negative denominator is made positive):\n\n        >>> M = DM([[2, 2, 0]], ZZ)\n        >>> den = ZZ(-4)\n        >>> M.cancel_denom(den)\n        (DomainMatrix([[-1, -1, 0]], (1, 3), ZZ), 2)\n\n        Any factor common to _all_ elements will be cancelled but there can\n        still be factors in common between _some_ elements of the matrix and\n        the denominator. To cancel factors between each element and the\n        denominator, use :meth:`cancel_denom_elementwise` or otherwise convert\n        to a field and use division:\n\n        >>> M = DM([[4, 6]], ZZ)\n        >>> den = ZZ(12)\n        >>> M.cancel_denom(den)\n        (DomainMatrix([[2, 3]], (1, 2), ZZ), 6)\n        >>> numers, denoms = M.cancel_denom_elementwise(den)\n        >>> numers\n        DomainMatrix([[1, 1]], (1, 2), ZZ)\n        >>> denoms\n        DomainMatrix([[3, 2]], (1, 2), ZZ)\n        >>> M.to_field() / den\n        DomainMatrix([[1/3, 1/2]], (1, 2), QQ)\n\n        See Also\n        ========\n\n        solve_den\n        inv_den\n        rref_den\n        cancel_denom_elementwise\n        '
        M = self
        K = self.domain
        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            return (M.copy(), denom)
        (elements, data) = M.to_flat_nz()
        if K.is_negative(denom):
            u = -K.one
        else:
            u = K.canonical_unit(denom)
        content = dup_content(elements, K)
        common = K.gcd(content, denom)
        if not K.is_one(content):
            common = K.gcd(content, denom)
            if not K.is_one(common):
                elements = dup_quo_ground(elements, common, K)
                denom = K.quo(denom, common)
        if not K.is_one(u):
            elements = dup_mul_ground(elements, u, K)
            denom = u * denom
        elif K.is_one(common):
            return (M.copy(), denom)
        M_cancelled = M.from_flat_nz(elements, data, K)
        return (M_cancelled, denom)

    def cancel_denom_elementwise(self, denom):
        if False:
            while True:
                i = 10
        '\n        Cancel factors between the elements of a matrix and a denominator.\n\n        Returns a matrix of numerators and matrix of denominators.\n\n        Requires ``gcd`` in the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import ZZ\n        >>> M = DM([[2, 3], [4, 12]], ZZ)\n        >>> denom = ZZ(6)\n        >>> numers, denoms = M.cancel_denom_elementwise(denom)\n        >>> numers.to_Matrix()\n        Matrix([\n        [1, 1],\n        [2, 2]])\n        >>> denoms.to_Matrix()\n        Matrix([\n        [3, 2],\n        [3, 1]])\n        >>> M_frac = (M.to_field() / denom).to_Matrix()\n        >>> M_frac\n        Matrix([\n        [1/3, 1/2],\n        [2/3,   2]])\n        >>> denoms_inverted = denoms.to_Matrix().applyfunc(lambda e: 1/e)\n        >>> numers.to_Matrix().multiply_elementwise(denoms_inverted) == M_frac\n        True\n\n        Use :meth:`cancel_denom` to cancel factors between the matrix and the\n        denominator while preserving the form of a matrix with a scalar\n        denominator.\n\n        See Also\n        ========\n\n        cancel_denom\n        '
        K = self.domain
        M = self
        if K.is_zero(denom):
            raise ZeroDivisionError('denominator is zero')
        elif K.is_one(denom):
            M_numers = M.copy()
            M_denoms = M.ones(M.shape, M.domain)
            return (M_numers, M_denoms)
        (elements, data) = M.to_flat_nz()
        cofactors = [K.cofactors(numer, denom) for numer in elements]
        (gcds, numers, denoms) = zip(*cofactors)
        M_numers = M.from_flat_nz(list(numers), data, K)
        M_denoms = M.from_flat_nz(list(denoms), data, K)
        return (M_numers, M_denoms)

    def content(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the gcd of the elements of the matrix.\n\n        Requires ``gcd`` in the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import ZZ\n        >>> M = DM([[2, 4], [4, 12]], ZZ)\n        >>> M.content()\n        2\n\n        See Also\n        ========\n\n        primitive\n        cancel_denom\n        '
        K = self.domain
        (elements, _) = self.to_flat_nz()
        return dup_content(elements, K)

    def primitive(self):
        if False:
            return 10
        '\n        Factor out gcd of the elements of a matrix.\n\n        Requires ``gcd`` in the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import ZZ\n        >>> M = DM([[2, 4], [4, 12]], ZZ)\n        >>> content, M_primitive = M.primitive()\n        >>> content\n        2\n        >>> M_primitive\n        DomainMatrix([[1, 2], [2, 6]], (2, 2), ZZ)\n        >>> content * M_primitive == M\n        True\n        >>> M_primitive.content() == ZZ(1)\n        True\n\n        See Also\n        ========\n\n        content\n        cancel_denom\n        '
        K = self.domain
        (elements, data) = self.to_flat_nz()
        (content, prims) = dup_primitive(elements, K)
        M_primitive = self.from_flat_nz(prims, data, K)
        return (content, M_primitive)

    def rref(self, *, method='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns reduced-row echelon form (RREF) and list of pivots.\n\n        If the domain is not a field then it will be converted to a field. See\n        :meth:`rref_den` for the fraction-free version of this routine that\n        returns RREF with denominator instead.\n\n        The domain must either be a field or have an associated fraction field\n        (see :meth:`to_field`).\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...     [QQ(2), QQ(-1), QQ(0)],\n        ...     [QQ(-1), QQ(2), QQ(-1)],\n        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)\n\n        >>> rref_matrix, rref_pivots = A.rref()\n        >>> rref_matrix\n        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)\n        >>> rref_pivots\n        (0, 1, 2)\n\n        Parameters\n        ==========\n\n        method : str, optional (default: 'auto')\n            The method to use to compute the RREF. The default is ``'auto'``,\n            which will attempt to choose the fastest method. The other options\n            are:\n\n            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with\n              division. If the domain is not a field then it will be converted\n              to a field with :meth:`to_field` first and RREF will be computed\n              by inverting the pivot elements in each row. This is most\n              efficient for very sparse matrices or for matrices whose elements\n              have complex denominators.\n\n            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan\n              elimination. Elimination is performed using exact division\n              (``exquo``) to control the growth of the coefficients. In this\n              case the current domain is always used for elimination but if\n              the domain is not a field then it will be converted to a field\n              at the end and divided by the denominator. This is most efficient\n              for dense matrices or for matrices with simple denominators.\n\n            - ``A.rref(method='CD')`` clears the denominators before using\n              fraction-free Gauss-Jordan elimination in the assoicated ring.\n              This is most efficient for dense matrices with very simple\n              denominators.\n\n            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and\n              ``A.rref(method='CD_dense')`` are the same as the above methods\n              except that the dense implementations of the algorithms are used.\n              By default ``A.rref(method='auto')`` will usually choose the\n              sparse implementations for RREF.\n\n            Regardless of which algorithm is used the returned matrix will\n            always have the same format (sparse or dense) as the input and its\n            domain will always be the field of fractions of the input domain.\n\n        Returns\n        =======\n\n        (DomainMatrix, list)\n            reduced-row echelon form and list of pivots for the DomainMatrix\n\n        See Also\n        ========\n\n        rref_den\n            RREF with denominator\n        sympy.polys.matrices.sdm.sdm_irref\n            Sparse implementation of ``method='GJ'``.\n        sympy.polys.matrices.sdm.sdm_rref_den\n            Sparse implementation of ``method='FF'`` and ``method='CD'``.\n        sympy.polys.matrices.dense.ddm_irref\n            Dense implementation of ``method='GJ'``.\n        sympy.polys.matrices.dense.ddm_irref_den\n            Dense implementation of ``method='FF'`` and ``method='CD'``.\n        clear_denoms\n            Clear denominators from a matrix, used by ``method='CD'`` and\n            by ``method='GJ'`` when the original domain is not a field.\n\n        "
        return _dm_rref(self, method=method)

    def rref_den(self, *, method='auto', keep_domain=True):
        if False:
            return 10
        "\n        Returns reduced-row echelon form with denominator and list of pivots.\n\n        Requires exact division in the ground domain (``exquo``).\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ, QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...     [ZZ(2), ZZ(-1), ZZ(0)],\n        ...     [ZZ(-1), ZZ(2), ZZ(-1)],\n        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)\n\n        >>> A_rref, denom, pivots = A.rref_den()\n        >>> A_rref\n        DomainMatrix([[6, 0, 0], [0, 6, 0], [0, 0, 6]], (3, 3), ZZ)\n        >>> denom\n        6\n        >>> pivots\n        (0, 1, 2)\n        >>> A_rref.to_field() / denom\n        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)\n        >>> A_rref.to_field() / denom == A.convert_to(QQ).rref()[0]\n        True\n\n        Parameters\n        ==========\n\n        method : str, optional (default: 'auto')\n            The method to use to compute the RREF. The default is ``'auto'``,\n            which will attempt to choose the fastest method. The other options\n            are:\n\n            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan\n              elimination. Elimination is performed using exact division\n              (``exquo``) to control the growth of the coefficients. In this\n              case the current domain is always used for elimination and the\n              result is always returned as a matrix over the current domain.\n              This is most efficient for dense matrices or for matrices with\n              simple denominators.\n\n            - ``A.rref(method='CD')`` clears denominators before using\n              fraction-free Gauss-Jordan elimination in the assoicated ring.\n              The result will be converted back to the original domain unless\n              ``keep_domain=False`` is passed in which case the result will be\n              over the ring used for elimination. This is most efficient for\n              dense matrices with very simple denominators.\n\n            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with\n              division. If the domain is not a field then it will be converted\n              to a field with :meth:`to_field` first and RREF will be computed\n              by inverting the pivot elements in each row. The result is\n              converted back to the original domain by clearing denominators\n              unless ``keep_domain=False`` is passed in which case the result\n              will be over the field used for elimination. This is most\n              efficient for very sparse matrices or for matrices whose elements\n              have complex denominators.\n\n            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and\n              ``A.rref(method='CD_dense')`` are the same as the above methods\n              except that the dense implementations of the algorithms are used.\n              By default ``A.rref(method='auto')`` will usually choose the\n              sparse implementations for RREF.\n\n            Regardless of which algorithm is used the returned matrix will\n            always have the same format (sparse or dense) as the input and if\n            ``keep_domain=True`` its domain will always be the same as the\n            input.\n\n        keep_domain : bool, optional\n            If True (the default), the domain of the returned matrix and\n            denominator are the same as the domain of the input matrix. If\n            False, the domain of the returned matrix might be changed to an\n            associated ring or field if the algorithm used a different domain.\n            This is useful for efficiency if the caller does not need the\n            result to be in the original domain e.g. it avoids clearing\n            denominators in the case of ``A.rref(method='GJ')``.\n\n        Returns\n        =======\n\n        (DomainMatrix, scalar, list)\n            Reduced-row echelon form, denominator and list of pivot indices.\n\n        See Also\n        ========\n\n        rref\n            RREF without denominator for field domains.\n        sympy.polys.matrices.sdm.sdm_irref\n            Sparse implementation of ``method='GJ'``.\n        sympy.polys.matrices.sdm.sdm_rref_den\n            Sparse implementation of ``method='FF'`` and ``method='CD'``.\n        sympy.polys.matrices.dense.ddm_irref\n            Dense implementation of ``method='GJ'``.\n        sympy.polys.matrices.dense.ddm_irref_den\n            Dense implementation of ``method='FF'`` and ``method='CD'``.\n        clear_denoms\n            Clear denominators from a matrix, used by ``method='CD'``.\n\n        "
        return _dm_rref_den(self, method=method, keep_domain=keep_domain)

    def columnspace(self):
        if False:
            while True:
                i = 10
        '\n        Returns the columnspace for the DomainMatrix\n\n        Returns\n        =======\n\n        DomainMatrix\n            The columns of this matrix form a basis for the columnspace.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [QQ(1), QQ(-1)],\n        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)\n        >>> A.columnspace()\n        DomainMatrix([[1], [2]], (2, 1), QQ)\n\n        '
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        (rref, pivots) = self.rref()
        (rows, cols) = self.shape
        return self.extract(range(rows), pivots)

    def rowspace(self):
        if False:
            print('Hello World!')
        '\n        Returns the rowspace for the DomainMatrix\n\n        Returns\n        =======\n\n        DomainMatrix\n            The rows of this matrix form a basis for the rowspace.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [QQ(1), QQ(-1)],\n        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)\n        >>> A.rowspace()\n        DomainMatrix([[1, -1]], (1, 2), QQ)\n\n        '
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        (rref, pivots) = self.rref()
        (rows, cols) = self.shape
        return self.extract(range(len(pivots)), range(cols))

    def nullspace(self, divide_last=False):
        if False:
            while True:
                i = 10
        '\n        Returns the nullspace for the DomainMatrix\n\n        Returns\n        =======\n\n        DomainMatrix\n            The rows of this matrix form a basis for the nullspace.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([\n        ...    [QQ(2), QQ(-2)],\n        ...    [QQ(4), QQ(-4)]], QQ)\n        >>> A.nullspace()\n        DomainMatrix([[1, 1]], (1, 2), QQ)\n\n        The returned matrix is a basis for the nullspace:\n\n        >>> A_null = A.nullspace().transpose()\n        >>> A * A_null\n        DomainMatrix([[0], [0]], (2, 1), QQ)\n        >>> rows, cols = A.shape\n        >>> nullity = rows - A.rank()\n        >>> A_null.shape == (cols, nullity)\n        True\n\n        Nullspace can also be computed for non-field rings. If the ring is not\n        a field then division is not used. Setting ``divide_last`` to True will\n        raise an error in this case:\n\n        >>> from sympy import ZZ\n        >>> B = DM([[6, -3],\n        ...         [4, -2]], ZZ)\n        >>> B.nullspace()\n        DomainMatrix([[3, 6]], (1, 2), ZZ)\n        >>> B.nullspace(divide_last=True)\n        Traceback (most recent call last):\n        ...\n        DMNotAField: Cannot normalize vectors over a non-field\n\n        Over a ring with ``gcd`` defined the nullspace can potentially be\n        reduced with :meth:`primitive`:\n\n        >>> B.nullspace().primitive()\n        (3, DomainMatrix([[1, 2]], (1, 2), ZZ))\n\n        A matrix over a ring can often be normalized by converting it to a\n        field but it is often a bad idea to do so:\n\n        >>> from sympy.abc import a, b, c\n        >>> from sympy import Matrix\n        >>> M = Matrix([[        a*b,       b + c,        c],\n        ...             [      a - b,         b*c,     c**2],\n        ...             [a*b + a - b, b*c + b + c, c**2 + c]])\n        >>> M.to_DM().domain\n        ZZ[a,b,c]\n        >>> M.to_DM().nullspace().to_Matrix().transpose()\n        Matrix([\n        [                             c**3],\n        [            -a*b*c**2 + a*c - b*c],\n        [a*b**2*c - a*b - a*c + b**2 + b*c]])\n\n        The unnormalized form here is nicer than the normalized form that\n        spreads a large denominator throughout the matrix:\n\n        >>> M.to_DM().to_field().nullspace(divide_last=True).to_Matrix().transpose()\n        Matrix([\n        [                   c**3/(a*b**2*c - a*b - a*c + b**2 + b*c)],\n        [(-a*b*c**2 + a*c - b*c)/(a*b**2*c - a*b - a*c + b**2 + b*c)],\n        [                                                          1]])\n\n        Parameters\n        ==========\n\n        divide_last : bool, optional\n            If False (the default), the vectors are not normalized and the RREF\n            is computed using :meth:`rref_den` and the denominator is\n            discarded. If True, then each row is divided by its final element;\n            the domain must be a field in this case.\n\n        See Also\n        ========\n\n        nullspace_from_rref\n        rref\n        rref_den\n        rowspace\n        '
        A = self
        K = A.domain
        if divide_last and (not K.is_Field):
            raise DMNotAField('Cannot normalize vectors over a non-field')
        if divide_last:
            (A_rref, pivots) = A.rref()
        else:
            (A_rref, den, pivots) = A.rref_den()
            u = K.canonical_unit(den)
            if u != K.one:
                A_rref *= u
        A_null = A_rref.nullspace_from_rref(pivots)
        return A_null

    def nullspace_from_rref(self, pivots=None):
        if False:
            while True:
                i = 10
        '\n        Compute nullspace from rref and pivots.\n\n        The domain of the matrix can be any domain.\n\n        The matrix must be in reduced row echelon form already. Otherwise the\n        result will be incorrect. Use :meth:`rref` or :meth:`rref_den` first\n        to get the reduced row echelon form or use :meth:`nullspace` instead.\n\n        See Also\n        ========\n\n        nullspace\n        rref\n        rref_den\n        sympy.polys.matrices.sdm.SDM.nullspace_from_rref\n        sympy.polys.matrices.ddm.DDM.nullspace_from_rref\n        '
        (null_rep, nonpivots) = self.rep.nullspace_from_rref(pivots)
        return self.from_rep(null_rep)

    def inv(self):
        if False:
            while True:
                i = 10
        '\n        Finds the inverse of the DomainMatrix if exists\n\n        Returns\n        =======\n\n        DomainMatrix\n            DomainMatrix after inverse\n\n        Raises\n        ======\n\n        ValueError\n            If the domain of DomainMatrix not a Field\n\n        DMNonSquareMatrixError\n            If the DomainMatrix is not a not Square DomainMatrix\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...     [QQ(2), QQ(-1), QQ(0)],\n        ...     [QQ(-1), QQ(2), QQ(-1)],\n        ...     [QQ(0), QQ(0), QQ(2)]], (3, 3), QQ)\n        >>> A.inv()\n        DomainMatrix([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [0, 0, 1/2]], (3, 3), QQ)\n\n        See Also\n        ========\n\n        neg\n\n        '
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        (m, n) = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        inv = self.rep.inv()
        return self.from_rep(inv)

    def det(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the determinant of a square :class:`DomainMatrix`.\n\n        Returns\n        =======\n\n        determinant: DomainElement\n            Determinant of the matrix.\n\n        Raises\n        ======\n\n        ValueError\n            If the domain of DomainMatrix is not a Field\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n\n        >>> A.det()\n        -2\n\n        '
        (m, n) = self.shape
        if m != n:
            raise DMNonSquareMatrixError
        return self.rep.det()

    def adj_det(self):
        if False:
            while True:
                i = 10
        '\n        Adjugate and determinant of a square :class:`DomainMatrix`.\n\n        Returns\n        =======\n\n        (adjugate, determinant) : (DomainMatrix, DomainScalar)\n            The adjugate matrix and determinant of this matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([\n        ...     [ZZ(1), ZZ(2)],\n        ...     [ZZ(3), ZZ(4)]], ZZ)\n        >>> adjA, detA = A.adj_det()\n        >>> adjA\n        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)\n        >>> detA\n        -2\n\n        See Also\n        ========\n\n        adjugate\n            Returns only the adjugate matrix.\n        det\n            Returns only the determinant.\n        inv_den\n            Returns a matrix/denominator pair representing the inverse matrix\n            but perhaps differing from the adjugate and determinant by a common\n            factor.\n        '
        (m, n) = self.shape
        I_m = self.eye((m, m), self.domain)
        (adjA, detA) = self.solve_den_charpoly(I_m, check=False)
        if self.rep.fmt == 'dense':
            adjA = adjA.to_dense()
        return (adjA, detA)

    def adjugate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adjugate of a square :class:`DomainMatrix`.\n\n        The adjugate matrix is the transpose of the cofactor matrix and is\n        related to the inverse by::\n\n            adj(A) = det(A) * A.inv()\n\n        Unlike the inverse matrix the adjugate matrix can be computed and\n        expressed without division or fractions in the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)\n        >>> A.adjugate()\n        DomainMatrix([[4, -2], [-3, 1]], (2, 2), ZZ)\n\n        Returns\n        =======\n\n        DomainMatrix\n            The adjugate matrix of this matrix with the same domain.\n\n        See Also\n        ========\n\n        adj_det\n        '
        (adjA, detA) = self.adj_det()
        return adjA

    def inv_den(self, method=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return the inverse as a :class:`DomainMatrix` with denominator.\n\n        Returns\n        =======\n\n        (inv, den) : (:class:`DomainMatrix`, :class:`~.DomainElement`)\n            The inverse matrix and its denominator.\n\n        This is more or less equivalent to :meth:`adj_det` except that ``inv``\n        and ``den`` are not guaranteed to be the adjugate and inverse. The\n        ratio ``inv/den`` is equivalent to ``adj/det`` but some factors\n        might be cancelled between ``inv`` and ``den``. In simple cases this\n        might just be a minus sign so that ``(inv, den) == (-adj, -det)`` but\n        factors more complicated than ``-1`` can also be cancelled.\n        Cancellation is not guaranteed to be complete so ``inv`` and ``den``\n        may not be on lowest terms. The denominator ``den`` will be zero if and\n        only if the determinant is zero.\n\n        If the actual adjugate and determinant are needed, use :meth:`adj_det`\n        instead. If the intention is to compute the inverse matrix or solve a\n        system of equations then :meth:`inv_den` is more efficient.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...     [ZZ(2), ZZ(-1), ZZ(0)],\n        ...     [ZZ(-1), ZZ(2), ZZ(-1)],\n        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)\n        >>> Ainv, den = A.inv_den()\n        >>> den\n        6\n        >>> Ainv\n        DomainMatrix([[4, 2, 1], [2, 4, 2], [0, 0, 3]], (3, 3), ZZ)\n        >>> A * Ainv == den * A.eye(A.shape, A.domain).to_dense()\n        True\n\n        Parameters\n        ==========\n\n        method : str, optional\n            The method to use to compute the inverse. Can be one of ``None``,\n            ``'rref'`` or ``'charpoly'``. If ``None`` then the method is\n            chosen automatically (see :meth:`solve_den` for details).\n\n        See Also\n        ========\n\n        inv\n        det\n        adj_det\n        solve_den\n        "
        I = self.eye(self.shape, self.domain)
        return self.solve_den(I, method=method)

    def solve_den(self, b, method=None):
        if False:
            while True:
                i = 10
        "\n        Solve matrix equation $Ax = b$ without fractions in the ground domain.\n\n        Examples\n        ========\n\n        Solve a matrix equation over the integers:\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)\n        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)\n        >>> xnum, xden = A.solve_den(b)\n        >>> xden\n        -2\n        >>> xnum\n        DomainMatrix([[8], [-9]], (2, 1), ZZ)\n        >>> A * xnum == xden * b\n        True\n\n        Solve a matrix equation over a polynomial ring:\n\n        >>> from sympy import ZZ\n        >>> from sympy.abc import x, y, z, a, b\n        >>> R = ZZ[x, y, z, a, b]\n        >>> M = DM([[x*y, x*z], [y*z, x*z]], R)\n        >>> b = DM([[a], [b]], R)\n        >>> M.to_Matrix()\n        Matrix([\n        [x*y, x*z],\n        [y*z, x*z]])\n        >>> b.to_Matrix()\n        Matrix([\n        [a],\n        [b]])\n        >>> xnum, xden = M.solve_den(b)\n        >>> xden\n        x**2*y*z - x*y*z**2\n        >>> xnum.to_Matrix()\n        Matrix([\n        [ a*x*z - b*x*z],\n        [-a*y*z + b*x*y]])\n        >>> M * xnum == xden * b\n        True\n\n        The solution can be expressed over a fraction field which will cancel\n        gcds between the denominator and the elements of the numerator:\n\n        >>> xsol = xnum.to_field() / xden\n        >>> xsol.to_Matrix()\n        Matrix([\n        [           (a - b)/(x*y - y*z)],\n        [(-a*z + b*x)/(x**2*z - x*z**2)]])\n        >>> (M * xsol).to_Matrix() == b.to_Matrix()\n        True\n\n        When solving a large system of equations this cancellation step might\n        be a lot slower than :func:`solve_den` itself. The solution can also be\n        expressed as a ``Matrix`` without attempting any polynomial\n        cancellation between the numerator and denominator giving a less\n        simplified result more quickly:\n\n        >>> xsol_uncancelled = xnum.to_Matrix() / xnum.domain.to_sympy(xden)\n        >>> xsol_uncancelled\n        Matrix([\n        [ (a*x*z - b*x*z)/(x**2*y*z - x*y*z**2)],\n        [(-a*y*z + b*x*y)/(x**2*y*z - x*y*z**2)]])\n        >>> from sympy import cancel\n        >>> cancel(xsol_uncancelled) == xsol.to_Matrix()\n        True\n\n        Parameters\n        ==========\n\n        self : :class:`DomainMatrix`\n            The ``m x n`` matrix $A$ in the equation $Ax = b$. Underdetermined\n            systems are not supported so ``m >= n``: $A$ should be square or\n            have more rows than columns.\n        b : :class:`DomainMatrix`\n            The ``n x m`` matrix $b$ for the rhs.\n        cp : list of :class:`~.DomainElement`, optional\n            The characteristic polynomial of the matrix $A$. If not given, it\n            will be computed using :meth:`charpoly`.\n        method: str, optional\n            The method to use for solving the system. Can be one of ``None``,\n            ``'charpoly'`` or ``'rref'``. If ``None`` (the default) then the\n            method will be chosen automatically.\n\n            The ``charpoly`` method uses :meth:`solve_den_charpoly` and can\n            only be used if the matrix is square. This method is division free\n            and can be used with any domain.\n\n            The ``rref`` method is fraction free but requires exact division\n            in the ground domain (``exquo``). This is also suitable for most\n            domains. This method can be used with overdetermined systems (more\n            equations than unknowns) but not underdetermined systems as a\n            unique solution is sought.\n\n        Returns\n        =======\n\n        (xnum, xden) : (DomainMatrix, DomainElement)\n            The solution of the equation $Ax = b$ as a pair consisting of an\n            ``n x m`` matrix numerator ``xnum`` and a scalar denominator\n            ``xden``.\n\n        The solution $x$ is given by ``x = xnum / xden``. The division free\n        invariant is ``A * xnum == xden * b``. If $A$ is square then the\n        denominator ``xden`` will be a divisor of the determinant $det(A)$.\n\n        Raises\n        ======\n\n        DMNonInvertibleMatrixError\n            If the system $Ax = b$ does not have a unique solution.\n\n        See Also\n        ========\n\n        solve_den_charpoly\n        solve_den_rref\n        inv_den\n        "
        (m, n) = self.shape
        (bm, bn) = b.shape
        if m != bm:
            raise DMShapeError('Matrix equation shape mismatch.')
        if method is None:
            method = 'rref'
        elif method == 'charpoly' and m != n:
            raise DMNonSquareMatrixError("method='charpoly' requires a square matrix.")
        if method == 'charpoly':
            (xnum, xden) = self.solve_den_charpoly(b)
        elif method == 'rref':
            (xnum, xden) = self.solve_den_rref(b)
        else:
            raise DMBadInputError("method should be 'rref' or 'charpoly'")
        return (xnum, xden)

    def solve_den_rref(self, b):
        if False:
            print('Hello World!')
        '\n        Solve matrix equation $Ax = b$ using fraction-free RREF\n\n        Solves the matrix equation $Ax = b$ for $x$ and returns the solution\n        as a numerator/denominator pair.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)\n        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)\n        >>> xnum, xden = A.solve_den_rref(b)\n        >>> xden\n        -2\n        >>> xnum\n        DomainMatrix([[8], [-9]], (2, 1), ZZ)\n        >>> A * xnum == xden * b\n        True\n\n        See Also\n        ========\n\n        solve_den\n        solve_den_charpoly\n        '
        A = self
        (m, n) = A.shape
        (bm, bn) = b.shape
        if m != bm:
            raise DMShapeError('Matrix equation shape mismatch.')
        if m < n:
            raise DMShapeError('Underdetermined matrix equation.')
        Aaug = A.hstack(b)
        (Aaug_rref, denom, pivots) = Aaug.rref_den()
        if len(pivots) != n or (pivots and pivots[-1] >= n):
            raise DMNonInvertibleMatrixError('Non-unique solution.')
        xnum = Aaug_rref[:n, n:]
        xden = denom
        return (xnum, xden)

    def solve_den_charpoly(self, b, cp=None, check=True):
        if False:
            while True:
                i = 10
        '\n        Solve matrix equation $Ax = b$ using the characteristic polynomial.\n\n        This method solves the square matrix equation $Ax = b$ for $x$ using\n        the characteristic polynomial without any division or fractions in the\n        ground domain.\n\n        Examples\n        ========\n\n        Solve a matrix equation over the integers:\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)\n        >>> b = DM([[ZZ(5)], [ZZ(6)]], ZZ)\n        >>> xnum, detA = A.solve_den_charpoly(b)\n        >>> detA\n        -2\n        >>> xnum\n        DomainMatrix([[8], [-9]], (2, 1), ZZ)\n        >>> A * xnum == detA * b\n        True\n\n        Parameters\n        ==========\n\n        self : DomainMatrix\n            The ``n x n`` matrix `A` in the equation `Ax = b`. Must be square\n            and invertible.\n        b : DomainMatrix\n            The ``n x m`` matrix `b` for the rhs.\n        cp : list, optional\n            The characteristic polynomial of the matrix `A` if known. If not\n            given, it will be computed using :meth:`charpoly`.\n        check : bool, optional\n            If ``True`` (the default) check that the determinant is not zero\n            and raise an error if it is. If ``False`` then if the determinant\n            is zero the return value will be equal to ``(A.adjugate()*b, 0)``.\n\n        Returns\n        =======\n\n        (xnum, detA) : (DomainMatrix, DomainElement)\n            The solution of the equation `Ax = b` as a matrix numerator and\n            scalar denominator pair. The denominator is equal to the\n            determinant of `A` and the numerator is ``adj(A)*b``.\n\n        The solution $x$ is given by ``x = xnum / detA``. The division free\n        invariant is ``A * xnum == detA * b``.\n\n        If ``b`` is the identity matrix, then ``xnum`` is the adjugate matrix\n        and we have ``A * adj(A) == detA * I``.\n\n        See Also\n        ========\n\n        solve_den\n            Main frontend for solving matrix equations with denominator.\n        solve_den_rref\n            Solve matrix equations using fraction-free RREF.\n        inv_den\n            Invert a matrix using the characteristic polynomial.\n        '
        (A, b) = self.unify(b)
        (m, n) = self.shape
        (mb, nb) = b.shape
        if m != n:
            raise DMNonSquareMatrixError('Matrix must be square')
        if mb != m:
            raise DMShapeError('Matrix and vector must have the same number of rows')
        (f, detA) = self.adj_poly_det(cp=cp)
        if check and (not detA):
            raise DMNonInvertibleMatrixError('Matrix is not invertible')
        adjA_b = self.eval_poly_mul(f, b)
        return (adjA_b, detA)

    def adj_poly_det(self, cp=None):
        if False:
            while True:
                i = 10
        '\n        Return the polynomial $p$ such that $p(A) = adj(A)$ and also the\n        determinant of $A$.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)\n        >>> p, detA = A.adj_poly_det()\n        >>> p\n        [-1, 5]\n        >>> p_A = A.eval_poly(p)\n        >>> p_A\n        DomainMatrix([[4, -2], [-3, 1]], (2, 2), QQ)\n        >>> p[0]*A**1 + p[1]*A**0 == p_A\n        True\n        >>> p_A == A.adjugate()\n        True\n        >>> A * A.adjugate() == detA * A.eye(A.shape, A.domain).to_dense()\n        True\n\n        See Also\n        ========\n\n        adjugate\n        eval_poly\n        adj_det\n        '
        A = self
        (m, n) = self.shape
        if m != n:
            raise DMNonSquareMatrixError('Matrix must be square')
        if cp is None:
            cp = A.charpoly()
        if len(cp) % 2:
            detA = cp[-1]
            f = [-cpi for cpi in cp[:-1]]
        else:
            detA = -cp[-1]
            f = cp[:-1]
        return (f, detA)

    def eval_poly(self, p):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate polynomial function of a matrix $p(A)$.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)\n        >>> p = [QQ(1), QQ(2), QQ(3)]\n        >>> p_A = A.eval_poly(p)\n        >>> p_A\n        DomainMatrix([[12, 14], [21, 33]], (2, 2), QQ)\n        >>> p_A == p[0]*A**2 + p[1]*A + p[2]*A**0\n        True\n\n        See Also\n        ========\n\n        eval_poly_mul\n        '
        A = self
        (m, n) = A.shape
        if m != n:
            raise DMNonSquareMatrixError('Matrix must be square')
        if not p:
            return self.zeros(self.shape, self.domain)
        elif len(p) == 1:
            return p[0] * self.eye(self.shape, self.domain)
        I = A.eye(A.shape, A.domain)
        p_A = p[0] * I
        for pi in p[1:]:
            p_A = A * p_A + pi * I
        return p_A

    def eval_poly_mul(self, p, B):
        if False:
            i = 10
            return i + 15
        "\n        Evaluate polynomial matrix product $p(A) \\times B$.\n\n        Evaluate the polynomial matrix product $p(A) \\times B$ using Horner's\n        method without creating the matrix $p(A)$ explicitly. If $B$ is a\n        column matrix then this method will only use matrix-vector multiplies\n        and no matrix-matrix multiplies are needed.\n\n        If $B$ is square or wide or if $A$ can be represented in a simpler\n        domain than $B$ then it might be faster to evaluate $p(A)$ explicitly\n        (see :func:`eval_poly`) and then multiply with $B$.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DM\n        >>> A = DM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], QQ)\n        >>> b = DM([[QQ(5)], [QQ(6)]], QQ)\n        >>> p = [QQ(1), QQ(2), QQ(3)]\n        >>> p_A_b = A.eval_poly_mul(p, b)\n        >>> p_A_b\n        DomainMatrix([[144], [303]], (2, 1), QQ)\n        >>> p_A_b == p[0]*A**2*b + p[1]*A*b + p[2]*b\n        True\n        >>> A.eval_poly_mul(p, b) == A.eval_poly(p)*b\n        True\n\n        See Also\n        ========\n\n        eval_poly\n        solve_den_charpoly\n        "
        A = self
        (m, n) = A.shape
        (mb, nb) = B.shape
        if m != n:
            raise DMNonSquareMatrixError('Matrix must be square')
        if mb != n:
            raise DMShapeError('Matrices are not aligned')
        if A.domain != B.domain:
            raise DMDomainError('Matrices must have the same domain')
        if not p:
            return B.zeros(B.shape, B.domain, fmt=B.rep.fmt)
        p_A_B = p[0] * B
        for p_i in p[1:]:
            p_A_B = A * p_A_B + p_i * B
        return p_A_B

    def lu(self):
        if False:
            return 10
        '\n        Returns Lower and Upper decomposition of the DomainMatrix\n\n        Returns\n        =======\n\n        (L, U, exchange)\n            L, U are Lower and Upper decomposition of the DomainMatrix,\n            exchange is the list of indices of rows exchanged in the\n            decomposition.\n\n        Raises\n        ======\n\n        ValueError\n            If the domain of DomainMatrix not a Field\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [QQ(1), QQ(-1)],\n        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)\n        >>> L, U, exchange = A.lu()\n        >>> L\n        DomainMatrix([[1, 0], [2, 1]], (2, 2), QQ)\n        >>> U\n        DomainMatrix([[1, -1], [0, 0]], (2, 2), QQ)\n        >>> exchange\n        []\n\n        See Also\n        ========\n\n        lu_solve\n\n        '
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        (L, U, swaps) = self.rep.lu()
        return (self.from_rep(L), self.from_rep(U), swaps)

    def lu_solve(self, rhs):
        if False:
            while True:
                i = 10
        '\n        Solver for DomainMatrix x in the A*x = B\n\n        Parameters\n        ==========\n\n        rhs : DomainMatrix B\n\n        Returns\n        =======\n\n        DomainMatrix\n            x in A*x = B\n\n        Raises\n        ======\n\n        DMShapeError\n            If the DomainMatrix A and rhs have different number of rows\n\n        ValueError\n            If the domain of DomainMatrix A not a Field\n\n        Examples\n        ========\n\n        >>> from sympy import QQ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [QQ(1), QQ(2)],\n        ...    [QQ(3), QQ(4)]], (2, 2), QQ)\n        >>> B = DomainMatrix([\n        ...    [QQ(1), QQ(1)],\n        ...    [QQ(0), QQ(1)]], (2, 2), QQ)\n\n        >>> A.lu_solve(B)\n        DomainMatrix([[-2, -1], [3/2, 1]], (2, 2), QQ)\n\n        See Also\n        ========\n\n        lu\n\n        '
        if self.shape[0] != rhs.shape[0]:
            raise DMShapeError('Shape')
        if not self.domain.is_Field:
            raise DMNotAField('Not a field')
        sol = self.rep.lu_solve(rhs.rep)
        return self.from_rep(sol)

    def _solve(A, b):
        if False:
            i = 10
            return i + 15
        if A.shape[0] != b.shape[0]:
            raise DMShapeError('Shape')
        if A.domain != b.domain or not A.domain.is_Field:
            raise DMNotAField('Not a field')
        Aaug = A.hstack(b)
        (Arref, pivots) = Aaug.rref()
        particular = Arref.from_rep(Arref.rep.particular())
        (nullspace_rep, nonpivots) = Arref[:, :-1].rep.nullspace()
        nullspace = Arref.from_rep(nullspace_rep)
        return (particular, nullspace)

    def charpoly(self):
        if False:
            return 10
        '\n        Characteristic polynomial of a square matrix.\n\n        Computes the characteristic polynomial in a fully expanded form using\n        division free arithmetic. If a factorization of the characteristic\n        polynomial is needed then it is more efficient to call\n        :meth:`charpoly_factor_list` than calling :meth:`charpoly` and then\n        factorizing the result.\n\n        Returns\n        =======\n\n        list: list of DomainElement\n            coefficients of the characteristic polynomial\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n\n        >>> A.charpoly()\n        [1, -5, -2]\n\n        See Also\n        ========\n\n        charpoly_factor_list\n            Compute the factorisation of the characteristic polynomial.\n        charpoly_factor_blocks\n            A partial factorisation of the characteristic polynomial that can\n            be computed more efficiently than either the full factorisation or\n            the fully expanded polynomial.\n        '
        M = self
        K = M.domain
        factors = M.charpoly_factor_blocks()
        cp = [K.one]
        for (f, mult) in factors:
            for _ in range(mult):
                cp = dup_mul(cp, f, K)
        return cp

    def charpoly_factor_list(self):
        if False:
            print('Hello World!')
        '\n        Full factorization of the characteristic polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import ZZ\n        >>> M = DM([[6, -1, 0, 0],\n        ...         [9, 12, 0, 0],\n        ...         [0,  0, 1, 2],\n        ...         [0,  0, 5, 6]], ZZ)\n\n        Compute the factorization of the characteristic polynomial:\n\n        >>> M.charpoly_factor_list()\n        [([1, -9], 2), ([1, -7, -4], 1)]\n\n        Use :meth:`charpoly` to get the unfactorized characteristic polynomial:\n\n        >>> M.charpoly()\n        [1, -25, 203, -495, -324]\n\n        The same calculations with ``Matrix``:\n\n        >>> M.to_Matrix().charpoly().as_expr()\n        lambda**4 - 25*lambda**3 + 203*lambda**2 - 495*lambda - 324\n        >>> M.to_Matrix().charpoly().as_expr().factor()\n        (lambda - 9)**2*(lambda**2 - 7*lambda - 4)\n\n        Returns\n        =======\n\n        list: list of pairs (factor, multiplicity)\n            A full factorization of the characteristic polynomial.\n\n        See Also\n        ========\n\n        charpoly\n            Expanded form of the characteristic polynomial.\n        charpoly_factor_blocks\n            A partial factorisation of the characteristic polynomial that can\n            be computed more efficiently.\n        '
        M = self
        K = M.domain
        factors = M.charpoly_factor_blocks()
        factors_irreducible = []
        for (factor_i, mult_i) in factors:
            (_, factors_list) = dup_factor_list(factor_i, K)
            for (factor_j, mult_j) in factors_list:
                factors_irreducible.append((factor_j, mult_i * mult_j))
        return _collect_factors(factors_irreducible)

    def charpoly_factor_blocks(self):
        if False:
            i = 10
            return i + 15
        '\n        Partial factorisation of the characteristic polynomial.\n\n        This factorisation arises from a block structure of the matrix (if any)\n        and so the factors are not guaranteed to be irreducible. The\n        :meth:`charpoly_factor_blocks` method is the most efficient way to get\n        a representation of the characteristic polynomial but the result is\n        neither fully expanded nor fully factored.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import ZZ\n        >>> M = DM([[6, -1, 0, 0],\n        ...         [9, 12, 0, 0],\n        ...         [0,  0, 1, 2],\n        ...         [0,  0, 5, 6]], ZZ)\n\n        This computes a partial factorization using only the block structure of\n        the matrix to reveal factors:\n\n        >>> M.charpoly_factor_blocks()\n        [([1, -18, 81], 1), ([1, -7, -4], 1)]\n\n        These factors correspond to the two diagonal blocks in the matrix:\n\n        >>> DM([[6, -1], [9, 12]], ZZ).charpoly()\n        [1, -18, 81]\n        >>> DM([[1, 2], [5, 6]], ZZ).charpoly()\n        [1, -7, -4]\n\n        Use :meth:`charpoly_factor_list` to get a complete factorization into\n        irreducibles:\n\n        >>> M.charpoly_factor_list()\n        [([1, -9], 2), ([1, -7, -4], 1)]\n\n        Use :meth:`charpoly` to get the expanded characteristic polynomial:\n\n        >>> M.charpoly()\n        [1, -25, 203, -495, -324]\n\n        Returns\n        =======\n\n        list: list of pairs (factor, multiplicity)\n            A partial factorization of the characteristic polynomial.\n\n        See Also\n        ========\n\n        charpoly\n            Compute the fully expanded characteristic polynomial.\n        charpoly_factor_list\n            Compute a full factorization of the characteristic polynomial.\n        '
        M = self
        if not M.is_square:
            raise DMNonSquareMatrixError('not square')
        components = M.scc()
        block_factors = []
        for indices in components:
            block = M.extract(indices, indices)
            block_factors.append((block.charpoly_base(), 1))
        return _collect_factors(block_factors)

    def charpoly_base(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Base case for :meth:`charpoly_factor_blocks` after block decomposition.\n\n        This method is used internally by :meth:`charpoly_factor_blocks` as the\n        base case for computing the characteristic polynomial of a block. It is\n        more efficient to call :meth:`charpoly_factor_blocks`, :meth:`charpoly`\n        or :meth:`charpoly_factor_list` rather than call this method directly.\n\n        This will use either the dense or the sparse implementation depending\n        on the sparsity of the matrix and will clear denominators if possible\n        before calling :meth:`charpoly_berk` to compute the characteristic\n        polynomial using the Berkowitz algorithm.\n\n        See Also\n        ========\n\n        charpoly\n        charpoly_factor_list\n        charpoly_factor_blocks\n        charpoly_berk\n        '
        M = self
        K = M.domain
        density = self.nnz() / self.shape[0] ** 2
        if density < 0.5:
            M = M.to_sparse()
        else:
            M = M.to_dense()
        clear_denoms = K.is_Field and K.has_assoc_Ring
        if clear_denoms:
            clear_denoms = True
            (d, M) = M.clear_denoms(convert=True)
            d = d.element
            K_f = K
            K_r = M.domain
        cp = M.charpoly_berk()
        if clear_denoms:
            cp = dup_convert(cp, K_r, K_f)
            p = [K_f.one, K_f.zero]
            q = [K_f.one / d]
            cp = dup_transform(cp, p, q, K_f)
        return cp

    def charpoly_berk(self):
        if False:
            print('Hello World!')
        'Compute the characteristic polynomial using the Berkowitz algorithm.\n\n        This method directly calls the underlying implementation of the\n        Berkowitz algorithm (:meth:`sympy.polys.matrices.dense.ddm_berk` or\n        :meth:`sympy.polys.matrices.sdm.sdm_berk`).\n\n        This is used by :meth:`charpoly` and other methods as the base case for\n        for computing the characteristic polynomial. However those methods will\n        apply other optimizations such as block decomposition, clearing\n        denominators and converting between dense and sparse representations\n        before calling this method. It is more efficient to call those methods\n        instead of this one but this method is provided for direct access to\n        the Berkowitz algorithm.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DM\n        >>> from sympy import QQ\n        >>> M = DM([[6, -1, 0, 0],\n        ...         [9, 12, 0, 0],\n        ...         [0,  0, 1, 2],\n        ...         [0,  0, 5, 6]], QQ)\n        >>> M.charpoly_berk()\n        [1, -25, 203, -495, -324]\n\n        See Also\n        ========\n\n        charpoly\n        charpoly_base\n        charpoly_factor_list\n        charpoly_factor_blocks\n        sympy.polys.matrices.dense.ddm_berk\n        sympy.polys.matrices.sdm.sdm_berk\n        '
        return self.rep.charpoly()

    @classmethod
    def eye(cls, shape, domain):
        if False:
            while True:
                i = 10
        '\n        Return identity matrix of size n or shape (m, n).\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> DomainMatrix.eye(3, QQ)\n        DomainMatrix({0: {0: 1}, 1: {1: 1}, 2: {2: 1}}, (3, 3), QQ)\n\n        '
        if isinstance(shape, int):
            shape = (shape, shape)
        return cls.from_rep(SDM.eye(shape, domain))

    @classmethod
    def diag(cls, diagonal, domain, shape=None):
        if False:
            return 10
        '\n        Return diagonal matrix with entries from ``diagonal``.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import ZZ\n        >>> DomainMatrix.diag([ZZ(5), ZZ(6)], ZZ)\n        DomainMatrix({0: {0: 5}, 1: {1: 6}}, (2, 2), ZZ)\n\n        '
        if shape is None:
            N = len(diagonal)
            shape = (N, N)
        return cls.from_rep(SDM.diag(diagonal, domain, shape))

    @classmethod
    def zeros(cls, shape, domain, *, fmt='sparse'):
        if False:
            print('Hello World!')
        'Returns a zero DomainMatrix of size shape, belonging to the specified domain\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> DomainMatrix.zeros((2, 3), QQ)\n        DomainMatrix({}, (2, 3), QQ)\n\n        '
        return cls.from_rep(SDM.zeros(shape, domain))

    @classmethod
    def ones(cls, shape, domain):
        if False:
            while True:
                i = 10
        'Returns a DomainMatrix of 1s, of size shape, belonging to the specified domain\n\n        Examples\n        ========\n\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> from sympy import QQ\n        >>> DomainMatrix.ones((2,3), QQ)\n        DomainMatrix([[1, 1, 1], [1, 1, 1]], (2, 3), QQ)\n\n        '
        return cls.from_rep(DDM.ones(shape, domain).to_dfm_or_ddm())

    def __eq__(A, B):
        if False:
            print('Hello World!')
        '\n        Checks for two DomainMatrix matrices to be equal or not\n\n        Parameters\n        ==========\n\n        A, B: DomainMatrix\n            to check equality\n\n        Returns\n        =======\n\n        Boolean\n            True for equal, else False\n\n        Raises\n        ======\n\n        NotImplementedError\n            If B is not a DomainMatrix\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> from sympy.polys.matrices import DomainMatrix\n        >>> A = DomainMatrix([\n        ...    [ZZ(1), ZZ(2)],\n        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)\n        >>> B = DomainMatrix([\n        ...    [ZZ(1), ZZ(1)],\n        ...    [ZZ(0), ZZ(1)]], (2, 2), ZZ)\n        >>> A.__eq__(A)\n        True\n        >>> A.__eq__(B)\n        False\n\n        '
        if not isinstance(A, type(B)):
            return NotImplemented
        return A.domain == B.domain and A.rep == B.rep

    def unify_eq(A, B):
        if False:
            i = 10
            return i + 15
        if A.shape != B.shape:
            return False
        if A.domain != B.domain:
            (A, B) = A.unify(B)
        return A == B

    def lll(A, delta=QQ(3, 4)):
        if False:
            return 10
        '\n        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.\n        See [1]_ and [2]_.\n\n        Parameters\n        ==========\n\n        delta : QQ, optional\n            The Lovász parameter. Must be in the interval (0.25, 1), with larger\n            values producing a more reduced basis. The default is 0.75 for\n            historical reasons.\n\n        Returns\n        =======\n\n        The reduced basis as a DomainMatrix over ZZ.\n\n        Throws\n        ======\n\n        DMValueError: if delta is not in the range (0.25, 1)\n        DMShapeError: if the matrix is not of shape (m, n) with m <= n\n        DMDomainError: if the matrix domain is not ZZ\n        DMRankError: if the matrix contains linearly dependent rows\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ, QQ\n        >>> from sympy.polys.matrices import DM\n        >>> x = DM([[1, 0, 0, 0, -20160],\n        ...         [0, 1, 0, 0, 33768],\n        ...         [0, 0, 1, 0, 39578],\n        ...         [0, 0, 0, 1, 47757]], ZZ)\n        >>> y = DM([[10, -3, -2, 8, -4],\n        ...         [3, -9, 8, 1, -11],\n        ...         [-3, 13, -9, -3, -9],\n        ...         [-12, -7, -11, 9, -1]], ZZ)\n        >>> assert x.lll(delta=QQ(5, 6)) == y\n\n        Notes\n        =====\n\n        The implementation is derived from the Maple code given in Figures 4.3\n        and 4.4 of [3]_ (pp.68-69). It uses the efficient method of only calculating\n        state updates as they are required.\n\n        See also\n        ========\n\n        lll_transform\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm\n        .. [2] https://web.archive.org/web/20221029115428/https://web.cs.elte.hu/~lovasz/scans/lll.pdf\n        .. [3] Murray R. Bremner, "Lattice Basis Reduction: An Introduction to the LLL Algorithm and Its Applications"\n\n        '
        return DomainMatrix.from_rep(A.rep.lll(delta=delta))

    def lll_transform(A, delta=QQ(3, 4)):
        if False:
            print('Hello World!')
        '\n        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm\n        and returns the reduced basis and transformation matrix.\n\n        Explanation\n        ===========\n\n        Parameters, algorithm and basis are the same as for :meth:`lll` except that\n        the return value is a tuple `(B, T)` with `B` the reduced basis and\n        `T` a transformation matrix. The original basis `A` is transformed to\n        `B` with `T*A == B`. If only `B` is needed then :meth:`lll` should be\n        used as it is a little faster.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ, QQ\n        >>> from sympy.polys.matrices import DM\n        >>> X = DM([[1, 0, 0, 0, -20160],\n        ...         [0, 1, 0, 0, 33768],\n        ...         [0, 0, 1, 0, 39578],\n        ...         [0, 0, 0, 1, 47757]], ZZ)\n        >>> B, T = X.lll_transform(delta=QQ(5, 6))\n        >>> T * X == B\n        True\n\n        See also\n        ========\n\n        lll\n\n        '
        (reduced, transform) = A.rep.lll_transform(delta=delta)
        return (DomainMatrix.from_rep(reduced), DomainMatrix.from_rep(transform))

def _collect_factors(factors_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Collect repeating factors and sort.\n\n    >>> from sympy.polys.matrices.domainmatrix import _collect_factors\n    >>> _collect_factors([([1, 2], 2), ([1, 4], 3), ([1, 2], 5)])\n    [([1, 4], 3), ([1, 2], 7)]\n    '
    factors = Counter()
    for (factor, exponent) in factors_list:
        factors[tuple(factor)] += exponent
    factors_list = [(list(f), e) for (f, e) in factors.items()]
    return _sort_factors(factors_list)