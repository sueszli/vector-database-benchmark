from collections import defaultdict
from operator import index as index_
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer, Rational
from sympy.core.sympify import _sympify, SympifyError
from sympy.core.singleton import S
from sympy.polys.domains import ZZ, QQ, GF, EXRAW
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent, as_int
from .common import classof, NonSquareMatrixError, NonInvertibleMatrixError
from .matrices import MatrixBase, MatrixKind, ShapeError

class RepMatrix(MatrixBase):
    """Matrix implementation based on DomainMatrix as an internal representation.

    The RepMatrix class is a superclass for Matrix, ImmutableMatrix,
    SparseMatrix and ImmutableSparseMatrix which are the main usable matrix
    classes in SymPy. Most methods on this class are simply forwarded to
    DomainMatrix.
    """
    _rep: DomainMatrix

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, RepMatrix):
            try:
                other = _sympify(other)
            except SympifyError:
                return NotImplemented
            if not isinstance(other, RepMatrix):
                return NotImplemented
        return self._rep.unify_eq(other._rep)

    def to_DM(self, domain=None, **kwargs):
        if False:
            while True:
                i = 10
        "Convert to a :class:`~.DomainMatrix`.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 2], [3, 4]])\n        >>> M.to_DM()\n        DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)\n\n        The :meth:`DomainMatrix.to_Matrix` method can be used to convert back:\n\n        >>> M.to_DM().to_Matrix() == M\n        True\n\n        The domain can be given explicitly or otherwise it will be chosen by\n        :func:`construct_domain`. Any keyword arguments (besides ``domain``)\n        are passed to :func:`construct_domain`:\n\n        >>> from sympy import QQ, symbols\n        >>> x = symbols('x')\n        >>> M = Matrix([[x, 1], [1, x]])\n        >>> M\n        Matrix([\n        [x, 1],\n        [1, x]])\n        >>> M.to_DM().domain\n        ZZ[x]\n        >>> M.to_DM(field=True).domain\n        ZZ(x)\n        >>> M.to_DM(domain=QQ[x]).domain\n        QQ[x]\n\n        See Also\n        ========\n\n        DomainMatrix\n        DomainMatrix.to_Matrix\n        DomainMatrix.convert_to\n        DomainMatrix.choose_domain\n        construct_domain\n        "
        if domain is not None:
            if kwargs:
                raise TypeError('Options cannot be used with domain parameter')
            return self._rep.convert_to(domain)
        rep = self._rep
        dom = rep.domain
        if not kwargs:
            if dom.is_ZZ:
                return rep.copy()
            elif dom.is_QQ:
                try:
                    return rep.convert_to(ZZ)
                except CoercionFailed:
                    pass
                return rep.copy()
        rep_dom = rep.choose_domain(**kwargs)
        if rep_dom.domain.is_EX:
            rep_dom = rep_dom.convert_to(EXRAW)
        return rep_dom

    @classmethod
    def _unify_element_sympy(cls, rep, element):
        if False:
            return 10
        domain = rep.domain
        element = _sympify(element)
        if domain != EXRAW:
            if element.is_Integer:
                new_domain = domain
            elif element.is_Rational:
                new_domain = QQ
            else:
                new_domain = EXRAW
            if new_domain != domain:
                rep = rep.convert_to(new_domain)
                domain = new_domain
            if domain != EXRAW:
                element = new_domain.from_sympy(element)
        if domain == EXRAW and (not isinstance(element, Expr)):
            sympy_deprecation_warning('\n                non-Expr objects in a Matrix is deprecated. Matrix represents\n                a mathematical matrix. To represent a container of non-numeric\n                entities, Use a list of lists, TableForm, NumPy array, or some\n                other data structure instead.\n                ', deprecated_since_version='1.9', active_deprecations_target='deprecated-non-expr-in-matrix', stacklevel=4)
        return (rep, element)

    @classmethod
    def _dod_to_DomainMatrix(cls, rows, cols, dod, types):
        if False:
            while True:
                i = 10
        if not all((issubclass(typ, Expr) for typ in types)):
            sympy_deprecation_warning('\n                non-Expr objects in a Matrix is deprecated. Matrix represents\n                a mathematical matrix. To represent a container of non-numeric\n                entities, Use a list of lists, TableForm, NumPy array, or some\n                other data structure instead.\n                ', deprecated_since_version='1.9', active_deprecations_target='deprecated-non-expr-in-matrix', stacklevel=6)
        rep = DomainMatrix(dod, (rows, cols), EXRAW)
        if all((issubclass(typ, Rational) for typ in types)):
            if all((issubclass(typ, Integer) for typ in types)):
                rep = rep.convert_to(ZZ)
            else:
                rep = rep.convert_to(QQ)
        return rep

    @classmethod
    def _flat_list_to_DomainMatrix(cls, rows, cols, flat_list):
        if False:
            i = 10
            return i + 15
        elements_dod = defaultdict(dict)
        for (n, element) in enumerate(flat_list):
            if element != 0:
                (i, j) = divmod(n, cols)
                elements_dod[i][j] = element
        types = set(map(type, flat_list))
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    @classmethod
    def _smat_to_DomainMatrix(cls, rows, cols, smat):
        if False:
            i = 10
            return i + 15
        elements_dod = defaultdict(dict)
        for ((i, j), element) in smat.items():
            if element != 0:
                elements_dod[i][j] = element
        types = set(map(type, smat.values()))
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep

    def flat(self):
        if False:
            print('Hello World!')
        return self._rep.to_sympy().to_list_flat()

    def _eval_tolist(self):
        if False:
            return 10
        return self._rep.to_sympy().to_list()

    def _eval_todok(self):
        if False:
            print('Hello World!')
        return self._rep.to_sympy().to_dok()

    def _eval_values(self):
        if False:
            print('Hello World!')
        return list(self.todok().values())

    def copy(self):
        if False:
            while True:
                i = 10
        return self._fromrep(self._rep.copy())

    @property
    def kind(self) -> MatrixKind:
        if False:
            for i in range(10):
                print('nop')
        domain = self._rep.domain
        element_kind: Kind
        if domain in (ZZ, QQ):
            element_kind = NumberKind
        elif domain == EXRAW:
            kinds = {e.kind for e in self.values()}
            if len(kinds) == 1:
                [element_kind] = kinds
            else:
                element_kind = UndefinedKind
        else:
            raise RuntimeError('Domain should only be ZZ, QQ or EXRAW')
        return MatrixKind(element_kind)

    def _eval_has(self, *patterns):
        if False:
            for i in range(10):
                print('nop')
        zhas = False
        dok = self.todok()
        if len(dok) != self.rows * self.cols:
            zhas = S.Zero.has(*patterns)
        return zhas or any((value.has(*patterns) for value in dok.values()))

    def _eval_is_Identity(self):
        if False:
            return 10
        if not all((self[i, i] == 1 for i in range(self.rows))):
            return False
        return len(self.todok()) == self.rows

    def _eval_is_symmetric(self, simpfunc):
        if False:
            print('Hello World!')
        diff = (self - self.T).applyfunc(simpfunc)
        return len(diff.values()) == 0

    def _eval_transpose(self):
        if False:
            return 10
        'Returns the transposed SparseMatrix of this SparseMatrix.\n\n        Examples\n        ========\n\n        >>> from sympy import SparseMatrix\n        >>> a = SparseMatrix(((1, 2), (3, 4)))\n        >>> a\n        Matrix([\n        [1, 2],\n        [3, 4]])\n        >>> a.T\n        Matrix([\n        [1, 3],\n        [2, 4]])\n        '
        return self._fromrep(self._rep.transpose())

    def _eval_col_join(self, other):
        if False:
            return 10
        return self._fromrep(self._rep.vstack(other._rep))

    def _eval_row_join(self, other):
        if False:
            i = 10
            return i + 15
        return self._fromrep(self._rep.hstack(other._rep))

    def _eval_extract(self, rowsList, colsList):
        if False:
            return 10
        return self._fromrep(self._rep.extract(rowsList, colsList))

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return _getitem_RepMatrix(self, key)

    @classmethod
    def _eval_zeros(cls, rows, cols):
        if False:
            i = 10
            return i + 15
        rep = DomainMatrix.zeros((rows, cols), ZZ)
        return cls._fromrep(rep)

    @classmethod
    def _eval_eye(cls, rows, cols):
        if False:
            print('Hello World!')
        rep = DomainMatrix.eye((rows, cols), ZZ)
        return cls._fromrep(rep)

    def _eval_add(self, other):
        if False:
            for i in range(10):
                print('nop')
        return classof(self, other)._fromrep(self._rep + other._rep)

    def _eval_matrix_mul(self, other):
        if False:
            print('Hello World!')
        return classof(self, other)._fromrep(self._rep * other._rep)

    def _eval_matrix_mul_elementwise(self, other):
        if False:
            for i in range(10):
                print('nop')
        (selfrep, otherrep) = self._rep.unify(other._rep)
        newrep = selfrep.mul_elementwise(otherrep)
        return classof(self, other)._fromrep(newrep)

    def _eval_scalar_mul(self, other):
        if False:
            for i in range(10):
                print('nop')
        (rep, other) = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.scalarmul(other))

    def _eval_scalar_rmul(self, other):
        if False:
            for i in range(10):
                print('nop')
        (rep, other) = self._unify_element_sympy(self._rep, other)
        return self._fromrep(rep.rscalarmul(other))

    def _eval_Abs(self):
        if False:
            while True:
                i = 10
        return self._fromrep(self._rep.applyfunc(abs))

    def _eval_conjugate(self):
        if False:
            while True:
                i = 10
        rep = self._rep
        domain = rep.domain
        if domain in (ZZ, QQ):
            return self.copy()
        else:
            return self._fromrep(rep.applyfunc(lambda e: e.conjugate()))

    def equals(self, other, failing_expression=False):
        if False:
            for i in range(10):
                print('nop')
        'Applies ``equals`` to corresponding elements of the matrices,\n        trying to prove that the elements are equivalent, returning True\n        if they are, False if any pair is not, and None (or the first\n        failing expression if failing_expression is True) if it cannot\n        be decided if the expressions are equivalent or not. This is, in\n        general, an expensive operation.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.abc import x\n        >>> A = Matrix([x*(x - 1), 0])\n        >>> B = Matrix([x**2 - x, 0])\n        >>> A == B\n        False\n        >>> A.simplify() == B.simplify()\n        True\n        >>> A.equals(B)\n        True\n        >>> A.equals(2)\n        False\n\n        See Also\n        ========\n        sympy.core.expr.Expr.equals\n        '
        if self.shape != getattr(other, 'shape', None):
            return False
        rv = True
        for i in range(self.rows):
            for j in range(self.cols):
                ans = self[i, j].equals(other[i, j], failing_expression)
                if ans is False:
                    return False
                elif ans is not True and rv is True:
                    rv = ans
        return rv

    def inv_mod(M, m):
        if False:
            i = 10
            return i + 15
        '\n        Returns the inverse of the integer matrix ``M`` modulo ``m``.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> A = Matrix(2, 2, [1, 2, 3, 4])\n        >>> A.inv_mod(5)\n        Matrix([\n        [3, 1],\n        [4, 2]])\n        >>> A.inv_mod(3)\n        Matrix([\n        [1, 1],\n        [0, 1]])\n\n        '
        if not M.is_square:
            raise NonSquareMatrixError()
        try:
            m = as_int(m)
        except ValueError:
            raise TypeError('inv_mod: modulus m must be an integer')
        K = GF(m, symmetric=False)
        try:
            dM = M.to_DM(K)
        except CoercionFailed:
            raise ValueError('inv_mod: matrix entries must be integers')
        try:
            dMi = dM.inv()
        except DMNonInvertibleMatrixError as exc:
            msg = f'Matrix is not invertible (mod {m})'
            raise NonInvertibleMatrixError(msg) from exc
        return dMi.to_Matrix()

    def lll(self, delta=0.75):
        if False:
            return 10
        'LLL-reduced basis for the rowspace of a matrix of integers.\n\n        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.\n\n        The implementation is provided by :class:`~DomainMatrix`. See\n        :meth:`~DomainMatrix.lll` for more details.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 0, 0, 0, -20160],\n        ...             [0, 1, 0, 0, 33768],\n        ...             [0, 0, 1, 0, 39578],\n        ...             [0, 0, 0, 1, 47757]])\n        >>> M.lll()\n        Matrix([\n        [ 10, -3,  -2,  8,  -4],\n        [  3, -9,   8,  1, -11],\n        [ -3, 13,  -9, -3,  -9],\n        [-12, -7, -11,  9,  -1]])\n\n        See Also\n        ========\n\n        lll_transform\n        sympy.polys.matrices.domainmatrix.DomainMatrix.lll\n        '
        delta = QQ.from_sympy(_sympify(delta))
        dM = self._rep.convert_to(ZZ)
        basis = dM.lll(delta=delta)
        return self._fromrep(basis)

    def lll_transform(self, delta=0.75):
        if False:
            while True:
                i = 10
        'LLL-reduced basis and transformation matrix.\n\n        Performs the Lenstra–Lenstra–Lovász (LLL) basis reduction algorithm.\n\n        The implementation is provided by :class:`~DomainMatrix`. See\n        :meth:`~DomainMatrix.lll_transform` for more details.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 0, 0, 0, -20160],\n        ...             [0, 1, 0, 0, 33768],\n        ...             [0, 0, 1, 0, 39578],\n        ...             [0, 0, 0, 1, 47757]])\n        >>> B, T = M.lll_transform()\n        >>> B\n        Matrix([\n        [ 10, -3,  -2,  8,  -4],\n        [  3, -9,   8,  1, -11],\n        [ -3, 13,  -9, -3,  -9],\n        [-12, -7, -11,  9,  -1]])\n        >>> T\n        Matrix([\n        [ 10, -3,  -2,  8],\n        [  3, -9,   8,  1],\n        [ -3, 13,  -9, -3],\n        [-12, -7, -11,  9]])\n\n        The transformation matrix maps the original basis to the LLL-reduced\n        basis:\n\n        >>> T * M == B\n        True\n\n        See Also\n        ========\n\n        lll\n        sympy.polys.matrices.domainmatrix.DomainMatrix.lll_transform\n        '
        delta = QQ.from_sympy(_sympify(delta))
        dM = self._rep.convert_to(ZZ)
        (basis, transform) = dM.lll_transform(delta=delta)
        B = self._fromrep(basis)
        T = self._fromrep(transform)
        return (B, T)

class MutableRepMatrix(RepMatrix):
    """Mutable matrix based on DomainMatrix as the internal representation"""
    is_zero = False

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return cls._new(*args, **kwargs)

    @classmethod
    def _new(cls, *args, copy=True, **kwargs):
        if False:
            while True:
                i = 10
        if copy is False:
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            (rows, cols, flat_list) = args
        else:
            (rows, cols, flat_list) = cls._handle_creation_inputs(*args, **kwargs)
            flat_list = list(flat_list)
        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)
        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        if False:
            while True:
                i = 10
        obj = super().__new__(cls)
        (obj.rows, obj.cols) = rep.shape
        obj._rep = rep
        return obj

    def copy(self):
        if False:
            return 10
        return self._fromrep(self._rep.copy())

    def as_mutable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.copy()

    def __setitem__(self, key, value):
        if False:
            while True:
                i = 10
        '\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, I, zeros, ones\n        >>> m = Matrix(((1, 2+I), (3, 4)))\n        >>> m\n        Matrix([\n        [1, 2 + I],\n        [3,     4]])\n        >>> m[1, 0] = 9\n        >>> m\n        Matrix([\n        [1, 2 + I],\n        [9,     4]])\n        >>> m[1, 0] = [[0, 1]]\n\n        To replace row r you assign to position r*m where m\n        is the number of columns:\n\n        >>> M = zeros(4)\n        >>> m = M.cols\n        >>> M[3*m] = ones(1, m)*2; M\n        Matrix([\n        [0, 0, 0, 0],\n        [0, 0, 0, 0],\n        [0, 0, 0, 0],\n        [2, 2, 2, 2]])\n\n        And to replace column c you can assign to position c:\n\n        >>> M[2] = ones(m, 1)*4; M\n        Matrix([\n        [0, 0, 4, 0],\n        [0, 0, 4, 0],\n        [0, 0, 4, 0],\n        [2, 2, 4, 2]])\n        '
        rv = self._setitem(key, value)
        if rv is not None:
            (i, j, value) = rv
            (self._rep, value) = self._unify_element_sympy(self._rep, value)
            self._rep.rep.setitem(i, j, value)

    def _eval_col_del(self, col):
        if False:
            return 10
        self._rep = DomainMatrix.hstack(self._rep[:, :col], self._rep[:, col + 1:])
        self.cols -= 1

    def _eval_row_del(self, row):
        if False:
            while True:
                i = 10
        self._rep = DomainMatrix.vstack(self._rep[:row, :], self._rep[row + 1:, :])
        self.rows -= 1

    def _eval_col_insert(self, col, other):
        if False:
            print('Hello World!')
        other = self._new(other)
        return self.hstack(self[:, :col], other, self[:, col:])

    def _eval_row_insert(self, row, other):
        if False:
            print('Hello World!')
        other = self._new(other)
        return self.vstack(self[:row, :], other, self[row:, :])

    def col_op(self, j, f):
        if False:
            i = 10
            return i + 15
        'In-place operation on col j using two-arg functor whose args are\n        interpreted as (self[i, j], i).\n\n        Examples\n        ========\n\n        >>> from sympy import eye\n        >>> M = eye(3)\n        >>> M.col_op(1, lambda v, i: v + 2*M[i, 0]); M\n        Matrix([\n        [1, 2, 0],\n        [0, 1, 0],\n        [0, 0, 1]])\n\n        See Also\n        ========\n        col\n        row_op\n        '
        for i in range(self.rows):
            self[i, j] = f(self[i, j], i)

    def col_swap(self, i, j):
        if False:
            for i in range(10):
                print('nop')
        'Swap the two given columns of the matrix in-place.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 0], [1, 0]])\n        >>> M\n        Matrix([\n        [1, 0],\n        [1, 0]])\n        >>> M.col_swap(0, 1)\n        >>> M\n        Matrix([\n        [0, 1],\n        [0, 1]])\n\n        See Also\n        ========\n\n        col\n        row_swap\n        '
        for k in range(0, self.rows):
            (self[k, i], self[k, j]) = (self[k, j], self[k, i])

    def row_op(self, i, f):
        if False:
            i = 10
            return i + 15
        'In-place operation on row ``i`` using two-arg functor whose args are\n        interpreted as ``(self[i, j], j)``.\n\n        Examples\n        ========\n\n        >>> from sympy import eye\n        >>> M = eye(3)\n        >>> M.row_op(1, lambda v, j: v + 2*M[0, j]); M\n        Matrix([\n        [1, 0, 0],\n        [2, 1, 0],\n        [0, 0, 1]])\n\n        See Also\n        ========\n        row\n        zip_row_op\n        col_op\n\n        '
        for j in range(self.cols):
            self[i, j] = f(self[i, j], j)

    def row_mult(self, i, factor):
        if False:
            i = 10
            return i + 15
        'Multiply the given row by the given factor in-place.\n\n        Examples\n        ========\n\n        >>> from sympy import eye\n        >>> M = eye(3)\n        >>> M.row_mult(1,7); M\n        Matrix([\n        [1, 0, 0],\n        [0, 7, 0],\n        [0, 0, 1]])\n\n        '
        for j in range(self.cols):
            self[i, j] *= factor

    def row_add(self, s, t, k):
        if False:
            return 10
        'Add k times row s (source) to row t (target) in place.\n\n        Examples\n        ========\n\n        >>> from sympy import eye\n        >>> M = eye(3)\n        >>> M.row_add(0, 2,3); M\n        Matrix([\n        [1, 0, 0],\n        [0, 1, 0],\n        [3, 0, 1]])\n        '
        for j in range(self.cols):
            self[t, j] += k * self[s, j]

    def row_swap(self, i, j):
        if False:
            return 10
        'Swap the two given rows of the matrix in-place.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[0, 1], [1, 0]])\n        >>> M\n        Matrix([\n        [0, 1],\n        [1, 0]])\n        >>> M.row_swap(0, 1)\n        >>> M\n        Matrix([\n        [1, 0],\n        [0, 1]])\n\n        See Also\n        ========\n\n        row\n        col_swap\n        '
        for k in range(0, self.cols):
            (self[i, k], self[j, k]) = (self[j, k], self[i, k])

    def zip_row_op(self, i, k, f):
        if False:
            while True:
                i = 10
        'In-place operation on row ``i`` using two-arg functor whose args are\n        interpreted as ``(self[i, j], self[k, j])``.\n\n        Examples\n        ========\n\n        >>> from sympy import eye\n        >>> M = eye(3)\n        >>> M.zip_row_op(1, 0, lambda v, u: v + 2*u); M\n        Matrix([\n        [1, 0, 0],\n        [2, 1, 0],\n        [0, 0, 1]])\n\n        See Also\n        ========\n        row\n        row_op\n        col_op\n\n        '
        for j in range(self.cols):
            self[i, j] = f(self[i, j], self[k, j])

    def copyin_list(self, key, value):
        if False:
            while True:
                i = 10
        'Copy in elements from a list.\n\n        Parameters\n        ==========\n\n        key : slice\n            The section of this matrix to replace.\n        value : iterable\n            The iterable to copy values from.\n\n        Examples\n        ========\n\n        >>> from sympy import eye\n        >>> I = eye(3)\n        >>> I[:2, 0] = [1, 2] # col\n        >>> I\n        Matrix([\n        [1, 0, 0],\n        [2, 1, 0],\n        [0, 0, 1]])\n        >>> I[1, :2] = [[3, 4]]\n        >>> I\n        Matrix([\n        [1, 0, 0],\n        [3, 4, 0],\n        [0, 0, 1]])\n\n        See Also\n        ========\n\n        copyin_matrix\n        '
        if not is_sequence(value):
            raise TypeError('`value` must be an ordered iterable, not %s.' % type(value))
        return self.copyin_matrix(key, type(self)(value))

    def copyin_matrix(self, key, value):
        if False:
            return 10
        'Copy in values from a matrix into the given bounds.\n\n        Parameters\n        ==========\n\n        key : slice\n            The section of this matrix to replace.\n        value : Matrix\n            The matrix to copy values from.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, eye\n        >>> M = Matrix([[0, 1], [2, 3], [4, 5]])\n        >>> I = eye(3)\n        >>> I[:3, :2] = M\n        >>> I\n        Matrix([\n        [0, 1, 0],\n        [2, 3, 0],\n        [4, 5, 1]])\n        >>> I[0, 1] = M\n        >>> I\n        Matrix([\n        [0, 0, 1],\n        [2, 2, 3],\n        [4, 4, 5]])\n\n        See Also\n        ========\n\n        copyin_list\n        '
        (rlo, rhi, clo, chi) = self.key2bounds(key)
        shape = value.shape
        (dr, dc) = (rhi - rlo, chi - clo)
        if shape != (dr, dc):
            raise ShapeError(filldedent("The Matrix `value` doesn't have the same dimensions as the in sub-Matrix given by `key`."))
        for i in range(value.rows):
            for j in range(value.cols):
                self[i + rlo, j + clo] = value[i, j]

    def fill(self, value):
        if False:
            print('Hello World!')
        'Fill self with the given value.\n\n        Notes\n        =====\n\n        Unless many values are going to be deleted (i.e. set to zero)\n        this will create a matrix that is slower than a dense matrix in\n        operations.\n\n        Examples\n        ========\n\n        >>> from sympy import SparseMatrix\n        >>> M = SparseMatrix.zeros(3); M\n        Matrix([\n        [0, 0, 0],\n        [0, 0, 0],\n        [0, 0, 0]])\n        >>> M.fill(1); M\n        Matrix([\n        [1, 1, 1],\n        [1, 1, 1],\n        [1, 1, 1]])\n\n        See Also\n        ========\n\n        zeros\n        ones\n        '
        value = _sympify(value)
        if not value:
            self._rep = DomainMatrix.zeros(self.shape, EXRAW)
        else:
            elements_dod = {i: {j: value for j in range(self.cols)} for i in range(self.rows)}
            self._rep = DomainMatrix(elements_dod, self.shape, EXRAW)

def _getitem_RepMatrix(self, key):
    if False:
        for i in range(10):
            print('nop')
    'Return portion of self defined by key. If the key involves a slice\n    then a list will be returned (if key is a single slice) or a matrix\n    (if key was a tuple involving a slice).\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, I\n    >>> m = Matrix([\n    ... [1, 2 + I],\n    ... [3, 4    ]])\n\n    If the key is a tuple that does not involve a slice then that element\n    is returned:\n\n    >>> m[1, 0]\n    3\n\n    When a tuple key involves a slice, a matrix is returned. Here, the\n    first column is selected (all rows, column 0):\n\n    >>> m[:, 0]\n    Matrix([\n    [1],\n    [3]])\n\n    If the slice is not a tuple then it selects from the underlying\n    list of elements that are arranged in row order and a list is\n    returned if a slice is involved:\n\n    >>> m[0]\n    1\n    >>> m[::2]\n    [1, 3]\n    '
    if isinstance(key, tuple):
        (i, j) = key
        try:
            return self._rep.getitem_sympy(index_(i), index_(j))
        except (TypeError, IndexError):
            if isinstance(i, Expr) and (not i.is_number) or (isinstance(j, Expr) and (not j.is_number)):
                if (j < 0) is True or (j >= self.shape[1]) is True or (i < 0) is True or ((i >= self.shape[0]) is True):
                    raise ValueError('index out of boundary')
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
            if isinstance(i, slice):
                i = range(self.rows)[i]
            elif is_sequence(i):
                pass
            else:
                i = [i]
            if isinstance(j, slice):
                j = range(self.cols)[j]
            elif is_sequence(j):
                pass
            else:
                j = [j]
            return self.extract(i, j)
    else:
        (rows, cols) = self.shape
        if not rows * cols:
            return [][key]
        rep = self._rep.rep
        domain = rep.domain
        is_slice = isinstance(key, slice)
        if is_slice:
            values = [rep.getitem(*divmod(n, cols)) for n in range(rows * cols)[key]]
        else:
            values = [rep.getitem(*divmod(index_(key), cols))]
        if domain != EXRAW:
            to_sympy = domain.to_sympy
            values = [to_sympy(val) for val in values]
        if is_slice:
            return values
        else:
            return values[0]