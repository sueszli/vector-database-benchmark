import mpmath as mp
from collections.abc import Callable
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import diff
from sympy.core.expr import Expr
from sympy.core.kind import _NumberKind, UndefinedKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, uniquely_named_symbol
from sympy.core.sympify import sympify, _sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta, LeviCivita
from sympy.polys import cancel
from sympy.printing import sstr
from sympy.printing.defaults import Printable
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import flatten, NotIterable, is_sequence, reshape
from sympy.utilities.misc import as_int, filldedent
from .common import MatrixCommon, MatrixError, NonSquareMatrixError, NonInvertibleMatrixError, ShapeError, MatrixKind, a2idx
from .utilities import _iszero, _is_zero_after_expand_mul, _simplify
from .determinant import _find_reasonable_pivot, _find_reasonable_pivot_naive, _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per, _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU, _minor, _minor_submatrix
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import _eigenvals, _eigenvects, _bidiagonalize, _bidiagonal_decomposition, _is_diagonalizable, _diagonalize, _is_positive_definite, _is_positive_semidefinite, _is_negative_definite, _is_negative_semidefinite, _is_indefinite, _jordan_form, _left_eigenvects, _singular_values
from .decompositions import _rank_decomposition, _cholesky, _LDLdecomposition, _LUdecomposition, _LUdecomposition_Simple, _LUdecompositionFF, _singular_value_decomposition, _QRdecomposition, _upper_hessenberg_decomposition
from .graph import _connected_components, _connected_components_decomposition, _strongly_connected_components, _strongly_connected_components_decomposition
from .solvers import _diagonal_solve, _lower_triangular_solve, _upper_triangular_solve, _cholesky_solve, _LDLsolve, _LUsolve, _QRsolve, _gauss_jordan_solve, _pinv_solve, _cramer_solve, _solve, _solve_least_squares
from .inverse import _pinv, _inv_ADJ, _inv_GE, _inv_LU, _inv_CH, _inv_LDL, _inv_QR, _inv, _inv_block

class DeferredVector(Symbol, NotIterable):
    """A vector whose components are deferred (e.g. for use with lambdify).

    Examples
    ========

    >>> from sympy import DeferredVector, lambdify
    >>> X = DeferredVector( 'X' )
    >>> X
    X
    >>> expr = (X[0] + 2, X[2] + 3)
    >>> func = lambdify( X, expr)
    >>> func( [1, 2, 3] )
    (3, 6)
    """

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        if i == -0:
            i = 0
        if i < 0:
            raise IndexError('DeferredVector index out of range')
        component_name = '%s[%d]' % (self.name, i)
        return Symbol(component_name)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return sstr(self)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "DeferredVector('%s')" % self.name

class MatrixDeterminant(MatrixCommon):
    """Provides basic matrix determinant operations. Should not be instantiated
    directly. See ``determinant.py`` for their implementations."""

    def _eval_det_bareiss(self, iszerofunc=_is_zero_after_expand_mul):
        if False:
            for i in range(10):
                print('nop')
        return _det_bareiss(self, iszerofunc=iszerofunc)

    def _eval_det_berkowitz(self):
        if False:
            while True:
                i = 10
        return _det_berkowitz(self)

    def _eval_det_lu(self, iszerofunc=_iszero, simpfunc=None):
        if False:
            print('Hello World!')
        return _det_LU(self, iszerofunc=iszerofunc, simpfunc=simpfunc)

    def _eval_det_bird(self):
        if False:
            while True:
                i = 10
        return _det_bird(self)

    def _eval_det_laplace(self):
        if False:
            print('Hello World!')
        return _det_laplace(self)

    def _eval_determinant(self):
        if False:
            i = 10
            return i + 15
        return _det(self)

    def adjugate(self, method='berkowitz'):
        if False:
            while True:
                i = 10
        return _adjugate(self, method=method)

    def charpoly(self, x='lambda', simplify=_simplify):
        if False:
            return 10
        return _charpoly(self, x=x, simplify=simplify)

    def cofactor(self, i, j, method='berkowitz'):
        if False:
            print('Hello World!')
        return _cofactor(self, i, j, method=method)

    def cofactor_matrix(self, method='berkowitz'):
        if False:
            while True:
                i = 10
        return _cofactor_matrix(self, method=method)

    def det(self, method='bareiss', iszerofunc=None):
        if False:
            return 10
        return _det(self, method=method, iszerofunc=iszerofunc)

    def per(self):
        if False:
            i = 10
            return i + 15
        return _per(self)

    def minor(self, i, j, method='berkowitz'):
        if False:
            for i in range(10):
                print('nop')
        return _minor(self, i, j, method=method)

    def minor_submatrix(self, i, j):
        if False:
            print('Hello World!')
        return _minor_submatrix(self, i, j)
    _find_reasonable_pivot.__doc__ = _find_reasonable_pivot.__doc__
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__ = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__ = _det_berkowitz.__doc__
    _eval_det_bird.__doc__ = _det_bird.__doc__
    _eval_det_laplace.__doc__ = _det_laplace.__doc__
    _eval_det_lu.__doc__ = _det_LU.__doc__
    _eval_determinant.__doc__ = _det.__doc__
    adjugate.__doc__ = _adjugate.__doc__
    charpoly.__doc__ = _charpoly.__doc__
    cofactor.__doc__ = _cofactor.__doc__
    cofactor_matrix.__doc__ = _cofactor_matrix.__doc__
    det.__doc__ = _det.__doc__
    per.__doc__ = _per.__doc__
    minor.__doc__ = _minor.__doc__
    minor_submatrix.__doc__ = _minor_submatrix.__doc__

class MatrixReductions(MatrixDeterminant):
    """Provides basic matrix row/column operations. Should not be instantiated
    directly. See ``reductions.py`` for some of their implementations."""

    def echelon_form(self, iszerofunc=_iszero, simplify=False, with_pivots=False):
        if False:
            while True:
                i = 10
        return _echelon_form(self, iszerofunc=iszerofunc, simplify=simplify, with_pivots=with_pivots)

    @property
    def is_echelon(self):
        if False:
            while True:
                i = 10
        return _is_echelon(self)

    def rank(self, iszerofunc=_iszero, simplify=False):
        if False:
            for i in range(10):
                print('nop')
        return _rank(self, iszerofunc=iszerofunc, simplify=simplify)

    def rref_rhs(self, rhs):
        if False:
            i = 10
            return i + 15
        "Return reduced row-echelon form of matrix, matrix showing\n        rhs after reduction steps. ``rhs`` must have the same number\n        of rows as ``self``.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, symbols\n        >>> r1, r2 = symbols('r1 r2')\n        >>> Matrix([[1, 1], [2, 1]]).rref_rhs(Matrix([r1, r2]))\n        (Matrix([\n        [1, 0],\n        [0, 1]]), Matrix([\n        [ -r1 + r2],\n        [2*r1 - r2]]))\n        "
        (r, _) = _rref(self.hstack(self, self.eye(self.rows), rhs))
        return (r[:, :self.cols], r[:, -rhs.cols:])

    def rref(self, iszerofunc=_iszero, simplify=False, pivots=True, normalize_last=True):
        if False:
            i = 10
            return i + 15
        return _rref(self, iszerofunc=iszerofunc, simplify=simplify, pivots=pivots, normalize_last=normalize_last)
    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__ = _is_echelon.__doc__
    rank.__doc__ = _rank.__doc__
    rref.__doc__ = _rref.__doc__

    def _normalize_op_args(self, op, col, k, col1, col2, error_str='col'):
        if False:
            while True:
                i = 10
        'Validate the arguments for a row/column operation.  ``error_str``\n        can be one of "row" or "col" depending on the arguments being parsed.'
        if op not in ['n->kn', 'n<->m', 'n->n+km']:
            raise ValueError("Unknown {} operation '{}'. Valid col operations are 'n->kn', 'n<->m', 'n->n+km'".format(error_str, op))
        self_cols = self.cols if error_str == 'col' else self.rows
        if op == 'n->kn':
            col = col if col is not None else col1
            if col is None or k is None:
                raise ValueError("For a {0} operation 'n->kn' you must provide the kwargs `{0}` and `k`".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
        elif op == 'n<->m':
            cols = {col, k, col1, col2}.difference([None])
            if len(cols) > 2:
                cols = {col, col1, col2}.difference([None])
            if len(cols) != 2:
                raise ValueError("For a {0} operation 'n<->m' you must provide the kwargs `{0}1` and `{0}2`".format(error_str))
            (col1, col2) = cols
            if not 0 <= col1 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col1))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))
        elif op == 'n->n+km':
            col = col1 if col is None else col
            col2 = col1 if col2 is None else col2
            if col is None or col2 is None or k is None:
                raise ValueError("For a {0} operation 'n->n+km' you must provide the kwargs `{0}`, `k`, and `{0}2`".format(error_str))
            if col == col2:
                raise ValueError("For a {0} operation 'n->n+km' `{0}` and `{0}2` must be different.".format(error_str))
            if not 0 <= col < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col))
            if not 0 <= col2 < self_cols:
                raise ValueError("This matrix does not have a {} '{}'".format(error_str, col2))
        else:
            raise ValueError('invalid operation %s' % repr(op))
        return (op, col, k, col1, col2)

    def _eval_col_op_multiply_col_by_const(self, col, k):
        if False:
            while True:
                i = 10

        def entry(i, j):
            if False:
                while True:
                    i = 10
            if j == col:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_swap(self, col1, col2):
        if False:
            return 10

        def entry(i, j):
            if False:
                print('Hello World!')
            if j == col1:
                return self[i, col2]
            elif j == col2:
                return self[i, col1]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_col_op_add_multiple_to_other_col(self, col, k, col2):
        if False:
            i = 10
            return i + 15

        def entry(i, j):
            if False:
                for i in range(10):
                    print('nop')
            if j == col:
                return self[i, j] + k * self[i, col2]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_swap(self, row1, row2):
        if False:
            print('Hello World!')

        def entry(i, j):
            if False:
                return 10
            if i == row1:
                return self[row2, j]
            elif i == row2:
                return self[row1, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_multiply_row_by_const(self, row, k):
        if False:
            for i in range(10):
                print('nop')

        def entry(i, j):
            if False:
                for i in range(10):
                    print('nop')
            if i == row:
                return k * self[i, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def _eval_row_op_add_multiple_to_other_row(self, row, k, row2):
        if False:
            i = 10
            return i + 15

        def entry(i, j):
            if False:
                i = 10
                return i + 15
            if i == row:
                return self[i, j] + k * self[row2, j]
            return self[i, j]
        return self._new(self.rows, self.cols, entry)

    def elementary_col_op(self, op='n->kn', col=None, k=None, col1=None, col2=None):
        if False:
            while True:
                i = 10
        'Performs the elementary column operation `op`.\n\n        `op` may be one of\n\n            * ``"n->kn"`` (column n goes to k*n)\n            * ``"n<->m"`` (swap column n and column m)\n            * ``"n->n+km"`` (column n goes to column n + k*column m)\n\n        Parameters\n        ==========\n\n        op : string; the elementary row operation\n        col : the column to apply the column operation\n        k : the multiple to apply in the column operation\n        col1 : one column of a column swap\n        col2 : second column of a column swap or column "m" in the column operation\n               "n->n+km"\n        '
        (op, col, k, col1, col2) = self._normalize_op_args(op, col, k, col1, col2, 'col')
        if op == 'n->kn':
            return self._eval_col_op_multiply_col_by_const(col, k)
        if op == 'n<->m':
            return self._eval_col_op_swap(col1, col2)
        if op == 'n->n+km':
            return self._eval_col_op_add_multiple_to_other_col(col, k, col2)

    def elementary_row_op(self, op='n->kn', row=None, k=None, row1=None, row2=None):
        if False:
            return 10
        'Performs the elementary row operation `op`.\n\n        `op` may be one of\n\n            * ``"n->kn"`` (row n goes to k*n)\n            * ``"n<->m"`` (swap row n and row m)\n            * ``"n->n+km"`` (row n goes to row n + k*row m)\n\n        Parameters\n        ==========\n\n        op : string; the elementary row operation\n        row : the row to apply the row operation\n        k : the multiple to apply in the row operation\n        row1 : one row of a row swap\n        row2 : second row of a row swap or row "m" in the row operation\n               "n->n+km"\n        '
        (op, row, k, row1, row2) = self._normalize_op_args(op, row, k, row1, row2, 'row')
        if op == 'n->kn':
            return self._eval_row_op_multiply_row_by_const(row, k)
        if op == 'n<->m':
            return self._eval_row_op_swap(row1, row2)
        if op == 'n->n+km':
            return self._eval_row_op_add_multiple_to_other_row(row, k, row2)

class MatrixSubspaces(MatrixReductions):
    """Provides methods relating to the fundamental subspaces of a matrix.
    Should not be instantiated directly. See ``subspaces.py`` for their
    implementations."""

    def columnspace(self, simplify=False):
        if False:
            i = 10
            return i + 15
        return _columnspace(self, simplify=simplify)

    def nullspace(self, simplify=False, iszerofunc=_iszero):
        if False:
            print('Hello World!')
        return _nullspace(self, simplify=simplify, iszerofunc=iszerofunc)

    def rowspace(self, simplify=False):
        if False:
            print('Hello World!')
        return _rowspace(self, simplify=simplify)

    def orthogonalize(cls, *vecs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return _orthogonalize(cls, *vecs, **kwargs)
    columnspace.__doc__ = _columnspace.__doc__
    nullspace.__doc__ = _nullspace.__doc__
    rowspace.__doc__ = _rowspace.__doc__
    orthogonalize.__doc__ = _orthogonalize.__doc__
    orthogonalize = classmethod(orthogonalize)

class MatrixEigen(MatrixSubspaces):
    """Provides basic matrix eigenvalue/vector operations.
    Should not be instantiated directly. See ``eigen.py`` for their
    implementations."""

    def eigenvals(self, error_when_incomplete=True, **flags):
        if False:
            print('Hello World!')
        return _eigenvals(self, error_when_incomplete=error_when_incomplete, **flags)

    def eigenvects(self, error_when_incomplete=True, iszerofunc=_iszero, **flags):
        if False:
            i = 10
            return i + 15
        return _eigenvects(self, error_when_incomplete=error_when_incomplete, iszerofunc=iszerofunc, **flags)

    def is_diagonalizable(self, reals_only=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return _is_diagonalizable(self, reals_only=reals_only, **kwargs)

    def diagonalize(self, reals_only=False, sort=False, normalize=False):
        if False:
            for i in range(10):
                print('nop')
        return _diagonalize(self, reals_only=reals_only, sort=sort, normalize=normalize)

    def bidiagonalize(self, upper=True):
        if False:
            i = 10
            return i + 15
        return _bidiagonalize(self, upper=upper)

    def bidiagonal_decomposition(self, upper=True):
        if False:
            i = 10
            return i + 15
        return _bidiagonal_decomposition(self, upper=upper)

    @property
    def is_positive_definite(self):
        if False:
            for i in range(10):
                print('nop')
        return _is_positive_definite(self)

    @property
    def is_positive_semidefinite(self):
        if False:
            print('Hello World!')
        return _is_positive_semidefinite(self)

    @property
    def is_negative_definite(self):
        if False:
            return 10
        return _is_negative_definite(self)

    @property
    def is_negative_semidefinite(self):
        if False:
            return 10
        return _is_negative_semidefinite(self)

    @property
    def is_indefinite(self):
        if False:
            i = 10
            return i + 15
        return _is_indefinite(self)

    def jordan_form(self, calc_transform=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return _jordan_form(self, calc_transform=calc_transform, **kwargs)

    def left_eigenvects(self, **flags):
        if False:
            i = 10
            return i + 15
        return _left_eigenvects(self, **flags)

    def singular_values(self):
        if False:
            print('Hello World!')
        return _singular_values(self)
    eigenvals.__doc__ = _eigenvals.__doc__
    eigenvects.__doc__ = _eigenvects.__doc__
    is_diagonalizable.__doc__ = _is_diagonalizable.__doc__
    diagonalize.__doc__ = _diagonalize.__doc__
    is_positive_definite.__doc__ = _is_positive_definite.__doc__
    is_positive_semidefinite.__doc__ = _is_positive_semidefinite.__doc__
    is_negative_definite.__doc__ = _is_negative_definite.__doc__
    is_negative_semidefinite.__doc__ = _is_negative_semidefinite.__doc__
    is_indefinite.__doc__ = _is_indefinite.__doc__
    jordan_form.__doc__ = _jordan_form.__doc__
    left_eigenvects.__doc__ = _left_eigenvects.__doc__
    singular_values.__doc__ = _singular_values.__doc__
    bidiagonalize.__doc__ = _bidiagonalize.__doc__
    bidiagonal_decomposition.__doc__ = _bidiagonal_decomposition.__doc__

class MatrixCalculus(MatrixCommon):
    """Provides calculus-related matrix operations."""

    def diff(self, *args, evaluate=True, **kwargs):
        if False:
            return 10
        'Calculate the derivative of each element in the matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.abc import x, y\n        >>> M = Matrix([[x, y], [1, 0]])\n        >>> M.diff(x)\n        Matrix([\n        [1, 0],\n        [0, 0]])\n\n        See Also\n        ========\n\n        integrate\n        limit\n        '
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        deriv = ArrayDerivative(self, *args, evaluate=evaluate)
        if not isinstance(self, Basic) and evaluate:
            return deriv.as_mutable()
        return deriv

    def _eval_derivative(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return self.applyfunc(lambda x: x.diff(arg))

    def integrate(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Integrate each element of the matrix.  ``args`` will\n        be passed to the ``integrate`` function.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.abc import x, y\n        >>> M = Matrix([[x, y], [1, 0]])\n        >>> M.integrate((x, ))\n        Matrix([\n        [x**2/2, x*y],\n        [     x,   0]])\n        >>> M.integrate((x, 0, 2))\n        Matrix([\n        [2, 2*y],\n        [2,   0]])\n\n        See Also\n        ========\n\n        limit\n        diff\n        '
        return self.applyfunc(lambda x: x.integrate(*args, **kwargs))

    def jacobian(self, X):
        if False:
            i = 10
            return i + 15
        "Calculates the Jacobian matrix (derivative of a vector-valued function).\n\n        Parameters\n        ==========\n\n        ``self`` : vector of expressions representing functions f_i(x_1, ..., x_n).\n        X : set of x_i's in order, it can be a list or a Matrix\n\n        Both ``self`` and X can be a row or a column matrix in any order\n        (i.e., jacobian() should always work).\n\n        Examples\n        ========\n\n        >>> from sympy import sin, cos, Matrix\n        >>> from sympy.abc import rho, phi\n        >>> X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])\n        >>> Y = Matrix([rho, phi])\n        >>> X.jacobian(Y)\n        Matrix([\n        [cos(phi), -rho*sin(phi)],\n        [sin(phi),  rho*cos(phi)],\n        [   2*rho,             0]])\n        >>> X = Matrix([rho*cos(phi), rho*sin(phi)])\n        >>> X.jacobian(Y)\n        Matrix([\n        [cos(phi), -rho*sin(phi)],\n        [sin(phi),  rho*cos(phi)]])\n\n        See Also\n        ========\n\n        hessian\n        wronskian\n        "
        if not isinstance(X, MatrixBase):
            X = self._new(X)
        if self.shape[0] == 1:
            m = self.shape[1]
        elif self.shape[1] == 1:
            m = self.shape[0]
        else:
            raise TypeError('``self`` must be a row or a column matrix')
        if X.shape[0] == 1:
            n = X.shape[1]
        elif X.shape[1] == 1:
            n = X.shape[0]
        else:
            raise TypeError('X must be a row or a column matrix')
        return self._new(m, n, lambda j, i: self[j].diff(X[i]))

    def limit(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the limit of each element in the matrix.\n        ``args`` will be passed to the ``limit`` function.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.abc import x, y\n        >>> M = Matrix([[x, y], [1, 0]])\n        >>> M.limit(x, 2)\n        Matrix([\n        [2, y],\n        [1, 0]])\n\n        See Also\n        ========\n\n        integrate\n        diff\n        '
        return self.applyfunc(lambda x: x.limit(*args))

class MatrixDeprecated(MatrixCommon):
    """A class to house deprecated matrix methods."""

    def berkowitz_charpoly(self, x=Dummy('lambda'), simplify=_simplify):
        if False:
            for i in range(10):
                print('nop')
        return self.charpoly(x=x)

    def berkowitz_det(self):
        if False:
            print('Hello World!')
        'Computes determinant using Berkowitz method.\n\n        See Also\n        ========\n\n        det\n        berkowitz\n        '
        return self.det(method='berkowitz')

    def berkowitz_eigenvals(self, **flags):
        if False:
            i = 10
            return i + 15
        'Computes eigenvalues of a Matrix using Berkowitz method.\n\n        See Also\n        ========\n\n        berkowitz\n        '
        return self.eigenvals(**flags)

    def berkowitz_minors(self):
        if False:
            print('Hello World!')
        'Computes principal minors using Berkowitz method.\n\n        See Also\n        ========\n\n        berkowitz\n        '
        (sign, minors) = (self.one, [])
        for poly in self.berkowitz():
            minors.append(sign * poly[-1])
            sign = -sign
        return tuple(minors)

    def berkowitz(self):
        if False:
            while True:
                i = 10
        from sympy.matrices import zeros
        berk = ((1,),)
        if not self:
            return berk
        if not self.is_square:
            raise NonSquareMatrixError()
        (A, N) = (self, self.rows)
        transforms = [0] * (N - 1)
        for n in range(N, 1, -1):
            (T, k) = (zeros(n + 1, n), n - 1)
            (R, C) = (-A[k, :k], A[:k, k])
            (A, a) = (A[:k, :k], -A[k, k])
            items = [C]
            for i in range(0, n - 2):
                items.append(A * items[i])
            for (i, B) in enumerate(items):
                items[i] = (R * B)[0, 0]
            items = [self.one, a] + items
            for i in range(n):
                T[i:, i] = items[:n - i + 1]
            transforms[k - 1] = T
        polys = [self._new([self.one, -A[0, 0]])]
        for (i, T) in enumerate(transforms):
            polys.append(T * polys[i])
        return berk + tuple(map(tuple, polys))

    def cofactorMatrix(self, method='berkowitz'):
        if False:
            print('Hello World!')
        return self.cofactor_matrix(method=method)

    def det_bareis(self):
        if False:
            while True:
                i = 10
        return _det_bareiss(self)

    def det_LU_decomposition(self):
        if False:
            i = 10
            return i + 15
        'Compute matrix determinant using LU decomposition.\n\n\n        Note that this method fails if the LU decomposition itself\n        fails. In particular, if the matrix has no inverse this method\n        will fail.\n\n        TODO: Implement algorithm for sparse matrices (SFF),\n        http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.\n\n        See Also\n        ========\n\n\n        det\n        det_bareiss\n        berkowitz_det\n        '
        return self.det(method='lu')

    def jordan_cell(self, eigenval, n):
        if False:
            return 10
        return self.jordan_block(size=n, eigenvalue=eigenval)

    def jordan_cells(self, calc_transformation=True):
        if False:
            print('Hello World!')
        (P, J) = self.jordan_form()
        return (P, J.get_diag_blocks())

    def minorEntry(self, i, j, method='berkowitz'):
        if False:
            print('Hello World!')
        return self.minor(i, j, method=method)

    def minorMatrix(self, i, j):
        if False:
            while True:
                i = 10
        return self.minor_submatrix(i, j)

    def permuteBkwd(self, perm):
        if False:
            while True:
                i = 10
        'Permute the rows of the matrix with the given permutation in reverse.'
        return self.permute_rows(perm, direction='backward')

    def permuteFwd(self, perm):
        if False:
            i = 10
            return i + 15
        'Permute the rows of the matrix with the given permutation.'
        return self.permute_rows(perm, direction='forward')

@Mul._kind_dispatcher.register(_NumberKind, MatrixKind)
def num_mat_mul(k1, k2):
    if False:
        i = 10
        return i + 15
    '\n    Return MatrixKind. The element kind is selected by recursive dispatching.\n    Do not need to dispatch in reversed order because KindDispatcher\n    searches for this automatically.\n    '
    if not isinstance(k2, MatrixKind):
        (k1, k2) = (k2, k1)
    elemk = Mul._kind_dispatcher(k1, k2.element_kind)
    return MatrixKind(elemk)

@Mul._kind_dispatcher.register(MatrixKind, MatrixKind)
def mat_mat_mul(k1, k2):
    if False:
        print('Hello World!')
    '\n    Return MatrixKind. The element kind is selected by recursive dispatching.\n    '
    elemk = Mul._kind_dispatcher(k1.element_kind, k2.element_kind)
    return MatrixKind(elemk)

class MatrixBase(MatrixDeprecated, MatrixCalculus, MatrixEigen, MatrixCommon, Printable):
    """Base class for matrix objects."""
    __array_priority__ = 11
    is_Matrix = True
    _class_priority = 3
    _sympify = staticmethod(sympify)
    zero = S.Zero
    one = S.One

    @property
    def kind(self) -> MatrixKind:
        if False:
            i = 10
            return i + 15
        elem_kinds = {e.kind for e in self.flat()}
        if len(elem_kinds) == 1:
            (elemkind,) = elem_kinds
        else:
            elemkind = UndefinedKind
        return MatrixKind(elemkind)

    def flat(self):
        if False:
            i = 10
            return i + 15
        return [self[i, j] for i in range(self.rows) for j in range(self.cols)]

    def __array__(self, dtype=object):
        if False:
            while True:
                i = 10
        from .dense import matrix2numpy
        return matrix2numpy(self, dtype=dtype)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of elements of ``self``.\n\n        Implemented mainly so bool(Matrix()) == False.\n        '
        return self.rows * self.cols

    def _matrix_pow_by_jordan_blocks(self, num):
        if False:
            return 10
        from sympy.matrices import diag, MutableMatrix

        def jordan_cell_power(jc, n):
            if False:
                return 10
            N = jc.shape[0]
            l = jc[0, 0]
            if l.is_zero:
                if N == 1 and n.is_nonnegative:
                    jc[0, 0] = l ** n
                elif not (n.is_integer and n.is_nonnegative):
                    raise NonInvertibleMatrixError('Non-invertible matrix can only be raised to a nonnegative integer')
                else:
                    for i in range(N):
                        jc[0, i] = KroneckerDelta(i, n)
            else:
                for i in range(N):
                    bn = binomial(n, i)
                    if isinstance(bn, binomial):
                        bn = bn._eval_expand_func()
                    jc[0, i] = l ** (n - i) * bn
            for i in range(N):
                for j in range(1, N - i):
                    jc[j, i + j] = jc[j - 1, i + j - 1]
        (P, J) = self.jordan_form()
        jordan_cells = J.get_diag_blocks()
        jordan_cells = [MutableMatrix(j) for j in jordan_cells]
        for j in jordan_cells:
            jordan_cell_power(j, num)
        return self._new(P.multiply(diag(*jordan_cells)).multiply(P.inv()))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        return 'Matrix(%s)' % str(self.tolist())

    def _format_str(self, printer=None):
        if False:
            for i in range(10):
                print('nop')
        if not printer:
            printer = StrPrinter()
        if S.Zero in self.shape:
            return 'Matrix(%s, %s, [])' % (self.rows, self.cols)
        if self.rows == 1:
            return 'Matrix([%s])' % self.table(printer, rowsep=',\n')
        return 'Matrix([\n%s])' % self.table(printer, rowsep=',\n')

    @classmethod
    def irregular(cls, ntop, *matrices, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return a matrix filled by the given matrices which\n      are listed in order of appearance from left to right, top to\n      bottom as they first appear in the matrix. They must fill the\n      matrix completely.\n\n      Examples\n      ========\n\n      >>> from sympy import ones, Matrix\n      >>> Matrix.irregular(3, ones(2,1), ones(3,3)*2, ones(2,2)*3,\n      ...   ones(1,1)*4, ones(2,2)*5, ones(1,2)*6, ones(1,2)*7)\n      Matrix([\n        [1, 2, 2, 2, 3, 3],\n        [1, 2, 2, 2, 3, 3],\n        [4, 2, 2, 2, 5, 5],\n        [6, 6, 7, 7, 5, 5]])\n      '
        ntop = as_int(ntop)
        b = [i.as_explicit() if hasattr(i, 'as_explicit') else i for i in matrices]
        q = list(range(len(b)))
        dat = [i.rows for i in b]
        active = [q.pop(0) for _ in range(ntop)]
        cols = sum([b[i].cols for i in active])
        rows = []
        while any(dat):
            r = []
            for (a, j) in enumerate(active):
                r.extend(b[j][-dat[j], :])
                dat[j] -= 1
                if dat[j] == 0 and q:
                    active[a] = q.pop(0)
            if len(r) != cols:
                raise ValueError(filldedent('\n                Matrices provided do not appear to fill\n                the space completely.'))
            rows.append(r)
        return cls._new(rows)

    @classmethod
    def _handle_ndarray(cls, arg):
        if False:
            print('Hello World!')
        arr = arg.__array__()
        if len(arr.shape) == 2:
            (rows, cols) = (arr.shape[0], arr.shape[1])
            flat_list = [cls._sympify(i) for i in arr.ravel()]
            return (rows, cols, flat_list)
        elif len(arr.shape) == 1:
            flat_list = [cls._sympify(i) for i in arr]
            return (arr.shape[0], 1, flat_list)
        else:
            raise NotImplementedError('SymPy supports just 1D and 2D matrices')

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return the number of rows, cols and flat matrix elements.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, I\n\n        Matrix can be constructed as follows:\n\n        * from a nested list of iterables\n\n        >>> Matrix( ((1, 2+I), (3, 4)) )\n        Matrix([\n        [1, 2 + I],\n        [3,     4]])\n\n        * from un-nested iterable (interpreted as a column)\n\n        >>> Matrix( [1, 2] )\n        Matrix([\n        [1],\n        [2]])\n\n        * from un-nested iterable with dimensions\n\n        >>> Matrix(1, 2, [1, 2] )\n        Matrix([[1, 2]])\n\n        * from no arguments (a 0 x 0 matrix)\n\n        >>> Matrix()\n        Matrix(0, 0, [])\n\n        * from a rule\n\n        >>> Matrix(2, 2, lambda i, j: i/(j + 1) )\n        Matrix([\n        [0,   0],\n        [1, 1/2]])\n\n        See Also\n        ========\n        irregular - filling a matrix with irregular blocks\n        '
        from sympy.matrices import SparseMatrix
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.matrices.expressions.blockmatrix import BlockMatrix
        flat_list = None
        if len(args) == 1:
            if isinstance(args[0], SparseMatrix):
                return (args[0].rows, args[0].cols, flatten(args[0].tolist()))
            elif isinstance(args[0], MatrixBase):
                return (args[0].rows, args[0].cols, args[0].flat())
            elif isinstance(args[0], Basic) and args[0].is_Matrix:
                return (args[0].rows, args[0].cols, args[0].as_explicit().flat())
            elif isinstance(args[0], mp.matrix):
                M = args[0]
                flat_list = [cls._sympify(x) for x in M]
                return (M.rows, M.cols, flat_list)
            elif hasattr(args[0], '__array__'):
                return cls._handle_ndarray(args[0])
            elif is_sequence(args[0]) and (not isinstance(args[0], DeferredVector)):
                dat = list(args[0])
                ismat = lambda i: isinstance(i, MatrixBase) and (evaluate or isinstance(i, (BlockMatrix, MatrixSymbol)))
                raw = lambda i: is_sequence(i) and (not ismat(i))
                evaluate = kwargs.get('evaluate', True)
                if evaluate:

                    def make_explicit(x):
                        if False:
                            return 10
                        'make Block and Symbol explicit'
                        if isinstance(x, BlockMatrix):
                            return x.as_explicit()
                        elif isinstance(x, MatrixSymbol) and all((_.is_Integer for _ in x.shape)):
                            return x.as_explicit()
                        else:
                            return x

                    def make_explicit_row(row):
                        if False:
                            return 10
                        if isinstance(row, (list, tuple)):
                            return [make_explicit(x) for x in row]
                        else:
                            return make_explicit(row)
                    if isinstance(dat, (list, tuple)):
                        dat = [make_explicit_row(row) for row in dat]
                if dat in ([], [[]]):
                    rows = cols = 0
                    flat_list = []
                elif not any((raw(i) or ismat(i) for i in dat)):
                    flat_list = [cls._sympify(i) for i in dat]
                    rows = len(flat_list)
                    cols = 1 if rows else 0
                elif evaluate and all((ismat(i) for i in dat)):
                    ncol = {i.cols for i in dat if any(i.shape)}
                    if ncol:
                        if len(ncol) != 1:
                            raise ValueError('mismatched dimensions')
                        flat_list = [_ for i in dat for r in i.tolist() for _ in r]
                        cols = ncol.pop()
                        rows = len(flat_list) // cols
                    else:
                        rows = cols = 0
                        flat_list = []
                elif evaluate and any((ismat(i) for i in dat)):
                    ncol = set()
                    flat_list = []
                    for i in dat:
                        if ismat(i):
                            flat_list.extend([k for j in i.tolist() for k in j])
                            if any(i.shape):
                                ncol.add(i.cols)
                        elif raw(i):
                            if i:
                                ncol.add(len(i))
                                flat_list.extend([cls._sympify(ij) for ij in i])
                        else:
                            ncol.add(1)
                            flat_list.append(i)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                    cols = ncol.pop()
                    rows = len(flat_list) // cols
                else:
                    flat_list = []
                    ncol = set()
                    rows = cols = 0
                    for row in dat:
                        if not is_sequence(row) and (not getattr(row, 'is_Matrix', False)):
                            raise ValueError('expecting list of lists')
                        if hasattr(row, '__array__'):
                            if 0 in row.shape:
                                continue
                        elif not row:
                            continue
                        if evaluate and all((ismat(i) for i in row)):
                            (r, c, flatT) = cls._handle_creation_inputs([i.T for i in row])
                            T = reshape(flatT, [c])
                            flat = [T[i][j] for j in range(c) for i in range(r)]
                            (r, c) = (c, r)
                        else:
                            r = 1
                            if getattr(row, 'is_Matrix', False):
                                c = 1
                                flat = [row]
                            else:
                                c = len(row)
                                flat = [cls._sympify(i) for i in row]
                        ncol.add(c)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                        flat_list.extend(flat)
                        rows += r
                    cols = ncol.pop() if ncol else 0
        elif len(args) == 3:
            rows = as_int(args[0])
            cols = as_int(args[1])
            if rows < 0 or cols < 0:
                raise ValueError('Cannot create a {} x {} matrix. Both dimensions must be positive'.format(rows, cols))
            if len(args) == 3 and isinstance(args[2], Callable):
                op = args[2]
                flat_list = []
                for i in range(rows):
                    flat_list.extend([cls._sympify(op(cls._sympify(i), cls._sympify(j))) for j in range(cols)])
            elif len(args) == 3 and is_sequence(args[2]):
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError('List length should be equal to rows*columns')
                flat_list = [cls._sympify(i) for i in flat_list]
        elif len(args) == 0:
            rows = cols = 0
            flat_list = []
        if flat_list is None:
            raise TypeError(filldedent('\n                Data type not understood; expecting list of lists\n                or lists of values.'))
        return (rows, cols, flat_list)

    def _setitem(self, key, value):
        if False:
            while True:
                i = 10
        'Helper to set value at location given by key.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, I, zeros, ones\n        >>> m = Matrix(((1, 2+I), (3, 4)))\n        >>> m\n        Matrix([\n        [1, 2 + I],\n        [3,     4]])\n        >>> m[1, 0] = 9\n        >>> m\n        Matrix([\n        [1, 2 + I],\n        [9,     4]])\n        >>> m[1, 0] = [[0, 1]]\n\n        To replace row r you assign to position r*m where m\n        is the number of columns:\n\n        >>> M = zeros(4)\n        >>> m = M.cols\n        >>> M[3*m] = ones(1, m)*2; M\n        Matrix([\n        [0, 0, 0, 0],\n        [0, 0, 0, 0],\n        [0, 0, 0, 0],\n        [2, 2, 2, 2]])\n\n        And to replace column c you can assign to position c:\n\n        >>> M[2] = ones(m, 1)*4; M\n        Matrix([\n        [0, 0, 4, 0],\n        [0, 0, 4, 0],\n        [0, 0, 4, 0],\n        [2, 2, 4, 2]])\n        '
        from .dense import Matrix
        is_slice = isinstance(key, slice)
        (i, j) = key = self.key2ij(key)
        is_mat = isinstance(value, MatrixBase)
        if isinstance(i, slice) or isinstance(j, slice):
            if is_mat:
                self.copyin_matrix(key, value)
                return
            if not isinstance(value, Expr) and is_sequence(value):
                self.copyin_list(key, value)
                return
            raise ValueError('unexpected value: %s' % value)
        else:
            if not is_mat and (not isinstance(value, Basic)) and is_sequence(value):
                value = Matrix(value)
                is_mat = True
            if is_mat:
                if is_slice:
                    key = (slice(*divmod(i, self.cols)), slice(*divmod(j, self.cols)))
                else:
                    key = (slice(i, i + value.rows), slice(j, j + value.cols))
                self.copyin_matrix(key, value)
            else:
                return (i, j, self._sympify(value))
            return

    def add(self, b):
        if False:
            while True:
                i = 10
        'Return self + b.'
        return self + b

    def condition_number(self):
        if False:
            print('Hello World!')
        'Returns the condition number of a matrix.\n\n        This is the maximum singular value divided by the minimum singular value\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, S\n        >>> A = Matrix([[1, 0, 0], [0, 10, 0], [0, 0, S.One/10]])\n        >>> A.condition_number()\n        100\n\n        See Also\n        ========\n\n        singular_values\n        '
        if not self:
            return self.zero
        singularvalues = self.singular_values()
        return Max(*singularvalues) / Min(*singularvalues)

    def copy(self):
        if False:
            print('Hello World!')
        '\n        Returns the copy of a matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> A = Matrix(2, 2, [1, 2, 3, 4])\n        >>> A.copy()\n        Matrix([\n        [1, 2],\n        [3, 4]])\n\n        '
        return self._new(self.rows, self.cols, self.flat())

    def cross(self, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the cross product of ``self`` and ``b`` relaxing the condition\n        of compatible dimensions: if each has 3 elements, a matrix of the\n        same type and shape as ``self`` will be returned. If ``b`` has the same\n        shape as ``self`` then common identities for the cross product (like\n        `a \\times b = - b \\times a`) will hold.\n\n        Parameters\n        ==========\n            b : 3x1 or 1x3 Matrix\n\n        See Also\n        ========\n\n        dot\n        hat\n        vee\n        multiply\n        multiply_elementwise\n        '
        from sympy.matrices.expressions.matexpr import MatrixExpr
        if not isinstance(b, (MatrixBase, MatrixExpr)):
            raise TypeError('{} must be a Matrix, not {}.'.format(b, type(b)))
        if not self.rows * self.cols == b.rows * b.cols == 3:
            raise ShapeError('Dimensions incorrect for cross product: %s x %s' % ((self.rows, self.cols), (b.rows, b.cols)))
        else:
            return self._new(self.rows, self.cols, (self[1] * b[2] - self[2] * b[1], self[2] * b[0] - self[0] * b[2], self[0] * b[1] - self[1] * b[0]))

    def hat(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the skew-symmetric matrix representing the cross product,\n        so that ``self.hat() * b`` is equivalent to  ``self.cross(b)``.\n\n        Examples\n        ========\n\n        Calling ``hat`` creates a skew-symmetric 3x3 Matrix from a 3x1 Matrix:\n\n        >>> from sympy import Matrix\n        >>> a = Matrix([1, 2, 3])\n        >>> a.hat()\n        Matrix([\n        [ 0, -3,  2],\n        [ 3,  0, -1],\n        [-2,  1,  0]])\n\n        Multiplying it with another 3x1 Matrix calculates the cross product:\n\n        >>> b = Matrix([3, 2, 1])\n        >>> a.hat() * b\n        Matrix([\n        [-4],\n        [ 8],\n        [-4]])\n\n        Which is equivalent to calling the ``cross`` method:\n\n        >>> a.cross(b)\n        Matrix([\n        [-4],\n        [ 8],\n        [-4]])\n\n        See Also\n        ========\n\n        dot\n        cross\n        vee\n        multiply\n        multiply_elementwise\n        '
        if self.shape != (3, 1):
            raise ShapeError('Dimensions incorrect, expected (3, 1), got ' + str(self.shape))
        else:
            (x, y, z) = self
            return self._new(3, 3, (0, -z, y, z, 0, -x, -y, x, 0))

    def vee(self):
        if False:
            i = 10
            return i + 15
        "\n        Return a 3x1 vector from a skew-symmetric matrix representing the cross product,\n        so that ``self * b`` is equivalent to  ``self.vee().cross(b)``.\n\n        Examples\n        ========\n\n        Calling ``vee`` creates a vector from a skew-symmetric Matrix:\n\n        >>> from sympy import Matrix\n        >>> A = Matrix([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])\n        >>> a = A.vee()\n        >>> a\n        Matrix([\n        [1],\n        [2],\n        [3]])\n\n        Calculating the matrix product of the original matrix with a vector\n        is equivalent to a cross product:\n\n        >>> b = Matrix([3, 2, 1])\n        >>> A * b\n        Matrix([\n        [-4],\n        [ 8],\n        [-4]])\n\n        >>> a.cross(b)\n        Matrix([\n        [-4],\n        [ 8],\n        [-4]])\n\n        ``vee`` can also be used to retrieve angular velocity expressions.\n        Defining a rotation matrix:\n\n        >>> from sympy import rot_ccw_axis3, trigsimp\n        >>> from sympy.physics.mechanics import dynamicsymbols\n        >>> theta = dynamicsymbols('theta')\n        >>> R = rot_ccw_axis3(theta)\n        >>> R\n        Matrix([\n        [cos(theta(t)), -sin(theta(t)), 0],\n        [sin(theta(t)),  cos(theta(t)), 0],\n        [            0,              0, 1]])\n\n        We can retrive the angular velocity:\n\n        >>> Omega = R.T * R.diff()\n        >>> Omega = trigsimp(Omega)\n        >>> Omega.vee()\n        Matrix([\n        [                      0],\n        [                      0],\n        [Derivative(theta(t), t)]])\n\n        See Also\n        ========\n\n        dot\n        cross\n        hat\n        multiply\n        multiply_elementwise\n        "
        if self.shape != (3, 3):
            raise ShapeError('Dimensions incorrect, expected (3, 3), got ' + str(self.shape))
        elif not self.is_anti_symmetric():
            raise ValueError('Matrix is not skew-symmetric')
        else:
            return self._new(3, 1, (self[2, 1], self[0, 2], self[1, 0]))

    @property
    def D(self):
        if False:
            while True:
                i = 10
        'Return Dirac conjugate (if ``self.rows == 4``).\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, I, eye\n        >>> m = Matrix((0, 1 + I, 2, 3))\n        >>> m.D\n        Matrix([[0, 1 - I, -2, -3]])\n        >>> m = (eye(4) + I*eye(4))\n        >>> m[0, 3] = 2\n        >>> m.D\n        Matrix([\n        [1 - I,     0,      0,      0],\n        [    0, 1 - I,      0,      0],\n        [    0,     0, -1 + I,      0],\n        [    2,     0,      0, -1 + I]])\n\n        If the matrix does not have 4 rows an AttributeError will be raised\n        because this property is only defined for matrices with 4 rows.\n\n        >>> Matrix(eye(2)).D\n        Traceback (most recent call last):\n        ...\n        AttributeError: Matrix has no attribute D.\n\n        See Also\n        ========\n\n        sympy.matrices.common.MatrixCommon.conjugate: By-element conjugation\n        sympy.matrices.common.MatrixCommon.H: Hermite conjugation\n        '
        from sympy.physics.matrices import mgamma
        if self.rows != 4:
            raise AttributeError
        return self.H * mgamma(0)

    def dot(self, b, hermitian=None, conjugate_convention=None):
        if False:
            i = 10
            return i + 15
        'Return the dot or inner product of two vectors of equal length.\n        Here ``self`` must be a ``Matrix`` of size 1 x n or n x 1, and ``b``\n        must be either a matrix of size 1 x n, n x 1, or a list/tuple of length n.\n        A scalar is returned.\n\n        By default, ``dot`` does not conjugate ``self`` or ``b``, even if there are\n        complex entries. Set ``hermitian=True`` (and optionally a ``conjugate_convention``)\n        to compute the hermitian inner product.\n\n        Possible kwargs are ``hermitian`` and ``conjugate_convention``.\n\n        If ``conjugate_convention`` is ``"left"``, ``"math"`` or ``"maths"``,\n        the conjugate of the first vector (``self``) is used.  If ``"right"``\n        or ``"physics"`` is specified, the conjugate of the second vector ``b`` is used.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n        >>> v = Matrix([1, 1, 1])\n        >>> M.row(0).dot(v)\n        6\n        >>> M.col(0).dot(v)\n        12\n        >>> v = [3, 2, 1]\n        >>> M.row(0).dot(v)\n        10\n\n        >>> from sympy import I\n        >>> q = Matrix([1*I, 1*I, 1*I])\n        >>> q.dot(q, hermitian=False)\n        -3\n\n        >>> q.dot(q, hermitian=True)\n        3\n\n        >>> q1 = Matrix([1, 1, 1*I])\n        >>> q.dot(q1, hermitian=True, conjugate_convention="maths")\n        1 - 2*I\n        >>> q.dot(q1, hermitian=True, conjugate_convention="physics")\n        1 + 2*I\n\n\n        See Also\n        ========\n\n        cross\n        multiply\n        multiply_elementwise\n        '
        from .dense import Matrix
        if not isinstance(b, MatrixBase):
            if is_sequence(b):
                if len(b) != self.cols and len(b) != self.rows:
                    raise ShapeError('Dimensions incorrect for dot product: %s, %s' % (self.shape, len(b)))
                return self.dot(Matrix(b))
            else:
                raise TypeError('`b` must be an ordered iterable or Matrix, not %s.' % type(b))
        if 1 not in self.shape or 1 not in b.shape:
            raise ShapeError
        if len(self) != len(b):
            raise ShapeError('Dimensions incorrect for dot product: %s, %s' % (self.shape, b.shape))
        mat = self
        n = len(mat)
        if mat.shape != (1, n):
            mat = mat.reshape(1, n)
        if b.shape != (n, 1):
            b = b.reshape(n, 1)
        if conjugate_convention is not None and hermitian is None:
            hermitian = True
        if hermitian and conjugate_convention is None:
            conjugate_convention = 'maths'
        if hermitian == True:
            if conjugate_convention in ('maths', 'left', 'math'):
                mat = mat.conjugate()
            elif conjugate_convention in ('physics', 'right'):
                b = b.conjugate()
            else:
                raise ValueError('Unknown conjugate_convention was entered. conjugate_convention must be one of the following: math, maths, left, physics or right.')
        return (mat * b)[0]

    def dual(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the dual of a matrix.\n\n        A dual of a matrix is:\n\n        ``(1/2)*levicivita(i, j, k, l)*M(k, l)`` summed over indices `k` and `l`\n\n        Since the levicivita method is anti_symmetric for any pairwise\n        exchange of indices, the dual of a symmetric matrix is the zero\n        matrix. Strictly speaking the dual defined here assumes that the\n        'matrix' `M` is a contravariant anti_symmetric second rank tensor,\n        so that the dual is a covariant second rank tensor.\n\n        "
        from sympy.matrices import zeros
        (M, n) = (self[:, :], self.rows)
        work = zeros(n)
        if self.is_symmetric():
            return work
        for i in range(1, n):
            for j in range(1, n):
                acum = 0
                for k in range(1, n):
                    acum += LeviCivita(i, j, 0, k) * M[0, k]
                work[i, j] = acum
                work[j, i] = -acum
        for l in range(1, n):
            acum = 0
            for a in range(1, n):
                for b in range(1, n):
                    acum += LeviCivita(0, l, a, b) * M[a, b]
            acum /= 2
            work[0, l] = -acum
            work[l, 0] = acum
        return work

    def _eval_matrix_exp_jblock(self):
        if False:
            i = 10
            return i + 15
        "A helper function to compute an exponential of a Jordan block\n        matrix\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Matrix\n        >>> l = Symbol('lamda')\n\n        A trivial example of 1*1 Jordan block:\n\n        >>> m = Matrix.jordan_block(1, l)\n        >>> m._eval_matrix_exp_jblock()\n        Matrix([[exp(lamda)]])\n\n        An example of 3*3 Jordan block:\n\n        >>> m = Matrix.jordan_block(3, l)\n        >>> m._eval_matrix_exp_jblock()\n        Matrix([\n        [exp(lamda), exp(lamda), exp(lamda)/2],\n        [         0, exp(lamda),   exp(lamda)],\n        [         0,          0,   exp(lamda)]])\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Matrix_function#Jordan_decomposition\n        "
        size = self.rows
        l = self[0, 0]
        exp_l = exp(l)
        bands = {i: exp_l / factorial(i) for i in range(size)}
        from .sparsetools import banded
        return self.__class__(banded(size, bands))

    def analytic_func(self, f, x):
        if False:
            i = 10
            return i + 15
        "\n        Computes f(A) where A is a Square Matrix\n        and f is an analytic function.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Matrix, S, log\n\n        >>> x = Symbol('x')\n        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])\n        >>> f = log(x)\n        >>> m.analytic_func(f, x)\n        Matrix([\n        [     0, log(2)],\n        [log(2),      0]])\n\n        Parameters\n        ==========\n\n        f : Expr\n            Analytic Function\n        x : Symbol\n            parameter of f\n\n        "
        (f, x) = (_sympify(f), _sympify(x))
        if not self.is_square:
            raise NonSquareMatrixError
        if not x.is_symbol:
            raise ValueError('{} must be a symbol.'.format(x))
        if x not in f.free_symbols:
            raise ValueError('{} must be a parameter of {}.'.format(x, f))
        if x in self.free_symbols:
            raise ValueError('{} must not be a parameter of {}.'.format(x, self))
        eigen = self.eigenvals()
        max_mul = max(eigen.values())
        derivative = {}
        dd = f
        for i in range(max_mul - 1):
            dd = diff(dd, x)
            derivative[i + 1] = dd
        n = self.shape[0]
        r = self.zeros(n)
        f_val = self.zeros(n, 1)
        row = 0
        for i in eigen:
            mul = eigen[i]
            f_val[row] = f.subs(x, i)
            if f_val[row].is_number and (not f_val[row].is_complex):
                raise ValueError('Cannot evaluate the function because the function {} is not analytic at the given eigenvalue {}'.format(f, f_val[row]))
            val = 1
            for a in range(n):
                r[row, a] = val
                val *= i
            if mul > 1:
                coe = [1 for ii in range(n)]
                deri = 1
                while mul > 1:
                    row = row + 1
                    mul -= 1
                    d_i = derivative[deri].subs(x, i)
                    if d_i.is_number and (not d_i.is_complex):
                        raise ValueError('Cannot evaluate the function because the derivative {} is not analytic at the given eigenvalue {}'.format(derivative[deri], d_i))
                    f_val[row] = d_i
                    for a in range(n):
                        if a - deri + 1 <= 0:
                            r[row, a] = 0
                            coe[a] = 0
                            continue
                        coe[a] = coe[a] * (a - deri + 1)
                        r[row, a] = coe[a] * pow(i, a - deri)
                    deri += 1
            row += 1
        c = r.solve(f_val)
        ans = self.zeros(n)
        pre = self.eye(n)
        for i in range(n):
            ans = ans + c[i] * pre
            pre *= self
        return ans

    def exp(self):
        if False:
            return 10
        "Return the exponential of a square matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Matrix\n\n        >>> t = Symbol('t')\n        >>> m = Matrix([[0, 1], [-1, 0]]) * t\n        >>> m.exp()\n        Matrix([\n        [    exp(I*t)/2 + exp(-I*t)/2, -I*exp(I*t)/2 + I*exp(-I*t)/2],\n        [I*exp(I*t)/2 - I*exp(-I*t)/2,      exp(I*t)/2 + exp(-I*t)/2]])\n        "
        if not self.is_square:
            raise NonSquareMatrixError('Exponentiation is valid only for square matrices')
        try:
            (P, J) = self.jordan_form()
            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError('Exponentiation is implemented only for matrices for which the Jordan normal form can be computed')
        blocks = [cell._eval_matrix_exp_jblock() for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)
        ret = P.multiply(eJ, dotprodsimp=None).multiply(P.inv(), dotprodsimp=None)
        if all((value.is_real for value in self.values())):
            return type(self)(re(ret))
        else:
            return type(self)(ret)

    def _eval_matrix_log_jblock(self):
        if False:
            print('Hello World!')
        "Helper function to compute logarithm of a jordan block.\n\n        Examples\n        ========\n\n        >>> from sympy import Symbol, Matrix\n        >>> l = Symbol('lamda')\n\n        A trivial example of 1*1 Jordan block:\n\n        >>> m = Matrix.jordan_block(1, l)\n        >>> m._eval_matrix_log_jblock()\n        Matrix([[log(lamda)]])\n\n        An example of 3*3 Jordan block:\n\n        >>> m = Matrix.jordan_block(3, l)\n        >>> m._eval_matrix_log_jblock()\n        Matrix([\n        [log(lamda),    1/lamda, -1/(2*lamda**2)],\n        [         0, log(lamda),         1/lamda],\n        [         0,          0,      log(lamda)]])\n        "
        size = self.rows
        l = self[0, 0]
        if l.is_zero:
            raise MatrixError('Could not take logarithm or reciprocal for the given eigenvalue {}'.format(l))
        bands = {0: log(l)}
        for i in range(1, size):
            bands[i] = -(-l) ** (-i) / i
        from .sparsetools import banded
        return self.__class__(banded(size, bands))

    def log(self, simplify=cancel):
        if False:
            for i in range(10):
                print('nop')
        'Return the logarithm of a square matrix.\n\n        Parameters\n        ==========\n\n        simplify : function, bool\n            The function to simplify the result with.\n\n            Default is ``cancel``, which is effective to reduce the\n            expression growing for taking reciprocals and inverses for\n            symbolic matrices.\n\n        Examples\n        ========\n\n        >>> from sympy import S, Matrix\n\n        Examples for positive-definite matrices:\n\n        >>> m = Matrix([[1, 1], [0, 1]])\n        >>> m.log()\n        Matrix([\n        [0, 1],\n        [0, 0]])\n\n        >>> m = Matrix([[S(5)/4, S(3)/4], [S(3)/4, S(5)/4]])\n        >>> m.log()\n        Matrix([\n        [     0, log(2)],\n        [log(2),      0]])\n\n        Examples for non positive-definite matrices:\n\n        >>> m = Matrix([[S(3)/4, S(5)/4], [S(5)/4, S(3)/4]])\n        >>> m.log()\n        Matrix([\n        [         I*pi/2, log(2) - I*pi/2],\n        [log(2) - I*pi/2,          I*pi/2]])\n\n        >>> m = Matrix(\n        ...     [[0, 0, 0, 1],\n        ...      [0, 0, 1, 0],\n        ...      [0, 1, 0, 0],\n        ...      [1, 0, 0, 0]])\n        >>> m.log()\n        Matrix([\n        [ I*pi/2,       0,       0, -I*pi/2],\n        [      0,  I*pi/2, -I*pi/2,       0],\n        [      0, -I*pi/2,  I*pi/2,       0],\n        [-I*pi/2,       0,       0,  I*pi/2]])\n        '
        if not self.is_square:
            raise NonSquareMatrixError('Logarithm is valid only for square matrices')
        try:
            if simplify:
                (P, J) = simplify(self).jordan_form()
            else:
                (P, J) = self.jordan_form()
            cells = J.get_diag_blocks()
        except MatrixError:
            raise NotImplementedError('Logarithm is implemented only for matrices for which the Jordan normal form can be computed')
        blocks = [cell._eval_matrix_log_jblock() for cell in cells]
        from sympy.matrices import diag
        eJ = diag(*blocks)
        if simplify:
            ret = simplify(P * eJ * simplify(P.inv()))
            ret = self.__class__(ret)
        else:
            ret = P * eJ * P.inv()
        return ret

    def is_nilpotent(self):
        if False:
            while True:
                i = 10
        'Checks if a matrix is nilpotent.\n\n        A matrix B is nilpotent if for some integer k, B**k is\n        a zero matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> a = Matrix([[0, 0, 0], [1, 0, 0], [1, 1, 0]])\n        >>> a.is_nilpotent()\n        True\n\n        >>> a = Matrix([[1, 0, 1], [1, 0, 0], [1, 1, 0]])\n        >>> a.is_nilpotent()\n        False\n        '
        if not self:
            return True
        if not self.is_square:
            raise NonSquareMatrixError('Nilpotency is valid only for square matrices')
        x = uniquely_named_symbol('x', self, modify=lambda s: '_' + s)
        p = self.charpoly(x)
        if p.args[0] == x ** self.rows:
            return True
        return False

    def key2bounds(self, keys):
        if False:
            for i in range(10):
                print('nop')
        "Converts a key with potentially mixed types of keys (integer and slice)\n        into a tuple of ranges and raises an error if any index is out of ``self``'s\n        range.\n\n        See Also\n        ========\n\n        key2ij\n        "
        (islice, jslice) = [isinstance(k, slice) for k in keys]
        if islice:
            if not self.rows:
                rlo = rhi = 0
            else:
                (rlo, rhi) = keys[0].indices(self.rows)[:2]
        else:
            rlo = a2idx(keys[0], self.rows)
            rhi = rlo + 1
        if jslice:
            if not self.cols:
                clo = chi = 0
            else:
                (clo, chi) = keys[1].indices(self.cols)[:2]
        else:
            clo = a2idx(keys[1], self.cols)
            chi = clo + 1
        return (rlo, rhi, clo, chi)

    def key2ij(self, key):
        if False:
            print('Hello World!')
        "Converts key into canonical form, converting integers or indexable\n        items into valid integers for ``self``'s range or returning slices\n        unchanged.\n\n        See Also\n        ========\n\n        key2bounds\n        "
        if is_sequence(key):
            if not len(key) == 2:
                raise TypeError('key must be a sequence of length 2')
            return [a2idx(i, n) if not isinstance(i, slice) else i for (i, n) in zip(key, self.shape)]
        elif isinstance(key, slice):
            return key.indices(len(self))[:2]
        else:
            return divmod(a2idx(key, len(self)), self.cols)

    def normalized(self, iszerofunc=_iszero):
        if False:
            for i in range(10):
                print('nop')
        'Return the normalized version of ``self``.\n\n        Parameters\n        ==========\n\n        iszerofunc : Function, optional\n            A function to determine whether ``self`` is a zero vector.\n            The default ``_iszero`` tests to see if each element is\n            exactly zero.\n\n        Returns\n        =======\n\n        Matrix\n            Normalized vector form of ``self``.\n            It has the same length as a unit vector. However, a zero vector\n            will be returned for a vector with norm 0.\n\n        Raises\n        ======\n\n        ShapeError\n            If the matrix is not in a vector form.\n\n        See Also\n        ========\n\n        norm\n        '
        if self.rows != 1 and self.cols != 1:
            raise ShapeError('A Matrix must be a vector to normalize.')
        norm = self.norm()
        if iszerofunc(norm):
            out = self.zeros(self.rows, self.cols)
        else:
            out = self.applyfunc(lambda i: i / norm)
        return out

    def norm(self, ord=None):
        if False:
            i = 10
            return i + 15
        "Return the Norm of a Matrix or Vector.\n\n        In the simplest case this is the geometric size of the vector\n        Other norms can be specified by the ord parameter\n\n\n        =====  ============================  ==========================\n        ord    norm for matrices             norm for vectors\n        =====  ============================  ==========================\n        None   Frobenius norm                2-norm\n        'fro'  Frobenius norm                - does not exist\n        inf    maximum row sum               max(abs(x))\n        -inf   --                            min(abs(x))\n        1      maximum column sum            as below\n        -1     --                            as below\n        2      2-norm (largest sing. value)  as below\n        -2     smallest singular value       as below\n        other  - does not exist              sum(abs(x)**ord)**(1./ord)\n        =====  ============================  ==========================\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, Symbol, trigsimp, cos, sin, oo\n        >>> x = Symbol('x', real=True)\n        >>> v = Matrix([cos(x), sin(x)])\n        >>> trigsimp( v.norm() )\n        1\n        >>> v.norm(10)\n        (sin(x)**10 + cos(x)**10)**(1/10)\n        >>> A = Matrix([[1, 1], [1, 1]])\n        >>> A.norm(1) # maximum sum of absolute values of A is 2\n        2\n        >>> A.norm(2) # Spectral norm (max of |Ax|/|x| under 2-vector-norm)\n        2\n        >>> A.norm(-2) # Inverse spectral norm (smallest singular value)\n        0\n        >>> A.norm() # Frobenius Norm\n        2\n        >>> A.norm(oo) # Infinity Norm\n        2\n        >>> Matrix([1, -2]).norm(oo)\n        2\n        >>> Matrix([-1, 2]).norm(-oo)\n        1\n\n        See Also\n        ========\n\n        normalized\n        "
        vals = list(self.values()) or [0]
        if S.One in self.shape:
            if ord in (2, None):
                return sqrt(Add(*(abs(i) ** 2 for i in vals)))
            elif ord == 1:
                return Add(*(abs(i) for i in vals))
            elif ord is S.Infinity:
                return Max(*[abs(i) for i in vals])
            elif ord is S.NegativeInfinity:
                return Min(*[abs(i) for i in vals])
            try:
                return Pow(Add(*(abs(i) ** ord for i in vals)), S.One / ord)
            except (NotImplementedError, TypeError):
                raise ValueError('Expected order to be Number, Symbol, oo')
        elif ord == 1:
            m = self.applyfunc(abs)
            return Max(*[sum(m.col(i)) for i in range(m.cols)])
        elif ord == 2:
            return Max(*self.singular_values())
        elif ord == -2:
            return Min(*self.singular_values())
        elif ord is S.Infinity:
            m = self.applyfunc(abs)
            return Max(*[sum(m.row(i)) for i in range(m.rows)])
        elif ord is None or (isinstance(ord, str) and ord.lower() in ['f', 'fro', 'frobenius', 'vector']):
            return self.vec().norm(ord=2)
        else:
            raise NotImplementedError('Matrix Norms under development')

    def print_nonzero(self, symb='X'):
        if False:
            while True:
                i = 10
        'Shows location of non-zero entries for fast shape lookup.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, eye\n        >>> m = Matrix(2, 3, lambda i, j: i*3+j)\n        >>> m\n        Matrix([\n        [0, 1, 2],\n        [3, 4, 5]])\n        >>> m.print_nonzero()\n        [ XX]\n        [XXX]\n        >>> m = eye(4)\n        >>> m.print_nonzero("x")\n        [x   ]\n        [ x  ]\n        [  x ]\n        [   x]\n\n        '
        s = []
        for i in range(self.rows):
            line = []
            for j in range(self.cols):
                if self[i, j] == 0:
                    line.append(' ')
                else:
                    line.append(str(symb))
            s.append('[%s]' % ''.join(line))
        print('\n'.join(s))

    def project(self, v):
        if False:
            while True:
                i = 10
        'Return the projection of ``self`` onto the line containing ``v``.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, S, sqrt\n        >>> V = Matrix([sqrt(3)/2, S.Half])\n        >>> x = Matrix([[1, 0]])\n        >>> V.project(x)\n        Matrix([[sqrt(3)/2, 0]])\n        >>> V.project(-x)\n        Matrix([[sqrt(3)/2, 0]])\n        '
        return v * (self.dot(v) / v.dot(v))

    def table(self, printer, rowstart='[', rowend=']', rowsep='\n', colsep=', ', align='right'):
        if False:
            return 10
        "\n        String form of Matrix as a table.\n\n        ``printer`` is the printer to use for on the elements (generally\n        something like StrPrinter())\n\n        ``rowstart`` is the string used to start each row (by default '[').\n\n        ``rowend`` is the string used to end each row (by default ']').\n\n        ``rowsep`` is the string used to separate rows (by default a newline).\n\n        ``colsep`` is the string used to separate columns (by default ', ').\n\n        ``align`` defines how the elements are aligned. Must be one of 'left',\n        'right', or 'center'.  You can also use '<', '>', and '^' to mean the\n        same thing, respectively.\n\n        This is used by the string printer for Matrix.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, StrPrinter\n        >>> M = Matrix([[1, 2], [-33, 4]])\n        >>> printer = StrPrinter()\n        >>> M.table(printer)\n        '[  1, 2]\\n[-33, 4]'\n        >>> print(M.table(printer))\n        [  1, 2]\n        [-33, 4]\n        >>> print(M.table(printer, rowsep=',\\n'))\n        [  1, 2],\n        [-33, 4]\n        >>> print('[%s]' % M.table(printer, rowsep=',\\n'))\n        [[  1, 2],\n        [-33, 4]]\n        >>> print(M.table(printer, colsep=' '))\n        [  1 2]\n        [-33 4]\n        >>> print(M.table(printer, align='center'))\n        [ 1 , 2]\n        [-33, 4]\n        >>> print(M.table(printer, rowstart='{', rowend='}'))\n        {  1, 2}\n        {-33, 4}\n        "
        if S.Zero in self.shape:
            return '[]'
        res = []
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                s = printer._print(self[i, j])
                res[-1].append(s)
                maxlen[j] = max(len(s), maxlen[j])
        align = {'left': 'ljust', 'right': 'rjust', 'center': 'center', '<': 'ljust', '>': 'rjust', '^': 'center'}[align]
        for (i, row) in enumerate(res):
            for (j, elem) in enumerate(row):
                row[j] = getattr(elem, align)(maxlen[j])
            res[i] = rowstart + colsep.join(row) + rowend
        return rowsep.join(res)

    def rank_decomposition(self, iszerofunc=_iszero, simplify=False):
        if False:
            print('Hello World!')
        return _rank_decomposition(self, iszerofunc=iszerofunc, simplify=simplify)

    def cholesky(self, hermitian=True):
        if False:
            while True:
                i = 10
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LDLdecomposition(self, hermitian=True):
        if False:
            print('Hello World!')
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def LUdecomposition(self, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
        if False:
            for i in range(10):
                print('nop')
        return _LUdecomposition(self, iszerofunc=iszerofunc, simpfunc=simpfunc, rankcheck=rankcheck)

    def LUdecomposition_Simple(self, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
        if False:
            print('Hello World!')
        return _LUdecomposition_Simple(self, iszerofunc=iszerofunc, simpfunc=simpfunc, rankcheck=rankcheck)

    def LUdecompositionFF(self):
        if False:
            print('Hello World!')
        return _LUdecompositionFF(self)

    def singular_value_decomposition(self):
        if False:
            for i in range(10):
                print('nop')
        return _singular_value_decomposition(self)

    def QRdecomposition(self):
        if False:
            print('Hello World!')
        return _QRdecomposition(self)

    def upper_hessenberg_decomposition(self):
        if False:
            return 10
        return _upper_hessenberg_decomposition(self)

    def diagonal_solve(self, rhs):
        if False:
            while True:
                i = 10
        return _diagonal_solve(self, rhs)

    def lower_triangular_solve(self, rhs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def upper_triangular_solve(self, rhs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('This function is implemented in DenseMatrix or SparseMatrix')

    def cholesky_solve(self, rhs):
        if False:
            for i in range(10):
                print('nop')
        return _cholesky_solve(self, rhs)

    def LDLsolve(self, rhs):
        if False:
            return 10
        return _LDLsolve(self, rhs)

    def LUsolve(self, rhs, iszerofunc=_iszero):
        if False:
            print('Hello World!')
        return _LUsolve(self, rhs, iszerofunc=iszerofunc)

    def QRsolve(self, b):
        if False:
            return 10
        return _QRsolve(self, b)

    def gauss_jordan_solve(self, B, freevar=False):
        if False:
            i = 10
            return i + 15
        return _gauss_jordan_solve(self, B, freevar=freevar)

    def pinv_solve(self, B, arbitrary_matrix=None):
        if False:
            print('Hello World!')
        return _pinv_solve(self, B, arbitrary_matrix=arbitrary_matrix)

    def cramer_solve(self, rhs, det_method='laplace'):
        if False:
            return 10
        return _cramer_solve(self, rhs, det_method=det_method)

    def solve(self, rhs, method='GJ'):
        if False:
            i = 10
            return i + 15
        return _solve(self, rhs, method=method)

    def solve_least_squares(self, rhs, method='CH'):
        if False:
            while True:
                i = 10
        return _solve_least_squares(self, rhs, method=method)

    def pinv(self, method='RD'):
        if False:
            return 10
        return _pinv(self, method=method)

    def inverse_ADJ(self, iszerofunc=_iszero):
        if False:
            i = 10
            return i + 15
        return _inv_ADJ(self, iszerofunc=iszerofunc)

    def inverse_BLOCK(self, iszerofunc=_iszero):
        if False:
            return 10
        return _inv_block(self, iszerofunc=iszerofunc)

    def inverse_GE(self, iszerofunc=_iszero):
        if False:
            i = 10
            return i + 15
        return _inv_GE(self, iszerofunc=iszerofunc)

    def inverse_LU(self, iszerofunc=_iszero):
        if False:
            while True:
                i = 10
        return _inv_LU(self, iszerofunc=iszerofunc)

    def inverse_CH(self, iszerofunc=_iszero):
        if False:
            for i in range(10):
                print('nop')
        return _inv_CH(self, iszerofunc=iszerofunc)

    def inverse_LDL(self, iszerofunc=_iszero):
        if False:
            for i in range(10):
                print('nop')
        return _inv_LDL(self, iszerofunc=iszerofunc)

    def inverse_QR(self, iszerofunc=_iszero):
        if False:
            print('Hello World!')
        return _inv_QR(self, iszerofunc=iszerofunc)

    def inv(self, method=None, iszerofunc=_iszero, try_block_diag=False):
        if False:
            for i in range(10):
                print('nop')
        return _inv(self, method=method, iszerofunc=iszerofunc, try_block_diag=try_block_diag)

    def connected_components(self):
        if False:
            return 10
        return _connected_components(self)

    def connected_components_decomposition(self):
        if False:
            for i in range(10):
                print('nop')
        return _connected_components_decomposition(self)

    def strongly_connected_components(self):
        if False:
            i = 10
            return i + 15
        return _strongly_connected_components(self)

    def strongly_connected_components_decomposition(self, lower=True):
        if False:
            i = 10
            return i + 15
        return _strongly_connected_components_decomposition(self, lower=lower)
    _sage_ = Basic._sage_
    rank_decomposition.__doc__ = _rank_decomposition.__doc__
    cholesky.__doc__ = _cholesky.__doc__
    LDLdecomposition.__doc__ = _LDLdecomposition.__doc__
    LUdecomposition.__doc__ = _LUdecomposition.__doc__
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    LUdecompositionFF.__doc__ = _LUdecompositionFF.__doc__
    singular_value_decomposition.__doc__ = _singular_value_decomposition.__doc__
    QRdecomposition.__doc__ = _QRdecomposition.__doc__
    upper_hessenberg_decomposition.__doc__ = _upper_hessenberg_decomposition.__doc__
    diagonal_solve.__doc__ = _diagonal_solve.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    cholesky_solve.__doc__ = _cholesky_solve.__doc__
    LDLsolve.__doc__ = _LDLsolve.__doc__
    LUsolve.__doc__ = _LUsolve.__doc__
    QRsolve.__doc__ = _QRsolve.__doc__
    gauss_jordan_solve.__doc__ = _gauss_jordan_solve.__doc__
    pinv_solve.__doc__ = _pinv_solve.__doc__
    cramer_solve.__doc__ = _cramer_solve.__doc__
    solve.__doc__ = _solve.__doc__
    solve_least_squares.__doc__ = _solve_least_squares.__doc__
    pinv.__doc__ = _pinv.__doc__
    inverse_ADJ.__doc__ = _inv_ADJ.__doc__
    inverse_GE.__doc__ = _inv_GE.__doc__
    inverse_LU.__doc__ = _inv_LU.__doc__
    inverse_CH.__doc__ = _inv_CH.__doc__
    inverse_LDL.__doc__ = _inv_LDL.__doc__
    inverse_QR.__doc__ = _inv_QR.__doc__
    inverse_BLOCK.__doc__ = _inv_block.__doc__
    inv.__doc__ = _inv.__doc__
    connected_components.__doc__ = _connected_components.__doc__
    connected_components_decomposition.__doc__ = _connected_components_decomposition.__doc__
    strongly_connected_components.__doc__ = _strongly_connected_components.__doc__
    strongly_connected_components_decomposition.__doc__ = _strongly_connected_components_decomposition.__doc__