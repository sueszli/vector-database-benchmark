from sympy.external.importtools import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.polys.domains import ZZ, QQ
from .exceptions import DMBadInputError, DMDomainError, DMNonSquareMatrixError, DMNonInvertibleMatrixError, DMRankError, DMShapeError, DMValueError
flint = import_module('flint')
__all__ = ['DFM']

@doctest_depends_on(ground_types=['flint'])
class DFM:
    """
    Dense FLINT matrix. This class is a wrapper for matrices from python-flint.

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.matrices.dfm import DFM
    >>> dfm = DFM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> dfm
    [[1, 2], [3, 4]]
    >>> dfm.rep
    [1, 2]
    [3, 4]
    >>> type(dfm.rep)  # doctest: +SKIP
    <class 'flint._flint.fmpz_mat'>

    Usually, the DFM class is not instantiated directly, but is created as the
    internal representation of :class:`~.DomainMatrix`. When
    `SYMPY_GROUND_TYPES` is set to `flint` and `python-flint` is installed, the
    :class:`DFM` class is used automatically as the internal representation of
    :class:`~.DomainMatrix` in dense format if the domain is supported by
    python-flint.

    >>> from sympy.polys.matrices.domainmatrix import DM
    >>> dM = DM([[1, 2], [3, 4]], ZZ)
    >>> dM.rep
    [[1, 2], [3, 4]]

    A :class:`~.DomainMatrix` can be converted to :class:`DFM` by calling the
    :meth:`to_dfm` method:

    >>> dM.to_dfm()
    [[1, 2], [3, 4]]

    """
    fmt = 'dense'
    is_DFM = True
    is_DDM = False

    def __new__(cls, rowslist, shape, domain):
        if False:
            for i in range(10):
                print('nop')
        'Construct from a nested list.'
        flint_mat = cls._get_flint_func(domain)
        if 0 not in shape:
            try:
                rep = flint_mat(rowslist)
            except (ValueError, TypeError):
                raise DMBadInputError(f'Input should be a list of list of {domain}')
        else:
            rep = flint_mat(*shape)
        return cls._new(rep, shape, domain)

    @classmethod
    def _new(cls, rep, shape, domain):
        if False:
            print('Hello World!')
        'Internal constructor from a flint matrix.'
        cls._check(rep, shape, domain)
        obj = object.__new__(cls)
        obj.rep = rep
        obj.shape = (obj.rows, obj.cols) = shape
        obj.domain = domain
        return obj

    def _new_rep(self, rep):
        if False:
            print('Hello World!')
        'Create a new DFM with the same shape and domain but a new rep.'
        return self._new(rep, self.shape, self.domain)

    @classmethod
    def _check(cls, rep, shape, domain):
        if False:
            print('Hello World!')
        repshape = (rep.nrows(), rep.ncols())
        if repshape != shape:
            raise DMBadInputError('Shape of rep does not match shape of DFM')
        if domain == ZZ and (not isinstance(rep, flint.fmpz_mat)):
            raise RuntimeError('Rep is not a flint.fmpz_mat')
        elif domain == QQ and (not isinstance(rep, flint.fmpq_mat)):
            raise RuntimeError('Rep is not a flint.fmpq_mat')
        elif domain not in (ZZ, QQ):
            raise NotImplementedError('Only ZZ and QQ are supported by DFM')

    @classmethod
    def _supports_domain(cls, domain):
        if False:
            print('Hello World!')
        'Return True if the given domain is supported by DFM.'
        return domain in (ZZ, QQ)

    @classmethod
    def _get_flint_func(cls, domain):
        if False:
            return 10
        'Return the flint matrix class for the given domain.'
        if domain == ZZ:
            return flint.fmpz_mat
        elif domain == QQ:
            return flint.fmpq_mat
        else:
            raise NotImplementedError('Only ZZ and QQ are supported by DFM')

    @property
    def _func(self):
        if False:
            for i in range(10):
                print('nop')
        'Callable to create a flint matrix of the same domain.'
        return self._get_flint_func(self.domain)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Return ``str(self)``.'
        return str(self.to_ddm())

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Return ``repr(self)``.'
        return f'DFM{repr(self.to_ddm())[3:]}'

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return ``self == other``.'
        if not isinstance(other, DFM):
            return NotImplemented
        return self.domain == other.domain and self.rep == other.rep

    @classmethod
    def from_list(cls, rowslist, shape, domain):
        if False:
            print('Hello World!')
        'Construct from a nested list.'
        return cls(rowslist, shape, domain)

    def to_list(self):
        if False:
            i = 10
            return i + 15
        'Convert to a nested list.'
        return self.rep.tolist()

    def copy(self):
        if False:
            return 10
        'Return a copy of self.'
        return self._new_rep(self._func(self.rep))

    def to_ddm(self):
        if False:
            i = 10
            return i + 15
        'Convert to a DDM.'
        return DDM.from_list(self.to_list(), self.shape, self.domain)

    def to_sdm(self):
        if False:
            i = 10
            return i + 15
        'Convert to a SDM.'
        return SDM.from_list(self.to_list(), self.shape, self.domain)

    def to_dfm(self):
        if False:
            i = 10
            return i + 15
        'Return self.'
        return self

    def to_dfm_or_ddm(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert to a :class:`DFM`.\n\n        This :class:`DFM` method exists to parallel the :class:`~.DDM` and\n        :class:`~.SDM` methods. For :class:`DFM` it will always return self.\n\n        See Also\n        ========\n\n        to_ddm\n        to_sdm\n        sympy.polys.matrices.domainmatrix.DomainMatrix.to_dfm_or_ddm\n        '
        return self

    @classmethod
    def from_ddm(cls, ddm):
        if False:
            while True:
                i = 10
        'Convert from a DDM.'
        return cls.from_list(ddm.to_list(), ddm.shape, ddm.domain)

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        if False:
            print('Hello World!')
        'Inverse of :meth:`to_list_flat`.'
        func = cls._get_flint_func(domain)
        try:
            rep = func(*shape, elements)
        except ValueError:
            raise DMBadInputError(f'Incorrect number of elements for shape {shape}')
        except TypeError:
            raise DMBadInputError(f'Input should be a list of {domain}')
        return cls(rep, shape, domain)

    def to_list_flat(self):
        if False:
            while True:
                i = 10
        'Convert to a flat list.'
        return self.rep.entries()

    def to_flat_nz(self):
        if False:
            while True:
                i = 10
        'Convert to a flat list of non-zeros.'
        return self.to_ddm().to_flat_nz()

    @classmethod
    def from_flat_nz(cls, elements, data, domain):
        if False:
            while True:
                i = 10
        'Inverse of :meth:`to_flat_nz`.'
        return DDM.from_flat_nz(elements, data, domain).to_dfm()

    def to_dok(self):
        if False:
            print('Hello World!')
        'Convert to a DOK.'
        return self.to_ddm().to_dok()

    def convert_to(self, domain):
        if False:
            print('Hello World!')
        'Convert to a new domain.'
        if domain == self.domain:
            return self.copy()
        elif domain == QQ and self.domain == ZZ:
            return self._new(flint.fmpq_mat(self.rep), self.shape, domain)
        elif domain == ZZ and self.domain == QQ:
            return self.to_ddm().convert_to(domain).to_dfm()
        else:
            raise NotImplementedError('Only ZZ and QQ are supported by DFM')

    def getitem(self, i, j):
        if False:
            for i in range(10):
                print('nop')
        'Get the ``(i, j)``-th entry.'
        (m, n) = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            return self.rep[i, j]
        except ValueError:
            raise IndexError(f'Invalid indices ({i}, {j}) for Matrix of shape {self.shape}')

    def setitem(self, i, j, value):
        if False:
            print('Hello World!')
        'Set the ``(i, j)``-th entry.'
        (m, n) = self.shape
        if i < 0:
            i += m
        if j < 0:
            j += n
        try:
            self.rep[i, j] = value
        except ValueError:
            raise IndexError(f'Invalid indices ({i}, {j}) for Matrix of shape {self.shape}')

    def _extract(self, i_indices, j_indices):
        if False:
            i = 10
            return i + 15
        'Extract a submatrix with no checking.'
        M = self.rep
        lol = [[M[i, j] for j in j_indices] for i in i_indices]
        shape = (len(i_indices), len(j_indices))
        return self.from_list(lol, shape, self.domain)

    def extract(self, rowslist, colslist):
        if False:
            for i in range(10):
                print('nop')
        'Extract a submatrix.'
        (m, n) = self.shape
        new_rows = []
        new_cols = []
        for i in rowslist:
            if i < 0:
                i_pos = i + m
            else:
                i_pos = i
            if not 0 <= i_pos < m:
                raise IndexError(f'Invalid row index {i} for Matrix of shape {self.shape}')
            new_rows.append(i_pos)
        for j in colslist:
            if j < 0:
                j_pos = j + n
            else:
                j_pos = j
            if not 0 <= j_pos < n:
                raise IndexError(f'Invalid column index {j} for Matrix of shape {self.shape}')
            new_cols.append(j_pos)
        return self._extract(new_rows, new_cols)

    def extract_slice(self, rowslice, colslice):
        if False:
            for i in range(10):
                print('nop')
        'Slice a DFM.'
        (m, n) = self.shape
        i_indices = range(m)[rowslice]
        j_indices = range(n)[colslice]
        return self._extract(i_indices, j_indices)

    def neg(self):
        if False:
            return 10
        'Negate a DFM matrix.'
        return self._new_rep(-self.rep)

    def add(self, other):
        if False:
            i = 10
            return i + 15
        'Add two DFM matrices.'
        return self._new_rep(self.rep + other.rep)

    def sub(self, other):
        if False:
            i = 10
            return i + 15
        'Subtract two DFM matrices.'
        return self._new_rep(self.rep - other.rep)

    def mul(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Multiply a DFM matrix from the right by a scalar.'
        return self._new_rep(self.rep * other)

    def rmul(self, other):
        if False:
            while True:
                i = 10
        'Multiply a DFM matrix from the left by a scalar.'
        return self._new_rep(other * self.rep)

    def mul_elementwise(self, other):
        if False:
            print('Hello World!')
        'Elementwise multiplication of two DFM matrices.'
        return self.to_ddm().mul_elementwise(other.to_ddm()).to_dfm()

    def matmul(self, other):
        if False:
            return 10
        'Multiply two DFM matrices.'
        shape = (self.rows, other.cols)
        return self._new(self.rep * other.rep, shape, self.domain)

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        'Negate a DFM matrix.'
        return self.neg()

    @classmethod
    def zeros(cls, shape, domain):
        if False:
            while True:
                i = 10
        'Return a zero DFM matrix.'
        func = cls._get_flint_func(domain)
        return cls._new(func(*shape), shape, domain)

    @classmethod
    def ones(cls, shape, domain):
        if False:
            print('Hello World!')
        'Return a one DFM matrix.'
        return DDM.ones(shape, domain).to_dfm()

    @classmethod
    def eye(cls, n, domain):
        if False:
            i = 10
            return i + 15
        'Return the identity matrix of size n.'
        return DDM.eye(n, domain).to_dfm()

    @classmethod
    def diag(cls, elements, domain):
        if False:
            i = 10
            return i + 15
        'Return a diagonal matrix.'
        return DDM.diag(elements, domain).to_dfm()

    def applyfunc(self, func, domain):
        if False:
            print('Hello World!')
        'Apply a function to each entry of a DFM matrix.'
        return self.to_ddm().applyfunc(func, domain).to_dfm()

    def transpose(self):
        if False:
            return 10
        'Transpose a DFM matrix.'
        return self._new(self.rep.transpose(), (self.cols, self.rows), self.domain)

    def hstack(self, *others):
        if False:
            for i in range(10):
                print('nop')
        'Horizontally stack matrices.'
        return self.to_ddm().hstack(*[o.to_ddm() for o in others]).to_dfm()

    def vstack(self, *others):
        if False:
            return 10
        'Vertically stack matrices.'
        return self.to_ddm().vstack(*[o.to_ddm() for o in others]).to_dfm()

    def diagonal(self):
        if False:
            i = 10
            return i + 15
        'Return the diagonal of a DFM matrix.'
        M = self.rep
        (m, n) = self.shape
        return [M[i, i] for i in range(min(m, n))]

    def is_upper(self):
        if False:
            while True:
                i = 10
        'Return ``True`` if the matrix is upper triangular.'
        M = self.rep
        for i in range(self.rows):
            for j in range(i):
                if M[i, j]:
                    return False
        return True

    def is_lower(self):
        if False:
            for i in range(10):
                print('nop')
        'Return ``True`` if the matrix is lower triangular.'
        M = self.rep
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if M[i, j]:
                    return False
        return True

    def is_diagonal(self):
        if False:
            print('Hello World!')
        'Return ``True`` if the matrix is diagonal.'
        return self.is_upper() and self.is_lower()

    def is_zero_matrix(self):
        if False:
            i = 10
            return i + 15
        'Return ``True`` if the matrix is the zero matrix.'
        M = self.rep
        for i in range(self.rows):
            for j in range(self.cols):
                if M[i, j]:
                    return False
        return True

    def nnz(self):
        if False:
            i = 10
            return i + 15
        'Return the number of non-zero elements in the matrix.'
        return self.to_ddm().nnz()

    def scc(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the strongly connected components of the matrix.'
        return self.to_ddm().scc()

    @doctest_depends_on(ground_types='flint')
    def det(self):
        if False:
            print('Hello World!')
        '\n        Compute the determinant of the matrix using FLINT.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 2], [3, 4]])\n        >>> dfm = M.to_DM().to_dfm()\n        >>> dfm\n        [[1, 2], [3, 4]]\n        >>> dfm.det()\n        -2\n\n        Notes\n        =====\n\n        Calls the ``.det()`` method of the underlying FLINT matrix.\n\n        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_det`` or\n        ``fmpq_mat_det`` respectively.\n\n        At the time of writing the implementation of ``fmpz_mat_det`` uses one\n        of several algorithms depending on the size of the matrix and bit size\n        of the entries. The algorithms used are:\n\n        - Cofactor for very small (up to 4x4) matrices.\n        - Bareiss for small (up to 25x25) matrices.\n        - Modular algorithms for larger matrices (up to 60x60) or for larger\n          matrices with large bit sizes.\n        - Modular "accelerated" for larger matrices (60x60 upwards) if the bit\n          size is smaller than the dimensions of the matrix.\n\n        The implementation of ``fmpq_mat_det`` clears denominators from each\n        row (not the whole matrix) and then calls ``fmpz_mat_det`` and divides\n        by the product of the denominators.\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.det\n            Higher level interface to compute the determinant of a matrix.\n        '
        return self.rep.det()

    @doctest_depends_on(ground_types='flint')
    def charpoly(self):
        if False:
            while True:
                i = 10
        "\n        Compute the characteristic polynomial of the matrix using FLINT.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 2], [3, 4]])\n        >>> dfm = M.to_DM().to_dfm()  # need ground types = 'flint'\n        >>> dfm\n        [[1, 2], [3, 4]]\n        >>> dfm.charpoly()\n        [1, -5, -2]\n\n        Notes\n        =====\n\n        Calls the ``.charpoly()`` method of the underlying FLINT matrix.\n\n        For :ref:`ZZ` or :ref:`QQ` this calls ``fmpz_mat_charpoly`` or\n        ``fmpq_mat_charpoly`` respectively.\n\n        At the time of writing the implementation of ``fmpq_mat_charpoly``\n        clears a denominator from the whole matrix and then calls\n        ``fmpz_mat_charpoly``. The coefficients of the characteristic\n        polynomial are then multiplied by powers of the denominator.\n\n        The ``fmpz_mat_charpoly`` method uses a modular algorithm with CRT\n        reconstruction. The modular algorithm uses ``nmod_mat_charpoly`` which\n        uses Berkowitz for small matrices and non-prime moduli or otherwise\n        the Danilevsky method.\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly\n            Higher level interface to compute the characteristic polynomial of\n            a matrix.\n        "
        return self.rep.charpoly().coeffs()[::-1]

    @doctest_depends_on(ground_types='flint')
    def inv(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the inverse of a matrix using FLINT.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, QQ\n        >>> M = Matrix([[1, 2], [3, 4]])\n        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)\n        >>> dfm\n        [[1, 2], [3, 4]]\n        >>> dfm.inv()\n        [[-2, 1], [3/2, -1/2]]\n        >>> dfm.matmul(dfm.inv())\n        [[1, 0], [0, 1]]\n\n        Notes\n        =====\n\n        Calls the ``.inv()`` method of the underlying FLINT matrix.\n\n        For now this will raise an error if the domain is :ref:`ZZ` but will\n        use the FLINT method for :ref:`QQ`.\n\n        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_inv`` and\n        ``fmpq_mat_inv`` respectively. The ``fmpz_mat_inv`` method computes an\n        inverse with denominator. This is implemented by calling\n        ``fmpz_mat_solve`` (see notes in :meth:`lu_solve` about the algorithm).\n\n        The ``fmpq_mat_inv`` method clears denominators from each row and then\n        multiplies those into the rhs identity matrix before calling\n        ``fmpz_mat_solve``.\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.inv\n            Higher level method for computing the inverse of a matrix.\n        '
        K = self.domain
        (m, n) = self.shape
        if m != n:
            raise DMNonSquareMatrixError('cannot invert a non-square matrix')
        if K == ZZ:
            raise DMDomainError('field expected, got %s' % K)
        elif K == QQ:
            try:
                return self._new_rep(self.rep.inv())
            except ZeroDivisionError:
                raise DMNonInvertibleMatrixError('matrix is not invertible')
        else:
            raise NotImplementedError('DFM.inv() is not implemented for %s' % K)

    def lu(self):
        if False:
            while True:
                i = 10
        'Return the LU decomposition of the matrix.'
        (L, U, swaps) = self.to_ddm().lu()
        return (L.to_dfm(), U.to_dfm(), swaps)

    @doctest_depends_on(ground_types='flint')
    def lu_solve(self, rhs):
        if False:
            return 10
        "\n        Solve a matrix equation using FLINT.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix, QQ\n        >>> M = Matrix([[1, 2], [3, 4]])\n        >>> dfm = M.to_DM().to_dfm().convert_to(QQ)\n        >>> dfm\n        [[1, 2], [3, 4]]\n        >>> rhs = Matrix([1, 2]).to_DM().to_dfm().convert_to(QQ)\n        >>> dfm.lu_solve(rhs)\n        [[0], [1/2]]\n\n        Notes\n        =====\n\n        Calls the ``.solve()`` method of the underlying FLINT matrix.\n\n        For now this will raise an error if the domain is :ref:`ZZ` but will\n        use the FLINT method for :ref:`QQ`.\n\n        The FLINT methods for :ref:`ZZ` and :ref:`QQ` are ``fmpz_mat_solve``\n        and ``fmpq_mat_solve`` respectively. The ``fmpq_mat_solve`` method\n        uses one of two algorithms:\n\n        - For small matrices (<25 rows) it clears denominators between the\n          matrix and rhs and uses ``fmpz_mat_solve``.\n        - For larger matrices it uses ``fmpq_mat_solve_dixon`` which is a\n          modular approach with CRT reconstruction over :ref:`QQ`.\n\n        The ``fmpz_mat_solve`` method uses one of four algorithms:\n\n        - For very small (<= 3x3) matrices it uses a Cramer's rule.\n        - For small (<= 15x15) matrices it uses a fraction-free LU solve.\n        - Otherwise it uses either Dixon or another multimodular approach.\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.lu_solve\n            Higher level interface to solve a matrix equation.\n        "
        if not self.domain == rhs.domain:
            raise DMDomainError('Domains must match: %s != %s' % (self.domain, rhs.domain))
        if not self.domain.is_Field:
            raise DMDomainError('Field expected, got %s' % self.domain)
        (m, n) = self.shape
        (j, k) = rhs.shape
        if m != j:
            raise DMShapeError('Matrix size mismatch: %s * %s vs %s * %s' % (m, n, j, k))
        sol_shape = (n, k)
        if m != n:
            return self.to_ddm().lu_solve(rhs.to_ddm()).to_dfm()
        try:
            sol = self.rep.solve(rhs.rep)
        except ZeroDivisionError:
            raise DMNonInvertibleMatrixError('Matrix det == 0; not invertible.')
        return self._new(sol, sol_shape, self.domain)

    def nullspace(self):
        if False:
            return 10
        'Return a basis for the nullspace of the matrix.'
        (ddm, nonpivots) = self.to_ddm().nullspace()
        return (ddm.to_dfm(), nonpivots)

    def nullspace_from_rref(self, pivots=None):
        if False:
            return 10
        'Return a basis for the nullspace of the matrix.'
        (sdm, nonpivots) = self.to_sdm().nullspace_from_rref(pivots=pivots)
        return (sdm.to_dfm(), nonpivots)

    def particular(self):
        if False:
            i = 10
            return i + 15
        'Return a particular solution to the system.'
        return self.to_ddm().particular().to_dfm()

    def _lll(self, transform=False, delta=0.99, eta=0.51, rep='zbasis', gram='approx'):
        if False:
            return 10
        'Call the fmpz_mat.lll() method but check rank to avoid segfaults.'

        def to_float(x):
            if False:
                while True:
                    i = 10
            if QQ.of_type(x):
                return float(x.numerator) / float(x.denominator)
            else:
                return float(x)
        delta = to_float(delta)
        eta = to_float(eta)
        if not 0.25 < delta < 1:
            raise DMValueError('delta must be between 0.25 and 1')
        (m, n) = self.shape
        if self.rep.rank() != m:
            raise DMRankError('Matrix must have full row rank for Flint LLL.')
        return self.rep.lll(transform=transform, delta=delta, eta=eta, rep=rep, gram=gram)

    @doctest_depends_on(ground_types='flint')
    def lll(self, delta=0.75):
        if False:
            print('Hello World!')
        'Compute LLL-reduced basis using FLINT.\n\n        See :meth:`lll_transform` for more information.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 2, 3], [4, 5, 6]])\n        >>> M.to_DM().to_dfm().lll()\n        [[2, 1, 0], [-1, 1, 3]]\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.lll\n            Higher level interface to compute LLL-reduced basis.\n        lll_transform\n            Compute LLL-reduced basis and transform matrix.\n        '
        if self.domain != ZZ:
            raise DMDomainError('ZZ expected, got %s' % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError('Matrix must not have more rows than columns.')
        rep = self._lll(delta=delta)
        return self._new_rep(rep)

    @doctest_depends_on(ground_types='flint')
    def lll_transform(self, delta=0.75):
        if False:
            while True:
                i = 10
        'Compute LLL-reduced basis and transform using FLINT.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> M = Matrix([[1, 2, 3], [4, 5, 6]]).to_DM().to_dfm()\n        >>> M_lll, T = M.lll_transform()\n        >>> M_lll\n        [[2, 1, 0], [-1, 1, 3]]\n        >>> T\n        [[-2, 1], [3, -1]]\n        >>> T.matmul(M) == M_lll\n        True\n\n        See Also\n        ========\n\n        sympy.polys.matrices.domainmatrix.DomainMatrix.lll\n            Higher level interface to compute LLL-reduced basis.\n        lll\n            Compute LLL-reduced basis without transform matrix.\n        '
        if self.domain != ZZ:
            raise DMDomainError('ZZ expected, got %s' % self.domain)
        elif self.rows > self.cols:
            raise DMShapeError('Matrix must not have more rows than columns.')
        (rep, T) = self._lll(transform=True, delta=delta)
        basis = self._new_rep(rep)
        T_dfm = self._new(T, (self.rows, self.rows), self.domain)
        return (basis, T_dfm)
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.ddm import SDM