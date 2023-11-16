class spmatrix:
    """This class provides a base class for all sparse matrix classes.

    It cannot be instantiated.  Most of the work is provided by subclasses.
    """

    @property
    def _bsr_container(self):
        if False:
            i = 10
            return i + 15
        from ._bsr import bsr_matrix
        return bsr_matrix

    @property
    def _coo_container(self):
        if False:
            i = 10
            return i + 15
        from ._coo import coo_matrix
        return coo_matrix

    @property
    def _csc_container(self):
        if False:
            while True:
                i = 10
        from ._csc import csc_matrix
        return csc_matrix

    @property
    def _csr_container(self):
        if False:
            return 10
        from ._csr import csr_matrix
        return csr_matrix

    @property
    def _dia_container(self):
        if False:
            i = 10
            return i + 15
        from ._dia import dia_matrix
        return dia_matrix

    @property
    def _dok_container(self):
        if False:
            return 10
        from ._dok import dok_matrix
        return dok_matrix

    @property
    def _lil_container(self):
        if False:
            return 10
        from ._lil import lil_matrix
        return lil_matrix

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        return self._mul_dispatch(other)

    def __rmul__(self, other):
        if False:
            i = 10
            return i + 15
        return self._rmul_dispatch(other)

    def __pow__(self, power):
        if False:
            i = 10
            return i + 15
        from .linalg import matrix_power
        return matrix_power(self, power)

    def set_shape(self, shape):
        if False:
            while True:
                i = 10
        'Set the shape of the matrix in-place'
        new_self = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_self.__dict__

    def get_shape(self):
        if False:
            print('Hello World!')
        'Get the shape of the matrix'
        return self._shape
    shape = property(fget=get_shape, fset=set_shape, doc='Shape of the matrix')

    def asfptype(self):
        if False:
            print('Hello World!')
        'Upcast matrix to a floating point format (if necessary)'
        return self._asfptype()

    def getmaxprint(self):
        if False:
            print('Hello World!')
        'Maximum number of elements to display when printed.'
        return self._getmaxprint()

    def getformat(self):
        if False:
            for i in range(10):
                print('nop')
        'Matrix storage format'
        return self.format

    def getnnz(self, axis=None):
        if False:
            print('Hello World!')
        'Number of stored values, including explicit zeros.\n\n        Parameters\n        ----------\n        axis : None, 0, or 1\n            Select between the number of values across the whole array, in\n            each column, or in each row.\n        '
        return self._getnnz(axis=axis)

    def getH(self):
        if False:
            print('Hello World!')
        "Return the Hermitian transpose of this matrix.\n\n        See Also\n        --------\n        numpy.matrix.getH : NumPy's implementation of `getH` for matrices\n        "
        return self.conjugate().transpose()

    def getcol(self, j):
        if False:
            return 10
        'Returns a copy of column j of the matrix, as an (m x 1) sparse\n        matrix (column vector).\n        '
        return self._getcol(j)

    def getrow(self, i):
        if False:
            print('Hello World!')
        'Returns a copy of row i of the matrix, as a (1 x n) sparse\n        matrix (row vector).\n        '
        return self._getrow(i)