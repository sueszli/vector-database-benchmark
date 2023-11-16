"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
import cvxpy.utilities.linalg as eig_util
from cvxpy.expressions.leaf import Leaf
from cvxpy.utilities import performance_utils as perf

class Constant(Leaf):
    """
    A constant value.

    Raw numerical constants (Python primite types, NumPy ndarrays,
    and NumPy matrices) are implicitly cast to constants via Expression
    operator overloading. For example, if ``x`` is an expression and
    ``c`` is a raw constant, then ``x + c`` creates an expression by
    casting ``c`` to a Constant.
    """

    def __init__(self, value) -> None:
        if False:
            for i in range(10):
                print('nop')
        if intf.is_sparse(value):
            self._value = intf.DEFAULT_SPARSE_INTF.const_to_matrix(value, convert_scalars=True)
            self._sparse = True
        else:
            self._value = intf.DEFAULT_INTF.const_to_matrix(value)
            self._sparse = False
        self._imag: Optional[bool] = None
        self._nonneg: Optional[bool] = None
        self._nonpos: Optional[bool] = None
        self._symm: Optional[bool] = None
        self._herm: Optional[bool] = None
        self._psd_test: Optional[bool] = None
        self._nsd_test: Optional[bool] = None
        self._cached_is_pos = None
        self._skew_symm = None
        super(Constant, self).__init__(intf.shape(self.value))

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The value as a string.\n        '
        if len(self.shape) == 2 and '\n' in str(self.value):
            return np.array2string(self.value, edgeitems=s.PRINT_EDGEITEMS, threshold=s.PRINT_THRESHOLD, formatter={'float': lambda x: f'{x:.2f}'})
        return str(self.value)

    def constants(self) -> List['Constant']:
        if False:
            print('Hello World!')
        'Returns self as a constant.\n        '
        return [self]

    def is_constant(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @property
    def value(self):
        if False:
            return 10
        'NumPy.ndarray or None: The numeric value of the constant.\n        '
        return self._value

    def is_pos(self) -> bool:
        if False:
            return 10
        'Returns whether the constant is elementwise positive.\n        '
        if self._cached_is_pos is None:
            if sp.issparse(self._value):
                self._cached_is_pos = False
            else:
                self._cached_is_pos = np.all(self._value > 0)
        return self._cached_is_pos

    @property
    def grad(self):
        if False:
            return 10
        'Gives the (sub/super)gradient of the expression w.r.t. each variable.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Returns:\n            A map of variable to SciPy CSC sparse matrix or None.\n        '
        return {}

    @property
    def shape(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        'Returns the (row, col) dimensions of the expression.\n        '
        return self._shape

    def canonicalize(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the graph implementation of the object.\n\n        Returns:\n            A tuple of (affine expression, [constraints]).\n        '
        obj = lu.create_const(self.value, self.shape, self._sparse)
        return (obj, [])

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        'Returns a string with information about the expression.\n        '
        return 'Constant(%s, %s, %s)' % (self.curvature, self.sign, self.shape)

    def is_nonneg(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression nonnegative?\n        '
        if self._nonneg is None:
            self._compute_attr()
        return self._nonneg

    def is_nonpos(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression nonpositive?\n        '
        if self._nonpos is None:
            self._compute_attr()
        return self._nonpos

    def is_imag(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the Leaf imaginary?\n        '
        if self._imag is None:
            self._compute_attr()
        return self._imag

    @perf.compute_once
    def is_complex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the Leaf complex valued?\n        '
        return np.iscomplexobj(self.value)

    @perf.compute_once
    def is_symmetric(self) -> bool:
        if False:
            return 10
        'Is the expression symmetric?\n        '
        if self.is_scalar():
            return True
        elif self.ndim == 2 and self.shape[0] == self.shape[1]:
            if self._symm is None:
                self._compute_symm_attr()
            return self._symm
        else:
            return False

    @perf.compute_once
    def is_hermitian(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression a Hermitian matrix?\n        '
        if self.is_scalar() and self.is_real():
            return True
        elif self.ndim == 2 and self.shape[0] == self.shape[1]:
            if self._herm is None:
                self._compute_symm_attr()
            return self._herm
        else:
            return False

    def _compute_attr(self) -> None:
        if False:
            print('Hello World!')
        'Compute the attributes of the constant related to complex/real, sign.\n        '
        (is_real, is_imag) = intf.is_complex(self.value)
        if self.is_complex():
            is_nonneg = is_nonpos = False
        else:
            (is_nonneg, is_nonpos) = intf.sign(self.value)
        self._imag = is_imag and (not is_real)
        self._nonpos = is_nonpos
        self._nonneg = is_nonneg

    def _compute_symm_attr(self) -> None:
        if False:
            i = 10
            return i + 15
        'Determine whether the constant is symmetric/Hermitian.\n        '
        (is_symm, is_herm) = intf.is_hermitian(self.value)
        self._symm = is_symm
        self._herm = is_herm

    def is_skew_symmetric(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self._skew_symm is None:
            self._skew_symm = intf.is_skew_symmetric(self.value)
        return self._skew_symm

    @perf.compute_once
    def is_psd(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression a positive semidefinite matrix?\n        '
        if self.is_scalar() and self.is_nonneg():
            return True
        elif self.is_scalar():
            return False
        elif self.ndim == 1:
            return False
        elif self.ndim == 2 and self.shape[0] != self.shape[1]:
            return False
        elif not self.is_hermitian():
            return False
        if self._psd_test is None:
            self._psd_test = eig_util.is_psd_within_tol(self.value, s.EIGVAL_TOL)
        return self._psd_test

    @perf.compute_once
    def is_nsd(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression a negative semidefinite matrix?\n        '
        if self.is_scalar() and self.is_nonpos():
            return True
        elif self.is_scalar():
            return False
        elif self.ndim == 1:
            return False
        elif self.ndim == 2 and self.shape[0] != self.shape[1]:
            return False
        elif not self.is_hermitian():
            return False
        if self._nsd_test is None:
            self._nsd_test = eig_util.is_psd_within_tol(-self.value, s.EIGVAL_TOL)
        return self._nsd_test