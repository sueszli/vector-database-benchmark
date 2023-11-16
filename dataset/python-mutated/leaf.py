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
from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from cvxpy import Constant, Parameter, Variable
    from cvxpy.atoms.atom import Atom
import numbers
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
import cvxpy.interface as intf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import expression
from cvxpy.settings import GENERAL_PROJECTION_TOL, PSD_NSD_PROJECTION_TOL, SPARSE_PROJECTION_TOL

class Leaf(expression.Expression):
    """
    A leaf node of an expression tree; i.e., a Variable, Constant, or Parameter.

    A leaf may carry *attributes* that constrain the set values permissible
    for it. Leafs can have no more than one attribute, with the exception
    that a leaf may be both ``nonpos`` and ``nonneg`` or both ``boolean``
    in some indices and ``integer`` in others.

    An error is raised if a leaf is assigned a value that contradicts
    one or more of its attributes. See the ``project`` method for a convenient
    way to project a value onto a leaf's domain.

    Parameters
    ----------
    shape : Iterable of ints or int
        The leaf dimensions. Either an integer n for a 1D shape, or an
        iterable where the semantics are the same as NumPy ndarray shapes.
        **Shapes cannot be more than 2D**.
    value : numeric type
        A value to assign to the leaf.
    nonneg : bool
        Is the variable constrained to be nonnegative?
    nonpos : bool
        Is the variable constrained to be nonpositive?
    complex : bool
        Is the variable complex valued?
    symmetric : bool
        Is the variable symmetric?
    diag : bool
        Is the variable diagonal?
    PSD : bool
        Is the variable constrained to be positive semidefinite?
    NSD : bool
        Is the variable constrained to be negative semidefinite?
    Hermitian : bool
        Is the variable Hermitian?
    boolean : bool or list of tuple
        Is the variable boolean? True, which constrains
        the entire Variable to be boolean, False, or a list of
        indices which should be constrained as boolean, where each
        index is a tuple of length exactly equal to the
        length of shape.
    integer : bool or list of tuple
        Is the variable integer? The semantics are the same as the
        boolean argument.
    sparsity : list of tuplewith
        Fixed sparsity pattern for the variable.
    pos : bool
        Is the variable positive?
    neg : bool
        Is the variable negative?
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, shape: int | Iterable[int, ...], value=None, nonneg: bool=False, nonpos: bool=False, complex: bool=False, imag: bool=False, symmetric: bool=False, diag: bool=False, PSD: bool=False, NSD: bool=False, hermitian: bool=False, boolean: bool=False, integer: bool=False, sparsity=None, pos: bool=False, neg: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(shape, numbers.Integral):
            shape = (int(shape),)
        elif len(shape) > 2:
            raise ValueError('Expressions of dimension greater than 2 are not supported.')
        for d in shape:
            if not isinstance(d, numbers.Integral) or d <= 0:
                raise ValueError('Invalid dimensions %s.' % (shape,))
        shape = tuple((np.int32(d) for d in shape))
        self._shape = shape
        if (PSD or NSD or symmetric or diag or hermitian) and (len(shape) != 2 or shape[0] != shape[1]):
            raise ValueError('Invalid dimensions %s. Must be a square matrix.' % (shape,))
        self.attributes = {'nonneg': nonneg, 'nonpos': nonpos, 'pos': pos, 'neg': neg, 'complex': complex, 'imag': imag, 'symmetric': symmetric, 'diag': diag, 'PSD': PSD, 'NSD': NSD, 'hermitian': hermitian, 'boolean': bool(boolean), 'integer': integer, 'sparsity': sparsity}
        if boolean:
            self.boolean_idx = boolean if not isinstance(boolean, bool) else list(np.ndindex(max(shape, (1,))))
        else:
            self.boolean_idx = []
        if integer:
            self.integer_idx = integer if not isinstance(integer, bool) else list(np.ndindex(max(shape, (1,))))
        else:
            self.integer_idx = []
        true_attr = sum((1 for (k, v) in self.attributes.items() if v))
        if boolean and integer:
            true_attr -= 1
        if true_attr > 1:
            raise ValueError('Cannot set more than one special attribute in %s.' % self.__class__.__name__)
        if value is not None:
            self.value = value
        self.args = []

    def _get_attr_str(self) -> str:
        if False:
            return 10
        'Get a string representing the attributes.\n        '
        attr_str = ''
        for (attr, val) in self.attributes.items():
            if attr != 'real' and val:
                attr_str += ', %s=%s' % (attr, val)
        return attr_str

    def copy(self, args=None, id_objects=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns a shallow copy of the object.\n\n        Used to reconstruct an object tree.\n\n        Parameters\n        ----------\n        args : list, optional\n            The arguments to reconstruct the object. If args=None, use the\n            current args of the object.\n\n        Returns\n        -------\n        Expression\n        '
        id_objects = {} if id_objects is None else id_objects
        if id(self) in id_objects:
            return id_objects[id(self)]
        return self

    def get_data(self) -> None:
        if False:
            print('Hello World!')
        'Leaves are not copied.\n        '

    @property
    def shape(self) -> tuple[int, ...]:
        if False:
            while True:
                i = 10
        ' tuple : The dimensions of the expression.\n        '
        return self._shape

    def variables(self) -> list[Variable]:
        if False:
            return 10
        'Default is empty list of Variables.\n        '
        return []

    def parameters(self) -> list[Parameter]:
        if False:
            i = 10
            return i + 15
        'Default is empty list of Parameters.\n        '
        return []

    def constants(self) -> list[Constant]:
        if False:
            return 10
        'Default is empty list of Constants.\n        '
        return []

    def is_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression convex?\n        '
        return True

    def is_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression concave?\n        '
        return True

    def is_log_log_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression log-log convex?\n        '
        return self.is_pos()

    def is_log_log_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression log-log concave?\n        '
        return self.is_pos()

    def is_nonneg(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression nonnegative?\n        '
        return self.attributes['nonneg'] or self.attributes['pos'] or self.attributes['boolean']

    def is_nonpos(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression nonpositive?\n        '
        return self.attributes['nonpos'] or self.attributes['neg']

    def is_pos(self) -> bool:
        if False:
            return 10
        'Is the expression positive?\n        '
        return self.attributes['pos']

    def is_neg(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression negative?\n        '
        return self.attributes['neg']

    def is_hermitian(self) -> bool:
        if False:
            return 10
        'Is the Leaf hermitian?\n        '
        return self.is_real() and self.is_symmetric() or self.attributes['hermitian'] or self.is_psd() or self.is_nsd()

    def is_symmetric(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the Leaf symmetric?\n        '
        return self.is_scalar() or any((self.attributes[key] for key in ['diag', 'symmetric', 'PSD', 'NSD']))

    def is_imag(self) -> bool:
        if False:
            return 10
        'Is the Leaf imaginary?\n        '
        return self.attributes['imag']

    def is_complex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the Leaf complex valued?\n        '
        return self.attributes['complex'] or self.is_imag() or self.attributes['hermitian']

    @property
    def domain(self) -> list[Constraint]:
        if False:
            for i in range(10):
                print('nop')
        'A list of constraints describing the closure of the region\n           where the expression is finite.\n        '
        domain = []
        if self.attributes['nonneg'] or self.attributes['pos']:
            domain.append(self >= 0)
        elif self.attributes['nonpos'] or self.attributes['neg']:
            domain.append(self <= 0)
        elif self.attributes['PSD']:
            domain.append(self >> 0)
        elif self.attributes['NSD']:
            domain.append(self << 0)
        return domain

    def project(self, val):
        if False:
            while True:
                i = 10
        'Project value onto the attribute set of the leaf.\n\n        A sensible idiom is ``leaf.value = leaf.project(val)``.\n\n        Parameters\n        ----------\n        val : numeric type\n            The value assigned.\n\n        Returns\n        -------\n        numeric type\n            The value rounded to the attribute type.\n        '
        if not self.is_complex():
            val = np.real(val)
        if self.attributes['nonpos'] and self.attributes['nonneg']:
            return 0 * val
        elif self.attributes['nonpos'] or self.attributes['neg']:
            return np.minimum(val, 0.0)
        elif self.attributes['nonneg'] or self.attributes['pos']:
            return np.maximum(val, 0.0)
        elif self.attributes['imag']:
            return np.imag(val) * 1j
        elif self.attributes['complex']:
            return val.astype(complex)
        elif self.attributes['boolean']:
            return np.round(np.clip(val, 0.0, 1.0))
        elif self.attributes['integer']:
            return np.round(val)
        elif self.attributes['diag']:
            if intf.is_sparse(val):
                val = val.diagonal()
            else:
                val = np.diag(val)
            return sp.diags([val], [0])
        elif self.attributes['hermitian']:
            return (val + np.conj(val).T) / 2.0
        elif any([self.attributes[key] for key in ['symmetric', 'PSD', 'NSD']]):
            if val.dtype.kind in 'ib':
                val = val.astype(float)
            val = val + val.T
            val /= 2.0
            if self.attributes['symmetric']:
                return val
            (w, V) = LA.eigh(val)
            if self.attributes['PSD']:
                bad = w < 0
                if not bad.any():
                    return val
                w[bad] = 0
            else:
                bad = w > 0
                if not bad.any():
                    return val
                w[bad] = 0
            return (V * w).dot(V.T)
        else:
            return val

    def save_value(self, val) -> None:
        if False:
            return 10
        self._value = val

    @property
    def value(self):
        if False:
            return 10
        'NumPy.ndarray or None: The numeric value of the parameter.\n        '
        return self._value

    @value.setter
    def value(self, val) -> None:
        if False:
            i = 10
            return i + 15
        self.save_value(self._validate_value(val))

    def project_and_assign(self, val) -> None:
        if False:
            print('Hello World!')
        'Project and assign a value to the variable.\n        '
        self.save_value(self.project(val))

    def _validate_value(self, val):
        if False:
            print('Hello World!')
        "Check that the value satisfies the leaf's symbolic attributes.\n\n        Parameters\n        ----------\n        val : numeric type\n            The value assigned.\n\n        Returns\n        -------\n        numeric type\n            The value converted to the proper matrix type.\n        "
        if val is not None:
            val = intf.convert(val)
            if intf.shape(val) != self.shape:
                raise ValueError('Invalid dimensions %s for %s value.' % (intf.shape(val), self.__class__.__name__))
            projection = self.project(val)
            delta = np.abs(val - projection)
            if intf.is_sparse(delta):
                close_enough = np.allclose(delta.data, 0, atol=SPARSE_PROJECTION_TOL)
            else:
                delta = np.array(delta)
                if self.attributes['PSD'] or self.attributes['NSD']:
                    close_enough = LA.norm(delta, ord=2) <= PSD_NSD_PROJECTION_TOL
                else:
                    close_enough = np.allclose(delta, 0, atol=GENERAL_PROJECTION_TOL)
            if not close_enough:
                if self.attributes['nonneg']:
                    attr_str = 'nonnegative'
                elif self.attributes['pos']:
                    attr_str = 'positive'
                elif self.attributes['nonpos']:
                    attr_str = 'nonpositive'
                elif self.attributes['neg']:
                    attr_str = 'negative'
                elif self.attributes['diag']:
                    attr_str = 'diagonal'
                elif self.attributes['PSD']:
                    attr_str = 'positive semidefinite'
                elif self.attributes['NSD']:
                    attr_str = 'negative semidefinite'
                elif self.attributes['imag']:
                    attr_str = 'imaginary'
                else:
                    attr_str = ([k for (k, v) in self.attributes.items() if v] + ['real'])[0]
                raise ValueError('%s value must be %s.' % (self.__class__.__name__, attr_str))
        return val

    def is_psd(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression a positive semidefinite matrix?\n        '
        return self.attributes['PSD']

    def is_nsd(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression a negative semidefinite matrix?\n        '
        return self.attributes['NSD']

    def is_diag(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression a diagonal matrix?\n        '
        return self.attributes['diag']

    def is_quadratic(self) -> bool:
        if False:
            while True:
                i = 10
        'Leaf nodes are always quadratic.\n        '
        return True

    def has_quadratic_term(self) -> bool:
        if False:
            return 10
        'Leaf nodes are not quadratic terms.\n        '
        return False

    def is_pwl(self) -> bool:
        if False:
            print('Hello World!')
        'Leaf nodes are always piecewise linear.\n        '
        return True

    def is_dpp(self, context: str='dcp') -> bool:
        if False:
            return 10
        'The expression is a disciplined parameterized expression.\n\n           context: dcp or dgp\n        '
        return True

    def atoms(self) -> list[Atom]:
        if False:
            i = 10
            return i + 15
        return []