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
import abc
from typing import TYPE_CHECKING, List, Tuple
if TYPE_CHECKING:
    from cvxpy.constraints.constraint import Constraint
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import interface as intf
from cvxpy import utilities as u
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.utilities import performance_utils as perf
from cvxpy.utilities.deterministic import unique_list

class Atom(Expression):
    """ Abstract base class for atoms. """
    __metaclass__ = abc.ABCMeta
    _allow_complex = False

    def __init__(self, *args) -> None:
        if False:
            i = 10
            return i + 15
        self.id = lu.get_id()
        if len(args) == 0:
            raise TypeError('No arguments given to %s.' % self.__class__.__name__)
        self.args = [Atom.cast_to_const(arg) for arg in args]
        self.validate_arguments()
        self._shape = self.shape_from_args()
        if len(self._shape) > 2:
            raise ValueError('Atoms must be at most 2D.')

    def name(self) -> str:
        if False:
            print('Hello World!')
        'Returns the string representation of the function call.\n        '
        if self.get_data() is None:
            data = []
        else:
            data = [str(elem) for elem in self.get_data()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join([arg.name() for arg in self.args] + data))

    def validate_arguments(self) -> None:
        if False:
            return 10
        'Raises an error if the arguments are invalid.\n        '
        if not self._allow_complex and any((arg.is_complex() for arg in self.args)):
            raise ValueError('Arguments to %s cannot be complex.' % self.__class__.__name__)

    @abc.abstractmethod
    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the shape of the expression.\n        '
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        return self._shape

    @abc.abstractmethod
    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Returns sign (is positive, is negative) of the expression.\n        '
        raise NotImplementedError()

    @perf.compute_once
    def is_nonneg(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression nonnegative?\n        '
        return self.sign_from_args()[0]

    @perf.compute_once
    def is_nonpos(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression nonpositive?\n        '
        return self.sign_from_args()[1]

    @perf.compute_once
    def is_imag(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression imaginary?\n        '
        return False

    @perf.compute_once
    def is_complex(self) -> bool:
        if False:
            return 10
        'Is the expression complex valued?\n        '
        return False

    @abc.abstractmethod
    def is_atom_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom convex?\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def is_atom_concave(self) -> bool:
        if False:
            return 10
        'Is the atom concave?\n        '
        raise NotImplementedError()

    def is_atom_affine(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom affine?\n        '
        return self.is_atom_concave() and self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom log-log convex?\n        '
        return False

    def is_atom_log_log_concave(self) -> bool:
        if False:
            return 10
        'Is the atom log-log concave?\n        '
        return False

    def is_atom_quasiconvex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom quasiconvex?\n        '
        return self.is_atom_convex()

    def is_atom_quasiconcave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom quasiconcave?\n        '
        return self.is_atom_concave()

    def is_atom_log_log_affine(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom log-log affine?\n        '
        return self.is_atom_log_log_concave() and self.is_atom_log_log_convex()

    @abc.abstractmethod
    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        raise NotImplementedError()

    @perf.compute_once
    def is_convex(self) -> bool:
        if False:
            return 10
        'Is the expression convex?\n        '
        if self.is_constant():
            return True
        elif self.is_atom_convex():
            for (idx, arg) in enumerate(self.args):
                if not (arg.is_affine() or (arg.is_convex() and self.is_incr(idx)) or (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def is_concave(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression concave?\n        '
        if self.is_constant():
            return True
        elif self.is_atom_concave():
            for (idx, arg) in enumerate(self.args):
                if not (arg.is_affine() or (arg.is_concave() and self.is_incr(idx)) or (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    def is_dpp(self, context='dcp') -> bool:
        if False:
            i = 10
            return i + 15
        'The expression is a disciplined parameterized expression.\n        '
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    @perf.compute_once
    def is_log_log_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression log-log convex?\n        '
        if self.is_log_log_constant():
            return True
        elif self.is_atom_log_log_convex():
            for (idx, arg) in enumerate(self.args):
                if not (arg.is_log_log_affine() or (arg.is_log_log_convex() and self.is_incr(idx)) or (arg.is_log_log_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def is_log_log_concave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression log-log concave?\n        '
        if self.is_log_log_constant():
            return True
        elif self.is_atom_log_log_concave():
            for (idx, arg) in enumerate(self.args):
                if not (arg.is_log_log_affine() or (arg.is_log_log_concave() and self.is_incr(idx)) or (arg.is_log_log_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def _non_const_idx(self) -> List[int]:
        if False:
            print('Hello World!')
        return [i for (i, arg) in enumerate(self.args) if not arg.is_constant()]

    @perf.compute_once
    def _is_real(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        non_const = self._non_const_idx()
        return self.is_scalar() and len(non_const) == 1 and self.args[non_const[0]].is_scalar()

    @perf.compute_once
    def is_quasiconvex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression quaisconvex?\n        '
        from cvxpy.atoms.max import max as max_atom
        if self.is_convex():
            return True
        if type(self) in (cvxtypes.maximum(), max_atom):
            return all((arg.is_quasiconvex() for arg in self.args))
        non_const = self._non_const_idx()
        if self._is_real() and self.is_incr(non_const[0]):
            return self.args[non_const[0]].is_quasiconvex()
        if self._is_real() and self.is_decr(non_const[0]):
            return self.args[non_const[0]].is_quasiconcave()
        if self.is_atom_quasiconvex():
            for (idx, arg) in enumerate(self.args):
                if not (arg.is_affine() or (arg.is_convex() and self.is_incr(idx)) or (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        return False

    @perf.compute_once
    def is_quasiconcave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression quasiconcave?\n        '
        from cvxpy.atoms.min import min as min_atom
        if self.is_concave():
            return True
        if type(self) in (cvxtypes.minimum(), min_atom):
            return all((arg.is_quasiconcave() for arg in self.args))
        non_const = self._non_const_idx()
        if self._is_real() and self.is_incr(non_const[0]):
            return self.args[non_const[0]].is_quasiconcave()
        if self._is_real() and self.is_decr(non_const[0]):
            return self.args[non_const[0]].is_quasiconvex()
        if self.is_atom_quasiconcave():
            for (idx, arg) in enumerate(self.args):
                if not (arg.is_affine() or (arg.is_concave() and self.is_incr(idx)) or (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        return False

    def canonicalize(self):
        if False:
            return 10
        'Represent the atom as an affine objective and conic constraints.\n        '
        if self.is_constant() and (not self.parameters()):
            return Constant(self.value).canonical_form
        else:
            arg_objs = []
            constraints = []
            for arg in self.args:
                (obj, constr) = arg.canonical_form
                arg_objs.append(obj)
                constraints += constr
            data = self.get_data()
            (graph_obj, graph_constr) = self.graph_implementation(arg_objs, self.shape, data)
            return (graph_obj, constraints + graph_constr)

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List['Constraint']]:
        if False:
            i = 10
            return i + 15
        'Reduces the atom to an affine expression and list of constraints.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        raise NotImplementedError()

    @property
    def value(self):
        if False:
            print('Hello World!')
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()

    def _value_impl(self):
        if False:
            while True:
                i = 10
        if 0 in self.shape:
            result = np.array([])
        else:
            arg_values = []
            for arg in self.args:
                arg_val = arg._value_impl()
                if arg_val is None and (not self.is_constant()):
                    return None
                else:
                    arg_values.append(arg_val)
            result = self.numeric(arg_values)
        return result

    @property
    def grad(self):
        if False:
            for i in range(10):
                print('nop')
        'Gives the (sub/super)gradient of the expression w.r.t. each variable.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n        None indicates variable values unknown or outside domain.\n\n        Returns:\n            A map of variable to SciPy CSC sparse matrix or None.\n        '
        if self.is_constant():
            return u.grad.constant_grad(self)
        arg_values = []
        for arg in self.args:
            if arg.value is None:
                return u.grad.error_grad(self)
            else:
                arg_values.append(arg.value)
        grad_self = self._grad(arg_values)
        result = {}
        for (idx, arg) in enumerate(self.args):
            grad_arg = arg.grad
            for key in grad_arg:
                if grad_arg[key] is None or grad_self[idx] is None:
                    result[key] = None
                else:
                    D = grad_arg[key] * grad_self[idx]
                    if not np.isscalar(D) and D.shape == (1, 1):
                        D = D[0, 0]
                    if key in result:
                        result[key] += D
                    else:
                        result[key] = D
        return result

    @abc.abstractmethod
    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        raise NotImplementedError()

    @property
    def domain(self) -> List['Constraint']:
        if False:
            print('Hello World!')
        'A list of constraints describing the closure of the region\n           where the expression is finite.\n        '
        return self._domain() + [con for arg in self.args for con in arg.domain]

    def _domain(self) -> List['Constraint']:
        if False:
            i = 10
            return i + 15
        'Returns constraints describing the domain of the atom.\n        '
        return []

    @staticmethod
    def numpy_numeric(numeric_func):
        if False:
            i = 10
            return i + 15
        "Wraps an atom's numeric function that requires numpy ndarrays as input.\n           Ensures both inputs and outputs are the correct matrix types.\n        "

        def new_numeric(self, values):
            if False:
                for i in range(10):
                    print('nop')
            interface = intf.DEFAULT_INTF
            values = [interface.const_to_matrix(v, convert_scalars=True) for v in values]
            result = numeric_func(self, values)
            return intf.DEFAULT_INTF.const_to_matrix(result)
        return new_numeric

    def atoms(self) -> List['Atom']:
        if False:
            return 10
        "A list of the atom types present amongst this atom's arguments.\n        "
        atom_list = []
        for arg in self.args:
            atom_list += arg.atoms()
        return unique_list(atom_list + [type(self)])