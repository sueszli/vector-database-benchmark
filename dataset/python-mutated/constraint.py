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
import numpy as np
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.expressions import cvxtypes

class Constraint(u.Canonical):
    """The base class for constraints.

    A constraint is an equality, inequality, or more generally a generalized
    inequality that is imposed upon a mathematical expression or a list of
    thereof.

    Parameters
    ----------
    args : list
        A list of expression trees.
    constr_id : int
        A unique id for the constraint.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, constr_id=None) -> None:
        if False:
            i = 10
            return i + 15
        self.args = args
        if constr_id is None:
            self.constr_id = lu.get_id()
        else:
            self.constr_id = constr_id
        self._construct_dual_variables(args)
        super(Constraint, self).__init__()

    def __str__(self):
        if False:
            return 10
        'Returns a string showing the mathematical constraint.\n        '
        return self.name()

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        'Returns a string with information about the constraint.\n        '
        return '%s(%s)' % (self.__class__.__name__, repr(self.args[0]))

    def _construct_dual_variables(self, args) -> None:
        if False:
            i = 10
            return i + 15
        self.dual_variables = [cvxtypes.variable()(arg.shape) for arg in args]

    @property
    def shape(self):
        if False:
            return 10
        'int : The shape of the constrained expression.'
        return self.args[0].shape

    @property
    def size(self):
        if False:
            while True:
                i = 10
        'int : The size of the constrained expression.'
        return self.args[0].size

    def is_real(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the Leaf real valued?\n        '
        return not self.is_complex()

    def is_imag(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the Leaf imaginary?\n        '
        return all((arg.is_imag() for arg in self.args))

    def is_complex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the Leaf complex valued?\n        '
        return any((arg.is_complex() for arg in self.args))

    @abc.abstractmethod
    def is_dcp(self, dpp: bool=False) -> bool:
        if False:
            print('Hello World!')
        'Checks whether the constraint is DCP.\n\n        Returns\n        -------\n        bool\n            True if the constraint is DCP, False otherwise.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def is_dgp(self, dpp: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks whether the constraint is DGP.\n\n        Returns\n        -------\n        bool\n            True if the constraint is DGP, False otherwise.\n        '
        raise NotImplementedError()

    def is_dpp(self, context='dcp') -> bool:
        if False:
            for i in range(10):
                print('nop')
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    @abc.abstractproperty
    def residual(self):
        if False:
            return 10
        'The residual of the constraint.\n\n        Returns\n        -------\n        NumPy.ndarray\n            The residual, or None if the constrained expression does not have\n            a value.\n        '
        raise NotImplementedError()

    def violation(self):
        if False:
            i = 10
            return i + 15
        "The numeric residual of the constraint.\n\n        The violation is defined as the distance between the constrained\n        expression's value and its projection onto the domain of the\n        constraint:\n\n        .. math::\n\n            ||\\Pi(v) - v||_2^2\n\n        where :math:`v` is the value of the constrained expression and\n        :math:`\\Pi` is the projection operator onto the constraint's domain .\n\n        Returns\n        -------\n        NumPy.ndarray\n            The residual value.\n\n        Raises\n        ------\n        ValueError\n            If the constrained expression does not have a value associated\n            with it.\n        "
        residual = self.residual
        if residual is None:
            raise ValueError('Cannot compute the violation of an constraint whose expression is None-valued.')
        return residual

    def value(self, tolerance: float=1e-08):
        if False:
            return 10
        'Checks whether the constraint violation is less than a tolerance.\n\n        Parameters\n        ----------\n            tolerance : float\n                The absolute tolerance to impose on the violation.\n\n        Returns\n        -------\n            bool\n                True if the violation is less than ``tolerance``, False\n                otherwise.\n\n        Raises\n        ------\n            ValueError\n                If the constrained expression does not have a value associated\n                with it.\n        '
        residual = self.residual
        if residual is None:
            raise ValueError('Cannot compute the value of an constraint whose expression is None-valued.')
        return np.all(residual <= tolerance)

    @property
    def id(self):
        if False:
            i = 10
            return i + 15
        'Wrapper for compatibility with variables.\n        '
        return self.constr_id

    @id.setter
    def id(self, value):
        if False:
            i = 10
            return i + 15
        self.constr_id = value

    def get_data(self):
        if False:
            print('Hello World!')
        'Data needed to copy.\n        '
        return [self.id]

    def __nonzero__(self):
        if False:
            while True:
                i = 10
        'Raises an exception when called.\n\n        Python 2 version.\n\n        Called when evaluating the truth value of the constraint.\n        Raising an error here prevents writing chained constraints.\n        '
        return self._chain_constraints()

    def _chain_constraints(self):
        if False:
            while True:
                i = 10
        'Raises an error due to chained constraints.\n        '
        raise Exception('Cannot evaluate the truth value of a constraint or chain constraints, e.g., 1 >= x >= 0.')

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        'Raises an exception when called.\n\n        Python 3 version.\n\n        Called when evaluating the truth value of the constraint.\n        Raising an error here prevents writing chained constraints.\n        '
        return self._chain_constraints()

    @property
    def dual_value(self):
        if False:
            while True:
                i = 10
        'NumPy.ndarray : The value of the dual variable.\n        '
        dual_vals = [dv.value for dv in self.dual_variables]
        if len(dual_vals) == 1:
            return dual_vals[0]
        else:
            return dual_vals

    def save_dual_value(self, value) -> None:
        if False:
            return 10
        "Save the value of the dual variable for the constraint's parent.\n        Args:\n            value: The value of the dual variable.\n        "
        self.dual_variables[0].save_value(value)