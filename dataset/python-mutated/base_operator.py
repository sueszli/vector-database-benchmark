"""
Abstract BaseOperator class.
"""
from __future__ import annotations
import copy
from abc import ABC
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from .mixins import GroupMixin

class BaseOperator(GroupMixin, ABC):
    """Abstract operator base class."""

    def __init__(self, input_dims: tuple | int | None=None, output_dims: tuple | int | None=None, num_qubits: int | None=None, shape: tuple | None=None, op_shape: OpShape | None=None):
        if False:
            return 10
        'Initialize a BaseOperator shape\n\n        Args:\n            input_dims (tuple or int or None): Optional, input dimensions.\n            output_dims (tuple or int or None): Optional, output dimensions.\n            num_qubits (int): Optional, the number of qubits of the operator.\n            shape (tuple): Optional, matrix shape for automatically determining\n                           qubit dimensions.\n            op_shape (OpShape): Optional, an OpShape object for operator dimensions.\n\n        .. note::\n\n            If `op_shape`` is specified it will take precedence over other\n            kwargs.\n        '
        self._qargs = None
        if op_shape:
            self._op_shape = op_shape
        else:
            self._op_shape = OpShape.auto(shape=shape, dims_l=output_dims, dims_r=input_dims, num_qubits=num_qubits)
    __array_priority__ = 20

    def __call__(self, *qargs):
        if False:
            i = 10
            return i + 15
        'Return a shallow copy with qargs attribute set'
        if len(qargs) == 1 and isinstance(qargs[0], (tuple, list)):
            qargs = qargs[0]
        n_qargs = len(qargs)
        if n_qargs not in self._op_shape.num_qargs:
            raise QiskitError(f'qargs does not match the number of operator qargs ({n_qargs} not in {self._op_shape.num_qargs})')
        ret = copy.copy(self)
        ret._qargs = tuple(qargs)
        return ret

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, type(self)) and self._op_shape == other._op_shape

    @property
    def qargs(self):
        if False:
            print('Hello World!')
        'Return the qargs for the operator.'
        return self._qargs

    @property
    def dim(self):
        if False:
            for i in range(10):
                print('nop')
        'Return tuple (input_shape, output_shape).'
        return (self._op_shape._dim_r, self._op_shape._dim_l)

    @property
    def num_qubits(self):
        if False:
            i = 10
            return i + 15
        'Return the number of qubits if a N-qubit operator or None otherwise.'
        return self._op_shape.num_qubits

    @property
    def _input_dim(self):
        if False:
            while True:
                i = 10
        'Return the total input dimension.'
        return self._op_shape._dim_r

    @property
    def _output_dim(self):
        if False:
            i = 10
            return i + 15
        'Return the total input dimension.'
        return self._op_shape._dim_l

    def reshape(self, input_dims: None | tuple | int=None, output_dims: None | tuple | int=None, num_qubits: None | int=None) -> BaseOperator:
        if False:
            return 10
        'Return a shallow copy with reshaped input and output subsystem dimensions.\n\n        Args:\n            input_dims (None or tuple): new subsystem input dimensions.\n                If None the original input dims will be preserved [Default: None].\n            output_dims (None or tuple): new subsystem output dimensions.\n                If None the original output dims will be preserved [Default: None].\n            num_qubits (None or int): reshape to an N-qubit operator [Default: None].\n\n        Returns:\n            BaseOperator: returns self with reshaped input and output dimensions.\n\n        Raises:\n            QiskitError: if combined size of all subsystem input dimension or\n                         subsystem output dimensions is not constant.\n        '
        new_shape = OpShape.auto(dims_l=output_dims, dims_r=input_dims, num_qubits=num_qubits, shape=self._op_shape.shape)
        ret = copy.copy(self)
        ret._op_shape = new_shape
        return ret

    def input_dims(self, qargs=None):
        if False:
            while True:
                i = 10
        'Return tuple of input dimension for specified subsystems.'
        return self._op_shape.dims_r(qargs)

    def output_dims(self, qargs=None):
        if False:
            for i in range(10):
                print('nop')
        'Return tuple of output dimension for specified subsystems.'
        return self._op_shape.dims_l(qargs)

    def copy(self):
        if False:
            i = 10
            return i + 15
        'Make a deep copy of current operator.'
        return copy.deepcopy(self)