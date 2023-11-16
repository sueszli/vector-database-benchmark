"""ComposedOp Class"""
from functools import partial, reduce
from typing import List, Optional, Union, cast, Dict
from numbers import Number
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func

class ComposedOp(ListOp):
    """Deprecated: A class for lazily representing compositions of Operators. Often Operators cannot be
    efficiently composed with one another, but may be manipulated further so that they can be
    composed later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be composed, and therefore if they reach a point in which they can be, such as after
    conversion to QuantumCircuits or matrices, they can be reduced by composition."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, oplist: List[OperatorBase], coeff: Union[complex, ParameterExpression]=1.0, abelian: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            oplist: The Operators being composed.\n            coeff: A coefficient multiplying the operator\n            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.\n        '
        super().__init__(oplist, combo_fn=partial(reduce, np.dot), coeff=coeff, abelian=abelian)

    @property
    def num_qubits(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.oplist[0].num_qubits

    @property
    def distributive(self) -> bool:
        if False:
            return 10
        return False

    @property
    def settings(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Return settings.'
        return {'oplist': self._oplist, 'coeff': self._coeff, 'abelian': self._abelian}

    def to_matrix(self, massive: bool=False) -> np.ndarray:
        if False:
            return 10
        OperatorBase._check_massive('to_matrix', True, self.num_qubits, massive)
        mat = self.coeff * reduce(np.dot, [np.asarray(op.to_matrix(massive=massive)) for op in self.oplist])
        if isinstance(mat, Number):
            mat = [mat]
        return np.asarray(mat, dtype=complex)

    def to_circuit(self) -> QuantumCircuit:
        if False:
            return 10
        'Returns the quantum circuit, representing the composed operator.\n\n        Returns:\n            The circuit representation of the composed operator.\n\n        Raises:\n            OpflowError: for operators where a single underlying circuit can not be obtained.\n        '
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..primitive_ops.primitive_op import PrimitiveOp
        circuit_op = self.to_circuit_op()
        if isinstance(circuit_op, (PrimitiveOp, CircuitStateFn)):
            return circuit_op.to_circuit()
        raise OpflowError('Conversion to_circuit supported only for operators, where a single underlying circuit can be produced.')

    def adjoint(self) -> 'ComposedOp':
        if False:
            i = 10
            return i + 15
        return ComposedOp([op.adjoint() for op in reversed(self.oplist)], coeff=self.coeff)

    def compose(self, other: OperatorBase, permutation: Optional[List[int]]=None, front: bool=False) -> OperatorBase:
        if False:
            print('Hello World!')
        (new_self, other) = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(ComposedOp, new_self)
        if front:
            return other.compose(new_self)
        if isinstance(other, ComposedOp):
            return ComposedOp(new_self.oplist + other.oplist, coeff=new_self.coeff * other.coeff)
        if not isinstance(new_self.oplist[-1], ComposedOp):
            comp_with_last = new_self.oplist[-1].compose(other)
            if not isinstance(comp_with_last, ComposedOp):
                new_oplist = new_self.oplist[0:-1] + [comp_with_last]
                return ComposedOp(new_oplist, coeff=new_self.coeff)
        return ComposedOp(new_self.oplist + [other], coeff=new_self.coeff)

    def eval(self, front: Optional[Union[str, dict, np.ndarray, OperatorBase, Statevector]]=None) -> Union[OperatorBase, complex]:
        if False:
            print('Hello World!')
        if self._is_empty():
            return 0.0
        from ..state_fns.state_fn import StateFn

        def tree_recursive_eval(r, l_arg):
            if False:
                while True:
                    i = 10
            if isinstance(r, list):
                return [tree_recursive_eval(r_op, l_arg) for r_op in r]
            else:
                return l_arg.eval(r)
        eval_list = self.oplist.copy()
        eval_list[0] = eval_list[0] * self.coeff
        if front and isinstance(front, OperatorBase):
            eval_list = eval_list + [front]
        elif front:
            eval_list = [StateFn(front, is_measurement=True)] + eval_list
        return reduce(tree_recursive_eval, reversed(eval_list))

    def non_distributive_reduce(self) -> OperatorBase:
        if False:
            print('Hello World!')
        'Reduce without attempting to expand all distributive compositions.\n\n        Returns:\n            The reduced Operator.\n        '
        reduced_ops = [op.reduce() for op in self.oplist]
        reduced_ops = reduce(lambda x, y: x.compose(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, ComposedOp) and len(reduced_ops.oplist) > 1:
            return reduced_ops
        else:
            return reduced_ops[0]

    def reduce(self) -> OperatorBase:
        if False:
            for i in range(10):
                print('nop')
        reduced_ops = [op.reduce() for op in self.oplist]
        if len(reduced_ops) == 0:
            return self.__class__([], coeff=self.coeff, abelian=self.abelian)

        def distribute_compose(l_arg, r):
            if False:
                while True:
                    i = 10
            if isinstance(l_arg, ListOp) and l_arg.distributive:
                return l_arg.__class__([distribute_compose(l_op * l_arg.coeff, r) for l_op in l_arg.oplist])
            if isinstance(r, ListOp) and r.distributive:
                return r.__class__([distribute_compose(l_arg, r_op * r.coeff) for r_op in r.oplist])
            else:
                return l_arg.compose(r)
        reduced_ops = reduce(distribute_compose, reduced_ops) * self.coeff
        if isinstance(reduced_ops, ListOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)