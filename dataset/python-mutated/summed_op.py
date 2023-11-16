"""SummedOp Class"""
from typing import List, Union, cast, Dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils.deprecation import deprecate_func

class SummedOp(ListOp):
    """Deprecated: A class for lazily representing sums of Operators. Often Operators cannot be
    efficiently added to one another, but may be manipulated further so that they can be
    later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be added together, and therefore if they reach a point in which they can be, such as after
    evaluation or conversion to matrices, they can be reduced by addition."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, oplist: List[OperatorBase], coeff: Union[complex, ParameterExpression]=1.0, abelian: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            oplist: The Operators being summed.\n            coeff: A coefficient multiplying the operator\n            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.\n        '
        super().__init__(oplist, combo_fn=lambda x: np.sum(x, axis=0), coeff=coeff, abelian=abelian)

    @property
    def num_qubits(self) -> int:
        if False:
            while True:
                i = 10
        return self.oplist[0].num_qubits

    @property
    def distributive(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    @property
    def settings(self) -> Dict:
        if False:
            print('Hello World!')
        'Return settings.'
        return {'oplist': self._oplist, 'coeff': self._coeff, 'abelian': self._abelian}

    def add(self, other: OperatorBase) -> 'SummedOp':
        if False:
            i = 10
            return i + 15
        "Return Operator addition of ``self`` and ``other``, overloaded by ``+``.\n\n        Note:\n            This appends ``other`` to ``self.oplist`` without checking ``other`` is already\n            included or not. If you want to simplify them, please use :meth:`simplify`.\n\n        Args:\n            other: An ``OperatorBase`` with the same number of qubits as self, and in the same\n                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type\n                of underlying function).\n\n        Returns:\n            A ``SummedOp`` equivalent to the sum of self and other.\n        "
        self_new_ops = self.oplist if self.coeff == 1 else [op.mul(self.coeff) for op in self.oplist]
        if isinstance(other, SummedOp):
            other_new_ops = other.oplist if other.coeff == 1 else [op.mul(other.coeff) for op in other.oplist]
        else:
            other_new_ops = [other]
        return SummedOp(self_new_ops + other_new_ops)

    def collapse_summands(self) -> 'SummedOp':
        if False:
            print('Hello World!')
        'Return Operator by simplifying duplicate operators.\n\n        E.g., ``SummedOp([2 * X ^ Y, X ^ Y]).collapse_summands() -> SummedOp([3 * X ^ Y])``.\n\n        Returns:\n            A simplified ``SummedOp`` equivalent to self.\n        '
        from ..primitive_ops.primitive_op import PrimitiveOp
        oplist = []
        coeffs = []
        for op in self.oplist:
            if isinstance(op, PrimitiveOp):
                new_op = PrimitiveOp(op.primitive)
                new_coeff = op.coeff * self.coeff
                if new_op in oplist:
                    index = oplist.index(new_op)
                    coeffs[index] += new_coeff
                else:
                    oplist.append(new_op)
                    coeffs.append(new_coeff)
            elif op in oplist:
                index = oplist.index(op)
                coeffs[index] += self.coeff
            else:
                oplist.append(op)
                coeffs.append(self.coeff)
        return SummedOp([op * coeff for (op, coeff) in zip(oplist, coeffs)])

    def reduce(self) -> OperatorBase:
        if False:
            i = 10
            return i + 15
        'Try collapsing list or trees of sums.\n\n        Tries to sum up duplicate operators and reduces the operators\n        in the sum.\n\n        Returns:\n            A collapsed version of self, if possible.\n        '
        if len(self.oplist) == 0:
            return SummedOp([], coeff=self.coeff, abelian=self.abelian)
        reduced_ops = sum((op.reduce() for op in self.oplist)) * self.coeff
        if isinstance(reduced_ops, SummedOp):
            reduced_ops = reduced_ops.collapse_summands()
        from ..primitive_ops.pauli_sum_op import PauliSumOp
        if isinstance(reduced_ops, PauliSumOp):
            reduced_ops = reduced_ops.reduce()
        if isinstance(reduced_ops, SummedOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)

    def to_circuit(self) -> QuantumCircuit:
        if False:
            i = 10
            return i + 15
        'Returns the quantum circuit, representing the SummedOp. In the first step,\n        the SummedOp is converted to MatrixOp. This is straightforward for most operators,\n        but it is not supported for operators containing parameterized PrimitiveOps (in that case,\n        OpflowError is raised). In the next step, the MatrixOp representation of SummedOp is\n        converted to circuit. In most cases, if the summands themselves are unitary operators,\n        the SummedOp itself is non-unitary and can not be converted to circuit. In that case,\n        ExtensionError is raised in the underlying modules.\n\n        Returns:\n            The circuit representation of the summed operator.\n\n        Raises:\n            OpflowError: if SummedOp can not be converted to MatrixOp (e.g. SummedOp is composed of\n            parameterized PrimitiveOps).\n        '
        from ..primitive_ops.matrix_op import MatrixOp
        matrix_op = self.to_matrix_op()
        if isinstance(matrix_op, MatrixOp):
            return matrix_op.to_circuit()
        raise OpflowError('The SummedOp can not be converted to circuit, because to_matrix_op did not return a MatrixOp.')

    def to_matrix_op(self, massive: bool=False) -> 'SummedOp':
        if False:
            i = 10
            return i + 15
        'Returns an equivalent Operator composed of only NumPy-based primitives, such as\n        ``MatrixOp`` and ``VectorStateFn``.'
        accum = self.oplist[0].to_matrix_op(massive=massive)
        for i in range(1, len(self.oplist)):
            accum += self.oplist[i].to_matrix_op(massive=massive)
        return cast(SummedOp, accum * self.coeff)

    def to_pauli_op(self, massive: bool=False) -> 'SummedOp':
        if False:
            print('Hello World!')
        from ..state_fns.state_fn import StateFn
        pauli_sum = SummedOp([op.to_pauli_op(massive=massive) if not isinstance(op, StateFn) else op for op in self.oplist], coeff=self.coeff, abelian=self.abelian).reduce()
        if isinstance(pauli_sum, SummedOp):
            return pauli_sum
        return pauli_sum.to_pauli_op()

    def equals(self, other: OperatorBase) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if other is equal to self.\n\n        Note:\n            This is not a mathematical check for equality.\n            If ``self`` and ``other`` implement the same operation but differ\n            in the representation (e.g. different type of summands)\n            ``equals`` will evaluate to ``False``.\n\n        Args:\n            other: The other operator to check for equality.\n\n        Returns:\n            True, if other and self are equal, otherwise False.\n\n        Examples:\n            >>> from qiskit.opflow import X, Z\n            >>> 2 * X == X + X\n            True\n            >>> X + Z == Z + X\n            True\n        '
        (self_reduced, other_reduced) = (self.reduce(), other.reduce())
        if not isinstance(other_reduced, type(self_reduced)):
            return False
        if not isinstance(self_reduced, SummedOp):
            return self_reduced == other_reduced
        self_reduced = cast(SummedOp, self_reduced)
        other_reduced = cast(SummedOp, other_reduced)
        if len(self_reduced.oplist) != len(other_reduced.oplist):
            return False
        if self_reduced.coeff != 1:
            self_reduced = SummedOp([op * self_reduced.coeff for op in self_reduced.oplist])
        if other_reduced.coeff != 1:
            other_reduced = SummedOp([op * other_reduced.coeff for op in other_reduced.oplist])
        return all((any((i == j for j in other_reduced)) for i in self_reduced))