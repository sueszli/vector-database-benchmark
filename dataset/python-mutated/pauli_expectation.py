"""PauliExpectation Class"""
import logging
from typing import Union
import numpy as np
from qiskit.opflow.converters.abelian_grouper import AbelianGrouper
from qiskit.opflow.converters.pauli_basis_change import PauliBasisChange
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.utils.deprecation import deprecate_func
logger = logging.getLogger(__name__)

class PauliExpectation(ExpectationBase):
    """
    An Expectation converter for Pauli-basis observables by changing Pauli measurements to a
    diagonal ({Z, I}^n) basis and appending circuit post-rotations to the measured state function.
    Optionally groups the Paulis with the same post-rotations (those that commute with one
    another, or form Abelian groups) into single measurements to reduce circuit execution
    overhead.

    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, group_paulis: bool=True) -> None:
        if False:
            return 10
        '\n        Args:\n            group_paulis: Whether to group the Pauli measurements into commuting sums, which all\n                have the same diagonalizing circuit.\n\n        '
        super().__init__()
        self._grouper = AbelianGrouper() if group_paulis else None

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Accepts an Operator and returns a new Operator with the Pauli measurements replaced by\n        diagonal Pauli post-rotation based measurements so they can be evaluated by sampling and\n        averaging.\n\n        Args:\n            operator: The operator to convert.\n\n        Returns:\n            The converted operator.\n        '
        if isinstance(operator, ListOp):
            return operator.traverse(self.convert).reduce()
        if isinstance(operator, OperatorStateFn) and operator.is_measurement:
            if isinstance(operator.primitive, (ListOp, PrimitiveOp)) and (not isinstance(operator.primitive, PauliSumOp)) and ({'Pauli', 'SparsePauliOp'} < operator.primitive_strings()):
                logger.warning('Measured Observable is not composed of only Paulis, converting to Pauli representation, which can be expensive.')
                pauli_obsv = operator.primitive.to_pauli_op(massive=False)
                operator = StateFn(pauli_obsv, is_measurement=True, coeff=operator.coeff)
            if self._grouper and isinstance(operator.primitive, (ListOp, PauliSumOp)):
                grouped = self._grouper.convert(operator.primitive)
                operator = StateFn(grouped, is_measurement=True, coeff=operator.coeff)
            cob = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
            return cob.convert(operator).reduce()
        return operator

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')

        def sum_variance(operator):
            if False:
                while True:
                    i = 10
            if isinstance(operator, ComposedOp):
                sfdict = operator.oplist[1]
                measurement = operator.oplist[0]
                average = np.asarray(measurement.eval(sfdict))
                variance = sum(((v * (np.asarray(measurement.eval(b)) - average)) ** 2 for (b, v) in sfdict.primitive.items()))
                return operator.coeff * variance
            elif isinstance(operator, ListOp):
                return operator.combo_fn([sum_variance(op) for op in operator.oplist])
            return 0.0
        return sum_variance(exp_op)