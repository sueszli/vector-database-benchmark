"""AerPauliExpectation Class"""
import logging
from functools import reduce
from operator import add
from typing import Union
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.list_ops.composed_op import ComposedOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils.deprecation import deprecate_func
logger = logging.getLogger(__name__)

class AerPauliExpectation(ExpectationBase):
    """An Expectation converter for using Aer's operator snapshot to
    take expectations of quantum state circuits over Pauli observables.

    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        'Accept an Operator and return a new Operator with the Pauli measurements replaced by\n        AerSnapshot-based expectation circuits.\n\n        Args:\n            operator: The operator to convert. If it contains non-hermitian terms, the\n                operator is decomposed into hermitian and anti-hermitian parts.\n\n        Returns:\n            The converted operator.\n        '
        if isinstance(operator, OperatorStateFn) and operator.is_measurement:
            if isinstance(operator.primitive, ListOp):
                is_herm = all((op.is_hermitian() for op in operator.primitive.oplist))
            else:
                is_herm = operator.primitive.is_hermitian()
            if not is_herm:
                pauli_sum_re = self._replace_pauli_sums(1 / 2 * (operator.primitive + operator.primitive.adjoint()).reduce()) * operator.coeff
                pauli_sum_im = self._replace_pauli_sums(1 / 2j * (operator.primitive - operator.primitive.adjoint()).reduce()) * operator.coeff
                pauli_sum = (pauli_sum_re + 1j * pauli_sum_im).reduce()
            else:
                pauli_sum = self._replace_pauli_sums(operator.primitive) * operator.coeff
            return pauli_sum
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert)
        else:
            return operator

    @classmethod
    def _replace_pauli_sums(cls, operator):
        if False:
            for i in range(10):
                print('nop')
        try:
            from qiskit.providers.aer.library import SaveExpectationValue
        except ImportError as ex:
            raise MissingOptionalLibraryError(libname='qiskit-aer', name='AerPauliExpectation', pip_install='pip install qiskit-aer') from ex
        if operator.__class__ == ListOp:
            return operator.traverse(cls._replace_pauli_sums)
        if isinstance(operator, PauliSumOp):
            save_instruction = SaveExpectationValue(operator.primitive, 'expval_measurement')
            return CircuitStateFn(save_instruction, coeff=operator.coeff, is_measurement=True, from_operator=True)
        if {'Pauli'} != operator.primitive_strings():
            logger.warning('Measured Observable is not composed of only Paulis, converting to Pauli representation, which can be expensive.')
            operator = operator.to_pauli_op(massive=False)
        if isinstance(operator, SummedOp):
            sparse_pauli = reduce(add, (meas.coeff * SparsePauliOp(meas.primitive) for meas in operator.oplist))
            save_instruction = SaveExpectationValue(sparse_pauli, 'expval_measurement')
            return CircuitStateFn(save_instruction, coeff=operator.coeff, is_measurement=True, from_operator=True)
        if isinstance(operator, PauliOp):
            sparse_pauli = operator.coeff * SparsePauliOp(operator.primitive)
            save_instruction = SaveExpectationValue(sparse_pauli, 'expval_measurement')
            return CircuitStateFn(save_instruction, is_measurement=True, from_operator=True)
        raise TypeError(f'Conversion of OperatorStateFn of {operator.__class__.__name__} is not defined.')

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute the variance of the expectation estimator. Because Aer takes this expectation\n        with matrix multiplication, the estimation is exact and the variance is always 0,\n        but we need to return those values in a way which matches the Operator's structure.\n\n        Args:\n            exp_op: The full expectation value Operator after sampling.\n\n        Returns:\n             The variances or lists thereof (if exp_op contains ListOps) of the expectation value\n             estimation, equal to 0.\n        "

        def sum_variance(operator):
            if False:
                return 10
            if isinstance(operator, ComposedOp):
                return 0.0
            elif isinstance(operator, ListOp):
                return operator.combo_fn([sum_variance(op) for op in operator.oplist])
            raise TypeError(f'Variance cannot be computed for {operator.__class__.__name__}.')
        return sum_variance(exp_op)