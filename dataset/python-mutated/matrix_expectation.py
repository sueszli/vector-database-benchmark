"""MatrixExpectation Class"""
from typing import Union
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.list_ops import ComposedOp, ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.utils.deprecation import deprecate_func

class MatrixExpectation(ExpectationBase):
    """An Expectation converter which converts Operator measurements to
    be matrix-based so they can be evaluated by matrix multiplication."""

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            return 10
        'Accept an Operator and return a new Operator with the Pauli measurements replaced by\n        Matrix based measurements.\n\n        Args:\n            operator: The operator to convert.\n\n        Returns:\n            The converted operator.\n        '
        if isinstance(operator, OperatorStateFn) and operator.is_measurement:
            return operator.to_matrix_op()
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert)
        else:
            return operator

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute the variance of the expectation estimator. Because this expectation\n        works by matrix multiplication, the estimation is exact and the variance is\n        always 0, but we need to return those values in a way which matches the Operator's\n        structure.\n\n        Args:\n            exp_op: The full expectation value Operator.\n\n        Returns:\n             The variances or lists thereof (if exp_op contains ListOps) of the expectation value\n             estimation, equal to 0.\n        "

        def sum_variance(operator):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(operator, ComposedOp):
                return 0.0
            elif isinstance(operator, ListOp):
                return operator.combo_fn([sum_variance(op) for op in operator.oplist])
            else:
                return 0.0
        return sum_variance(exp_op)