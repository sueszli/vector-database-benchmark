"""The CVaR (Conditional Value at Risk) expectation class."""
from typing import Optional, Union
from qiskit.opflow.expectations.aer_pauli_expectation import AerPauliExpectation
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.expectations.pauli_expectation import PauliExpectation
from qiskit.opflow.list_ops import ComposedOp, ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns import CVaRMeasurement, OperatorStateFn
from qiskit.utils.deprecation import deprecate_func

class CVaRExpectation(ExpectationBase):
    """Deprecated: Compute the Conditional Value at Risk (CVaR) expectation value.

    The standard approach to calculating the expectation value of a Hamiltonian w.r.t. a
    state is to take the sample mean of the measurement outcomes. This corresponds to an estimator
    of the energy. However in several problem settings with a diagonal Hamiltonian, e.g.
    in combinatorial optimization where the Hamiltonian encodes a cost function, we are not
    interested in calculating the energy but in the lowest possible value we can find.

    To this end, we might consider using the best observed sample as a cost function during
    variational optimization. The issue here, is that this can result in a non-smooth optimization
    surface. To resolve this issue, we can smooth the optimization surface by using not just the
    best observed sample, but instead average over some fraction of best observed samples.
    This is exactly what the CVaR estimator accomplishes [1].

    It is empirically shown, that this can lead to faster convergence for combinatorial
    optimization problems.

    Let :math:`\\alpha` be a real number in :math:`[0,1]` which specifies the fraction of best
    observed samples which are used to compute the objective function. Observe that if
    :math:`\\alpha = 1`, CVaR is equivalent to a standard expectation value. Similarly,
    if :math:`\\alpha = 0`, then CVaR corresponds to using the best observed sample.
    Intermediate values of :math:`\\alpha` interpolate between these two objective functions.

    References:

        [1]: Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., and Woerner, S.,
             "Improving Variational Quantum Optimization using CVaR"
             `arXiv:1907.04769 <https://arxiv.org/abs/1907.04769>`_

    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, alpha: float, expectation: Optional[ExpectationBase]=None) -> None:
        if False:
            return 10
        '\n        Args:\n            alpha: The alpha value describing the quantile considered in the expectation value.\n            expectation: An expectation object to compute the expectation value. Defaults\n                to the PauliExpectation calculation.\n\n        Raises:\n            NotImplementedError: If the ``expectation`` is an AerPauliExpecation.\n        '
        super().__init__()
        self.alpha = alpha
        if isinstance(expectation, AerPauliExpectation):
            raise NotImplementedError('AerPauliExpecation currently not supported.')
        if expectation is None:
            expectation = PauliExpectation()
        self.expectation = expectation

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            while True:
                i = 10
        'Return an expression that computes the CVaR expectation upon calling ``eval``.\n        Args:\n            operator: The operator to convert.\n\n        Returns:\n            The converted operator.\n        '
        expectation = self.expectation.convert(operator)

        def replace_with_cvar(operator):
            if False:
                i = 10
                return i + 15
            if isinstance(operator, OperatorStateFn) and operator.is_measurement:
                return CVaRMeasurement(operator.primitive, alpha=self.alpha)
            elif isinstance(operator, ListOp):
                return operator.traverse(replace_with_cvar)
            return operator
        return replace_with_cvar(expectation)

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        if False:
            i = 10
            return i + 15
        'Returns the variance of the CVaR calculation\n\n        Args:\n            exp_op: The operator whose evaluation yields an expectation\n                of some StateFn against a diagonal observable.\n\n        Returns:\n            The variance of the CVaR estimate corresponding to the converted\n                exp_op.\n        Raises:\n            ValueError: If the exp_op does not correspond to an expectation value.\n        '

        def cvar_variance(operator):
            if False:
                i = 10
                return i + 15
            if isinstance(operator, ComposedOp):
                sfdict = operator.oplist[1]
                measurement = operator.oplist[0]
                return measurement.eval_variance(sfdict)
            elif isinstance(operator, ListOp):
                return operator.combo_fn([cvar_variance(op) for op in operator.oplist])
            raise ValueError('Input operator does not correspond to a value expectation value.')
        cvar_op = self.convert(exp_op)
        return cvar_variance(cvar_op)