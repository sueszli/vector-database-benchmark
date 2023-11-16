"""The module for Quantum the Fisher Information."""
from typing import List, Union, Optional
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit._utils import sort_parameters
from qiskit.utils.deprecation import deprecate_func
from ..list_ops.list_op import ListOp
from ..expectations.pauli_expectation import PauliExpectation
from ..state_fns.circuit_state_fn import CircuitStateFn
from .qfi_base import QFIBase
from .circuit_qfis import CircuitQFI

class QFI(QFIBase):
    """Deprecated: Compute the Quantum Fisher Information (QFI).

    Computes the QFI given a pure, parameterized quantum state, where QFI is:

    .. math::

        \\mathrm{QFI}_{kl}= 4 \\mathrm{Re}[\\langle \\partial_k \\psi | \\partial_l \\psi \\rangle
            − \\langle\\partial_k \\psi | \\psi \\rangle \\langle\\psi | \\partial_l \\psi \\rangle].

    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, qfi_method: Union[str, CircuitQFI]='lin_comb_full'):
        if False:
            i = 10
            return i + 15
        super().__init__(qfi_method=qfi_method)

    def convert(self, operator: CircuitStateFn, params: Optional[Union[ParameterExpression, ParameterVector, List[ParameterExpression]]]=None) -> ListOp:
        if False:
            while True:
                i = 10
        '\n        Args:\n            operator: The operator corresponding to the quantum state \\|ψ(ω)〉for which we compute\n                the QFI\n            params: The parameters we are computing the QFI wrt: ω\n                If not explicitly passed, they are inferred from the operator and sorted by name.\n\n        Returns:\n            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl\n\n        Raises:\n            ValueError: If operator is not parameterized.\n        '
        if len(operator.parameters) == 0:
            raise ValueError('The operator we are taking the gradient of is not parameterized!')
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        if params is None:
            params = sort_parameters(operator.parameters)
        return self.qfi_method.convert(cleaned_op, params)