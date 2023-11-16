"""ExpectationFactory Class"""
import logging
from typing import Optional, Union
from qiskit import BasicAer
from qiskit.opflow.expectations.aer_pauli_expectation import AerPauliExpectation
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow.expectations.matrix_expectation import MatrixExpectation
from qiskit.opflow.expectations.pauli_expectation import PauliExpectation
from qiskit.opflow.operator_base import OperatorBase
from qiskit.providers import Backend
from qiskit.utils.backend_utils import is_aer_qasm, is_statevector_backend
from qiskit.utils import QuantumInstance, optionals
from qiskit.utils.deprecation import deprecate_func
logger = logging.getLogger(__name__)

class ExpectationFactory:
    """Deprecated:  factory class for convenient automatic selection of an Expectation based on the
    Operator to be converted and backend used to sample the expectation value.
    """

    @staticmethod
    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def build(operator: OperatorBase, backend: Optional[Union[Backend, QuantumInstance]]=None, include_custom: bool=True) -> ExpectationBase:
        if False:
            return 10
        '\n        A factory method for convenient automatic selection of an Expectation based on the\n        Operator to be converted and backend used to sample the expectation value.\n\n        Args:\n            operator: The Operator whose expectation value will be taken.\n            backend: The backend which will be used to sample the expectation value.\n            include_custom: Whether the factory will include the (Aer) specific custom\n                expectations if their behavior against the backend might not be as expected.\n                For instance when using Aer qasm_simulator with paulis the Aer snapshot can\n                be used but the outcome lacks shot noise and hence does not intuitively behave\n                overall as people might expect when choosing a qasm_simulator. It is however\n                fast as long as the more state vector like behavior is acceptable.\n\n        Returns:\n            The expectation algorithm which best fits the Operator and backend.\n\n        Raises:\n            ValueError: If operator is not of a composition for which we know the best Expectation\n                method.\n        '
        backend_to_check = backend.backend if isinstance(backend, QuantumInstance) else backend
        primitives = operator.primitive_strings()
        if primitives in ({'Pauli'}, {'SparsePauliOp'}):
            if backend_to_check is None:
                if optionals.HAS_AER:
                    from qiskit_aer import AerSimulator
                    backend_to_check = AerSimulator()
                elif operator.num_qubits <= 16:
                    backend_to_check = BasicAer.get_backend('statevector_simulator')
                else:
                    logger.warning("%d qubits is a very large expectation value. Consider installing Aer to use Aer's fast expectation, which will perform better here. We'll use the BasicAer qasm backend for this expectation to avoid having to construct the %dx%d operator matrix.", operator.num_qubits, 2 ** operator.num_qubits, 2 ** operator.num_qubits)
                    backend_to_check = BasicAer.get_backend('qasm_simulator')
            if is_aer_qasm(backend_to_check) and include_custom:
                return AerPauliExpectation()
            elif is_statevector_backend(backend_to_check):
                if operator.num_qubits >= 16:
                    logger.warning("Note: Using a statevector_simulator with %d qubits can be very expensive. Consider using the Aer qasm_simulator instead to take advantage of Aer's built-in fast Pauli Expectation", operator.num_qubits)
                return MatrixExpectation()
            else:
                return PauliExpectation()
        elif primitives == {'Matrix'}:
            return MatrixExpectation()
        else:
            raise ValueError('Expectations of Mixed Operators not yet supported.')