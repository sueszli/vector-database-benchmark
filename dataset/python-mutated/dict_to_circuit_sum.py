"""DictToCircuitSum Class"""
from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.dict_state_fn import DictStateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.utils.deprecation import deprecate_func

class DictToCircuitSum(ConverterBase):
    """
    Deprecated: Converts ``DictStateFns`` or ``VectorStateFns`` to equivalent ``CircuitStateFns``
    or sums thereof. The behavior of this class can be mostly replicated by calling ``to_circuit_op``
    on an Operator, but with the added control of choosing whether to convert only ``DictStateFns``
    or ``VectorStateFns``, rather than both.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, traverse: bool=True, convert_dicts: bool=True, convert_vectors: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            traverse: Whether to recurse down into Operators with internal sub-operators for\n                conversion.\n            convert_dicts: Whether to convert VectorStateFn.\n            convert_vectors: Whether to convert DictStateFns.\n        '
        super().__init__()
        self._traverse = traverse
        self._convert_dicts = convert_dicts
        self._convert_vectors = convert_vectors

    def convert(self, operator: OperatorBase) -> OperatorBase:
        if False:
            print('Hello World!')
        'Convert the Operator to ``CircuitStateFns``, recursively if ``traverse`` is True.\n\n        Args:\n            operator: The Operator to convert\n\n        Returns:\n            The converted Operator.\n        '
        if isinstance(operator, DictStateFn) and self._convert_dicts:
            return CircuitStateFn.from_dict(operator.primitive)
        if isinstance(operator, VectorStateFn) and self._convert_vectors:
            return CircuitStateFn.from_vector(operator.to_matrix(massive=True))
        elif isinstance(operator, ListOp) and 'Dict' in operator.primitive_strings():
            return operator.traverse(self.convert)
        else:
            return operator