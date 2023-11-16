"""Base transpiler passes."""
from __future__ import annotations
import abc
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable
from inspect import signature
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.base_tasks import GenericPass, PassManagerIR
from qiskit.passmanager.compilation_status import PropertySet, RunState, PassManagerState
from .exceptions import TranspilerError
from .layout import TranspileLayout

class MetaPass(abc.ABCMeta):
    """Metaclass for transpiler passes.

    Enforces the creation of some fields in the pass while allowing passes to
    override ``__init__``.
    """

    def __call__(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass_instance = type.__call__(cls, *args, **kwargs)
        pass_instance._hash = hash(MetaPass._freeze_init_parameters(cls, args, kwargs))
        return pass_instance

    @staticmethod
    def _freeze_init_parameters(class_, args, kwargs):
        if False:
            return 10
        self_guard = object()
        init_signature = signature(class_.__init__)
        bound_signature = init_signature.bind(self_guard, *args, **kwargs)
        arguments = [('class_.__name__', class_.__name__)]
        for (name, value) in bound_signature.arguments.items():
            if value == self_guard:
                continue
            if isinstance(value, Hashable):
                arguments.append((name, type(value), value))
            else:
                arguments.append((name, type(value), repr(value)))
        return frozenset(arguments)

class BasePass(GenericPass, metaclass=MetaPass):
    """Base class for transpiler passes."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.preserves: Iterable[GenericPass] = []
        self._hash = hash(None)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._hash

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return hash(self) == hash(other)

    @abstractmethod
    def run(self, dag: DAGCircuit):
        if False:
            for i in range(10):
                print('nop')
        'Run a pass on the DAGCircuit. This is implemented by the pass developer.\n\n        Args:\n            dag: the dag on which the pass is run.\n\n        Raises:\n            NotImplementedError: when this is left unimplemented for a pass.\n        '
        raise NotImplementedError

    @property
    def is_transformation_pass(self):
        if False:
            print('Hello World!')
        'Check if the pass is a transformation pass.\n\n        If the pass is a TransformationPass, that means that the pass can manipulate the DAG,\n        but cannot modify the property set (but it can be read).\n        '
        return isinstance(self, TransformationPass)

    @property
    def is_analysis_pass(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the pass is an analysis pass.\n\n        If the pass is an AnalysisPass, that means that the pass can analyze the DAG and write\n        the results of that analysis in the property set. Modifications on the DAG are not allowed\n        by this kind of pass.\n        '
        return isinstance(self, AnalysisPass)

    def __call__(self, circuit: QuantumCircuit, property_set: PropertySet | dict | None=None) -> QuantumCircuit:
        if False:
            return 10
        'Runs the pass on circuit.\n\n        Args:\n            circuit: The dag on which the pass is run.\n            property_set: Input/output property set. An analysis pass\n                might change the property set in-place.\n\n        Returns:\n            If on transformation pass, the resulting QuantumCircuit.\n            If analysis pass, the input circuit.\n        '
        property_set_ = None
        if isinstance(property_set, dict):
            property_set_ = PropertySet(property_set)
        if isinstance(property_set_, PropertySet):
            self.property_set = property_set_
        result = self.run(circuit_to_dag(circuit))
        result_circuit = circuit
        if isinstance(property_set, dict):
            property_set.clear()
            property_set.update(self.property_set)
        if isinstance(result, DAGCircuit):
            result_circuit = dag_to_circuit(result, copy_operations=False)
        elif result is None:
            result_circuit = circuit.copy()
        if self.property_set['layout']:
            result_circuit._layout = TranspileLayout(initial_layout=self.property_set['layout'], input_qubit_mapping=self.property_set['original_qubit_indices'], final_layout=self.property_set['final_layout'], _input_qubit_count=len(circuit.qubits), _output_qubit_list=result_circuit.qubits)
        if self.property_set['clbit_write_latency'] is not None:
            result_circuit._clbit_write_latency = self.property_set['clbit_write_latency']
        if self.property_set['conditional_latency'] is not None:
            result_circuit._conditional_latency = self.property_set['conditional_latency']
        if self.property_set['node_start_time']:
            topological_start_times = []
            start_times = self.property_set['node_start_time']
            for dag_node in result.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            result_circuit._op_start_times = topological_start_times
        return result_circuit

class AnalysisPass(BasePass):
    """An analysis pass: change property set, not DAG."""

class TransformationPass(BasePass):
    """A transformation pass: change DAG, not property set."""

    def execute(self, passmanager_ir: PassManagerIR, state: PassManagerState, callback: Callable=None) -> tuple[PassManagerIR, PassManagerState]:
        if False:
            print('Hello World!')
        (new_dag, state) = super().execute(passmanager_ir=passmanager_ir, state=state, callback=callback)
        if state.workflow_status.previous_run == RunState.SUCCESS:
            if isinstance(new_dag, DAGCircuit):
                new_dag.calibrations = passmanager_ir.calibrations
            else:
                raise TranspilerError(f'Transformation passes should return a transformed dag.The pass {self.__class__.__name__} is returning a {type(new_dag)}')
        return (new_dag, state)

    def update_status(self, state: PassManagerState, run_state: RunState) -> PassManagerState:
        if False:
            while True:
                i = 10
        state = super().update_status(state, run_state)
        if run_state == RunState.SUCCESS:
            state.workflow_status.completed_passes.intersection_update(set(self.preserves))
        return state