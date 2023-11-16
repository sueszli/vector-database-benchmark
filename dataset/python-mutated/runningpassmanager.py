"""RunningPassManager class for the transpiler.
This object holds the state of a pass manager during running-time."""
from __future__ import annotations
import inspect
import logging
from functools import wraps
from typing import Callable
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.passmanager.compilation_status import PropertySet, WorkflowStatus, PassManagerState
from qiskit.passmanager.base_tasks import Task
from qiskit.passmanager.exceptions import PassManagerError
from qiskit.utils.deprecation import deprecate_func
from qiskit.passmanager.flow_controllers import BaseController, FlowController, FlowControllerLinear, ConditionalController, DoWhileController
from .exceptions import TranspilerError
from .layout import TranspileLayout
logger = logging.getLogger(__name__)

class RunningPassManager(FlowControllerLinear):
    """A RunningPassManager is a running pass manager.

    .. warning::

        :class:`.RunningPassManager` will be deprecated in the future release.
        As of Qiskit Terra 0.25 this class becomes a subclass of the flow controller
        with extra methods for backward compatibility.
        Relying on a subclass of the running pass manager might break your code stack.
    """

    @deprecate_func(since='0.45.0', additional_msg='Building the pipline of the tasks is responsibility of PassManager. RunningPassManager should not modify prepared pipeline at running time.')
    def append(self, passes: Task | list[Task], **flow_controller_conditions):
        if False:
            print('Hello World!')
        'Append a passes to the schedule of passes.\n\n        Args:\n            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of\n                passes that are controlled by the same flow controller. If a single pass is\n                provided, the pass set will only have that pass a single element.\n                It is also possible to append a :class:`.BaseFlowController` instance and\n                the rest of the parameter will be ignored.\n            flow_controller_conditions: Dictionary of control flow plugins.\n                Following built-in controllers are available by default:\n\n                * do_while: The passes repeat until the callable returns False.\n                * condition: The passes run only if the callable returns True.\n        '
        if not isinstance(passes, BaseController):
            normalized_controller = passes
        else:
            normalized_controller = FlowController.controller_factory(passes=passes, options=self._options, **flow_controller_conditions)
        super().append(normalized_controller)

    @deprecate_func(since='0.45.0', additional_msg='Now RunningPassManager is a subclass of flow controller.', pending=True)
    def run(self, circuit: QuantumCircuit, output_name: str=None, callback: Callable=None) -> QuantumCircuit:
        if False:
            print('Hello World!')
        'Run all the passes on a QuantumCircuit\n\n        Args:\n            circuit: Circuit to transform via all the registered passes.\n            output_name: The output circuit name. If not given, the same as the input circuit.\n            callback: A callback function that will be called after each pass execution.\n\n        Returns:\n            QuantumCircuit: Transformed circuit.\n        '
        initial_status = WorkflowStatus()
        property_set = PropertySet()
        state = PassManagerState(workflow_status=initial_status, property_set=property_set)
        passmanager_ir = circuit_to_dag(circuit)
        (passmanager_ir, state) = super().execute(passmanager_ir=passmanager_ir, state=state, callback=callback)
        out_circuit = dag_to_circuit(passmanager_ir, copy_operations=False)
        out_circuit.name = output_name
        if state.property_set['layout'] is not None:
            circuit._layout = TranspileLayout(initial_layout=state.property_set['layout'], input_qubit_mapping=state.property_set['original_qubit_indices'], final_layout=state.property_set['final_layout'])
        circuit._clbit_write_latency = state.property_set['clbit_write_latency']
        circuit._conditional_latency = state.property_set['conditional_latency']
        if state.property_set['node_start_time']:
            topological_start_times = []
            start_times = state.property_set['node_start_time']
            for dag_node in passmanager_ir.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            circuit._op_start_times = topological_start_times
        return circuit

def _replace_error(meth):
    if False:
        return 10

    @wraps(meth)
    def wrapper(*meth_args, **meth_kwargs):
        if False:
            print('Hello World!')
        try:
            return meth(*meth_args, **meth_kwargs)
        except PassManagerError as ex:
            raise TranspilerError(ex.message) from ex
    return wrapper
for (_name, _method) in inspect.getmembers(RunningPassManager, predicate=inspect.isfunction):
    if _name.startswith('_'):
        continue
    _wrapped = _replace_error(_method)
    setattr(RunningPassManager, _name, _wrapped)