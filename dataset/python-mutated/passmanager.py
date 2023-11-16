"""Manager for a set of Passes and their scheduling during transpilation."""
from __future__ import annotations
import inspect
import io
import re
import warnings
from collections.abc import Iterator, Iterable, Callable
from functools import wraps
from typing import Union, List, Any
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.passmanager import BasePassManager
from qiskit.passmanager.base_tasks import Task, BaseController
from qiskit.passmanager.flow_controllers import FlowController
from qiskit.passmanager.exceptions import PassManagerError
from qiskit.utils.deprecation import deprecate_arg
from .basepasses import BasePass
from .exceptions import TranspilerError
from .layout import TranspileLayout
from .runningpassmanager import RunningPassManager
_CircuitsT = Union[List[QuantumCircuit], QuantumCircuit]

class PassManager(BasePassManager):
    """Manager for a set of Passes and their scheduling during transpilation."""

    def __init__(self, passes: Task | list[Task]=(), max_iteration: int=1000):
        if False:
            while True:
                i = 10
        'Initialize an empty pass manager object.\n\n        Args:\n            passes: A pass set to be added to the pass manager schedule.\n            max_iteration: The maximum number of iterations the schedule will be looped if the\n                condition is not met.\n        '
        self._pass_sets = []
        super().__init__(tasks=passes, max_iteration=max_iteration)

    def _passmanager_frontend(self, input_program: QuantumCircuit, **kwargs) -> DAGCircuit:
        if False:
            while True:
                i = 10
        return circuit_to_dag(input_program, copy_operations=True)

    def _passmanager_backend(self, passmanager_ir: DAGCircuit, in_program: QuantumCircuit, **kwargs) -> QuantumCircuit:
        if False:
            return 10
        out_program = dag_to_circuit(passmanager_ir, copy_operations=False)
        out_name = kwargs.get('output_name', None)
        if out_name is not None:
            out_program.name = out_name
        if self.property_set['layout'] is not None:
            out_program._layout = TranspileLayout(initial_layout=self.property_set['layout'], input_qubit_mapping=self.property_set['original_qubit_indices'], final_layout=self.property_set['final_layout'], _input_qubit_count=len(in_program.qubits), _output_qubit_list=out_program.qubits)
        out_program._clbit_write_latency = self.property_set['clbit_write_latency']
        out_program._conditional_latency = self.property_set['conditional_latency']
        if self.property_set['node_start_time']:
            topological_start_times = []
            start_times = self.property_set['node_start_time']
            for dag_node in passmanager_ir.topological_op_nodes():
                topological_start_times.append(start_times[dag_node])
            out_program._op_start_times = topological_start_times
        return out_program

    @deprecate_arg(name='max_iteration', since='0.25', additional_msg="'max_iteration' can be set in the constructor.", pending=True, package_name='qiskit-terra')
    def append(self, passes: Task | list[Task], max_iteration: int=None, **flow_controller_conditions: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Append a Pass Set to the schedule of passes.\n\n        Args:\n            passes: A set of passes (a pass set) to be added to schedule. A pass set is a list of\n                passes that are controlled by the same flow controller. If a single pass is\n                provided, the pass set will only have that pass a single element.\n                It is also possible to append a :class:`.BaseFlowController` instance and\n                the rest of the parameter will be ignored.\n            max_iteration: max number of iterations of passes.\n            flow_controller_conditions: Dictionary of control flow plugins.\n                Following built-in controllers are available by default:\n\n                * do_while: The passes repeat until the callable returns False.  Corresponds to\n                  :class:`.DoWhileController`.\n                * condition: The passes run only if the callable returns True.  Corresponds to\n                  :class:`.ConditionalController`.\n\n                In general, you have more control simply by creating the controller you want and\n                passing it to :meth:`append`.\n\n        Raises:\n            TranspilerError: if a pass in passes is not a proper pass.\n        '
        if max_iteration:
            self.max_iteration = max_iteration
        if isinstance(passes, Task):
            passes = [passes]
        self._pass_sets.append({'passes': passes, 'flow_controllers': flow_controller_conditions})
        if flow_controller_conditions:
            passes = _legacy_build_flow_controller(passes, options={'max_iteration': self.max_iteration}, **flow_controller_conditions)
        super().append(passes)

    @deprecate_arg(name='max_iteration', since='0.25', additional_msg="'max_iteration' can be set in the constructor.", pending=True, package_name='qiskit-terra')
    def replace(self, index: int, passes: Task | list[Task], max_iteration: int=None, **flow_controller_conditions: Any) -> None:
        if False:
            while True:
                i = 10
        'Replace a particular pass in the scheduler.\n\n        Args:\n            index: Pass index to replace, based on the position in passes().\n            passes: A pass set to be added to the pass manager schedule.\n            max_iteration: max number of iterations of passes.\n            flow_controller_conditions: Dictionary of control flow plugins.\n                See :meth:`qiskit.transpiler.PassManager.append` for details.\n        '
        if max_iteration:
            self.max_iteration = max_iteration
        if isinstance(passes, Task):
            passes = [passes]
        try:
            self._pass_sets[index] = {'passes': passes, 'flow_controllers': flow_controller_conditions}
        except IndexError as ex:
            raise PassManagerError(f'Index to replace {index} does not exists') from ex
        if flow_controller_conditions:
            passes = _legacy_build_flow_controller(passes, options={'max_iteration': self.max_iteration}, **flow_controller_conditions)
        super().replace(index, passes)

    def remove(self, index: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().remove(index)
        del self._pass_sets[index]

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        new_passmanager = super().__getitem__(index)
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __add__(self, other):
        if False:
            while True:
                i = 10
        new_passmanager = super().__add__(other)
        if isinstance(other, self.__class__):
            new_passmanager._pass_sets = self._pass_sets
            new_passmanager._pass_sets += other._pass_sets
        return new_passmanager

    def to_flow_controller(self) -> RunningPassManager:
        if False:
            return 10
        flatten_tasks = list(self._flatten_tasks(self._tasks))
        return RunningPassManager(flatten_tasks)

    def run(self, circuits: _CircuitsT, output_name: str | None=None, callback: Callable=None) -> _CircuitsT:
        if False:
            while True:
                i = 10
        "Run all the passes on the specified ``circuits``.\n\n        Args:\n            circuits: Circuit(s) to transform via all the registered passes.\n            output_name: The output circuit name. If ``None``, it will be set to the same as the\n                input circuit name.\n            callback: A callback function that will be called after each pass execution. The\n                function will be called with 5 keyword arguments::\n\n                    pass_ (Pass): the pass being run\n                    dag (DAGCircuit): the dag output of the pass\n                    time (float): the time to execute the pass\n                    property_set (PropertySet): the property set\n                    count (int): the index for the pass execution\n\n                .. note::\n\n                    Beware that the keyword arguments here are different to those used by the\n                    generic :class:`.BasePassManager`.  This pass manager will translate those\n                    arguments into the form described above.\n\n                The exact arguments pass expose the internals of the pass\n                manager and are subject to change as the pass manager internals\n                change. If you intend to reuse a callback function over\n                multiple releases be sure to check that the arguments being\n                passed are the same.\n\n                To use the callback feature you define a function that will\n                take in kwargs dict and access the variables. For example::\n\n                    def callback_func(**kwargs):\n                        pass_ = kwargs['pass_']\n                        dag = kwargs['dag']\n                        time = kwargs['time']\n                        property_set = kwargs['property_set']\n                        count = kwargs['count']\n                        ...\n\n        Returns:\n            The transformed circuit(s).\n        "
        if callback is not None:
            callback = _legacy_style_callback(callback)
        return super().run(in_programs=circuits, callback=callback, output_name=output_name)

    def draw(self, filename=None, style=None, raw=False):
        if False:
            return 10
        'Draw the pass manager.\n\n        This function needs `pydot <https://github.com/erocarrera/pydot>`__, which in turn needs\n        `Graphviz <https://www.graphviz.org/>`__ to be installed.\n\n        Args:\n            filename (str): file path to save image to.\n            style (dict): keys are the pass classes and the values are the colors to make them. An\n                example can be seen in the DEFAULT_STYLE. An ordered dict can be used to ensure\n                a priority coloring when pass falls into multiple categories. Any values not\n                included in the provided dict will be filled in from the default dict.\n            raw (bool): If ``True``, save the raw Dot output instead of the image.\n\n        Returns:\n            Optional[PassManager]: an in-memory representation of the pass manager, or ``None``\n            if no image was generated or `Pillow <https://pypi.org/project/Pillow/>`__\n            is not installed.\n\n        Raises:\n            ImportError: when nxpd or pydot not installed.\n        '
        from qiskit.visualization import pass_manager_drawer
        return pass_manager_drawer(self, filename=filename, style=style, raw=raw)

    def passes(self) -> list[dict[str, BasePass]]:
        if False:
            return 10
        'Return a list structure of the appended passes and its options.\n\n        Returns:\n            A list of pass sets, as defined in ``append()``.\n        '
        ret = []
        for pass_set in self._pass_sets:
            item = {'passes': pass_set['passes']}
            if pass_set['flow_controllers']:
                item['flow_controllers'] = set(pass_set['flow_controllers'].keys())
            else:
                item['flow_controllers'] = {}
            ret.append(item)
        return ret

class StagedPassManager(PassManager):
    """A Pass manager pipeline built up of individual stages

    This class enables building a compilation pipeline out of fixed stages.
    Each ``StagedPassManager`` defines a list of stages which are executed in
    a fixed order, and each stage is defined as a standalone :class:`~.PassManager`
    instance. There are also ``pre_`` and ``post_`` stages for each defined stage.
    This enables easily composing and replacing different stages and also adding
    hook points to enable programmatic modifications to a pipeline. When using a staged
    pass manager you are not able to modify the individual passes and are only able
    to modify stages.

    By default instances of ``StagedPassManager`` define a typical full compilation
    pipeline from an abstract virtual circuit to one that is optimized and
    capable of running on the specified backend. The default pre-defined stages are:

    #. ``init`` - any initial passes that are run before we start embedding the circuit to the backend
    #. ``layout`` - This stage runs layout and maps the virtual qubits in the
       circuit to the physical qubits on a backend
    #. ``routing`` - This stage runs after a layout has been run and will insert any
       necessary gates to move the qubit states around until it can be run on
       backend's coupling map.
    #. ``translation`` - Perform the basis gate translation, in other words translate the gates
       in the circuit to the target backend's basis set
    #. ``optimization`` - The main optimization loop, this will typically run in a loop trying to
       optimize the circuit until a condition (such as fixed depth) is reached.
    #. ``scheduling`` - Any hardware aware scheduling passes

    .. note::

        For backwards compatibility the relative positioning of these default
        stages will remain stable moving forward. However, new stages may be
        added to the default stage list in between current stages. For example,
        in a future release a new phase, something like ``logical_optimization``, could be added
        immediately after the existing ``init`` stage in the default stage list.
        This would preserve compatibility for pre-existing ``StagedPassManager``
        users as the relative positions of the stage are preserved so the behavior
        will not change between releases.

    These stages will be executed in order and any stage set to ``None`` will be skipped.
    If a stage is provided multiple times (i.e. at diferent relative positions), the
    associated passes, including pre and post, will run once per declaration.
    If a :class:`~qiskit.transpiler.PassManager` input is being used for more than 1 stage here
    (for example in the case of a :class:`~.Pass` that covers both Layout and Routing) you will
    want to set that to the earliest stage in sequence that it covers.
    """
    invalid_stage_regex = re.compile('\\s|\\+|\\-|\\*|\\/|\\\\|\\%|\\<|\\>|\\@|\\!|\\~|\\^|\\&|\\:|\\[|\\]|\\{|\\}|\\(|\\)')

    def __init__(self, stages: Iterable[str] | None=None, **kwargs) -> None:
        if False:
            return 10
        "Initialize a new StagedPassManager object\n\n        Args:\n            stages (Iterable[str]): An optional list of stages to use for this\n                instance. If this is not specified the default stages list\n                ``['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling']`` is\n                used. After instantiation, the final list will be immutable and stored as tuple.\n                If a stage is provided multiple times (i.e. at diferent relative positions), the\n                associated passes, including pre and post, will run once per declaration.\n            kwargs: The initial :class:`~.PassManager` values for any stages\n                defined in ``stages``. If a argument is not defined the\n                stages will default to ``None`` indicating an empty/undefined\n                stage.\n\n        Raises:\n            AttributeError: If a stage in the input keyword arguments is not defined.\n            ValueError: If an invalid stage name is specified.\n        "
        stages = stages or ['init', 'layout', 'routing', 'translation', 'optimization', 'scheduling']
        self._validate_stages(stages)
        super().__setattr__('_stages', tuple(stages))
        super().__setattr__('_expanded_stages', tuple(self._generate_expanded_stages()))
        super().__init__()
        self._validate_init_kwargs(kwargs)
        for stage in set(self.expanded_stages):
            pm = kwargs.get(stage, None)
            setattr(self, stage, pm)

    def _validate_stages(self, stages: Iterable[str]) -> None:
        if False:
            i = 10
            return i + 15
        invalid_stages = [stage for stage in stages if self.invalid_stage_regex.search(stage) is not None]
        if invalid_stages:
            with io.StringIO() as msg:
                msg.write(f'The following stage names are not valid: {invalid_stages[0]}')
                for invalid_stage in invalid_stages[1:]:
                    msg.write(f', {invalid_stage}')
                raise ValueError(msg.getvalue())

    def _validate_init_kwargs(self, kwargs: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        expanded_stages = set(self.expanded_stages)
        for stage in kwargs.keys():
            if stage not in expanded_stages:
                raise AttributeError(f'{stage} is not a valid stage.')

    @property
    def stages(self) -> tuple[str, ...]:
        if False:
            return 10
        'Pass manager stages'
        return self._stages

    @property
    def expanded_stages(self) -> tuple[str, ...]:
        if False:
            i = 10
            return i + 15
        'Expanded Pass manager stages including ``pre_`` and ``post_`` phases.'
        return self._expanded_stages

    def _generate_expanded_stages(self) -> Iterator[str]:
        if False:
            i = 10
            return i + 15
        for stage in self.stages:
            yield ('pre_' + stage)
            yield stage
            yield ('post_' + stage)

    def _update_passmanager(self) -> None:
        if False:
            while True:
                i = 10
        self._tasks = []
        self._pass_sets = []
        for stage in self.expanded_stages:
            pm = getattr(self, stage, None)
            if pm is not None:
                self._tasks += pm._tasks
                self._pass_sets.extend(pm._pass_sets)

    def __setattr__(self, attr, value):
        if False:
            while True:
                i = 10
        if value == self and attr in self.expanded_stages:
            raise TranspilerError('Recursive definition of StagedPassManager disallowed.')
        super().__setattr__(attr, value)
        if attr in self.expanded_stages:
            self._update_passmanager()

    def append(self, passes: Task | list[Task], max_iteration: int=None, **flow_controller_conditions: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def replace(self, index: int, passes: BasePass | list[BasePass], max_iteration: int=None, **flow_controller_conditions: Any) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def remove(self, index: int) -> None:
        if False:
            return 10
        raise NotImplementedError

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        self._update_passmanager()
        new_passmanager = PassManager(max_iteration=self.max_iteration)
        new_passmanager._tasks = self._tasks[index]
        _pass_sets = self._pass_sets[index]
        if isinstance(_pass_sets, dict):
            _pass_sets = [_pass_sets]
        new_passmanager._pass_sets = _pass_sets
        return new_passmanager

    def __len__(self):
        if False:
            print('Hello World!')
        self._update_passmanager()
        return super().__len__()

    def __setitem__(self, index, item):
        if False:
            return 10
        raise NotImplementedError

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def passes(self) -> list[dict[str, BasePass]]:
        if False:
            while True:
                i = 10
        self._update_passmanager()
        return super().passes()

    def run(self, circuits: _CircuitsT, output_name: str | None=None, callback: Callable | None=None) -> _CircuitsT:
        if False:
            for i in range(10):
                print('nop')
        self._update_passmanager()
        return super().run(circuits, output_name, callback)

    def draw(self, filename=None, style=None, raw=False):
        if False:
            return 10
        'Draw the staged pass manager.'
        from qiskit.visualization import staged_pass_manager_drawer
        return staged_pass_manager_drawer(self, filename=filename, style=style, raw=raw)

def _replace_error(meth):
    if False:
        return 10

    @wraps(meth)
    def wrapper(*meth_args, **meth_kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            return meth(*meth_args, **meth_kwargs)
        except PassManagerError as ex:
            raise TranspilerError(ex.message) from ex
    return wrapper
for (_name, _method) in inspect.getmembers(PassManager, predicate=inspect.isfunction):
    if _name.startswith('_'):
        continue
    _wrapped = _replace_error(_method)
    setattr(PassManager, _name, _wrapped)

def _legacy_style_callback(callback: Callable):
    if False:
        i = 10
        return i + 15

    def _wrapped_callable(task, passmanager_ir, property_set, running_time, count):
        if False:
            return 10
        callback(pass_=task, dag=passmanager_ir, time=running_time, property_set=property_set, count=count)
    return _wrapped_callable

def _legacy_build_flow_controller(tasks: list[Task], options: dict[str, Any], **flow_controller_conditions) -> BaseController:
    if False:
        for i in range(10):
            print('nop')
    'A legacy method to build flow controller with keyword arguments.\n\n    Args:\n        tasks: A list of tasks fed into custom flow controllers.\n        options: Option for flow controllers.\n        flow_controller_conditions: Callables keyed on the alias of the flow controller.\n\n    Returns:\n        A built controller.\n    '
    warnings.warn('Building a flow controller with keyword arguments is going to be deprecated. Custom controllers must be explicitly instantiated and appended to the task list.', PendingDeprecationWarning, stacklevel=3)
    if isinstance(tasks, Task):
        tasks = [tasks]
    if any((not isinstance(t, Task) for t in tasks)):
        raise TypeError('Added tasks are not all valid pass manager task types.')
    for alias in FlowController.hierarchy[::-1]:
        if alias not in flow_controller_conditions:
            continue
        class_type = FlowController.registered_controllers[alias]
        init_kwargs = {'options': options, alias: flow_controller_conditions.pop(alias)}
        tasks = class_type(tasks, **init_kwargs)
    return tasks