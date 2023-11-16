"""Built-in pass flow controllers."""
from __future__ import annotations
import logging
from collections.abc import Callable, Iterable, Generator
from typing import Type, Any
from qiskit.utils.deprecation import deprecate_func
from .base_tasks import BaseController, Task
from .compilation_status import PassManagerState, PropertySet
from .exceptions import PassManagerError
logger = logging.getLogger(__name__)

class FlowControllerLinear(BaseController):
    """A standard flow controller that runs tasks one after the other."""

    def __init__(self, tasks: Task | Iterable[Task]=(), *, options: dict[str, Any] | None=None):
        if False:
            while True:
                i = 10
        super().__init__(options)
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)

    @property
    def passes(self) -> list[Task]:
        if False:
            print('Hello World!')
        'Alias of tasks for backward compatibility.'
        return list(self.tasks)

    @deprecate_func(since='0.45.0', additional_msg='All tasks must be provided at construction time of the controller object.')
    def append(self, passes: Task | list[Task]):
        if False:
            i = 10
            return i + 15
        'Add new task to pipeline.\n\n        Args:\n            passes: A new task or list of tasks to add.\n        '
        if not isinstance(passes, Iterable):
            passes = [passes]
        tasks = list(self.tasks)
        for task in passes:
            if not isinstance(task, Task):
                raise TypeError(f'New task {task} is not a valid pass manager pass or flow controller.')
            tasks.append(task)
        self.tasks = tuple(tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        if False:
            print('Hello World!')
        for task in self.tasks:
            state = (yield task)

class DoWhileController(BaseController):
    """Run the given tasks in a loop until the ``do_while`` condition on the property set becomes
    ``False``.

    The given tasks will always run at least once, and on iteration of the loop, all the
    tasks will be run (with the exception of a failure state being set)."""

    def __init__(self, tasks: Task | Iterable[Task]=(), do_while: Callable[[PropertySet], bool]=None, *, options: dict[str, Any] | None=None):
        if False:
            print('Hello World!')
        super().__init__(options)
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)
        self.do_while = do_while

    @property
    def passes(self) -> list[Task]:
        if False:
            while True:
                i = 10
        'Alias of tasks for backward compatibility.'
        return list(self.tasks)

    @deprecate_func(since='0.45.0', additional_msg='All tasks must be provided at construction time of the controller object.')
    def append(self, passes: Task | list[Task]):
        if False:
            for i in range(10):
                print('nop')
        'Add new task to pipeline.\n\n        Args:\n            passes: A new task or list of tasks to add.\n        '
        if not isinstance(passes, Iterable):
            passes = [passes]
        tasks = list(self.tasks)
        for task in passes:
            if not isinstance(task, Task):
                raise TypeError(f'New task {task} is not a valid pass manager pass or flow controller.')
            tasks.append(task)
        self.tasks = tuple(tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        if False:
            while True:
                i = 10
        max_iteration = self._options.get('max_iteration', 1000)
        for _ in range(max_iteration):
            for task in self.tasks:
                state = (yield task)
            if not self.do_while(state.property_set):
                return
            state.workflow_status.completed_passes.difference_update(self.tasks)
        raise PassManagerError('Maximum iteration reached. max_iteration=%i' % max_iteration)

class ConditionalController(BaseController):
    """A flow controller runs the pipeline once if the condition is true, or does nothing if the
    condition is false."""

    def __init__(self, tasks: Task | Iterable[Task]=(), condition: Callable[[PropertySet], bool]=None, *, options: dict[str, Any] | None=None):
        if False:
            print('Hello World!')
        super().__init__(options)
        if not isinstance(tasks, Iterable):
            tasks = [tasks]
        self.tasks: tuple[Task] = tuple(tasks)
        self.condition = condition

    @property
    def passes(self) -> list[Task]:
        if False:
            return 10
        'Alias of tasks for backward compatibility.'
        return list(self.tasks)

    @deprecate_func(since='0.45.0', additional_msg='All tasks must be provided at construction time of the controller object.')
    def append(self, passes: Task | list[Task]):
        if False:
            for i in range(10):
                print('nop')
        'Add new task to pipeline.\n\n        Args:\n            passes: A new task or list of tasks to add.\n        '
        if not isinstance(passes, Iterable):
            passes = [passes]
        tasks = list(self.tasks)
        for task in passes:
            if not isinstance(task, Task):
                raise TypeError(f'New task {task} is not a valid pass manager pass or flow controller.')
            tasks.append(task)
        self.tasks = tuple(tasks)

    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        if False:
            i = 10
            return i + 15
        if self.condition(state.property_set):
            for task in self.tasks:
                state = (yield task)

class FlowController(BaseController):
    """A legacy factory for other flow controllers.

    .. warning::

        This class is primarily for compatibility with legacy versions of Qiskit, and in general,
        you should prefer simply instantiating the controller you want, and adding it to the
        relevant :class:`.PassManager` or other controller.  Its use is deprecated.

    This allows syntactic sugar for writing pipelines. For example::

        FlowController.add_flow_controller("my_condition", CustomController)

        controller = FlowController.controller_factory(
            [PassA(), PassB()],
            {"max_iteration": 1000},
            condition=lambda prop_set: prop_set["x"] == 0,
            do_while=lambda prop_set: prop_set["x"] < 100,
            my_condition=lambda prop_set: prop_set["y"] = "abc",
        )

    This creates a nested flow controller that runs when the value :code:`x` in the
    :class:`.PropertySet` is zero and repeats the pipeline until the value becomes 100.
    In each innermost loop, the custom iteration condition provided by
    the ``CustomController`` is also evaluated.

    .. warning::

        :class:`.BaseController` must be directly subclassed to define a custom flow controller.
        This class provides a controller factory method, which consumes a class variable
        :attr:`.registered_controllers`. Subclassing FlowController may cause
        unexpected behavior in the factory method.
        Note that factory method implicitly determines the priority of the builtin controllers
        when multiple controllers are called together,
        and the behavior of generated controller is hardly debugged.
    """
    registered_controllers = {'condition': ConditionalController, 'do_while': DoWhileController}
    hierarchy = ['condition', 'do_while']

    @classmethod
    @deprecate_func(since='0.45.0', additional_msg='Controller object must be explicitly instantiated. Building controller with keyword arguments may yield race condition when multiple keyword arguments are provided together, which is likely unsafe.')
    def controller_factory(cls, passes: Task | list[Task], options: dict, **controllers):
        if False:
            while True:
                i = 10
        'Create a new flow controller with normalization.\n\n        Args:\n            passes: A list of optimization tasks.\n            options: Option for this flow controller.\n            controllers: Dictionary of controller callables keyed on flow controller alias.\n\n        Returns:\n            An instance of normalized flow controller.\n        '
        if None in controllers.values():
            raise PassManagerError('The controller needs a callable. Value cannot be None.')
        if isinstance(passes, BaseController):
            instance = passes
        else:
            instance = FlowControllerLinear(passes, options=options)
        if controllers:
            for alias in cls.hierarchy[::-1]:
                if alias not in controllers:
                    continue
                class_type = cls.registered_controllers[alias]
                init_kwargs = {'options': options, alias: controllers.pop(alias)}
                instance = class_type([instance], **init_kwargs)
        return instance

    @classmethod
    @deprecate_func(since='0.45.0', additional_msg='Controller factory method is deprecated and managing the custom flow controllers with alias no longer helps building the task pipeline. Controllers must be explicitly instantiated and appended to the pipeline.')
    def add_flow_controller(cls, name: str, controller: Type[BaseController]):
        if False:
            return 10
        'Adds a flow controller.\n\n        Args:\n            name: Alias of controller class in the namespace.\n            controller: Flow controller class.\n        '
        cls.registered_controllers[name] = controller
        if name not in cls.hierarchy:
            cls.hierarchy.append(name)

    @classmethod
    @deprecate_func(since='0.45.0', additional_msg='Controller factory method is deprecated and managing the custom flow controllers with alias no longer helps building the task pipeline. Controllers must be explicitly instantiated and appended to the pipeline.')
    def remove_flow_controller(cls, name: str):
        if False:
            i = 10
            return i + 15
        'Removes a flow controller.\n\n        Args:\n            name: Alias of the controller to remove.\n\n        Raises:\n            KeyError: If the controller to remove was not registered.\n        '
        if name not in cls.hierarchy:
            raise KeyError('Flow controller not found: %s' % name)
        del cls.registered_controllers[name]
        cls.hierarchy.remove(name)