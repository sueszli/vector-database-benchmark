"""Baseclasses for the Qiskit passmanager optimization tasks."""
from __future__ import annotations
import logging
import time
from abc import abstractmethod, ABC
from collections.abc import Iterable, Callable, Generator
from typing import Any
from .compilation_status import RunState, PassManagerState, PropertySet
logger = logging.getLogger(__name__)
PassManagerIR = Any

class Task(ABC):
    """An interface of the pass manager task.

    The task takes a Qiskit IR, and outputs new Qiskit IR after some operation on it.
    A task can rely on the :class:`.PropertySet` to communicate intermediate data among tasks.
    """

    @abstractmethod
    def execute(self, passmanager_ir: PassManagerIR, state: PassManagerState, callback: Callable=None) -> tuple[PassManagerIR, PassManagerState]:
        if False:
            print('Hello World!')
        'Execute optimization task for input Qiskit IR.\n\n        Args:\n            passmanager_ir: Qiskit IR to optimize.\n            state: State associated with workflow execution by the pass manager itself.\n            callback: A callback function which is caller per execution of optimization task.\n\n        Returns:\n            Optimized Qiskit IR and state of the workflow.\n        '
        pass

class GenericPass(Task, ABC):
    """Base class of a single pass manager task.

    A pass instance can read and write to the provided :class:`.PropertySet`,
    and may modify the input pass manager IR.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.property_set = PropertySet()
        self.requires: Iterable[Task] = []

    def name(self) -> str:
        if False:
            return 10
        'Name of the pass.'
        return self.__class__.__name__

    def execute(self, passmanager_ir: PassManagerIR, state: PassManagerState, callback: Callable=None) -> tuple[PassManagerIR, PassManagerState]:
        if False:
            while True:
                i = 10
        self.property_set = state.property_set
        if self.requires:
            from .flow_controllers import FlowControllerLinear
            (passmanager_ir, state) = FlowControllerLinear(self.requires).execute(passmanager_ir=passmanager_ir, state=state, callback=callback)
        run_state = None
        ret = None
        start_time = time.time()
        try:
            if self not in state.workflow_status.completed_passes:
                ret = self.run(passmanager_ir)
                run_state = RunState.SUCCESS
            else:
                run_state = RunState.SKIP
        except Exception:
            run_state = RunState.FAIL
            raise
        finally:
            ret = ret or passmanager_ir
            if run_state != RunState.SKIP:
                running_time = time.time() - start_time
                logger.info('Pass: %s - %.5f (ms)', self.name(), running_time * 1000)
                if callback is not None:
                    callback(task=self, passmanager_ir=ret, property_set=state.property_set, running_time=running_time, count=state.workflow_status.count)
        return (ret, self.update_status(state, run_state))

    def update_status(self, state: PassManagerState, run_state: RunState) -> PassManagerState:
        if False:
            for i in range(10):
                print('nop')
        'Update workflow status.\n\n        Args:\n            state: Pass manager state to update.\n            run_state: Completion status of current task.\n\n        Returns:\n            Updated pass manager state.\n        '
        state.workflow_status.previous_run = run_state
        if run_state == RunState.SUCCESS:
            state.workflow_status.count += 1
            state.workflow_status.completed_passes.add(self)
        return state

    @abstractmethod
    def run(self, passmanager_ir: PassManagerIR) -> PassManagerIR:
        if False:
            return 10
        'Run optimization task.\n\n        Args:\n            passmanager_ir: Qiskit IR to optimize.\n\n        Returns:\n            Optimized Qiskit IR.\n        '
        pass

class BaseController(Task, ABC):
    """Base class of controller.

    A controller is built with a collection of pass manager tasks,
    and a subclass provides a custom logic to choose next task to run.
    Note a controller can be nested into another controller,
    and a controller itself doesn't provide any subroutine to modify the input IR.
    """

    def __init__(self, options: dict[str, Any] | None=None):
        if False:
            while True:
                i = 10
        'Create new flow controller.\n\n        Args:\n            options: Option for this flow controller.\n        '
        self._options = options or {}

    @abstractmethod
    def iter_tasks(self, state: PassManagerState) -> Generator[Task, PassManagerState, None]:
        if False:
            return 10
        'A custom logic to choose a next task to run.\n\n        Controller subclass can consume the state to build a proper task pipeline.  The updated\n        state after a task execution will be fed back in as the "return" value of any ``yield``\n        statements.  This indicates the order of task execution is only determined at running time.\n        This method is not allowed to mutate the given state object.\n\n        Args:\n            state: The state of the passmanager workflow at the beginning of this flow controller\'s\n                execution.\n\n        Receives:\n            state: the state of pass manager after the execution of the last task that was yielded.\n                The generator does not need to inspect this if it is irrelevant to its logic, nor\n                update it.\n\n        Yields:\n            Task: Next task to run.\n        '
        pass

    def execute(self, passmanager_ir: PassManagerIR, state: PassManagerState, callback: Callable=None) -> tuple[PassManagerIR, PassManagerState]:
        if False:
            print('Hello World!')
        task_generator = self.iter_tasks(state)
        try:
            next_task = task_generator.send(None)
        except StopIteration:
            return (passmanager_ir, state)
        while True:
            (passmanager_ir, state) = next_task.execute(passmanager_ir=passmanager_ir, state=state, callback=callback)
            try:
                next_task = task_generator.send(state)
            except StopIteration:
                break
        return (passmanager_ir, state)