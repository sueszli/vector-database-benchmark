import abc
import logging
from enum import Enum
from typing import Callable, Dict, List, Optional, Type, TYPE_CHECKING
from dataclasses import dataclass, field
from golem_messages.datastructures import stats as dt_stats
from apps.core.task.coretaskstate import TaskDefinition, Options
from golem.task.helpers import calculate_subtask_payment
from golem.task.taskstate import TaskState, SubtaskStatus
from golem.marketplace import ProviderMarketStrategy, RequestorMarketStrategy, DEFAULT_REQUESTOR_MARKET_STRATEGY, DEFAULT_PROVIDER_MARKET_STRATEGY
if TYPE_CHECKING:
    import golem_messages
    from golem_messages.datastructures.tasks import TaskHeader
    from apps.core.task.coretaskstate import TaskDefinition, Options
    from golem.task.taskstate import TaskState
logger = logging.getLogger('golem.task')

class AcceptClientVerdict(Enum):
    ACCEPTED = 0
    REJECTED = 1
    SHOULD_WAIT = 2

class TaskPurpose(Enum):
    TESTING = 'testing'
    REQUESTING = 'requesting'

class TaskTypeInfo(object):
    """ Information about task that allows to define and build a new task"""

    def __init__(self, name: str, definition: 'Type[TaskDefinition]', options: 'Type[Options]', task_builder_type: 'Type[TaskBuilder]') -> None:
        if False:
            return 10
        self.name = name
        self.options = options
        self.definition = definition
        self.task_builder_type = task_builder_type

    def for_purpose(self, purpose: TaskPurpose) -> 'TaskTypeInfo':
        if False:
            i = 10
            return i + 15
        return self

    @classmethod
    def get_preview(cls, task, single=False):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        normalized task type name\n        '
        return self.name.lower()

class TaskBuilder(abc.ABC):
    TASK_CLASS: Type['Task']

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def build(self) -> 'Task':
        if False:
            print('Hello World!')
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def build_definition(cls, task_type: TaskTypeInfo, dictionary, minimal=False) -> 'TaskDefinition':
        if False:
            i = 10
            return i + 15
        ' Build task defintion from dictionary with described options.\n        :param dict dictionary: described all options need to build a task\n        :param bool minimal: if this option is set too True, then only minimal\n        definition that can be used for task testing can be build. Otherwise\n        all necessary options must be specified in dictionary\n        '
        raise NotImplementedError

    @staticmethod
    def build_dictionary(definition: 'TaskDefinition') -> dict:
        if False:
            for i in range(10):
                print('nop')
        return definition.to_dict()

class TaskEventListener(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def notify_update_task(self, task_id):
        if False:
            i = 10
            return i + 15
        pass

@dataclass
class TaskResult:
    files: List[str] = field(default_factory=list)
    stats: dt_stats.ProviderStats = dt_stats.ProviderStats()

class Task(abc.ABC):
    REQUESTOR_MARKET_STRATEGY: Type[RequestorMarketStrategy] = DEFAULT_REQUESTOR_MARKET_STRATEGY
    PROVIDER_MARKET_STRATEGY: Type[ProviderMarketStrategy] = DEFAULT_PROVIDER_MARKET_STRATEGY

    class ExtraData(object):

        def __init__(self, ctd=None, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.ctd = ctd
            for (key, value) in kwargs.items():
                setattr(self, key, value)

    def __init__(self, header: 'TaskHeader', task_definition: 'TaskDefinition') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.header = header
        self.task_definition = task_definition
        self.listeners = []

    def __getstate__(self):
        if False:
            print('Hello World!')
        state = self.__dict__.copy()
        del state['listeners']
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.__dict__ = state
        self.listeners = []

    def __repr__(self):
        if False:
            return 10
        return '<Task: %r>' % (self.header,)

    @classmethod
    def calculate_subtask_budget(cls, task_definition: 'TaskDefinition'):
        if False:
            i = 10
            return i + 15
        '\n        calculate the per-job budget based on the task definition\n        :param task_definition:\n        :return: single job (subtask) budget [ GNT wei ]\n        '
        return calculate_subtask_payment(task_definition.max_price, task_definition.subtask_timeout)

    @property
    def price(self) -> int:
        if False:
            return 10
        return self.subtask_price * self.get_total_tasks()

    @property
    def subtask_price(self):
        if False:
            for i in range(10):
                print('nop')
        return self.calculate_subtask_budget(self.task_definition)

    def register_listener(self, listener):
        if False:
            return 10
        if not isinstance(listener, TaskEventListener):
            raise TypeError("Incorrect 'listener' type: {}. Should be: TaskEventListener".format(type(listener)))
        self.listeners.append(listener)

    def unregister_listener(self, listener):
        if False:
            for i in range(10):
                print('nop')
        if listener in self.listeners:
            self.listeners.remove(listener)
        else:
            logger.warning("Trying to unregister listener that wasn't registered.")

    @abc.abstractmethod
    def initialize(self, dir_manager):
        if False:
            return 10
        'Called after adding a new task, may initialize or create\n        some resources or do other required operations.\n        :param DirManager dir_manager: DirManager instance for accessing\n        temp dir for this task\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def query_extra_data(self, perf_index: float, node_id: Optional[str]=None, node_name: Optional[str]=None) -> 'ExtraData':
        if False:
            print('Hello World!')
        ' Called when a node asks with given parameters asks for a new\n        subtask to compute.\n        :param perf_index: performance that given node declares\n        :param node_id: id of a node that wants to get a next subtask\n        :param node_name: name of a node that wants to get a next subtask\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def query_extra_data_for_test_task(self) -> 'golem_messages.message.ComputeTaskDef':
        if False:
            return 10
        raise NotImplementedError

    @abc.abstractmethod
    def needs_computation(self) -> bool:
        if False:
            i = 10
            return i + 15
        ' Return information if there are still some subtasks\n        that may be dispended\n        :return bool: True if there are still subtask that should be computed,\n        False otherwise\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def finished_computation(self) -> bool:
        if False:
            i = 10
            return i + 15
        ' Return information if tasks has been fully computed\n        :return bool: True if there is all tasks has been computed and verified\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def computation_finished(self, subtask_id: str, task_result: TaskResult, verification_finished: Callable[[], None]) -> None:
        if False:
            return 10
        ' Inform about finished subtask\n        :param subtask_id: finished subtask id\n        :param task_result: task result, can be binary data or list of files\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def computation_failed(self, subtask_id: str, ban_node: bool=True):
        if False:
            i = 10
            return i + 15
        ' Inform that computation of a task with given id has failed\n        :param subtask_id:\n        :param ban_node: Whether to ban this node from this task\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def verify_subtask(self, subtask_id):
        if False:
            i = 10
            return i + 15
        ' Verify given subtask\n        :param subtask_id:\n        :return bool: True if a subtask passed verification, False otherwise\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def verify_task(self):
        if False:
            print('Hello World!')
        ' Verify whole task after computation\n        :return bool: True if task passed verification, False otherwise\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_tasks(self) -> int:
        if False:
            print('Hello World!')
        ' Return total number of tasks that should be computed\n        :return int: number should be greater than 0\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def get_active_tasks(self) -> int:
        if False:
            print('Hello World!')
        ' Return number of tasks that are currently being computed\n        :return int: number should be between 0 and a result of get_total_tasks\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def get_tasks_left(self) -> int:
        if False:
            print('Hello World!')
        ' Return number of tasks that still should be computed\n        :return int: number should be between 0 and a result of get_total_tasks\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def restart(self):
        if False:
            while True:
                i = 10
        ' Restart all subtask computation for this task '
        raise NotImplementedError

    @abc.abstractmethod
    def restart_subtask(self, subtask_id, new_state: Optional[SubtaskStatus]=None):
        if False:
            for i in range(10):
                print('nop')
        ' Restart subtask with given id '
        raise NotImplementedError

    @abc.abstractmethod
    def abort(self):
        if False:
            print('Hello World!')
        ' Abort task and all computations '
        raise NotImplementedError

    @abc.abstractmethod
    def get_progress(self) -> float:
        if False:
            print('Hello World!')
        ' Return task computations progress\n        :return float: Return number between 0.0 and 1.0.\n        '
        raise NotImplementedError

    def get_resources(self) -> list:
        if False:
            print('Hello World!')
        ' Return list of files that are needed to compute this task.'
        return []

    @abc.abstractmethod
    def update_task_state(self, task_state: 'TaskState'):
        if False:
            for i in range(10):
                print('nop')
        ' Update some task information taking into account new state.\n        :param TaskState task_state:\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def get_trust_mod(self, subtask_id) -> int:
        if False:
            return 10
        ' Return trust modifier for given subtask. This number may be taken\n        into account during increasing or decreasing trust for given node\n        after successful or failed computation.\n        :param subtask_id:\n        :return int:\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def add_resources(self, resources: set):
        if False:
            while True:
                i = 10
        ' Add resources to a task\n        :param resources:\n        '
        raise NotImplementedError

    def get_stdout(self, subtask_id) -> str:
        if False:
            print('Hello World!')
        ' Return stdout received after computation of subtask_id,\n        if there is no data available\n        return empty string\n        :param subtask_id:\n        :return str:\n        '
        return ''

    def get_stderr(self, subtask_id) -> str:
        if False:
            print('Hello World!')
        ' Return stderr received after computation of subtask_id,\n        if there is no data available\n        return emtpy string\n        :param subtask_id:\n        :return str:\n        '
        return ''

    def get_results(self, subtask_id) -> List:
        if False:
            for i in range(10):
                print('nop')
        ' Return list of files containing results for subtask with given id\n        :param subtask_id:\n        :return list:\n        '
        return []

    def result_incoming(self, subtask_id):
        if False:
            print('Hello World!')
        ' Informs that a computed task result is being retrieved\n        :param subtask_id:\n        :return:\n        '
        pass

    def get_output_names(self) -> List:
        if False:
            while True:
                i = 10
        ' Return list of files containing final import task results\n        :return list:\n        '
        return []

    def get_output_states(self) -> List:
        if False:
            while True:
                i = 10
        ' Return list of states of final task results\n        :return list:\n        '
        return []

    @abc.abstractmethod
    def copy_subtask_results(self, subtask_id: str, old_subtask_info: dict, results: TaskResult) -> None:
        if False:
            while True:
                i = 10
        '\n        Copy results of a single subtask from another task\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def to_dictionary(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abc.abstractmethod
    def should_accept_client(self, node_id: str, offer_hash: str) -> AcceptClientVerdict:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @abc.abstractmethod
    def accept_client(self, node_id: str, offer_hash: str, num_subtasks: int=1) -> AcceptClientVerdict:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def get_finishing_subtasks(self, node_id: str) -> List[dict]:
        if False:
            while True:
                i = 10
        return []

    def external_verify_subtask(self, subtask_id, verdict):
        if False:
            while True:
                i = 10
        '\n        Verify subtask results\n        '
        return None

    def subtask_status_updated(self, subtask_id: str) -> None:
        if False:
            print('Hello World!')
        pass

class ResultMetadata:

    def __init__(self, compute_time: float) -> None:
        if False:
            i = 10
            return i + 15
        self.compute_time: float = compute_time