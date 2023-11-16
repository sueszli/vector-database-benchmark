import copy
import logging
import os
import pickle
import shutil
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Type, TYPE_CHECKING
from zipfile import ZipFile
from golem_messages import message
from pydispatch import dispatcher
from twisted.internet.defer import Deferred
from twisted.internet.threads import deferToThread
from apps.appsmanager import AppsManager
from apps.core.task.coretask import CoreTask
from apps.wasm.environment import WasmTaskEnvironment
from golem import model
from golem.clientconfigdescriptor import ClientConfigDescriptor
from golem.core.common import get_timestamp_utc, HandleForwardedError, HandleKeyError, short_node_id, to_unicode, update_dict
from golem.marketplace import ProviderBrassMarketStrategy, ProviderWasmMarketStrategy, ProviderMarketStrategy
from golem.manager.nodestatesnapshot import LocalTaskStateSnapshot
from golem.network import nodeskeeper
from golem.ranking.manager.database_manager import update_provider_efficiency, update_provider_efficacy
from golem.resource.dirmanager import DirManager
from golem.resource.hyperdrive.resourcesmanager import HyperdriveResourceManager
from golem.rpc import utils as rpc_utils
from golem.task.result.resultmanager import EncryptedResultPackageManager
from golem.task.taskbase import TaskEventListener, Task, TaskPurpose, AcceptClientVerdict, TaskResult
from golem.task.helpers import calculate_subtask_payment
from golem.task.taskkeeper import CompTaskKeeper
from golem.task.taskrequestorstats import RequestorTaskStatsManager
from golem.task.taskstate import TaskState, TaskStatus, SubtaskStatus, SubtaskState, Operation, TaskOp, SubtaskOp, OtherOp
from golem.task.timer import ProviderComputeTimers
if TYPE_CHECKING:
    from typing import Tuple
    from apps.appsmanager import App
    from apps.core.task.coretaskstate import TaskDefinition
    from golem.task.taskbase import TaskTypeInfo, TaskBuilder
logger = logging.getLogger(__name__)

def log_subtask_key_error(*args, **kwargs):
    if False:
        return 10
    logger.warning('This is not my subtask %r', args[1])
    logger.debug('Subtask not found', exc_info=True)
    return None

def log_generic_key_error(err):
    if False:
        while True:
            i = 10
    logger.warning('Subtask key error: %r', err)
    return None

def log_task_key_error(*args, **kwargs):
    if False:
        return 10
    logger.warning('This is not my task %r', args[1])
    logger.debug('Task not found', exc_info=True)
    return None

class TaskManager(TaskEventListener):
    """ Keeps and manages information about requested tasks
    Requestor uses TaskManager to assign task to providers
    """
    handle_task_key_error = HandleKeyError(log_task_key_error)
    handle_subtask_key_error = HandleKeyError(log_subtask_key_error)
    handle_generic_key_error = HandleForwardedError(KeyError, log_generic_key_error)

    class Error(Exception):
        pass

    class AlreadyRestartedError(Error):
        pass

    def __init__(self, node, keys_auth, root_path, config_desc: ClientConfigDescriptor, tasks_dir='tasks', apps_manager=AppsManager(), finished_cb=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.apps_manager: AppsManager = apps_manager
        apps: 'List[App]' = list(apps_manager.apps.values())
        task_types: 'List[TaskTypeInfo]' = [app.task_type_info() for app in apps]
        self.task_types: 'Dict[str, TaskTypeInfo]' = {t.id: t for t in task_types}
        self.node = node
        self.keys_auth = keys_auth
        self.tasks: Dict[str, Task] = {}
        self.tasks_states: Dict[str, TaskState] = {}
        self.subtask2task_mapping: Dict[str, str] = {}
        tasks_dir = Path(tasks_dir)
        self.tasks_dir = tasks_dir / 'tmanager'
        if not self.tasks_dir.is_dir():
            self.tasks_dir.mkdir(parents=True)
        self.root_path = root_path
        self.dir_manager = DirManager(self.get_task_manager_root())
        resource_manager = HyperdriveResourceManager(self.dir_manager, resource_dir_method=self.dir_manager.get_task_temporary_dir, client_kwargs={'host': config_desc.hyperdrive_rpc_address, 'port': config_desc.hyperdrive_rpc_port})
        self.task_result_manager = EncryptedResultPackageManager(resource_manager)
        self.comp_task_keeper = CompTaskKeeper(tasks_dir)
        self.requestor_stats_manager = RequestorTaskStatsManager()
        self.provider_stats_manager = self.comp_task_keeper.provider_stats_manager
        self.finished_cb = finished_cb
        self.restore_tasks()

    def get_task_manager_root(self):
        if False:
            print('Hello World!')
        return self.root_path

    def create_task_definition(self, dictionary, test=False) -> 'Tuple[TaskDefinition, Type[TaskBuilder]]':
        if False:
            return 10
        purpose = TaskPurpose.TESTING if test else TaskPurpose.REQUESTING
        is_requesting = purpose == TaskPurpose.REQUESTING
        type_name = dictionary['type'].lower()
        compute_on = dictionary.get('compute_on', 'cpu').lower()
        if type_name == 'blender' and is_requesting and (compute_on == 'gpu'):
            type_name = type_name + '_nvgpu'
        task_type: 'TaskTypeInfo' = self.task_types[type_name].for_purpose(purpose)
        builder_type: 'Type[TaskBuilder]' = task_type.task_builder_type
        definition: 'TaskDefinition' = builder_type.build_definition(task_type, dictionary, test)
        definition.concent_enabled = dictionary.get('concent_enabled', False)
        definition.task_id = CoreTask.create_task_id(self.keys_auth.public_key)
        return (definition, builder_type)

    def create_task(self, dictionary, test=False):
        if False:
            print('Hello World!')
        (definition, builder_type) = self.create_task_definition(dictionary, test)
        task = builder_type(self.node, definition, self.dir_manager).build()
        task_id = definition.task_id
        if not test:
            logger.info('Creating task. type=%r, id=%s', type(task), task_id)
            self.tasks[task_id] = task
            self.tasks_states[task_id] = TaskState(task)
        return task

    def initialize_task(self, task: Task):
        if False:
            while True:
                i = 10
        task.initialize(self.dir_manager)

    def get_task_definition_dict(self, task: Task):
        if False:
            return 10
        if isinstance(task, dict):
            return task
        definition = task.task_definition
        task_type = self.task_types[definition.task_type.lower()]
        return task_type.task_builder_type.build_dictionary(definition)

    def add_new_task(self, task: Task, estimated_fee: int=0) -> None:
        if False:
            i = 10
            return i + 15
        task_id = task.header.task_id
        task_state = self.tasks_states.get(task_id)
        if not task_state:
            task_state = TaskState(task)
            self.tasks[task_id] = task
            self.tasks_states[task_id] = task_state
        if task_state.status is not TaskStatus.creating:
            raise RuntimeError('Task {} has already been added'.format(task_id))
        task.header.task_owner = self.node
        self.sign_task_header(task.header)
        task.register_listener(self)
        task_state.status = TaskStatus.notStarted
        task_state.time_started = time.time()
        task_state.estimated_fee = estimated_fee
        logger.info('Task %s added', task_id)
        self._create_task_output_dir(task.task_definition)
        self.notice_task_updated(task_id, op=TaskOp.CREATED, persist=False)

    @handle_task_key_error
    def task_creation_failed(self, task_id: str, reason: str) -> None:
        if False:
            while True:
                i = 10
        logger.error('Cannot create task. task_id=%s : %s', task_id, reason)
        task_state = self.tasks_states[task_id]
        task_state.status = TaskStatus.errorCreating
        task_state.status_message = reason

    @handle_task_key_error
    def increase_task_mask(self, task_id: str, num_bits: int=1) -> None:
        if False:
            print('Hello World!')
        ' Increase mask for given task i.e. make it more restrictive '
        task_header = copy.deepcopy(self.tasks[task_id].header)
        try:
            task_header.mask.increase(num_bits)
        except ValueError:
            logger.exception('Wrong number of bits for mask increase')
        else:
            self.sign_task_header(task_header)
            try:
                self.tasks[task_id].header = task_header
            except KeyError:
                pass

    @handle_task_key_error
    def decrease_task_mask(self, task_id: str, num_bits: int=1) -> None:
        if False:
            while True:
                i = 10
        ' Decrease mask for given task i.e. make it less restrictive '
        task_header = copy.deepcopy(self.tasks[task_id].header)
        try:
            task_header.mask.decrease(num_bits)
        except ValueError:
            logger.exception('Wrong number of bits for mask decrease')
        else:
            self.sign_task_header(task_header)
            try:
                self.tasks[task_id].header = task_header
            except KeyError:
                pass

    @handle_task_key_error
    def start_task(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        task_state = self.tasks_states[task_id]
        if not task_state.status.is_preparing():
            raise RuntimeError('Task {} has already been started'.format(task_id))
        task_state.status = TaskStatus.waiting
        self.notice_task_updated(task_id, op=TaskOp.STARTED)
        logger.info('Task started. task_id=%r', task_id)

    def _dump_filepath(self, task_id):
        if False:
            while True:
                i = 10
        return self.tasks_dir / ('%s.pickle' % (task_id,))

    def dump_task(self, task_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('DUMP TASK %r', task_id)
        filepath = self._dump_filepath(task_id)
        try:
            data = (self.tasks[task_id], self.tasks_states[task_id])
            logger.debug('DUMPING TASK %r', filepath)
            with filepath.open('wb') as f:
                pickle.dump(data, f, protocol=2)
            logger.debug('TASK %s DUMPED in %r', task_id, filepath)
        except Exception:
            logger.exception('DUMP ERROR task_id: %r task: %r state: %r', task_id, self.tasks.get(task_id, '<not found>'), self.tasks_states.get(task_id, '<not found>'))
            if filepath.exists():
                filepath.unlink()
            raise

    def remove_dump(self, task_id: str):
        if False:
            while True:
                i = 10
        filepath = self._dump_filepath(task_id)
        try:
            filepath.unlink()
            logger.debug('TASK DUMP with id %s REMOVED from %r', task_id, filepath)
        except (FileNotFoundError, OSError) as e:
            logger.warning("Couldn't remove dump file: %s - %s", filepath, e)

    def _create_task_output_dir(self, task_def: 'TaskDefinition'):
        if False:
            return 10
        "\n        Creates the output directory for a task along with any parents,\n        if necessary. The path is obtained from `output_file` field in the\n        task's definition.\n        For example, for an output file with the following path:\n        `/some/output/dir/result.png` the created directory will be:\n        `/some/output/dir`.\n        "
        output_dir = self._get_task_output_dir(task_def)
        if not output_dir:
            return
        output_dir.mkdir(parents=True, exist_ok=True)

    def _try_remove_task_output_dir(self, task_def: 'TaskDefinition'):
        if False:
            print('Hello World!')
        '\n        Attempts to remove the output directory from a given task definition.\n        This will only succeed if the directory is empty.\n        '
        output_dir = self._get_task_output_dir(task_def)
        if not output_dir:
            return
        try:
            output_dir.rmdir()
        except OSError:
            pass

    @staticmethod
    def _get_task_output_dir(task_def: 'TaskDefinition') -> Optional[Path]:
        if False:
            while True:
                i = 10
        if not task_def.output_file:
            return None
        return Path(task_def.output_file).resolve().parent

    def restore_tasks(self) -> None:
        if False:
            while True:
                i = 10
        logger.debug('SEARCHING FOR TASKS TO RESTORE')
        broken_paths = set()
        for path in self.tasks_dir.iterdir():
            if not path.suffix == '.pickle':
                continue
            logger.debug('RESTORE TASKS %r', path)
            task_id = None
            with path.open('rb') as f:
                try:
                    task: Task
                    state: TaskState
                    (task, state) = pickle.load(f)
                except Exception:
                    logger.exception('Problem restoring task from: %s', path)
                    broken_paths.add(path)
                else:
                    task.register_listener(self)
                    task_id = task.header.task_id
                    self.tasks[task_id] = task
                    self.tasks_states[task_id] = state
                    for sub in state.subtask_states.values():
                        self.subtask2task_mapping[sub.subtask_id] = task_id
                    logger.debug('TASK %s RESTORED from %r', task_id, path)
            if task_id is not None:
                self.notice_task_updated(task_id, op=TaskOp.RESTORED, persist=False)
        for path in broken_paths:
            path.unlink()

    def got_wants_to_compute(self, task_id: str):
        if False:
            while True:
                i = 10
        '\n        Updates number of offers to compute task.\n\n        For statistical purposes only, real processing of the offer is done\n        elsewhere. Silently ignores wrong task ids.\n\n        :param str task_id: id of the task in the offer\n        :return: Nothing\n        :rtype: None\n        '
        if task_id in self.tasks:
            self.notice_task_updated(task_id, op=TaskOp.WORK_OFFER_RECEIVED, persist=False)

    def task_being_created(self, task_id: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        task_status = self.tasks_states[task_id].status
        return task_status.is_creating()

    def task_finished(self, task_id: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        task_status = self.tasks_states[task_id].status
        return task_status.is_completed()

    def task_needs_computation(self, task_id: str) -> bool:
        if False:
            return 10
        if self.task_being_created(task_id) or self.task_finished(task_id):
            task_status = self.tasks_states[task_id].status
            logger.debug('task is not active: %(task_id)s, status: %(task_status)s', {'task_id': task_id, 'task_status': task_status})
            return False
        task = self.tasks[task_id]
        if not task.needs_computation():
            logger.info(f'no more computation needed: {task_id}')
            return False
        return True

    def get_next_subtask(self, node_id: str, task_id: str, estimated_performance: float, price: int, offer_hash: str) -> Optional[message.tasks.ComputeTaskDef]:
        if False:
            return 10
        ' Assign next subtask from task <task_id> to node with given\n        id <node_id>.\n        :return ComputeTaskDef that describe assigned subtask\n        or None. It is recommended to call is_my_task and should_wait_for_node\n        before this to find the reason why the task is not able to be picked up\n        '
        logger.debug('get_next_subtask(%r, %r, %r, %r)', node_id, task_id, estimated_performance, price)
        if node_id == self.keys_auth.key_id:
            logger.warning('No subtasks for self')
            return None
        if not self.is_my_task(task_id):
            return None
        if not self.check_next_subtask(task_id, price):
            return None
        if not self.task_needs_computation(task_id):
            return None
        if self.should_wait_for_node(task_id, node_id, offer_hash):
            return None
        task = self.tasks[task_id]
        if task.get_progress() == 1.0:
            logger.error('Task already computed. task_id=%r, node_id=%r', task_id, node_id)
            return None
        extra_data = task.query_extra_data(estimated_performance, node_id, '')
        ctd = extra_data.ctd

        def check_compute_task_def():
            if False:
                print('Hello World!')
            if not isinstance(ctd, message.tasks.ComputeTaskDef) or not ctd['subtask_id']:
                logger.debug('check ctd: ctd not instance or not subtask_id')
                return False
            if task_id != ctd['task_id'] or ctd['subtask_id'] in self.subtask2task_mapping:
                logger.debug('check ctd: %r != %r or %r in self.subtask2task_maping', task_id, ctd['task_id'], ctd['subtask_id'])
                return False
            if ctd['subtask_id'] in self.tasks_states[ctd['task_id']].subtask_states:
                logger.debug('check ctd: subtask_states')
                return False
            return True
        if not check_compute_task_def():
            return None
        self.subtask2task_mapping[ctd['subtask_id']] = task_id
        self.__add_subtask_to_tasks_states(node_id, ctd, price)
        self.notice_task_updated(task_id, subtask_id=ctd['subtask_id'], op=SubtaskOp.ASSIGNED)
        logger.debug('Subtask generated. task=%s, node=%s, ctd=%s', task_id, short_node_id(node_id), ctd)
        ProviderComputeTimers.start(ctd['subtask_id'])
        return ctd

    def is_my_task(self, task_id: str) -> bool:
        if False:
            return 10
        ' Check if the task ID is known by this node. '
        return task_id in self.tasks

    def is_task_active(self, task_id: str) -> bool:
        if False:
            while True:
                i = 10
        if task_id not in self.tasks_states:
            return False
        return self.tasks_states[task_id].status.is_active()

    def should_wait_for_node(self, task_id: str, node_id: str, offer_hash: str) -> bool:
        if False:
            while True:
                i = 10
        ' Check if the node has too many tasks assigned already '
        if not self.is_my_task(task_id):
            logger.debug('Not my task. task_id=%s, node=%s', task_id, short_node_id(node_id))
            return False
        task = self.tasks[task_id]
        verdict = task.should_accept_client(node_id, offer_hash)
        logger.debug('Should accept client verdict. verdict=%s, task=%s, node=%s', verdict, task_id, short_node_id(node_id))
        if verdict == AcceptClientVerdict.SHOULD_WAIT:
            logger.warning('Waiting for results from %s on %s', short_node_id(node_id), task_id)
            return True
        elif verdict == AcceptClientVerdict.REJECTED:
            logger.warning('Client has failed on subtask within this task and is banned from it. node_id=%s, task_id=%s', short_node_id(node_id), task_id)
            return True
        return False

    def check_next_subtask(self, task_id: str, price: int) -> bool:
        if False:
            print('Hello World!')
        'Check next subtask from task <task_id> with given price limit'
        logger.debug('check_next_subtask(%r, %r)', task_id, price)
        if not self.is_my_task(task_id):
            logger.info('Cannot find task in my tasks. task_id=%s', task_id)
            return False
        task = self.tasks[task_id]
        if task.header.max_price < price:
            logger.debug('Requested price too high. task_id=%(task_id)s, task.header.max_price=%(task_price)s, requested_price=%(price)s', {'task_id': task_id, 'price': price, 'task_price': task.header.max_price})
            return False
        return True

    def copy_results(self, old_task_id: str, new_task_id: str, subtask_ids_to_copy: Iterable[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('copy_results. old_task_id=%r, new_task_id=%r', old_task_id, new_task_id)
        try:
            old_task = self.tasks[old_task_id]
            new_task = self.tasks[new_task_id]
            assert isinstance(old_task, CoreTask)
            assert isinstance(new_task, CoreTask)
        except (KeyError, AssertionError):
            logger.exception('Cannot copy results from task %r to %r', old_task_id, new_task_id)
            return
        subtasks_to_copy = {subtask['start_task']: subtask for subtask in map(lambda id_: old_task.subtasks_given[id_], subtask_ids_to_copy)}
        new_subtasks_ids = []
        while new_task.needs_computation():
            extra_data = new_task.query_extra_data(0, node_id=str(uuid.uuid4()))
            new_subtask_id = extra_data.ctd['subtask_id']
            self.subtask2task_mapping[new_subtask_id] = new_task_id
            self.__add_subtask_to_tasks_states(node_id='', price=0, ctd=extra_data.ctd)
            new_subtasks_ids.append(new_subtask_id)
        logger.debug('copy_results. new_subtasks_ids=%r', new_subtasks_ids)
        for new_subtask_id in new_subtasks_ids:
            self.tasks_states[new_task_id].subtask_states[new_subtask_id].status = SubtaskStatus.failure
            new_task.subtasks_given[new_subtask_id]['status'] = SubtaskStatus.failure
            new_task.subtask_status_updated(new_subtask_id)
        new_task.num_failed_subtasks = new_task.get_total_tasks() - len(subtasks_to_copy)

        def handle_copy_error(subtask_id, error):
            if False:
                while True:
                    i = 10
            logger.error('Cannot copy result of subtask %r: %r', subtask_id, error)
            self.restart_subtask(subtask_id)
        for (new_subtask_id, new_subtask) in new_task.subtasks_given.items():
            old_subtask = subtasks_to_copy.get(new_subtask['start_task'])
            if old_subtask:
                deferred = self._copy_subtask_results(old_task=old_task, new_task=new_task, old_subtask=old_subtask, new_subtask=new_subtask)
                deferred.addErrback(partial(handle_copy_error, new_subtask_id))
            else:
                self.restart_subtask(new_subtask_id)

    def _copy_subtask_results(self, old_task: CoreTask, new_task: CoreTask, old_subtask: dict, new_subtask: dict) -> Deferred:
        if False:
            while True:
                i = 10
        old_task_id = old_task.header.task_id
        new_task_id = new_task.header.task_id
        assert isinstance(old_task.tmp_dir, str)
        assert isinstance(new_task.tmp_dir, str)
        old_tmp_dir = Path(old_task.tmp_dir)
        new_tmp_dir = Path(new_task.tmp_dir)
        old_subtask_id = old_subtask['subtask_id']
        new_subtask_id = new_subtask['subtask_id']

        def copy_and_extract_zips():
            if False:
                return 10
            old_result_path = old_tmp_dir / '{}.{}.zip'.format(old_task_id, old_subtask_id)
            new_result_path = new_tmp_dir / '{}.{}.zip'.format(new_task_id, new_subtask_id)
            shutil.copy(old_result_path, new_result_path)
            subtask_result_dir = new_tmp_dir / new_subtask_id
            os.makedirs(subtask_result_dir)
            with ZipFile(new_result_path, 'r') as zf:
                zf.extractall(subtask_result_dir)
                return [str(subtask_result_dir / name) for name in zf.namelist() if name != '.package_desc']

        def after_results_extracted(results):
            if False:
                for i in range(10):
                    print('nop')
            new_task.copy_subtask_results(new_subtask_id, old_subtask, TaskResult(files=results))
            self.__set_subtask_state_finished(new_subtask_id)
            new_task.subtask_status_updated(new_subtask_id)
            self.notice_task_updated(task_id=new_task_id, subtask_id=new_subtask_id, op=SubtaskOp.FINISHED)
        deferred = deferToThread(copy_and_extract_zips)
        deferred.addCallback(after_results_extracted)
        return deferred

    def get_tasks_headers(self):
        if False:
            print('Hello World!')
        ret = []
        for (tid, task) in self.tasks.items():
            status = self.tasks_states[tid].status
            if task.needs_computation() and status.is_active():
                ret.append(task.header)
        return ret

    def get_trust_mod(self, subtask_id):
        if False:
            print('Hello World!')
        if subtask_id in self.subtask2task_mapping:
            task_id = self.subtask2task_mapping[subtask_id]
            return self.tasks[task_id].get_trust_mod(subtask_id)
        logger.warning('Cannot get trust mod for subtask_id=%s', subtask_id)
        return 0

    def update_task_signatures(self):
        if False:
            print('Hello World!')
        for task in list(self.tasks.values()):
            self.sign_task_header(task.header)

    def sign_task_header(self, task_header):
        if False:
            i = 10
            return i + 15
        task_header.sign(private_key=self.keys_auth._private_key)

    def verify_subtask(self, subtask_id):
        if False:
            while True:
                i = 10
        logger.debug('verify_subtask. subtask_id=%r', subtask_id)
        if subtask_id in self.subtask2task_mapping:
            task_id = self.subtask2task_mapping[subtask_id]
            return self.tasks[task_id].verify_subtask(subtask_id)
        return False

    def get_node_id_for_subtask(self, subtask_id):
        if False:
            while True:
                i = 10
        if subtask_id not in self.subtask2task_mapping:
            return None
        task = self.subtask2task_mapping[subtask_id]
        subtask_state = self.tasks_states[task].subtask_states[subtask_id]
        return subtask_state.node_id

    @handle_subtask_key_error
    def computed_task_received(self, subtask_id: str, result: TaskResult, verification_finished: Callable[[], None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.debug('Computed task received. subtask_id=%s', subtask_id)
        task_id: str = self.subtask2task_mapping[subtask_id]
        subtask_state: SubtaskState = self.tasks_states[task_id].subtask_states[subtask_id]
        subtask_status: SubtaskStatus = subtask_state.status
        if not subtask_status.is_computed():
            logger.warning('Result for subtask with invalid status. subtask_id=%s, status=%s', subtask_id, subtask_status.value)
            self.notice_task_updated(task_id, subtask_id=subtask_id, op=OtherOp.UNEXPECTED)
            verification_finished()
            return
        subtask_state.status = SubtaskStatus.verifying

        @TaskManager.handle_generic_key_error
        def verification_finished_():
            if False:
                while True:
                    i = 10
            logger.debug('Verification finished. subtask_id=%s', subtask_id)
            ss = self.__set_subtask_state_finished(subtask_id)
            if not self.tasks[task_id].verify_subtask(subtask_id):
                logger.debug('Subtask %r not accepted\n', subtask_id)
                ss.status = SubtaskStatus.failure
                ss.stderr = '[GOLEM] Not accepted'
                self.notice_task_updated(task_id, subtask_id=subtask_id, op=SubtaskOp.NOT_ACCEPTED)
                verification_finished()
                return
            self.notice_task_updated(task_id, subtask_id=subtask_id, op=SubtaskOp.FINISHED)
            verification_finished()
            if self.tasks_states[task_id].status.is_active():
                if not self.tasks[task_id].finished_computation():
                    self.tasks_states[task_id].status = TaskStatus.computing
                elif self.tasks[task_id].verify_task():
                    logger.info('Task finished! task_id=%r', task_id)
                    self.tasks_states[task_id].status = TaskStatus.finished
                    self.notice_task_updated(task_id, op=TaskOp.FINISHED)
                else:
                    logger.warning('Task finished but was not accepted. task_id=%r', task_id)
                    self.notice_task_updated(task_id, op=TaskOp.NOT_ACCEPTED)
        self.notice_task_updated(task_id, subtask_id=subtask_id, op=SubtaskOp.VERIFYING)
        self.tasks[task_id].computation_finished(subtask_id, result, verification_finished_)

    @handle_subtask_key_error
    def __set_subtask_state_finished(self, subtask_id: str) -> SubtaskState:
        if False:
            i = 10
            return i + 15
        task_id = self.subtask2task_mapping[subtask_id]
        ss = self.tasks_states[task_id].subtask_states[subtask_id]
        ss.progress = 1.0
        ss.status = SubtaskStatus.finished
        ss.stdout = self.tasks[task_id].get_stdout(subtask_id)
        ss.stderr = self.tasks[task_id].get_stderr(subtask_id)
        ss.results = self.tasks[task_id].get_results(subtask_id)
        return ss

    @handle_subtask_key_error
    def task_computation_failure(self, subtask_id: str, err: object, ban_node: bool=True) -> bool:
        if False:
            for i in range(10):
                print('nop')
        task_id = self.subtask2task_mapping[subtask_id]
        task = self.tasks[task_id]
        task_state = self.tasks_states[task_id]
        subtask_state = task_state.subtask_states[subtask_id]
        subtask_status = subtask_state.status
        if not subtask_status.is_computed():
            logger.warning("Subtask %s status cannot be changed from '%s' to '%s'", subtask_id, subtask_status.value, SubtaskStatus.failure)
            self.notice_task_updated(task_id, subtask_id=subtask_id, op=OtherOp.UNEXPECTED)
            return False
        task.computation_failed(subtask_id, ban_node)
        subtask_state.progress = 1.0
        subtask_state.status = SubtaskStatus.failure
        subtask_state.stderr = str(err)
        self.notice_task_updated(task_id, subtask_id=subtask_id, op=SubtaskOp.FAILED)
        return True

    @handle_subtask_key_error
    def task_computation_cancelled(self, subtask_id: str, err: message.tasks.CannotComputeTask.REASON, timeout: float) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if err is message.tasks.CannotComputeTask.REASON.OfferCancelled:
            self.restart_subtask(subtask_id, new_status=SubtaskStatus.cancelled)
            return True
        task_id = self.subtask2task_mapping[subtask_id]
        task_state = self.tasks_states[task_id]
        subtask_state = task_state.subtask_states[subtask_id]
        ban_node = subtask_state.time_started + timeout < time.time()
        return self.task_computation_failure(subtask_id, f"Task computation rejected: {(err.value if err else 'unknown')}", ban_node)

    def task_result_incoming(self, subtask_id):
        if False:
            print('Hello World!')
        try:
            task_id = self.subtask2task_mapping[subtask_id]
        except KeyError:
            logger.error('Unknown subtask. subtask_id=%s', subtask_id)
            return
        try:
            task = self.tasks[task_id]
        except KeyError:
            logger.error('Unknown task. task_id=%s', task_id)
            return
        subtask_state = self.tasks_states[task_id].subtask_states[subtask_id]
        task.result_incoming(subtask_id)
        subtask_state.status = SubtaskStatus.downloading
        self.notice_task_updated(task_id, subtask_id=subtask_id, op=SubtaskOp.RESULT_DOWNLOADING)

    def check_timeouts(self):
        if False:
            for i in range(10):
                print('nop')
        nodes_with_timeouts = []
        for t in list(self.tasks.values()):
            th = t.header
            if not self.tasks_states[th.task_id].status.is_active():
                continue
            cur_time = int(get_timestamp_utc())
            ts = self.tasks_states[th.task_id]
            for s in list(ts.subtask_states.values()):
                if s.status.is_computed():
                    if cur_time > s.deadline:
                        logger.info('Subtask %r dies with status %r', s.subtask_id, s.status.value)
                        s.status = SubtaskStatus.timeout
                        nodes_with_timeouts.append(s.node_id)
                        t.computation_failed(s.subtask_id)
                        s.stderr = '[GOLEM] Timeout'
                        self.notice_task_updated(th.task_id, subtask_id=s.subtask_id, op=SubtaskOp.TIMEOUT)
            if cur_time > th.deadline:
                logger.info('Task %r dies', th.task_id)
                self.tasks_states[th.task_id].status = TaskStatus.timeout
                self.notice_task_updated(th.task_id, op=TaskOp.TIMEOUT)
                self._try_remove_task_output_dir(t.task_definition)
        return nodes_with_timeouts

    def get_progresses(self):
        if False:
            print('Hello World!')
        tasks_progresses = {}
        for t in list(self.tasks.values()):
            task_id = t.header.task_id
            task_state = self.tasks_states[task_id]
            task_status = task_state.status
            in_progress = not TaskStatus.is_completed(task_status)
            logger.info('Collecting progress %r %r %r', task_id, task_status, in_progress)
            if in_progress:
                ltss = LocalTaskStateSnapshot(task_id, t.get_total_tasks(), t.get_active_tasks(), t.get_progress())
                tasks_progresses[task_id] = ltss
        return tasks_progresses

    @handle_task_key_error
    def assert_task_can_be_restarted(self, task_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        task_state = self.tasks_states[task_id]
        if task_state.status == TaskStatus.restarted:
            raise self.AlreadyRestartedError()

    @handle_task_key_error
    def put_task_in_restarted_state(self, task_id, clear_tmp=True):
        if False:
            i = 10
            return i + 15
        "\n        When restarting task, it's put in a final state 'restarted' and\n        a new one is created.\n        "
        self.assert_task_can_be_restarted(task_id)
        if clear_tmp:
            self.dir_manager.clear_temporary(task_id)
        task_state = self.tasks_states[task_id]
        task_state.status = TaskStatus.restarted
        for ss in self.tasks_states[task_id].subtask_states.values():
            if ss.status != SubtaskStatus.failure:
                ss.status = SubtaskStatus.restarted
        logger.info('Task %s put into restarted state', task_id)
        self.notice_task_updated(task_id, op=TaskOp.RESTARTED)

    @handle_task_key_error
    def put_task_in_failed_state(self, task_id: str, task_status=TaskStatus.errorCreating) -> None:
        if False:
            return 10
        assert not task_status.is_active()
        task_state = self.tasks_states[task_id]
        if task_state.status.is_completed():
            logger.debug("Task is already completed. Won't change status. current_status=%(current_status)s, refused_status=%(refused_status)s", {'current_status': task_state.status, 'refused_status': task_status})
            return
        task_state.status = task_status
        logger.info('Task %s put into failed state. task_status=%s', task_id, task_state)
        self.notice_task_updated(task_id, op=TaskOp.ABORTED)

    @handle_subtask_key_error
    def restart_subtask(self, subtask_id: str, new_status: SubtaskStatus=SubtaskStatus.restarted):
        if False:
            print('Hello World!')
        task_id = self.subtask2task_mapping[subtask_id]
        logger.debug('Restart subtask. subtask_id=%s, new_status=%s, task_id=%s', subtask_id, new_status, task_id)
        self.tasks[task_id].restart_subtask(subtask_id, new_state=new_status)
        task_state = self.tasks_states[task_id]
        task_state.status = TaskStatus.computing
        subtask_state = task_state.subtask_states[subtask_id]
        subtask_state.status = new_status
        subtask_state.stderr = f'[GOLEM] {new_status.value}'
        self.notice_task_updated(task_id, subtask_id=subtask_id, op=SubtaskOp.RESTARTED)

    @handle_task_key_error
    def abort_task(self, task_id):
        if False:
            while True:
                i = 10
        self.tasks[task_id].abort()
        self.tasks_states[task_id].status = TaskStatus.aborted
        for sub in list(self.tasks_states[task_id].subtask_states.values()):
            del self.subtask2task_mapping[sub.subtask_id]
        self.tasks_states[task_id].subtask_states.clear()
        self.notice_task_updated(task_id, op=TaskOp.ABORTED)

    @rpc_utils.expose('comp.task.subtasks.frames')
    @handle_task_key_error
    def get_output_states(self, task_id):
        if False:
            return 10
        return self.tasks[task_id].get_output_states()

    @handle_task_key_error
    def delete_task(self, task_id):
        if False:
            while True:
                i = 10
        for sub in list(self.tasks_states[task_id].subtask_states.values()):
            del self.subtask2task_mapping[sub.subtask_id]
        self.tasks_states[task_id].subtask_states.clear()
        self.tasks[task_id].unregister_listener(self)
        del self.tasks[task_id]
        del self.tasks_states[task_id]
        self.dir_manager.clear_temporary(task_id)
        self.remove_dump(task_id)
        if self.finished_cb:
            self.finished_cb()

    @handle_task_key_error
    def query_task_state(self, task_id):
        if False:
            i = 10
            return i + 15
        ts = self.tasks_states[task_id]
        t = self.tasks[task_id]
        ts.progress = t.get_progress()
        ts.elapsed_time = time.time() - ts.time_started
        if ts.progress > 0.0:
            proportion = ts.elapsed_time / ts.progress
            ts.remaining_time = proportion - ts.elapsed_time
        else:
            ts.remaining_time = None
        t.update_task_state(ts)
        return ts

    def subtask_to_task(self, subtask_id: str, local_role: model.Actor) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if local_role == model.Actor.Provider:
            return self.comp_task_keeper.subtask_to_task.get(subtask_id)
        elif local_role == model.Actor.Requestor:
            return self.subtask2task_mapping.get(subtask_id)
        return None

    def get_subtasks(self, task_id) -> Optional[List[str]]:
        if False:
            return 10
        '\n        Get all subtasks related to given task id\n        :param task_id: Task ID\n        :return: list of all subtasks related with @task_id or None\n                 if @task_id is not known\n        '
        task_state = self.tasks_states.get(task_id)
        if not task_state:
            return None
        subtask_states = list(task_state.subtask_states.values())
        return [subtask_state.subtask_id for subtask_state in subtask_states]

    @rpc_utils.expose('comp.task.verify_subtask')
    def external_verify_subtask(self, subtask_id, verdict):
        if False:
            print('Hello World!')
        logger.info('external_verify_subtask. subtask_id=%r', subtask_id)
        if subtask_id in self.subtask2task_mapping:
            task_id = self.subtask2task_mapping[subtask_id]
            return self.tasks[task_id].external_verify_subtask(subtask_id, verdict)
        else:
            raise ValueError('Not my subtask')

    def get_frame_subtasks(self, task_id: str, frame) -> Optional[FrozenSet[str]]:
        if False:
            print('Hello World!')
        task: Optional[Task] = self.tasks.get(task_id)
        if not task:
            return None
        if not isinstance(task, CoreTask):
            return None
        return frozenset(task.get_subtasks(frame).keys())

    def get_task_id(self, subtask_id):
        if False:
            return 10
        return self.subtask2task_mapping[subtask_id]

    def get_task_dict(self, task_id) -> Optional[Dict]:
        if False:
            while True:
                i = 10
        task = self.tasks.get(task_id)
        if not task:
            return None
        task_type_name = task.task_definition.task_type.lower()
        task_type = self.task_types[task_type_name]
        state = self.query_task_state(task.header.task_id)
        dictionary = {'preview': task_type.get_preview(task, single=True)}
        return update_dict(dictionary, task.to_dictionary(), state.to_dictionary(), self.get_task_definition_dict(task))

    def get_tasks_dict(self) -> List[Dict]:
        if False:
            return 10
        task_ids = list(self.tasks.keys())
        mapped = map(self.get_task_dict, task_ids)
        filtered = filter(None, mapped)
        return list(filtered)

    def get_subtask_dict(self, subtask_id):
        if False:
            i = 10
            return i + 15
        task_id = self.subtask2task_mapping[subtask_id]
        task_state = self.tasks_states[task_id]
        subtask = task_state.subtask_states[subtask_id]
        return subtask.to_dict()

    def get_subtasks_dict(self, task_id):
        if False:
            return 10
        task_state = self.tasks_states[task_id]
        subtasks = task_state.subtask_states
        if subtasks:
            return [subtask.to_dict() for subtask in subtasks.values()]
        return None

    @rpc_utils.expose('comp.task.subtasks.borders')
    def get_subtasks_borders(self, task_id, part=1):
        if False:
            while True:
                i = 10
        task = self.tasks[task_id]
        task_type_name = task.task_definition.task_type.lower()
        task_type = self.task_types[task_type_name]
        subtasks_count = task.get_total_tasks()
        return {to_unicode(subtask_id): task_type.get_task_border(extra_data, task.task_definition, subtasks_count, as_path=True) for (subtask_id, extra_data) in task.get_subtasks(part).items()}

    def get_task_preview(self, task_id, single=False):
        if False:
            return 10
        task = self.tasks[task_id]
        task_type_name = task.task_definition.task_type.lower()
        task_type = self.task_types[task_type_name]
        return task_type.get_preview(task, single=single)

    def add_comp_task_request(self, task_header: message.tasks.TaskHeader, budget: int, performance: float, num_subtasks: int):
        if False:
            while True:
                i = 10
        ' Add a header of a task which this node may try to compute '
        self.comp_task_keeper.add_request(task_header, budget, performance, num_subtasks)

    def __add_subtask_to_tasks_states(self, node_id, ctd, price: int):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('add_subtask_to_tasks_states(%r, %r)', node_id, ctd)
        node_info = nodeskeeper.get(node_id)
        ss = SubtaskState(subtask_id=ctd['subtask_id'], node_id=node_id, node_name=node_info.node_name if node_info else '', price=price, deadline=ctd['deadline'], extra_data=ctd['extra_data'])
        self.tasks_states[ctd['task_id']].subtask_states[ctd['subtask_id']] = ss

    def notify_update_task(self, task_id):
        if False:
            print('Hello World!')
        self.notice_task_updated(task_id)

    @handle_task_key_error
    def notice_task_updated(self, task_id: str, subtask_id: Optional[str]=None, op: Optional[Operation]=None, persist: bool=True):
        if False:
            print('Hello World!')
        'Called when a task is modified, saves the task and\n        propagates information\n\n        Whenever task is changed `notice_task_updated` should be called\n        to save the task - if the change is save-worthy, as specified\n        by the `persist` parameter - and propagate information about\n        changed task to other parts of the system.\n\n        Most of the calls are save-worthy, but a minority is not: for\n        instance when the work offer is received, the task does not\n        change so saving it does not make sense, but it still makes\n        sense to let other parts of the system know about the change.\n        Also, when a number of minor changes are always followed by a\n        major one, as it is with restarting a frame task, it does not\n        make sense to store all the partial changes, so only the\n        final one is considered save-worthy.\n\n        :param str task_id: id of the updated task\n        :param str subtask_id: if the operation done on the\n          task is related to a subtask, id of that subtask\n        :param Operation op: performed operation\n        :param bool persist: should the task be persisted now\n        '
        logger.debug('Notice task updated. task_id=%s, subtask_id=%s,op=%s, persist=%s', task_id, subtask_id, op, persist)
        if persist:
            self.dump_task(task_id)
        task_state = self.tasks_states.get(task_id)
        dispatcher.send(signal='golem.taskmanager', event='task_status_updated', task_id=task_id, task_state=task_state, subtask_id=subtask_id, op=op)
        self._stop_timers(task_id, subtask_id, op)
        self._update_subtask_statistics(task_id, subtask_id, op)
        if self.finished_cb and persist and op and op.task_related() and op.is_completed():
            self.finished_cb()

    def _stop_timers(self, task_id: str, subtask_id: Optional[str]=None, op: Optional[Operation]=None):
        if False:
            return 10
        if subtask_id and isinstance(op, SubtaskOp) and op.is_completed():
            ProviderComputeTimers.finish(subtask_id)
        elif isinstance(op, TaskOp) and op in (TaskOp.ABORTED, TaskOp.TIMEOUT, TaskOp.RESTARTED):
            for _subtask_id in self.tasks_states[task_id].subtask_states:
                ProviderComputeTimers.finish(_subtask_id)

    def _update_subtask_statistics(self, task_id: str, subtask_id: Optional[str]=None, op: Optional[Operation]=None) -> None:
        if False:
            print('Hello World!')
        if not (subtask_id and isinstance(op, SubtaskOp) and op.is_completed()):
            return
        try:
            self._update_provider_statistics(task_id, subtask_id, op)
        except (KeyError, ValueError) as e:
            logger.error('Unable to update statistics for subtask %s: %r', subtask_id, e)
        try:
            self._update_provider_reputation(task_id, subtask_id, op)
        except (KeyError, ValueError) as e:
            logger.error('Unable to update reputation for subtask %s: %r', subtask_id, e)
        ProviderComputeTimers.remove(subtask_id)

    def _update_provider_statistics(self, task_id: str, subtask_id: str, op: SubtaskOp) -> None:
        if False:
            print('Hello World!')
        logger.debug('_update_provider_statistics. task_id=%r, subtask_id=%r,op=%r', task_id, subtask_id, op)
        header = self.tasks[task_id].header
        subtask_state = self.tasks_states[task_id].subtask_states[subtask_id]
        computation_price = calculate_subtask_payment(subtask_state.price, header.subtask_timeout)
        computation_time = ProviderComputeTimers.time(subtask_id)
        if not computation_time:
            logger.warning('Could not obtain computation time for subtask: %r', subtask_id)
            return
        computation_time = int(round(computation_time))
        dispatcher.send(signal='golem.subtask', event='finished', timed_out=op == SubtaskOp.TIMEOUT, subtask_count=header.subtasks_count, subtask_timeout=header.subtask_timeout, subtask_price=computation_price, subtask_computation_time=computation_time)

    def _update_provider_reputation(self, task_id: str, subtask_id: str, op: SubtaskOp) -> None:
        if False:
            for i in range(10):
                print('nop')
        timeout = self.tasks[task_id].header.subtask_timeout
        subtask_state = self.tasks_states[task_id].subtask_states[subtask_id]
        node_id = subtask_state.node_id
        logger.debug('_update_provider_reputation. task_id=%r, subtask_id=%r,op=%r, subtask_state=%r', task_id, subtask_id, op, subtask_state)
        update_provider_efficacy(node_id, op)
        computation_time = ProviderComputeTimers.time(subtask_id)
        if not computation_time:
            logger.warning('Could not obtain computation time for subtask: %r', subtask_id)
            return
        update_provider_efficiency(node_id, timeout, computation_time)

    @staticmethod
    def get_provider_market_strategy_for_env(env_id: str) -> Type[ProviderMarketStrategy]:
        if False:
            return 10
        if env_id == WasmTaskEnvironment.ENV_ID:
            return ProviderWasmMarketStrategy
        return ProviderBrassMarketStrategy