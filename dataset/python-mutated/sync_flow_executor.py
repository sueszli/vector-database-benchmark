"""Executor for SyncFlows"""
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import RLock
from typing import Callable, List, Optional, Set
from uuid import uuid4
from botocore.exceptions import ClientError
from samcli.lib.providers.exceptions import MissingLocalDefinition
from samcli.lib.sync.exceptions import InfraSyncRequiredError, InvalidRuntimeDefinitionForFunction, MissingFunctionBuildDefinition, MissingPhysicalResourceError, NoLayerVersionsFoundError, SyncFlowException
from samcli.lib.sync.sync_flow import SyncFlow
from samcli.lib.telemetry.event import EventName, EventTracker, EventType
from samcli.lib.utils.colors import Colored
from samcli.lib.utils.lock_distributor import LockDistributor, LockDistributorType
LOG = logging.getLogger(__name__)
HELP_TEXT_FOR_SYNC_INFRA = ' Try sam sync without --code or sam deploy.'

@dataclass(frozen=True, eq=True)
class SyncFlowTask:
    """Data struct for individual SyncFlow execution tasks"""
    sync_flow: SyncFlow
    dedup: bool

@dataclass(frozen=True, eq=True)
class SyncFlowResult:
    """Data struct for SyncFlow results"""
    sync_flow: SyncFlow
    dependent_sync_flows: List[SyncFlow]

@dataclass(frozen=True, eq=True)
class SyncFlowFuture:
    """Data struct for SyncFlow futures"""
    sync_flow: SyncFlow
    future: Future

def default_exception_handler(sync_flow_exception: SyncFlowException) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Default exception handler for SyncFlowExecutor\n    This will try log and parse common SyncFlow exceptions.\n\n    Parameters\n    ----------\n    sync_flow_exception : SyncFlowException\n        SyncFlowException containing exception to be handled and SyncFlow that raised it\n\n    Raises\n    ------\n    exception\n        Unhandled exception\n    '
    exception = sync_flow_exception.exception
    if isinstance(exception, MissingPhysicalResourceError):
        LOG.error('Cannot find resource %s in remote.%s', exception.resource_identifier, HELP_TEXT_FOR_SYNC_INFRA)
    elif isinstance(exception, InfraSyncRequiredError):
        LOG.error('Cannot code sync for %s due to: %s.%s', exception.resource_identifier, exception.reason, HELP_TEXT_FOR_SYNC_INFRA)
    elif isinstance(exception, ClientError) and exception.response.get('Error', dict()).get('Code', '') == 'ResourceNotFoundException':
        LOG.error('Cannot find resource in remote.%s', HELP_TEXT_FOR_SYNC_INFRA)
        LOG.error(exception.response.get('Error', dict()).get('Message', ''))
    elif isinstance(exception, NoLayerVersionsFoundError):
        LOG.error('Cannot find any versions for layer %s.%s', exception.layer_name_arn, HELP_TEXT_FOR_SYNC_INFRA)
    elif isinstance(exception, MissingFunctionBuildDefinition):
        LOG.error('Cannot find build definition for function %s.%s', exception.function_logical_id, HELP_TEXT_FOR_SYNC_INFRA)
    elif isinstance(exception, InvalidRuntimeDefinitionForFunction):
        LOG.error('No Runtime information found for function resource named %s', exception.function_logical_id)
    elif isinstance(exception, MissingLocalDefinition):
        LOG.error('Resource %s does not have %s specified. Skipping the sync.%s', exception.resource_identifier, exception.property_name, HELP_TEXT_FOR_SYNC_INFRA)
    else:
        raise exception

class SyncFlowExecutor:
    """Executor for SyncFlows
    Can be used with ThreadPoolExecutor or ProcessPoolExecutor with/without manager
    """
    _flow_queue: Queue
    _flow_queue_lock: RLock
    _lock_distributor: LockDistributor
    _running_flag: bool
    _color: Colored
    _running_futures: Set[SyncFlowFuture]

    def __init__(self) -> None:
        if False:
            return 10
        self._flow_queue = Queue()
        self._lock_distributor = LockDistributor(LockDistributorType.THREAD)
        self._running_flag = False
        self._flow_queue_lock = RLock()
        self._color = Colored()
        self._running_futures = set()

    def _add_sync_flow_task(self, task: SyncFlowTask) -> None:
        if False:
            while True:
                i = 10
        'Add SyncFlowTask to the queue\n\n        Parameters\n        ----------\n        task : SyncFlowTask\n            SyncFlowTask to be added.\n        '
        with self._flow_queue_lock:
            if task.dedup and task.sync_flow in [task.sync_flow for task in self._flow_queue.queue]:
                LOG.debug('Found the same SyncFlow in queue. Skip adding.')
                return
            task.sync_flow.set_locks_with_distributor(self._lock_distributor)
            self._flow_queue.put(task)

    def add_sync_flow(self, sync_flow: SyncFlow, dedup: bool=True) -> None:
        if False:
            while True:
                i = 10
        'Add a SyncFlow to queue to be executed\n        Locks will be set with LockDistributor\n\n        Parameters\n        ----------\n        sync_flow : SyncFlow\n            SyncFlow to be executed\n        dedup : bool\n            SyncFlow will not be added if this flag is True and has a duplicate in the queue\n        '
        self._add_sync_flow_task(SyncFlowTask(sync_flow, dedup))

    def is_running(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        bool\n            Is executor running\n        '
        return self._running_flag

    def _can_exit(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        bool\n            Can executor be safely exited\n        '
        return not self._running_futures and self._flow_queue.empty()

    def execute(self, exception_handler: Optional[Callable[[SyncFlowException], None]]=default_exception_handler) -> None:
        if False:
            i = 10
            return i + 15
        'Blocking execution of the SyncFlows\n\n        Parameters\n        ----------\n        exception_handler : Optional[Callable[[Exception], None]], optional\n            Function to be called if an exception is raised during the execution of a SyncFlow,\n            by default default_exception_handler.__func__\n        '
        self._running_flag = True
        with ThreadPoolExecutor() as executor:
            self._running_futures.clear()
            while True:
                self._execute_step(executor, exception_handler)
                if self._can_exit():
                    LOG.debug('No more SyncFlows in executor. Stopping.')
                    break
                time.sleep(0.1)
        self._running_flag = False

    def _execute_step(self, executor: ThreadPoolExecutor, exception_handler: Optional[Callable[[SyncFlowException], None]]) -> None:
        if False:
            i = 10
            return i + 15
        'A single step in the execution flow\n\n        Parameters\n        ----------\n        executor : ThreadPoolExecutor\n            THreadPoolExecutor to be used for execution\n        exception_handler : Optional[Callable[[SyncFlowException], None]]\n            Exception handler\n        '
        with self._flow_queue_lock:
            deferred_tasks = list()
            while not self._flow_queue.empty():
                sync_flow_task: SyncFlowTask = self._flow_queue.get()
                sync_flow_future = self._submit_sync_flow_task(executor, sync_flow_task)
                if sync_flow_future:
                    self._running_futures.add(sync_flow_future)
                    LOG.info(self._color.color_log(msg=f'Syncing {sync_flow_future.sync_flow.log_name}...', color='cyan'), extra=dict(markup=True))
                else:
                    deferred_tasks.append(sync_flow_task)
            for task in deferred_tasks:
                self._add_sync_flow_task(task)
        for sync_flow_future in set(self._running_futures):
            if self._handle_result(sync_flow_future, exception_handler):
                self._running_futures.remove(sync_flow_future)

    def _submit_sync_flow_task(self, executor: ThreadPoolExecutor, sync_flow_task: SyncFlowTask) -> Optional[SyncFlowFuture]:
        if False:
            for i in range(10):
                print('nop')
        'Submit SyncFlowTask to be executed by ThreadPoolExecutor\n        and return its future\n\n        Parameters\n        ----------\n        executor : ThreadPoolExecutor\n            THreadPoolExecutor to be used for execution\n        sync_flow_task : SyncFlowTask\n            SyncFlowTask to be executed.\n\n        Returns\n        -------\n        Optional[SyncFlowFuture]\n            Returns SyncFlowFuture generated by the SyncFlowTask.\n            Can be None if the task cannot be executed yet.\n        '
        sync_flow = sync_flow_task.sync_flow
        if sync_flow in [future.sync_flow for future in self._running_futures]:
            return None
        sync_flow_future = SyncFlowFuture(sync_flow=sync_flow, future=executor.submit(SyncFlowExecutor._sync_flow_execute_wrapper, sync_flow))
        return sync_flow_future

    def _handle_result(self, sync_flow_future: SyncFlowFuture, exception_handler: Optional[Callable[[SyncFlowException], None]]) -> bool:
        if False:
            return 10
        'Checks and handles the result of a SyncFlowFuture\n\n        Parameters\n        ----------\n        sync_flow_future : SyncFlowFuture\n            The SyncFlowFuture that needs to be handled\n        exception_handler : Optional[Callable[[SyncFlowException], None]]\n            Exception handler that will be called if an exception is raised within the SyncFlow\n\n        Returns\n        -------\n        bool\n            Returns True if the SyncFlowFuture was finished and successfully handled, False otherwise.\n        '
        future = sync_flow_future.future
        if not future.done():
            return False
        exception = future.exception()
        if exception and isinstance(exception, SyncFlowException) and exception_handler:
            exception_handler(exception)
        else:
            sync_flow_result: SyncFlowResult = future.result()
            for dependent_sync_flow in sync_flow_result.dependent_sync_flows:
                self.add_sync_flow(dependent_sync_flow)
            LOG.info(self._color.color_log(msg=f'Finished syncing {sync_flow_result.sync_flow.log_name}.', color='green'), extra=dict(markup=True))
        return True

    @staticmethod
    def _sync_flow_execute_wrapper(sync_flow: SyncFlow) -> SyncFlowResult:
        if False:
            i = 10
            return i + 15
        'Simple wrapper method for executing SyncFlow and converting all Exceptions into SyncFlowException\n\n        Parameters\n        ----------\n        sync_flow : SyncFlow\n            SyncFlow to be executed\n\n        Returns\n        -------\n        SyncFlowResult\n            SyncFlowResult for the SyncFlow executed\n\n        Raises\n        ------\n        SyncFlowException\n        '
        dependent_sync_flows = []
        sync_types = EventType.get_accepted_values(EventName.SYNC_FLOW_START)
        sync_type: Optional[str] = type(sync_flow).__name__
        thread_id = uuid4()
        if sync_type not in sync_types:
            sync_type = None
        try:
            if sync_type:
                EventTracker.track_event('SyncFlowStart', sync_type, thread_id=thread_id)
            dependent_sync_flows = sync_flow.execute()
        except ClientError as e:
            if e.response.get('Error', dict()).get('Code', '') == 'ResourceNotFoundException':
                raise SyncFlowException(sync_flow, MissingPhysicalResourceError()) from e
            raise SyncFlowException(sync_flow, e) from e
        except Exception as e:
            raise SyncFlowException(sync_flow, e) from e
        finally:
            if sync_type:
                EventTracker.track_event('SyncFlowEnd', sync_type, thread_id=thread_id)
        return SyncFlowResult(sync_flow=sync_flow, dependent_sync_flows=dependent_sync_flows)