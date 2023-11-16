from typing import Dict, List, Iterator, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import time
from collections import defaultdict
import ray
from ray.exceptions import RayTaskError, RayError
from ray.workflow.common import WorkflowRef, WorkflowExecutionMetadata, WorkflowStatus, TaskID
from ray.workflow.exceptions import WorkflowCancellationError, WorkflowExecutionError
from ray.workflow.task_executor import get_task_executor, _BakedWorkflowInputs
from ray.workflow.workflow_state import WorkflowExecutionState, TaskExecutionMetadata, Task
if TYPE_CHECKING:
    from ray.workflow.workflow_context import WorkflowTaskContext
    from ray.workflow.workflow_storage import WorkflowStorage
logger = logging.getLogger(__name__)

class WorkflowExecutor:

    def __init__(self, state: WorkflowExecutionState):
        if False:
            for i in range(10):
                print('nop')
        "The core logic of executing a workflow.\n\n        This class is responsible for:\n\n        - Dependency resolving.\n        - Task scheduling.\n        - Reference counting.\n        - Garbage collection.\n        - Continuation handling and scheduling.\n        - Error handling.\n        - Responding callbacks.\n\n        It borrows some design of event loop in asyncio,\n        e.g., 'run_until_complete'.\n\n        Args:\n            state: The initial state of the workflow.\n        "
        self._state = state
        self._completion_queue = asyncio.Queue()
        self._task_done_callbacks: Dict[TaskID, List[asyncio.Future]] = defaultdict(list)

    def is_running(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'The state is running, if there are tasks to be run or running tasks.'
        return bool(self._state.frontier_to_run or self._state.running_frontier)

    def get_state(self) -> WorkflowExecutionState:
        if False:
            return 10
        return self._state

    @property
    def output_task_id(self) -> TaskID:
        if False:
            i = 10
            return i + 15
        return self._state.output_task_id

    async def run_until_complete(self, job_id: str, context: 'WorkflowTaskContext', wf_store: 'WorkflowStorage'):
        """Drive the state util it completes.

        Args:
            job_id: The Ray JobID for logging properly.
            context: The context of workflow execution.
            wf_store: The store for the workflow.

        # TODO(suquark): move job_id inside context
        """
        workflow_id = context.workflow_id
        wf_store.update_workflow_status(WorkflowStatus.RUNNING)
        logger.info(f'Workflow job [id={workflow_id}] started.')
        self._state.construct_scheduling_plan(self._state.output_task_id)
        self._state.init_context(context)
        while self.is_running():
            queued_tasks = self._poll_queued_tasks()
            for task_id in queued_tasks:
                self._submit_ray_task(task_id, job_id=job_id)
                self._post_process_submit_task(task_id, wf_store)
            self._garbage_collect()
            ready_futures = await self._poll_ready_tasks()
            await asyncio.gather(*[self._handle_ready_task(fut, workflow_id=workflow_id, wf_store=wf_store) for fut in ready_futures])
            del ready_futures
        wf_store.update_workflow_status(WorkflowStatus.SUCCESSFUL)
        logger.info(f"Workflow '{workflow_id}' completes successfully.")
        for (task_id, futures) in self._task_done_callbacks.items():
            err = ValueError(f"The workflow haven't yet produced output of task '{task_id}' after workflow execution completes.")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(err)

    def cancel(self) -> None:
        if False:
            i = 10
            return i + 15
        'Cancel the running workflow.'
        for (fut, workflow_ref) in self._state.running_frontier.items():
            fut.cancel()
            try:
                ray.cancel(workflow_ref.ref, force=True)
            except Exception:
                pass

    def _poll_queued_tasks(self) -> List[TaskID]:
        if False:
            while True:
                i = 10
        tasks = []
        while True:
            task_id = self._state.pop_frontier_to_run()
            if task_id is None:
                break
            tasks.append(task_id)
        return tasks

    def _submit_ray_task(self, task_id: TaskID, job_id: str) -> None:
        if False:
            print('Hello World!')
        'Submit a workflow task as a Ray task.'
        state = self._state
        baked_inputs = _BakedWorkflowInputs(args=state.task_input_args[task_id], workflow_refs=[state.get_input(d) for d in state.upstream_dependencies[task_id]])
        task = state.tasks[task_id]
        executor = get_task_executor(task.options)
        (metadata_ref, output_ref) = executor(task.func_body, state.task_context[task_id], job_id, task_id, baked_inputs, task.options)
        future = asyncio.wrap_future(metadata_ref.future())
        future.add_done_callback(self._completion_queue.put_nowait)
        state.insert_running_frontier(future, WorkflowRef(task_id, ref=output_ref))
        state.task_execution_metadata[task_id] = TaskExecutionMetadata(submit_time=time.time())

    def _post_process_submit_task(self, task_id: TaskID, store: 'WorkflowStorage') -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update dependencies and reference count etc. after task submission.'
        state = self._state
        if task_id in state.continuation_root:
            if state.tasks[task_id].options.checkpoint:
                store.update_continuation_output_link(state.continuation_root[task_id], task_id)
        else:
            for c in state.upstream_dependencies[task_id]:
                state.reference_set[c].remove(task_id)
                if not state.reference_set[c]:
                    del state.reference_set[c]
                    state.free_outputs.add(c)

    def _garbage_collect(self) -> None:
        if False:
            return 10
        'Garbage collect the output refs of tasks.\n\n        Currently, this is done after task submission, because when a task\n        starts, we no longer needs its inputs (i.e. outputs from other tasks).\n\n        # TODO(suquark): We may need to improve garbage collection\n        #  when taking more fault tolerant cases into consideration.\n        '
        state = self._state
        while state.free_outputs:
            gc_task_id = state.free_outputs.pop()
            assert state.get_input(gc_task_id) is not None
            state.output_map.pop(gc_task_id, None)

    async def _poll_ready_tasks(self) -> List[asyncio.Future]:
        cq = self._completion_queue
        ready_futures = []
        rf = await cq.get()
        ready_futures.append(rf)
        while not cq.empty():
            ready_futures.append(cq.get_nowait())
        return ready_futures

    def _iter_callstack(self, task_id: TaskID) -> Iterator[Tuple[TaskID, Task]]:
        if False:
            while True:
                i = 10
        state = self._state
        while task_id in state.task_context and task_id in state.tasks:
            yield (task_id, state.tasks[task_id])
            task_id = state.task_context[task_id].creator_task_id

    def _retry_failed_task(self, workflow_id: str, failed_task_id: TaskID, exc: Exception) -> bool:
        if False:
            i = 10
            return i + 15
        state = self._state
        is_application_error = isinstance(exc, RayTaskError)
        options = state.tasks[failed_task_id].options
        if not is_application_error or options.retry_exceptions:
            if state.task_retries[failed_task_id] < options.max_retries:
                state.task_retries[failed_task_id] += 1
                logger.info(f'Retry [{workflow_id}@{failed_task_id}] ({state.task_retries[failed_task_id]}/{options.max_retries})')
                state.construct_scheduling_plan(failed_task_id)
                return True
        return False

    async def _catch_failed_task(self, workflow_id: str, failed_task_id: TaskID, exc: Exception) -> bool:
        is_application_error = isinstance(exc, RayTaskError)
        exception_catcher = None
        if is_application_error:
            for (t, task) in self._iter_callstack(failed_task_id):
                if task.options.catch_exceptions:
                    exception_catcher = t
                    break
        if exception_catcher is not None:
            logger.info(f"Exception raised by '{workflow_id}@{failed_task_id}' is caught by '{workflow_id}@{exception_catcher}'")
            await self._post_process_ready_task(exception_catcher, metadata=WorkflowExecutionMetadata(), output_ref=WorkflowRef(failed_task_id, ray.put((None, exc))))
            return True
        return False

    async def _handle_ready_task(self, fut: asyncio.Future, workflow_id: str, wf_store: 'WorkflowStorage') -> None:
        """Handle ready task, especially about its exception."""
        state = self._state
        output_ref = state.pop_running_frontier(fut)
        task_id = output_ref.task_id
        try:
            metadata: WorkflowExecutionMetadata = fut.result()
            state.task_execution_metadata[task_id].finish_time = time.time()
            logger.info(f'Task status [{WorkflowStatus.SUCCESSFUL}]\t[{workflow_id}@{task_id}]')
            await self._post_process_ready_task(task_id, metadata, output_ref)
        except asyncio.CancelledError:
            wf_store.update_workflow_status(WorkflowStatus.CANCELED)
            logger.warning(f"Workflow '{workflow_id}' is cancelled.")
            err = WorkflowCancellationError(workflow_id)
            self._broadcast_exception(err)
            raise err from None
        except Exception as e:
            if isinstance(e, RayTaskError):
                reason = 'an exception raised by the task'
            elif isinstance(e, RayError):
                reason = 'a system error'
            else:
                reason = 'an unknown error'
            logger.error(f'Task status [{WorkflowStatus.FAILED}] due to {reason}.\t[{workflow_id}@{task_id}]')
            is_application_error = isinstance(e, RayTaskError)
            options = state.tasks[task_id].options
            if not is_application_error or options.retry_exceptions:
                if state.task_retries[task_id] < options.max_retries:
                    state.task_retries[task_id] += 1
                    logger.info(f'Retry [{workflow_id}@{task_id}] ({state.task_retries[task_id]}/{options.max_retries})')
                    state.construct_scheduling_plan(task_id)
                    return
            exception_catcher = None
            if is_application_error:
                for (t, task) in self._iter_callstack(task_id):
                    if task.options.catch_exceptions:
                        exception_catcher = t
                        break
            if exception_catcher is not None:
                logger.info(f"Exception raised by '{workflow_id}@{task_id}' is caught by '{workflow_id}@{exception_catcher}'")
                await self._post_process_ready_task(exception_catcher, metadata=WorkflowExecutionMetadata(), output_ref=WorkflowRef(task_id, ray.put((None, e))))
                return
            wf_store.update_workflow_status(WorkflowStatus.FAILED)
            logger.error(f"Workflow '{workflow_id}' failed due to {e}")
            err = WorkflowExecutionError(workflow_id)
            err.__cause__ = e
            self._broadcast_exception(err)
            raise err

    async def _post_process_ready_task(self, task_id: TaskID, metadata: WorkflowExecutionMetadata, output_ref: WorkflowRef) -> None:
        state = self._state
        state.task_retries.pop(task_id, None)
        if metadata.is_output_workflow:
            sub_workflow_state: WorkflowExecutionState = await output_ref.ref
            sub_workflow_state.init_context(state.task_context[task_id])
            state.merge_state(sub_workflow_state)
            continuation_task_id = sub_workflow_state.output_task_id
            state.append_continuation(task_id, continuation_task_id)
            if continuation_task_id in self._task_done_callbacks:
                self._task_done_callbacks[state.continuation_root[continuation_task_id]].extend(self._task_done_callbacks.pop(continuation_task_id))
            state.construct_scheduling_plan(sub_workflow_state.output_task_id)
        else:
            target_task_id = state.continuation_root.get(task_id, task_id)
            state.output_map[target_task_id] = output_ref
            if state.tasks[task_id].options.checkpoint:
                state.checkpoint_map[target_task_id] = WorkflowRef(task_id)
            state.done_tasks.add(target_task_id)
            if target_task_id in self._task_done_callbacks:
                for callback in self._task_done_callbacks[target_task_id]:
                    callback.set_result(output_ref)
            for m in state.reference_set[target_task_id]:
                state.pending_input_set[m].remove(target_task_id)
                if not state.pending_input_set[m]:
                    state.append_frontier_to_run(m)

    def _broadcast_exception(self, err: Exception):
        if False:
            while True:
                i = 10
        for (_, futures) in self._task_done_callbacks.items():
            for fut in futures:
                if not fut.done():
                    fut.set_exception(err)

    def get_task_output_async(self, task_id: Optional[TaskID]) -> asyncio.Future:
        if False:
            for i in range(10):
                print('nop')
        'Get the output of a task asynchronously.\n\n        Args:\n            task_id: The ID of task the callback associates with.\n\n        Returns:\n            A callback in the form of a future that associates with the task.\n        '
        state = self._state
        if self._task_done_callbacks[task_id]:
            return self._task_done_callbacks[task_id][0]
        fut = asyncio.Future()
        task_id = state.continuation_root.get(task_id, task_id)
        output = state.get_input(task_id)
        if output is not None:
            fut.set_result(output)
        elif task_id in state.done_tasks:
            fut.set_exception(ValueError(f"Task '{task_id}' is done but neither in memory or in storage could we find its output. It could because its in memory output has been garbage collected and the task did notcheckpoint its output."))
        else:
            self._task_done_callbacks[task_id].append(fut)
        return fut