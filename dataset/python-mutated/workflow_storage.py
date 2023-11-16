"""
This module is higher-level abstraction of storage directly used by
workflows.
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import TaskID, WorkflowStatus, WorkflowTaskRuntimeOptions
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
logger = logging.getLogger(__name__)
ArgsType = Tuple[List[Any], Dict[str, Any]]
WORKFLOW_ROOT = 'workflows'
OBJECTS_DIR = 'objects'
STEPS_DIR = 'tasks'
STEP_INPUTS_METADATA = 'inputs.json'
STEP_USER_METADATA = 'user_task_metadata.json'
STEP_PRERUN_METADATA = 'pre_task_metadata.json'
STEP_POSTRUN_METADATA = 'post_task_metadata.json'
STEP_OUTPUTS_METADATA = 'outputs.json'
STEP_ARGS = 'args.pkl'
STEP_OUTPUT = 'output.pkl'
STEP_EXCEPTION = 'exception.pkl'
STEP_FUNC_BODY = 'func_body.pkl'
CLASS_BODY = 'class_body.pkl'
WORKFLOW_META = 'workflow_meta.json'
WORKFLOW_USER_METADATA = 'user_run_metadata.json'
WORKFLOW_PRERUN_METADATA = 'pre_run_metadata.json'
WORKFLOW_POSTRUN_METADATA = 'post_run_metadata.json'
WORKFLOW_PROGRESS = 'progress.json'
WORKFLOW_STATUS_DIR = '__status__'
WORKFLOW_STATUS_DIRTY_DIR = 'dirty'
DUPLICATE_NAME_COUNTER = 'duplicate_name_counter'

@dataclass
class TaskInspectResult:
    output_object_valid: bool = False
    output_task_id: Optional[TaskID] = None
    args_valid: bool = False
    func_body_valid: bool = False
    workflow_refs: Optional[List[str]] = None
    task_options: Optional[WorkflowTaskRuntimeOptions] = None
    task_raised_exception: bool = False

    def is_recoverable(self) -> bool:
        if False:
            while True:
                i = 10
        return self.output_object_valid or self.output_task_id or (self.args_valid and self.workflow_refs is not None and self.func_body_valid)

class WorkflowIndexingStorage:
    """Access and maintenance the indexing of workflow status.

    It runs a protocol that guarantees we can recover from any interrupted
    status updating. This protocol is **not thread-safe** for updating the
    status of the same workflow, currently it is executed by workflow management
    actor with a single thread.

    Here is how the protocol works:

    Update the status of a workflow
    1. Load workflow status from workflow data. If it is the same as the new status,
       return.
    2. Check if the workflow status updating is dirty. If it is, fix the
       workflow status; otherwise, mark the workflow status updating dirty.
    3. Update status in the workflow metadata.
    4. Insert the workflow ID key in the status indexing directory of the new status.
    5. Delete the workflow ID key in the status indexing directory of
       the previous status.
    6. Remove the workflow status updating dirty mark.

    Load a status of a workflow
    1. Read the status of the workflow from the workflow metadata.
    2. Return the status.

    List the status of all workflows
    1. Get status of all workflows by listing workflow ID keys in each workflow
       status indexing directory.
    2. List all workflows with dirty updating status. Get their status from
       workflow data. Override the status of the corresponding workflow.
    3. Return all the status.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._storage = storage.get_client(WORKFLOW_ROOT)

    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus):
        if False:
            return 10
        'Update the status of the workflow.\n        Try fixing indexing if workflow status updating was marked dirty.\n\n        This method is NOT thread-safe. It is handled by the workflow management actor.\n        '
        prev_status = self.load_workflow_status(workflow_id)
        if prev_status != status:
            if self._storage.get_info(self._key_workflow_status_dirty(workflow_id)) is not None:
                self._storage.put(self._key_workflow_with_status(workflow_id, prev_status), b'')
                for s in WorkflowStatus:
                    if s != prev_status:
                        self._storage.delete(self._key_workflow_with_status(workflow_id, s))
            else:
                self._storage.put(self._key_workflow_status_dirty(workflow_id), b'')
            self._storage.put(self._key_workflow_metadata(workflow_id), json.dumps({'status': status.value}).encode())
            self._storage.put(self._key_workflow_with_status(workflow_id, status), b'')
            if prev_status is not WorkflowStatus.NONE:
                self._storage.delete(self._key_workflow_with_status(workflow_id, prev_status))
            self._storage.delete(self._key_workflow_status_dirty(workflow_id))

    def load_workflow_status(self, workflow_id: str):
        if False:
            print('Hello World!')
        'Load the committed workflow status.'
        raw_data = self._storage.get(self._key_workflow_metadata(workflow_id))
        if raw_data is not None:
            metadata = json.loads(raw_data)
            return WorkflowStatus(metadata['status'])
        return WorkflowStatus.NONE

    def list_workflow(self, status_filter: Optional[Set[WorkflowStatus]]=None) -> List[Tuple[str, WorkflowStatus]]:
        if False:
            i = 10
            return i + 15
        'List workflow status. Override status of the workflows whose status updating\n        were marked dirty with the workflow status from workflow metadata.\n\n        Args:\n            status_filter: If given, only returns workflow with that status. This can\n                be a single status or set of statuses.\n        '
        if status_filter is None:
            status_filter = set(WorkflowStatus)
            status_filter.discard(WorkflowStatus.NONE)
        elif not isinstance(status_filter, set):
            raise TypeError("'status_filter' should either be 'None' or a set.")
        elif WorkflowStatus.NONE in status_filter:
            raise ValueError("'WorkflowStatus.NONE' is not a valid filter value.")
        results = {}
        for status in status_filter:
            try:
                for p in self._storage.list(self._key_workflow_with_status('', status)):
                    workflow_id = p.base_name
                    results[workflow_id] = status
            except FileNotFoundError:
                pass
        try:
            for p in self._storage.list(self._key_workflow_status_dirty('')):
                workflow_id = p.base_name
                results.pop(workflow_id, None)
                status = self.load_workflow_status(workflow_id)
                if status in status_filter:
                    results[workflow_id] = status
        except FileNotFoundError:
            pass
        return list(results.items())

    def delete_workflow_status(self, workflow_id: str):
        if False:
            while True:
                i = 10
        'Delete status indexing for the workflow.'
        for status in WorkflowStatus:
            self._storage.delete(self._key_workflow_with_status(workflow_id, status))
        self._storage.delete(self._key_workflow_status_dirty(workflow_id))

    def _key_workflow_with_status(self, workflow_id: str, status: WorkflowStatus):
        if False:
            return 10
        'A key whose existence marks the status of the workflow.'
        return os.path.join(WORKFLOW_STATUS_DIR, status.value, workflow_id)

    def _key_workflow_status_dirty(self, workflow_id: str):
        if False:
            print('Hello World!')
        'A key marks the workflow status dirty, because it is under change.'
        return os.path.join(WORKFLOW_STATUS_DIR, WORKFLOW_STATUS_DIRTY_DIR, workflow_id)

    def _key_workflow_metadata(self, workflow_id: str):
        if False:
            return 10
        return os.path.join(workflow_id, WORKFLOW_META)

class WorkflowStorage:
    """Access workflow in storage. This is a higher-level abstraction,
    which does not care about the underlining storage implementation."""

    def __init__(self, workflow_id: str):
        if False:
            print('Hello World!')
        self._storage = storage.get_client(os.path.join(WORKFLOW_ROOT, workflow_id))
        self._status_storage = WorkflowIndexingStorage()
        self._workflow_id = workflow_id

    def load_task_output(self, task_id: TaskID) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Load the output of the workflow task from checkpoint.\n\n        Args:\n            task_id: ID of the workflow task.\n\n        Returns:\n            Output of the workflow task.\n        '
        tasks = [self._get(self._key_task_output(task_id), no_exception=True), self._get(self._key_task_exception(task_id), no_exception=True)]
        ((output_ret, output_err), (exception_ret, exception_err)) = tasks
        if output_err is None:
            return output_ret
        if exception_err is None:
            raise exception_ret
        raise output_err

    def save_workflow_execution_state(self, creator_task_id: TaskID, state: WorkflowExecutionState) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Save a workflow execution state.\n        Typically, the state is translated from a Ray DAG.\n\n        Args:\n            creator_task_id: The ID of the task that creates the state.\n            state: The state converted from the DAG.\n        '
        assert creator_task_id != state.output_task_id
        for (task_id, task) in state.tasks.items():
            metadata = {**task.to_dict(), 'workflow_refs': state.upstream_dependencies[task_id]}
            self._put(self._key_task_input_metadata(task_id), metadata, True)
            self._put(self._key_task_user_metadata(task_id), task.user_metadata, True)
            workflow_id = self._workflow_id
            serialization.dump_to_storage(self._key_task_function_body(task_id), task.func_body, workflow_id, self)
            with serialization_context.workflow_args_keeping_context():
                args_obj = ray.get(state.task_input_args[task_id])
            serialization.dump_to_storage(self._key_task_args(task_id), args_obj, workflow_id, self)
        self._put(self._key_task_output_metadata(creator_task_id), {'output_task_id': state.output_task_id}, True)

    def save_task_output(self, task_id: TaskID, ret: Any, *, exception: Optional[Exception]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'When a workflow task returns,\n        1. If the returned object is a workflow, this means we are a nested\n           workflow. We save the output metadata that points to the workflow.\n        2. Otherwise, checkpoint the output.\n\n        Args:\n            task_id: The ID of the workflow task. If it is an empty string,\n                it means we are in the workflow job driver process.\n            ret: The returned object from a workflow task.\n            exception: This task should throw exception.\n        '
        if exception is None:
            ret = ray.get(ret) if isinstance(ret, ray.ObjectRef) else ret
            serialization.dump_to_storage(self._key_task_output(task_id), ret, self._workflow_id, storage=self)
        else:
            assert ret is None
            serialization.dump_to_storage(self._key_task_exception(task_id), exception, self._workflow_id, storage=self)

    def load_task_func_body(self, task_id: TaskID) -> Callable:
        if False:
            while True:
                i = 10
        'Load the function body of the workflow task.\n\n        Args:\n            task_id: ID of the workflow task.\n\n        Returns:\n            A callable function.\n        '
        return self._get(self._key_task_function_body(task_id))

    def gen_task_id(self, task_name: str) -> int:
        if False:
            return 10

        def _gen_task_id():
            if False:
                for i in range(10):
                    print('nop')
            key = self._key_num_tasks_with_name(task_name)
            try:
                val = self._get(key, True)
                self._put(key, val + 1, True)
                return val + 1
            except KeyNotFoundError:
                self._put(key, 0, True)
                return 0
        return _gen_task_id()

    def load_task_args(self, task_id: TaskID) -> ray.ObjectRef:
        if False:
            return 10
        'Load the input arguments of the workflow task. This must be\n        done under a serialization context, otherwise the arguments would\n        not be reconstructed successfully.\n\n        Args:\n            task_id: ID of the workflow task.\n\n        Returns:\n            An object ref of the input args.\n        '
        with serialization_context.workflow_args_keeping_context():
            x = self._get(self._key_task_args(task_id))
        return ray.put(x)

    def save_object_ref(self, obj_ref: ray.ObjectRef) -> None:
        if False:
            while True:
                i = 10
        'Save the object ref.\n\n        Args:\n            obj_ref: The object reference\n\n        Returns:\n            None\n        '
        return self._save_object_ref(obj_ref)

    def load_object_ref(self, object_id: str) -> ray.ObjectRef:
        if False:
            for i in range(10):
                print('nop')
        'Load the input object ref.\n\n        Args:\n            object_id: The hex ObjectID.\n\n        Returns:\n            The object ref.\n        '

        def _load_obj_ref() -> ray.ObjectRef:
            if False:
                i = 10
                return i + 15
            data = self._get(self._key_obj_id(object_id))
            ref = _put_obj_ref.remote((data,))
            return ref
        return _load_obj_ref()

    def update_continuation_output_link(self, continuation_root_id: TaskID, latest_continuation_task_id: TaskID) -> None:
        if False:
            i = 10
            return i + 15
        'Update the link of the continuation output. The link points\n        to the ID of the latest finished continuation task.\n\n        Args:\n            continuation_root_id: The ID of the task that returns all later\n                continuations.\n            latest_continuation_task_id: The ID of the latest finished\n                continuation task.\n        '
        try:
            metadata = self._get(self._key_task_output_metadata(continuation_root_id), True)
        except KeyNotFoundError:
            metadata = {}
        if latest_continuation_task_id != metadata.get('output_task_id') and latest_continuation_task_id != metadata.get('dynamic_output_task_id'):
            metadata['dynamic_output_task_id'] = latest_continuation_task_id
            self._put(self._key_task_output_metadata(continuation_root_id), metadata, True)

    def _locate_output_task_id(self, task_id: TaskID) -> str:
        if False:
            i = 10
            return i + 15
        metadata = self._get(self._key_task_output_metadata(task_id), True)
        return metadata.get('dynamic_output_task_id') or metadata['output_task_id']

    def get_entrypoint_task_id(self) -> TaskID:
        if False:
            for i in range(10):
                print('nop')
        'Load the entrypoint task ID of the workflow.\n\n        Returns:\n            The ID of the entrypoint task.\n        '
        try:
            return self._locate_output_task_id('')
        except Exception as e:
            raise ValueError(f'Fail to get entrypoint task ID from workflow[id={self._workflow_id}]') from e

    def _locate_output_in_storage(self, task_id: TaskID) -> Optional[TaskID]:
        if False:
            return 10
        result = self.inspect_task(task_id)
        while isinstance(result.output_task_id, str):
            task_id = result.output_task_id
            result = self.inspect_task(result.output_task_id)
        if result.output_object_valid:
            return task_id
        return None

    def inspect_output(self, task_id: TaskID) -> Optional[TaskID]:
        if False:
            return 10
        "Get the actual checkpointed output for a task, represented by the ID of\n        the task that actually keeps the checkpoint.\n\n        Raises:\n            ValueError: The workflow does not exist or the workflow state is not valid.\n\n        Args:\n            task_id: The ID of the task we are looking for its checkpoint.\n\n        Returns:\n            The ID of the task that actually keeps the checkpoint.\n                'None' if the checkpoint does not exist.\n        "
        status = self.load_workflow_status()
        if status == WorkflowStatus.NONE:
            raise ValueError(f"No such workflow '{self._workflow_id}'")
        if status == WorkflowStatus.CANCELED:
            raise ValueError(f'Workflow {self._workflow_id} is canceled')
        if status == WorkflowStatus.RESUMABLE:
            raise ValueError(f'Workflow {self._workflow_id} is in resumable status, please resume it')
        if task_id is None:
            task_id = self.get_entrypoint_task_id()
        return self._locate_output_in_storage(task_id)

    def inspect_task(self, task_id: TaskID) -> TaskInspectResult:
        if False:
            return 10
        '\n        Get the status of a workflow task. The status indicates whether\n        the workflow task can be recovered etc.\n\n        Args:\n            task_id: The ID of a workflow task\n\n        Returns:\n            The status of the task.\n        '
        return self._inspect_task(task_id)

    def _inspect_task(self, task_id: TaskID) -> TaskInspectResult:
        if False:
            while True:
                i = 10
        items = self._scan(self._key_task_prefix(task_id), ignore_errors=True)
        keys = set(items)
        if STEP_OUTPUT in keys:
            return TaskInspectResult(output_object_valid=True)
        if STEP_OUTPUTS_METADATA in keys:
            output_task_id = self._locate_output_task_id(task_id)
            return TaskInspectResult(output_task_id=output_task_id)
        try:
            metadata = self._get(self._key_task_input_metadata(task_id), True)
            return TaskInspectResult(args_valid=STEP_ARGS in keys, func_body_valid=STEP_FUNC_BODY in keys, workflow_refs=metadata['workflow_refs'], task_options=WorkflowTaskRuntimeOptions.from_dict(metadata['task_options']), task_raised_exception=STEP_EXCEPTION in keys)
        except Exception:
            return TaskInspectResult(args_valid=STEP_ARGS in keys, func_body_valid=STEP_FUNC_BODY in keys, task_raised_exception=STEP_EXCEPTION in keys)

    def _save_object_ref(self, identifier: str, obj_ref: ray.ObjectRef):
        if False:
            print('Hello World!')
        data = ray.get(obj_ref)
        self._put(self._key_obj_id(identifier), data)

    def load_actor_class_body(self) -> type:
        if False:
            for i in range(10):
                print('nop')
        'Load the class body of the virtual actor.\n\n        Raises:\n            DataLoadError: if we fail to load the class body.\n        '
        return self._get(self._key_class_body())

    def save_actor_class_body(self, cls: type) -> None:
        if False:
            while True:
                i = 10
        'Save the class body of the virtual actor.\n\n        Args:\n            cls: The class body used by the virtual actor.\n\n        Raises:\n            DataSaveError: if we fail to save the class body.\n        '
        self._put(self._key_class_body(), cls)

    def save_task_prerun_metadata(self, task_id: TaskID, metadata: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        'Save pre-run metadata of the current task.\n\n        Args:\n            task_id: ID of the workflow task.\n            metadata: pre-run metadata of the current task.\n\n        Raises:\n            DataSaveError: if we fail to save the pre-run metadata.\n        '
        self._put(self._key_task_prerun_metadata(task_id), metadata, True)

    def save_task_postrun_metadata(self, task_id: TaskID, metadata: Dict[str, Any]):
        if False:
            return 10
        'Save post-run metadata of the current task.\n\n        Args:\n            task_id: ID of the workflow task.\n            metadata: post-run metadata of the current task.\n\n        Raises:\n            DataSaveError: if we fail to save the post-run metadata.\n        '
        self._put(self._key_task_postrun_metadata(task_id), metadata, True)

    def save_workflow_user_metadata(self, metadata: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        'Save user metadata of the current workflow.\n\n        Args:\n            metadata: user metadata of the current workflow.\n\n        Raises:\n            DataSaveError: if we fail to save the user metadata.\n        '
        self._put(self._key_workflow_user_metadata(), metadata, True)

    def load_task_metadata(self, task_id: TaskID) -> Dict[str, Any]:
        if False:
            return 10
        'Load the metadata of the given task.\n\n        Returns:\n            The metadata of the given task.\n        '

        def _load_task_metadata():
            if False:
                i = 10
                return i + 15
            if not self._scan(self._key_task_prefix(task_id), ignore_errors=True):
                if not self._scan('', ignore_errors=True):
                    raise ValueError("No such workflow_id '{}'".format(self._workflow_id))
                else:
                    raise ValueError("No such task_id '{}' in workflow '{}'".format(task_id, self._workflow_id))
            tasks = [self._get(self._key_task_input_metadata(task_id), True, True), self._get(self._key_task_prerun_metadata(task_id), True, True), self._get(self._key_task_postrun_metadata(task_id), True, True)]
            ((input_metadata, _), (prerun_metadata, _), (postrun_metadata, _)) = tasks
            input_metadata = input_metadata or {}
            prerun_metadata = prerun_metadata or {}
            postrun_metadata = postrun_metadata or {}
            metadata = input_metadata
            metadata['stats'] = {**prerun_metadata, **postrun_metadata}
            return metadata
        return _load_task_metadata()

    def load_workflow_metadata(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Load the metadata of the current workflow.\n\n        Returns:\n            The metadata of the current workflow.\n        '

        def _load_workflow_metadata():
            if False:
                for i in range(10):
                    print('nop')
            if not self._scan('', ignore_errors=True):
                raise ValueError("No such workflow_id '{}'".format(self._workflow_id))
            tasks = [self._get(self._key_workflow_metadata(), True, True), self._get(self._key_workflow_user_metadata(), True, True), self._get(self._key_workflow_prerun_metadata(), True, True), self._get(self._key_workflow_postrun_metadata(), True, True)]
            ((status_metadata, _), (user_metadata, _), (prerun_metadata, _), (postrun_metadata, _)) = tasks
            status_metadata = status_metadata or {}
            user_metadata = user_metadata or {}
            prerun_metadata = prerun_metadata or {}
            postrun_metadata = postrun_metadata or {}
            metadata = status_metadata
            metadata['user_metadata'] = user_metadata
            metadata['stats'] = {**prerun_metadata, **postrun_metadata}
            return metadata
        return _load_workflow_metadata()

    def list_workflow(self, status_filter: Optional[Set[WorkflowStatus]]=None) -> List[Tuple[str, WorkflowStatus]]:
        if False:
            i = 10
            return i + 15
        'List all workflows matching a given status filter.\n\n        Args:\n            status_filter: If given, only returns workflow with that status. This can\n                be a single status or set of statuses.\n        '
        return self._status_storage.list_workflow(status_filter)

    def delete_workflow(self) -> None:
        if False:
            print('Hello World!')
        self._status_storage.delete_workflow_status(self._workflow_id)
        found = self._storage.delete_dir('')
        if not found:
            raise WorkflowNotFoundError(self._workflow_id)

    def update_workflow_status(self, status: WorkflowStatus):
        if False:
            return 10
        'Update the status of the workflow.\n        This method is NOT thread-safe. It is handled by the workflow management actor.\n        '
        self._status_storage.update_workflow_status(self._workflow_id, status)
        if status == WorkflowStatus.RUNNING:
            self._put(self._key_workflow_prerun_metadata(), {'start_time': time.time()}, True)
        elif status in (WorkflowStatus.SUCCESSFUL, WorkflowStatus.FAILED):
            self._put(self._key_workflow_postrun_metadata(), {'end_time': time.time()}, True)

    def load_workflow_status(self):
        if False:
            print('Hello World!')
        'Load workflow status. If we find the previous status updating failed,\n        fix it with redo-log transaction recovery.'
        return self._status_storage.load_workflow_status(self._workflow_id)

    def _put(self, key: str, data: Any, is_json: bool=False) -> str:
        if False:
            return 10
        'Serialize and put an object in the object store.\n\n        Args:\n            key: The key of the object.\n            data: The data to be stored.\n            is_json: If true, json encode the data, otherwise pickle it.\n        '
        try:
            if not is_json:
                serialization.dump_to_storage(key, data, self._workflow_id, storage=self)
            else:
                serialized_data = json.dumps(data).encode()
                self._storage.put(key, serialized_data)
        except Exception as e:
            raise DataSaveError from e
        return key

    def _get(self, key: str, is_json: bool=False, no_exception: bool=False) -> Any:
        if False:
            while True:
                i = 10
        err = None
        ret = None
        try:
            unmarshaled = self._storage.get(key)
            if unmarshaled is None:
                raise KeyNotFoundError
            if is_json:
                ret = json.loads(unmarshaled.decode())
            else:
                ret = cloudpickle.loads(unmarshaled)
        except KeyNotFoundError as e:
            err = e
        except Exception as e:
            err = DataLoadError()
            err.__cause__ = e
        if no_exception:
            return (ret, err)
        elif err is None:
            return ret
        else:
            raise err

    def _scan(self, prefix: str, ignore_errors: bool=False) -> List[str]:
        if False:
            return 10
        try:
            return [p.base_name for p in self._storage.list(prefix)]
        except Exception as e:
            if ignore_errors:
                return []
            raise e

    def _exists(self, key: str) -> bool:
        if False:
            i = 10
            return i + 15
        return self._storage.get_info(key) is not None

    def _key_task_input_metadata(self, task_id):
        if False:
            return 10
        return os.path.join(STEPS_DIR, task_id, STEP_INPUTS_METADATA)

    def _key_task_user_metadata(self, task_id):
        if False:
            i = 10
            return i + 15
        return os.path.join(STEPS_DIR, task_id, STEP_USER_METADATA)

    def _key_task_prerun_metadata(self, task_id):
        if False:
            i = 10
            return i + 15
        return os.path.join(STEPS_DIR, task_id, STEP_PRERUN_METADATA)

    def _key_task_postrun_metadata(self, task_id):
        if False:
            while True:
                i = 10
        return os.path.join(STEPS_DIR, task_id, STEP_POSTRUN_METADATA)

    def _key_task_output(self, task_id):
        if False:
            while True:
                i = 10
        return os.path.join(STEPS_DIR, task_id, STEP_OUTPUT)

    def _key_task_exception(self, task_id):
        if False:
            print('Hello World!')
        return os.path.join(STEPS_DIR, task_id, STEP_EXCEPTION)

    def _key_task_output_metadata(self, task_id):
        if False:
            i = 10
            return i + 15
        return os.path.join(STEPS_DIR, task_id, STEP_OUTPUTS_METADATA)

    def _key_task_function_body(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(STEPS_DIR, task_id, STEP_FUNC_BODY)

    def _key_task_args(self, task_id):
        if False:
            print('Hello World!')
        return os.path.join(STEPS_DIR, task_id, STEP_ARGS)

    def _key_obj_id(self, object_id):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(OBJECTS_DIR, object_id)

    def _key_task_prefix(self, task_id):
        if False:
            return 10
        return os.path.join(STEPS_DIR, task_id, '')

    def _key_class_body(self):
        if False:
            print('Hello World!')
        return os.path.join(CLASS_BODY)

    def _key_workflow_metadata(self):
        if False:
            return 10
        return os.path.join(WORKFLOW_META)

    def _key_workflow_user_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(WORKFLOW_USER_METADATA)

    def _key_workflow_prerun_metadata(self):
        if False:
            return 10
        return os.path.join(WORKFLOW_PRERUN_METADATA)

    def _key_workflow_postrun_metadata(self):
        if False:
            while True:
                i = 10
        return os.path.join(WORKFLOW_POSTRUN_METADATA)

    def _key_num_tasks_with_name(self, task_name):
        if False:
            return 10
        return os.path.join(DUPLICATE_NAME_COUNTER, task_name)

def get_workflow_storage(workflow_id: Optional[str]=None) -> WorkflowStorage:
    if False:
        for i in range(10):
            print('nop')
    'Get the storage for the workflow.\n\n    Args:\n        workflow_id: The ID of the storage.\n\n    Returns:\n        A workflow storage.\n    '
    if workflow_id is None:
        workflow_id = workflow_context.get_workflow_task_context().workflow_id
    return WorkflowStorage(workflow_id)

def _load_object_ref(paths: List[str], wf_storage: WorkflowStorage) -> ObjectRef:
    if False:
        for i in range(10):
            print('nop')

    @ray.remote(num_cpus=0)
    def load_ref(paths: List[str], wf_storage: WorkflowStorage):
        if False:
            for i in range(10):
                print('nop')
        return wf_storage._get(paths)
    return load_ref.remote(paths, wf_storage)

@ray.remote(num_cpus=0)
def _put_obj_ref(ref: Tuple[ObjectRef]):
    if False:
        while True:
            i = 10
    "\n    Return a ref to an object ref. (This can't be done with\n    `ray.put(obj_ref)`).\n\n    "
    return ref[0]