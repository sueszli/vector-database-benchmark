from __future__ import annotations
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from airflow.models.taskinstance import SimpleTaskInstance

class CallbackRequest:
    """
    Base Class with information about the callback to be executed.

    :param full_filepath: File Path to use to run the callback
    :param msg: Additional Message that can be used for logging
    :param processor_subdir: Directory used by Dag Processor when parsed the dag.
    """

    def __init__(self, full_filepath: str, processor_subdir: str | None=None, msg: str | None=None):
        if False:
            return 10
        self.full_filepath = full_filepath
        self.processor_subdir = processor_subdir
        self.msg = msg

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __repr__(self):
        if False:
            return 10
        return str(self.__dict__)

    def to_json(self) -> str:
        if False:
            i = 10
            return i + 15
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str: str):
        if False:
            i = 10
            return i + 15
        json_object = json.loads(json_str)
        return cls(**json_object)

class TaskCallbackRequest(CallbackRequest):
    """
    Task callback status information.

    A Class with information about the success/failure TI callback to be executed. Currently, only failure
    callbacks (when tasks are externally killed) and Zombies are run via DagFileProcessorProcess.

    :param full_filepath: File Path to use to run the callback
    :param simple_task_instance: Simplified Task Instance representation
    :param is_failure_callback: Flag to determine whether it is a Failure Callback or Success Callback
    :param msg: Additional Message that can be used for logging to determine failure/zombie
    :param processor_subdir: Directory used by Dag Processor when parsed the dag.
    """

    def __init__(self, full_filepath: str, simple_task_instance: SimpleTaskInstance, is_failure_callback: bool | None=True, processor_subdir: str | None=None, msg: str | None=None):
        if False:
            return 10
        super().__init__(full_filepath=full_filepath, processor_subdir=processor_subdir, msg=msg)
        self.simple_task_instance = simple_task_instance
        self.is_failure_callback = is_failure_callback

    def to_json(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        from airflow.serialization.serialized_objects import BaseSerialization
        val = BaseSerialization.serialize(self.__dict__, strict=True)
        return json.dumps(val)

    @classmethod
    def from_json(cls, json_str: str):
        if False:
            return 10
        from airflow.serialization.serialized_objects import BaseSerialization
        val = json.loads(json_str)
        return cls(**BaseSerialization.deserialize(val))

class DagCallbackRequest(CallbackRequest):
    """
    A Class with information about the success/failure DAG callback to be executed.

    :param full_filepath: File Path to use to run the callback
    :param dag_id: DAG ID
    :param run_id: Run ID for the DagRun
    :param processor_subdir: Directory used by Dag Processor when parsed the dag.
    :param is_failure_callback: Flag to determine whether it is a Failure Callback or Success Callback
    :param msg: Additional Message that can be used for logging
    """

    def __init__(self, full_filepath: str, dag_id: str, run_id: str, processor_subdir: str | None, is_failure_callback: bool | None=True, msg: str | None=None):
        if False:
            i = 10
            return i + 15
        super().__init__(full_filepath=full_filepath, processor_subdir=processor_subdir, msg=msg)
        self.dag_id = dag_id
        self.run_id = run_id
        self.is_failure_callback = is_failure_callback

class SlaCallbackRequest(CallbackRequest):
    """
    A class with information about the SLA callback to be executed.

    :param full_filepath: File Path to use to run the callback
    :param dag_id: DAG ID
    :param processor_subdir: Directory used by Dag Processor when parsed the dag.
    """

    def __init__(self, full_filepath: str, dag_id: str, processor_subdir: str | None, msg: str | None=None):
        if False:
            i = 10
            return i + 15
        super().__init__(full_filepath, processor_subdir=processor_subdir, msg=msg)
        self.dag_id = dag_id