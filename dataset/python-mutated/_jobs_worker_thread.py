import threading
from dataclasses import dataclass
import queue
from typing import Optional, Any, Callable, Sequence, Union
from uuid import UUID
from sslyze.plugins.scan_commands import ScanCommand
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

@dataclass(frozen=True)
class CompletedScanJob:
    parent_server_scan_request_uuid: UUID
    for_scan_command: ScanCommand
    return_value: Optional[Any]
    exception: Optional[Exception]

@dataclass(frozen=True)
class QueuedScanJob:
    parent_server_scan_request_uuid: UUID
    for_scan_command: ScanCommand
    function_to_call: Callable
    function_arguments: Sequence[Any]

class WorkerThreadNoMoreJobsSentinel:
    pass
WorkerQueueType: TypeAlias = 'queue.Queue[Union[WorkerThreadNoMoreJobsSentinel, QueuedScanJob]]'

class JobsWorkerThread(threading.Thread):

    def __init__(self, jobs_queue_in: WorkerQueueType, completed_jobs_queue_out: 'queue.Queue[CompletedScanJob]'):
        if False:
            print('Hello World!')
        super().__init__()
        self._jobs_queue_in = jobs_queue_in
        self._completed_jobs_queue_out = completed_jobs_queue_out
        self.daemon = True

    def run(self) -> None:
        if False:
            while True:
                i = 10
        while True:
            job_to_complete = self._jobs_queue_in.get(block=True)
            if isinstance(job_to_complete, WorkerThreadNoMoreJobsSentinel):
                self._jobs_queue_in.task_done()
                break
            try:
                return_value = job_to_complete.function_to_call(*job_to_complete.function_arguments)
                self._completed_jobs_queue_out.put(CompletedScanJob(parent_server_scan_request_uuid=job_to_complete.parent_server_scan_request_uuid, for_scan_command=job_to_complete.for_scan_command, return_value=return_value, exception=None))
            except Exception as e:
                self._completed_jobs_queue_out.put(CompletedScanJob(parent_server_scan_request_uuid=job_to_complete.parent_server_scan_request_uuid, for_scan_command=job_to_complete.for_scan_command, return_value=None, exception=e))
            self._jobs_queue_in.task_done()