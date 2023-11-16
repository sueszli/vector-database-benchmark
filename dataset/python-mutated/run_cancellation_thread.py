import threading
from typing import Tuple, cast
import dagster._check as check
from dagster._core.instance import DagsterInstance, InstanceRef
from dagster._core.storage.dagster_run import DagsterRun, DagsterRunStatus
from dagster._utils import send_interrupt

def _kill_on_cancel(instance_ref: InstanceRef, run_id, shutdown_event):
    if False:
        for i in range(10):
            print('nop')
    check.inst_param(instance_ref, 'instance_ref', InstanceRef)
    check.str_param(run_id, 'run_id')
    with DagsterInstance.from_ref(instance_ref) as instance:
        while not shutdown_event.is_set():
            shutdown_event.wait(instance.cancellation_thread_poll_interval_seconds)
            run = cast(DagsterRun, check.inst(instance.get_run_by_id(run_id), DagsterRun, 'Run not found for cancellation thread'))
            if run.status in [DagsterRunStatus.CANCELING, DagsterRunStatus.CANCELED]:
                print(f'Detected run status {run.status}, sending interrupt to main thread')
                send_interrupt()
                return

def start_run_cancellation_thread(instance: DagsterInstance, run_id) -> Tuple[threading.Thread, threading.Event]:
    if False:
        return 10
    print('Starting run cancellation thread')
    shutdown_event = threading.Event()
    thread = threading.Thread(target=_kill_on_cancel, args=(instance.get_ref(), run_id, shutdown_event), name='kill-on-cancel')
    thread.start()
    return (thread, shutdown_event)