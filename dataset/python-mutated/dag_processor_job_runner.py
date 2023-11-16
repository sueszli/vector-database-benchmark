from __future__ import annotations
from typing import TYPE_CHECKING, Any
from airflow.jobs.base_job_runner import BaseJobRunner
from airflow.jobs.job import Job, perform_heartbeat
from airflow.utils.log.logging_mixin import LoggingMixin
if TYPE_CHECKING:
    from airflow.dag_processing.manager import DagFileProcessorManager

def empty_callback(_: Any) -> None:
    if False:
        print('Hello World!')
    pass

class DagProcessorJobRunner(BaseJobRunner, LoggingMixin):
    """
    DagProcessorJobRunner is a job runner that runs a DagFileProcessorManager processor.

    :param job: Job instance to use
    :param processor: DagFileProcessorManager instance to use
    """
    job_type = 'DagProcessorJob'

    def __init__(self, job: Job, processor: DagFileProcessorManager, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(job)
        self.processor = processor
        self.processor.heartbeat = lambda : perform_heartbeat(job=self.job, heartbeat_callback=empty_callback, only_if_necessary=True)

    def _execute(self) -> int | None:
        if False:
            while True:
                i = 10
        self.log.info('Starting the Dag Processor Job')
        try:
            self.processor.start()
        except Exception:
            self.log.exception('Exception when executing DagProcessorJob')
            raise
        finally:
            self.processor.terminate()
            self.processor.end()
        return None