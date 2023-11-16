from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.jobs.job import Job
    from airflow.serialization.pydantic.job import JobPydantic

class BaseJobRunner:
    """Abstract class for job runners to derive from."""
    job_type = 'undefined'

    def __init__(self, job: Job) -> None:
        if False:
            for i in range(10):
                print('nop')
        if job.job_type and job.job_type != self.job_type:
            raise Exception(f'The job is already assigned a different job_type: {job.job_type}.This is a bug and should be reported.')
        job.job_type = self.job_type
        self.job: Job = job

    def _execute(self) -> int | None:
        if False:
            while True:
                i = 10
        '\n        Execute the logic connected to the runner.\n\n        This method should be overridden by subclasses.\n\n        :meta private:\n        :return: return code if available, otherwise None\n        '
        raise NotImplementedError()

    @provide_session
    def heartbeat_callback(self, session: Session=NEW_SESSION) -> None:
        if False:
            return 10
        '\n        Execute callback during heartbeat.\n\n        This method can be overwritten by the runners.\n        '

    @classmethod
    @provide_session
    def most_recent_job(cls, session: Session=NEW_SESSION) -> Job | JobPydantic | None:
        if False:
            print('Hello World!')
        'Return the most recent job of this type, if any, based on last heartbeat received.'
        from airflow.jobs.job import most_recent_job
        return most_recent_job(cls.job_type, session=session)