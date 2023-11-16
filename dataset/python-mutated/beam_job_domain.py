"""Domain objects related to Apache Beam jobs."""
from __future__ import annotations
import datetime
from core import utils
from core.jobs import base_jobs
from typing import Dict, List, Type, Union

class BeamJob:
    """Encapsulates the definition of an Apache Beam job.

    Attributes:
        name: str. The name of the class that implements the job's logic.
    """

    def __init__(self, job_class: Type[base_jobs.JobBase]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initializes a new instance of BeamJob.\n\n        Args:\n            job_class: type(JobBase). The JobBase subclass which implements the\n                job's logic.\n        "
        self._job_class = job_class

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        "Returns the name of the class that implements the job's logic.\n\n        Returns:\n            str. The name of the job class.\n        "
        return self._job_class.__name__

    def to_dict(self) -> Dict[str, Union[str, List[str]]]:
        if False:
            i = 10
            return i + 15
        "Returns a dict representation of the BeamJob.\n\n        Returns:\n            dict(str: *). The dict has the following structure:\n                name: str. The name of the class that implements the job's\n                    logic.\n        "
        return {'name': self.name}

class BeamJobRun:
    """Encapsulates an individual execution of an Apache Beam job.

    Attributes:
        job_id: str. The ID of the job execution.
        job_name: str. The name of the job class that implements the job's
            logic.
        job_state: str. The state of the job at the time the model was last
            updated.
        job_started_on: datetime. The time at which the job was started.
        job_updated_on: datetime. The time at which the job's state was last
            updated.
        job_is_synchronous: bool. Whether the job has been run synchronously.
            Synchronous jobs are similar to function calls that return
            immediately. Asynchronous jobs are similar to JavaScript Promises
            that return nothing immediately but then _eventually_ produce a
            result.
    """

    def __init__(self, job_id: str, job_name: str, job_state: str, job_started_on: datetime.datetime, job_updated_on: datetime.datetime, job_is_synchronous: bool) -> None:
        if False:
            return 10
        "Initializes a new BeamJobRun instance.\n\n        Args:\n            job_id: str. The ID of the job execution.\n            job_name: str. The name of the job class that implements the job's\n                logic.\n            job_state: str. The state of the job at the time the model was last\n                updated.\n            job_started_on: datetime. The time at which the job was started.\n            job_updated_on: datetime. The time at which the job's state was last\n                updated.\n            job_is_synchronous: bool. Whether the job has been run\n                synchronously.\n        "
        self.job_id = job_id
        self.job_name = job_name
        self.job_state = job_state
        self.job_started_on = job_started_on
        self.job_updated_on = job_updated_on
        self.job_is_synchronous = job_is_synchronous

    def to_dict(self) -> Dict[str, Union[bool, float, str, List[str]]]:
        if False:
            i = 10
            return i + 15
        "Returns a dict representation of the BeamJobRun.\n\n        Returns:\n            dict(str: *). The dict has the following structure:\n                job_id: str. The ID of the job execution.\n                job_name: str. The name of the job class that implements the\n                    job's logic.\n                job_state: str. The state of the job at the time the model was\n                    last updated.\n                job_started_on_msecs: float. The number of milliseconds since\n                    UTC epoch at which the job was created.\n                job_updated_on_msecs: float. The number of milliseconds since\n                    UTC epoch at which the job's state was last updated.\n                job_is_synchronous: bool. Whether the job has been run\n                    synchronously.\n        "
        return {'job_id': self.job_id, 'job_name': self.job_name, 'job_state': self.job_state, 'job_started_on_msecs': utils.get_time_in_millisecs(self.job_started_on), 'job_updated_on_msecs': utils.get_time_in_millisecs(self.job_updated_on), 'job_is_synchronous': self.job_is_synchronous}

class AggregateBeamJobRunResult:
    """Encapsulates the complete result of an Apache Beam job run.

    Attributes:
        stdout: str. The standard output produced by the job.
        stderr: str. The error output produced by the job.
    """

    def __init__(self, stdout: str, stderr: str) -> None:
        if False:
            while True:
                i = 10
        'Initializes a new instance of AggregateBeamJobRunResult.\n\n        Args:\n            stdout: str. The standard output produced by the job.\n            stderr: str. The error output produced by the job.\n        '
        self.stdout = stdout
        self.stderr = stderr

    def to_dict(self) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a dict representation of the AggregateBeamJobRunResult.\n\n        Returns:\n            dict(str: str). The dict structure is:\n                stdout: str. The standard output produced by the job.\n                stderr: str. The error output produced by the job.\n        '
        return {'stdout': self.stdout, 'stderr': self.stderr}