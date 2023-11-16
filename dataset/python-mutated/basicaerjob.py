"""This module implements the job class used by Basic Aer Provider."""
import warnings
from qiskit.providers import JobStatus
from qiskit.providers.job import JobV1

class BasicAerJob(JobV1):
    """BasicAerJob class."""
    _async = False

    def __init__(self, backend, job_id, result):
        if False:
            return 10
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        if False:
            return 10
        'Submit the job to the backend for execution.\n\n        Raises:\n            JobError: if trying to re-submit the job.\n        '
        return

    def result(self, timeout=None):
        if False:
            i = 10
            return i + 15
        'Get job result .\n\n        Returns:\n            qiskit.result.Result: Result object\n        '
        if timeout is not None:
            warnings.warn("The timeout kwarg doesn't have any meaning with BasicAer because execution is synchronous and the result already exists when run() returns.", UserWarning)
        return self._result

    def status(self):
        if False:
            for i in range(10):
                print('nop')
        "Gets the status of the job by querying the Python's future\n\n        Returns:\n            qiskit.providers.JobStatus: The current JobStatus\n        "
        return JobStatus.DONE

    def backend(self):
        if False:
            return 10
        'Return the instance of the backend used for this job.'
        return self._backend