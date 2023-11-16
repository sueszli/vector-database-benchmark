from enum import Enum
from pyflink.java_gateway import get_gateway
__all__ = ['JobStatus']

class JobStatus(Enum):
    """
    Possible states of a job once it has been accepted by the job manager.

    :data:`CREATED`:

    Job is newly created, no task has started to run.

    :data:`RUNNING`:

    Some tasks are scheduled or running, some may be pending, some may be finished.

    :data:`FAILING`:

    The job has failed and is currently waiting for the cleanup to complete.

    :data:`FAILED`:

    The job has failed with a non-recoverable task failure.

    :data:`CANCELLING`:

    Job is being cancelled.

    :data:`CANCELED`:

    Job has been cancelled.

    :data:`FINISHED`:

    All of the job's tasks have successfully finished.

    :data:`RESTARTING`:

    The job is currently undergoing a reset and total restart.

    :data:`SUSPENDED`:

    The job has been suspended which means that it has been stopped but not been removed from a
    potential HA job store.

    :data:`RECONCILING`:

    The job is currently reconciling and waits for task execution report to recover state.

    .. versionadded:: 1.11.0
    """
    CREATED = 0
    RUNNING = 1
    FAILING = 2
    FAILED = 3
    CANCELLING = 4
    CANCELED = 5
    FINISHED = 6
    RESTARTING = 7
    SUSPENDED = 8
    RECONCILING = 9

    def is_globally_terminal_state(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks whether this state is <i>globally terminal</i>. A globally terminal job\n        is complete and cannot fail any more and will not be restarted or recovered by another\n        standby master node.\n\n        When a globally terminal state has been reached, all recovery data for the job is\n        dropped from the high-availability services.\n\n        :return: ``True`` if this job status is globally terminal, ``False`` otherwise.\n\n        .. versionadded:: 1.11.0\n        '
        return self._to_j_job_status().isGloballyTerminalState()

    def is_terminal_state(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Checks whether this state is locally terminal. Locally terminal refers to the\n        state of a job's execution graph within an executing JobManager. If the execution graph\n        is locally terminal, the JobManager will not continue executing or recovering the job.\n\n        The only state that is locally terminal, but not globally terminal is SUSPENDED,\n        which is typically entered when the executing JobManager loses its leader status.\n\n        :return: ``True`` if this job status is terminal, ``False`` otherwise.\n\n        .. versionadded:: 1.11.0\n        "
        return self._to_j_job_status().isTerminalState()

    @staticmethod
    def _from_j_job_status(j_job_status) -> 'JobStatus':
        if False:
            return 10
        return JobStatus[j_job_status.name()]

    def _to_j_job_status(self):
        if False:
            i = 10
            return i + 15
        gateway = get_gateway()
        JJobStatus = gateway.jvm.org.apache.flink.api.common.JobStatus
        return getattr(JJobStatus, self.name)