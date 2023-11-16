from pyflink.common.completable_future import CompletableFuture
from pyflink.common.job_execution_result import JobExecutionResult
from pyflink.common.job_id import JobID
from pyflink.common.job_status import JobStatus
__all__ = ['JobClient']

class JobClient(object):
    """
    A client that is scoped to a specific job.

    .. versionadded:: 1.11.0
    """

    def __init__(self, j_job_client):
        if False:
            return 10
        self._j_job_client = j_job_client

    def get_job_id(self) -> JobID:
        if False:
            i = 10
            return i + 15
        '\n        Returns the JobID that uniquely identifies the job this client is scoped to.\n\n        :return: JobID, or null if the job has been executed on a runtime without JobIDs\n                 or if the execution failed.\n\n        .. versionadded:: 1.11.0\n        '
        return JobID(self._j_job_client.getJobID())

    def get_job_status(self) -> CompletableFuture:
        if False:
            return 10
        '\n        Requests the JobStatus of the associated job.\n\n        :return: A CompletableFuture containing the JobStatus of the associated job.\n\n        .. versionadded:: 1.11.0\n        '
        return CompletableFuture(self._j_job_client.getJobStatus(), JobStatus._from_j_job_status)

    def cancel(self) -> CompletableFuture:
        if False:
            for i in range(10):
                print('nop')
        '\n        Cancels the associated job.\n\n        :return: A CompletableFuture for canceling the associated job.\n\n        .. versionadded:: 1.11.0\n        '
        return CompletableFuture(self._j_job_client.cancel())

    def stop_with_savepoint(self, advance_to_end_of_event_time: bool, savepoint_directory: str=None) -> CompletableFuture:
        if False:
            i = 10
            return i + 15
        '\n        Stops the associated job on Flink cluster.\n\n        Stopping works only for streaming programs. Be aware, that the job might continue to run\n        for a while after sending the stop command, because after sources stopped to emit data all\n        operators need to finish processing.\n\n        :param advance_to_end_of_event_time: Flag indicating if the source should inject a\n                                             MAX_WATERMARK in the pipeline.\n        :param savepoint_directory: Directory the savepoint should be written to.\n        :return: A CompletableFuture containing the path where the savepoint is located.\n\n        .. versionadded:: 1.11.0\n        '
        return CompletableFuture(self._j_job_client.stopWithSavepoint(advance_to_end_of_event_time, savepoint_directory), str)

    def trigger_savepoint(self, savepoint_directory: str=None) -> CompletableFuture:
        if False:
            i = 10
            return i + 15
        '\n        Triggers a savepoint for the associated job. The savepoint will be written to the given\n        savepoint directory.\n\n        :param savepoint_directory: Directory the savepoint should be written to.\n        :return: A CompletableFuture containing the path where the savepoint is located.\n\n        .. versionadded:: 1.11.0\n        '
        return CompletableFuture(self._j_job_client.triggerSavepoint(savepoint_directory), str)

    def get_accumulators(self) -> CompletableFuture:
        if False:
            while True:
                i = 10
        '\n        Requests the accumulators of the associated job. Accumulators can be requested while it\n        is running or after it has finished. The class loader is used to deserialize the incoming\n        accumulator results.\n\n        :param class_loader: Class loader used to deserialize the incoming accumulator results.\n        :return: A CompletableFuture containing the accumulators of the associated job.\n\n        .. versionadded:: 1.11.0\n        '
        return CompletableFuture(self._j_job_client.getAccumulators(), dict)

    def get_job_execution_result(self) -> CompletableFuture:
        if False:
            while True:
                i = 10
        '\n        Returns the JobExecutionResult result of the job execution of the submitted job.\n\n        :return: A CompletableFuture containing the JobExecutionResult result of the job execution.\n\n        .. versionadded:: 1.11.0\n        '
        return CompletableFuture(self._j_job_client.getJobExecutionResult(), JobExecutionResult)