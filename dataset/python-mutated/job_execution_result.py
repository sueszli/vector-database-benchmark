from typing import Dict, Any
from pyflink.common.job_id import JobID
__all__ = ['JobExecutionResult']

class JobExecutionResult(object):
    """
    The result of a job execution. Gives access to the execution time of the job,
    and to all accumulators created by this job.

    .. versionadded:: 1.11.0
    """

    def __init__(self, j_job_execution_result):
        if False:
            return 10
        self._j_job_execution_result = j_job_execution_result

    def get_job_id(self) -> JobID:
        if False:
            return 10
        '\n        Returns the JobID assigned to the job by the Flink runtime.\n\n        :return: JobID, or null if the job has been executed on a runtime without JobIDs\n                 or if the execution failed.\n\n        .. versionadded:: 1.11.0\n        '
        return JobID(self._j_job_execution_result.getJobID())

    def get_net_runtime(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the net execution time of the job, i.e., the execution time in the parallel system,\n        without the pre-flight steps like the optimizer.\n\n        :return: The net execution time in milliseconds.\n\n        .. versionadded:: 1.11.0\n        '
        return self._j_job_execution_result.getNetRuntime()

    def get_accumulator_result(self, accumulator_name: str):
        if False:
            while True:
                i = 10
        '\n        Gets the accumulator with the given name. Returns None, if no accumulator with\n        that name was produced.\n\n        :param accumulator_name: The name of the accumulator.\n        :return: The value of the accumulator with the given name.\n\n        .. versionadded:: 1.11.0\n        '
        return self.get_all_accumulator_results().get(accumulator_name)

    def get_all_accumulator_results(self) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Gets all accumulators produced by the job. The map contains the accumulators as\n        mappings from the accumulator name to the accumulator value.\n\n        :return: The dict which the keys are names of the accumulator and the values\n                 are values of the accumulator produced by the job.\n\n        .. versionadded:: 1.11.0\n        '
        j_result_map = self._j_job_execution_result.getAllAccumulatorResults()
        accumulators = {}
        for key in j_result_map:
            accumulators[key] = j_result_map[key]
        return accumulators

    def __str__(self):
        if False:
            print('Hello World!')
        '\n        Convert JobExecutionResult to a string, if possible.\n\n        .. versionadded:: 1.11.0\n        '
        return self._j_job_execution_result.toString()