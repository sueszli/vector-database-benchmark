"""Provides an Apache Beam API for operating on NDB models."""
from __future__ import annotations
from core.jobs.types import job_run_result
import apache_beam as beam
import result
from typing import Any, Optional, Tuple

class ResultsToJobRunResults(beam.PTransform):
    """Transforms result.Result into job_run_result.JobRunResult."""

    def __init__(self, prefix: Optional[str]=None, label: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes the ResultsToJobRunResults PTransform.\n\n        Args:\n            prefix: str|None. The prefix for the result string.\n            label: str|None. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.prefix = '%s ' % prefix if prefix else ''

    @beam.typehints.no_annotations
    def _transform_result_to_job_run_result(self, result_item: result.Result[Any, Any]) -> job_run_result.JobRunResult:
        if False:
            while True:
                i = 10
        'Transforms Result objects into JobRunResult objects. When the result\n        is Ok then transform it into stdout, if the result is Err transform it\n        into stderr.\n\n        Args:\n            result_item: Result. The result object.\n\n        Returns:\n            JobRunResult. The JobRunResult object.\n        '
        if isinstance(result_item, result.Ok):
            return job_run_result.JobRunResult.as_stdout('%sSUCCESS:' % self.prefix)
        else:
            return job_run_result.JobRunResult.as_stderr('%sERROR: "%s":' % (self.prefix, result_item.value))

    @staticmethod
    def _add_count_to_job_run_result(job_result_and_count: Tuple[job_run_result.JobRunResult, int]) -> job_run_result.JobRunResult:
        if False:
            for i in range(10):
                print('nop')
        'Adds count to the stdout or stderr of the JobRunResult.\n\n        Args:\n            job_result_and_count: tuple(JobRunResult, int). Tuple containing\n                unique JobRunResult and their counts.\n\n        Returns:\n            JobRunResult. JobRunResult objects with counts added.\n        '
        (job_result, count) = job_result_and_count
        if job_result.stdout:
            job_result.stdout += ' %s' % str(count)
        if job_result.stderr:
            job_result.stderr += ' %s' % str(count)
        return job_result

    @beam.typehints.no_annotations
    def expand(self, results: beam.PCollection[result.Result[Any, Any]]) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            while True:
                i = 10
        'Transforms Result objects into unique JobRunResult objects and\n        adds counts to them.\n\n        Args:\n            results: PCollection. Sequence of Result objects.\n\n        Returns:\n            PCollection. Sequence of unique JobRunResult objects with count.\n        '
        return results | 'Transform result to job run result' >> beam.Map(self._transform_result_to_job_run_result) | 'Count all elements' >> beam.combiners.Count.PerElement() | 'Add count to job run result' >> beam.Map(self._add_count_to_job_run_result)

class CountObjectsToJobRunResult(beam.PTransform):
    """Transform that counts number of objects in a sequence and puts
    the count into job_run_result.JobRunResult.
    """

    def __init__(self, prefix: Optional[str]=None, label: Optional[str]=None) -> None:
        if False:
            return 10
        'Initializes the ResultsToJobRunResults PTransform.\n\n        Args:\n            prefix: str|None. The prefix for the result string.\n            label: str|None. The label of the PTransform.\n        '
        super().__init__(label=label)
        self.prefix = '%s ' % prefix if prefix else ''

    def expand(self, objects: beam.PCollection[Any]) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            for i in range(10):
                print('nop')
        'Counts items in collection and puts the count into a job run result.\n\n        Args:\n            objects: PCollection. Sequence of any objects.\n\n        Returns:\n            PCollection. Sequence of one JobRunResult with count.\n        '
        return objects | 'Count all new models' >> beam.combiners.Count.Globally() | 'Only create result for non-zero number of objects' >> beam.Filter(lambda x: x > 0) | 'Add count to job run result' >> beam.Map(lambda object_count: job_run_result.JobRunResult.as_stdout('%sSUCCESS: %s' % (self.prefix, object_count)))