"""Unit tests for core.jobs.transforms.results_transforms."""
from __future__ import annotations
from core.jobs import job_test_utils
from core.jobs.transforms import results_transforms
import apache_beam as beam
import result

class DrainResultsOnErrorTests(job_test_utils.PipelinedTestBase):

    def test_error_results_returns_empty_collection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        transform_result = self.pipeline | beam.Create([result.Ok(('id_1', None)), result.Ok(('id_2', None)), result.Err(('id_3', None))]) | results_transforms.DrainResultsOnError()
        self.assert_pcoll_empty(transform_result)

    def test_ok_results_returns_unchanged_collection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        transform_result = self.pipeline | beam.Create([result.Ok(('id_1', None)), result.Ok(('id_2', None)), result.Ok(('id_3', None))]) | results_transforms.DrainResultsOnError()
        self.assert_pcoll_equal(transform_result, [result.Ok(('id_1', None)), result.Ok(('id_2', None)), result.Ok(('id_3', None))])

    def test_zero_objects_correctly_outputs(self) -> None:
        if False:
            return 10
        transform_result = self.pipeline | beam.Create([]) | results_transforms.DrainResultsOnError()
        self.assert_pcoll_empty(transform_result)