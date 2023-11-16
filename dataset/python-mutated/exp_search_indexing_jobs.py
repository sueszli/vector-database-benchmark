"""Jobs that are run by CRON scheduler."""
from __future__ import annotations
from core.domain import exp_domain
from core.domain import exp_fetchers
from core.domain import search_services
from core.jobs import base_jobs
from core.jobs.io import ndb_io
from core.jobs.transforms import job_result_transforms
from core.jobs.types import job_run_result
from core.platform import models
import apache_beam as beam
import result
from typing import Final, Iterable, List
MYPY = False
if MYPY:
    from mypy_imports import exp_models
    from mypy_imports import search_services as platform_search_services
(exp_models,) = models.Registry.import_models([models.Names.EXPLORATION])
platform_search_services = models.Registry.import_search_services()

class IndexExplorationsInSearchJob(base_jobs.JobBase):
    """Job that indexes the explorations in Elastic Search."""
    MAX_BATCH_SIZE: Final = 1000

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            i = 10
            return i + 15
        "Returns a PCollection of 'SUCCESS' or 'FAILURE' results from\n        the Elastic Search.\n\n        Returns:\n            PCollection. A PCollection of 'SUCCESS' or 'FAILURE' results from\n            the Elastic Search.\n        "
        return self.pipeline | 'Get all non-deleted models' >> ndb_io.GetModels(exp_models.ExpSummaryModel.get_all(include_deleted=False)) | 'Convert ExpSummaryModels to domain objects' >> beam.Map(exp_fetchers.get_exploration_summary_from_model) | 'Split models into batches' >> beam.transforms.util.BatchElements(max_batch_size=self.MAX_BATCH_SIZE) | 'Index batches of models' >> beam.ParDo(IndexExplorationSummaries()) | 'Count the output' >> job_result_transforms.ResultsToJobRunResults()

class IndexExplorationSummaries(beam.DoFn):
    """DoFn to index exploration summaries."""

    def process(self, exp_summary: List[exp_domain.ExplorationSummary]) -> Iterable[result.Result[None, Exception]]:
        if False:
            for i in range(10):
                print('nop')
        'Index exploration summaries and catch any errors.\n\n        Args:\n            exp_summary: list(ExplorationSummary). List of Exp Summary domain\n                objects to be indexed.\n\n        Yields:\n            JobRunResult. List containing one element, which is either SUCCESS,\n            or FAILURE.\n        '
        try:
            search_services.index_exploration_summaries(exp_summary)
            for _ in exp_summary:
                yield result.Ok()
        except platform_search_services.SearchException as e:
            yield result.Err(e)