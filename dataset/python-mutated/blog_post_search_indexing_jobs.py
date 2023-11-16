"""Jobs that are run by CRON scheduler."""
from __future__ import annotations
from core.domain import blog_domain
from core.domain import blog_services
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
    from mypy_imports import blog_models
    from mypy_imports import search_services as platform_search_services
(blog_models,) = models.Registry.import_models([models.Names.BLOG])
platform_search_services = models.Registry.import_search_services()

class IndexBlogPostsInSearchJob(base_jobs.JobBase):
    """Job that indexes the blog posts in Elastic Search."""
    MAX_BATCH_SIZE: Final = 1000

    def run(self) -> beam.PCollection[job_run_result.JobRunResult]:
        if False:
            return 10
        "Returns a PCollection of 'SUCCESS' or 'FAILURE' results from\n        the Elastic Search.\n\n        Returns:\n            PCollection. A PCollection of 'SUCCESS' or 'FAILURE' results from\n            the Elastic Search.\n        "
        return self.pipeline | 'Get all non-deleted models' >> ndb_io.GetModels(blog_models.BlogPostSummaryModel.get_all(include_deleted=False)) | 'Convert BlogPostSummaryModels to domain objects' >> beam.Map(blog_services.get_blog_post_summary_from_model) | 'Split models into batches' >> beam.transforms.util.BatchElements(max_batch_size=self.MAX_BATCH_SIZE) | 'Index batches of models' >> beam.ParDo(IndexBlogPostSummaries()) | 'Count the output' >> job_result_transforms.ResultsToJobRunResults()

class IndexBlogPostSummaries(beam.DoFn):
    """DoFn to index blog post summaries."""

    def process(self, blog_post_summaries: List[blog_domain.BlogPostSummary]) -> Iterable[result.Result[None, Exception]]:
        if False:
            i = 10
            return i + 15
        'Index blog post summaries and catch any errors.\n\n        Args:\n            blog_post_summaries: list(BlogPostSummaries). List of Blog Post\n                Summary domain objects to be indexed.\n\n        Yields:\n            JobRunResult. List containing one element, which is either SUCCESS,\n            or FAILURE.\n        '
        try:
            search_services.index_blog_post_summaries(blog_post_summaries)
            for _ in blog_post_summaries:
                yield result.Ok()
        except platform_search_services.SearchException as e:
            yield result.Err(e)