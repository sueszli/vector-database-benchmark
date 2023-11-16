"""Unit tests for jobs.batch_jobs.blog_post_search_indexing_jobs."""
from __future__ import annotations
import datetime
import math
from core import utils
from core.domain import search_services
from core.jobs import job_test_utils
from core.jobs.batch_jobs import blog_post_search_indexing_jobs
from core.jobs.types import job_run_result
from core.platform import models
from typing import Dict, Final, List, Tuple, Type, Union
MYPY = False
if MYPY:
    from mypy_imports import blog_models
    from mypy_imports import search_services as platform_search_services
(blog_models,) = models.Registry.import_models([models.Names.BLOG])
platform_search_services = models.Registry.import_search_services()
StatsType = List[Tuple[str, List[Dict[str, Union[bool, int, str]]]]]

class IndexBlogPostSummariesInSearchJobTests(job_test_utils.JobTestBase):
    JOB_CLASS: Type[blog_post_search_indexing_jobs.IndexBlogPostsInSearchJob] = blog_post_search_indexing_jobs.IndexBlogPostsInSearchJob
    USER_ID_1: Final = 'id_1'
    USERNAME: Final = 'someUsername'

    def test_empty_storage(self) -> None:
        if False:
            return 10
        self.assert_job_output_is_empty()

    def test_indexes_non_deleted_model(self) -> None:
        if False:
            i = 10
            return i + 15
        blog_summary = self.create_model(blog_models.BlogPostSummaryModel, id='abcd', author_id=self.USER_ID_1, deleted=False, title='title', summary='blog_post_summary', url_fragment='sample-url-fragment', tags=['tag1', 'tag2'], thumbnail_filename='xyzabc', published_on=datetime.datetime.utcnow())
        blog_summary.update_timestamps()
        blog_summary.put()
        add_docs_to_index_swap = self.swap_with_checks(platform_search_services, 'add_documents_to_index', lambda _, __: None, expected_args=[([{'id': 'abcd', 'title': 'title', 'tags': ['tag1', 'tag2'], 'rank': math.floor(utils.get_time_in_millisecs(blog_summary.published_on))}], search_services.SEARCH_INDEX_BLOG_POSTS)])
        with add_docs_to_index_swap:
            self.assert_job_output_is([job_run_result.JobRunResult.as_stdout('SUCCESS: 1')])

    def test_indexes_non_deleted_models(self) -> None:
        if False:
            i = 10
            return i + 15
        date_time_now = datetime.datetime.utcnow()
        for i in range(5):
            blog_summary = self.create_model(blog_models.BlogPostSummaryModel, id='abcd%s' % i, author_id=self.USER_ID_1, deleted=False, title='title', summary='blog_post_summary', url_fragment='sample-url-fragment', tags=['tag1', 'tag2'], thumbnail_filename='xyzabc', published_on=date_time_now)
            blog_summary.update_timestamps()
            blog_summary.put()
        add_docs_to_index_swap = self.swap_with_checks(platform_search_services, 'add_documents_to_index', lambda _, __: None, expected_args=[([{'id': 'abcd%s' % i, 'title': 'title', 'tags': ['tag1', 'tag2'], 'rank': math.floor(utils.get_time_in_millisecs(blog_summary.published_on))}], search_services.SEARCH_INDEX_BLOG_POSTS) for i in range(5)])
        max_batch_size_swap = self.swap(blog_post_search_indexing_jobs.IndexBlogPostsInSearchJob, 'MAX_BATCH_SIZE', 1)
        with add_docs_to_index_swap, max_batch_size_swap:
            self.assert_job_output_is([job_run_result.JobRunResult.as_stdout('SUCCESS: 5')])

    def test_reports_failed_when_indexing_fails(self) -> None:
        if False:
            print('Hello World!')
        blog_summary = self.create_model(blog_models.BlogPostSummaryModel, id='abcd', author_id=self.USER_ID_1, deleted=False, title='title', summary='blog_post_summary', url_fragment='sample-url-fragment', tags=['tag1', 'tag2'], thumbnail_filename='xyzabc', published_on=datetime.datetime.utcnow())
        blog_summary.update_timestamps()
        blog_summary.put()

        def add_docs_to_index_mock(unused_documents: Dict[str, Union[int, str, List[str]]], unused_index_name: str) -> None:
            if False:
                print('Hello World!')
            raise platform_search_services.SearchException('search exception')
        add_docs_to_index_swap = self.swap_with_checks(platform_search_services, 'add_documents_to_index', add_docs_to_index_mock, expected_args=[([{'id': 'abcd', 'title': 'title', 'tags': ['tag1', 'tag2'], 'rank': math.floor(utils.get_time_in_millisecs(blog_summary.published_on))}], search_services.SEARCH_INDEX_BLOG_POSTS)])
        with add_docs_to_index_swap:
            self.assert_job_output_is([job_run_result.JobRunResult.as_stderr('ERROR: "search exception": 1')])

    def test_skips_deleted_model(self) -> None:
        if False:
            print('Hello World!')
        blog_summary = self.create_model(blog_models.BlogPostSummaryModel, id='abcd', author_id=self.USER_ID_1, deleted=True, title='title', summary='blog_post_summary', url_fragment='sample-url-fragment', tags=['tag1', 'tag2'], thumbnail_filename='xyzabc', published_on=datetime.datetime.utcnow())
        blog_summary.update_timestamps()
        blog_summary.put()
        add_docs_to_index_swap = self.swap_with_checks(platform_search_services, 'add_documents_to_index', lambda _, __: None, called=False)
        with add_docs_to_index_swap:
            self.assert_job_output_is_empty()

    def test_skips_draft_blog_post_model(self) -> None:
        if False:
            print('Hello World!')
        blog_summary = self.create_model(blog_models.BlogPostSummaryModel, id='abcd', author_id=self.USER_ID_1, deleted=False, title='title', summary='blog_post_summary', url_fragment='sample-url-fragment', tags=['tag1', 'tag2'], thumbnail_filename='xyzabc', published_on=None)
        blog_summary.update_timestamps()
        blog_summary.put()
        add_docs_to_index_swap = self.swap_with_checks(platform_search_services, 'add_documents_to_index', lambda _, __: None, expected_args=[([], search_services.SEARCH_INDEX_BLOG_POSTS)])
        with add_docs_to_index_swap:
            self.assert_job_output_is([job_run_result.JobRunResult.as_stdout('SUCCESS: 1')])