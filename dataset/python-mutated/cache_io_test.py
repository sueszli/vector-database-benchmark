"""Unit tests for jobs.io.cache_io."""
from __future__ import annotations
from core.domain import caching_services
from core.jobs import job_test_utils
from core.jobs.io import cache_io
import apache_beam as beam

class FlushCacheTests(job_test_utils.PipelinedTestBase):

    def test_cache_is_flushed(self) -> None:
        if False:
            print('Hello World!')
        items = [1] * 100
        called_functions = {'flush_caches': False}

        class MockMemoryCachingServices:

            @staticmethod
            def flush_caches() -> None:
                if False:
                    print('Hello World!')
                'Flush cache.'
                called_functions['flush_caches'] = True
        with self.swap(caching_services, 'memory_cache_services', MockMemoryCachingServices):
            self.assert_pcoll_equal(self.pipeline | beam.Create(items) | cache_io.FlushCache(), [None])
        self.assertTrue(called_functions['flush_caches'])