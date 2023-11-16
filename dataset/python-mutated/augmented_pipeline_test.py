"""Tests for augmented_pipeline module."""
import unittest
import apache_beam as beam
from apache_beam.runners.interactive import augmented_pipeline as ap
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie

class CacheableTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ie.new_env()

    def test_find_all_cacheables(self):
        if False:
            return 10
        p = beam.Pipeline()
        cacheable_pcoll_1 = p | beam.Create([1, 2, 3])
        cacheable_pcoll_2 = cacheable_pcoll_1 | beam.Map(lambda x: x * x)
        ib.watch(locals())
        aug_p = ap.AugmentedPipeline(p)
        cacheables = aug_p.cacheables()
        self.assertIn(cacheable_pcoll_1, cacheables)
        self.assertIn(cacheable_pcoll_2, cacheables)

    def test_ignore_cacheables(self):
        if False:
            while True:
                i = 10
        p = beam.Pipeline()
        cacheable_pcoll_1 = p | 'cacheable_pcoll_1' >> beam.Create([1, 2, 3])
        cacheable_pcoll_2 = p | 'cacheable_pcoll_2' >> beam.Create([4, 5, 6])
        ib.watch(locals())
        aug_p = ap.AugmentedPipeline(p, (cacheable_pcoll_1,))
        cacheables = aug_p.cacheables()
        self.assertIn(cacheable_pcoll_1, cacheables)
        self.assertNotIn(cacheable_pcoll_2, cacheables)

    def test_ignore_pcoll_from_other_pipeline(self):
        if False:
            i = 10
            return i + 15
        p = beam.Pipeline()
        p2 = beam.Pipeline()
        cacheable_from_p2 = p2 | beam.Create([1, 2, 3])
        ib.watch(locals())
        aug_p = ap.AugmentedPipeline(p)
        cacheables = aug_p.cacheables()
        self.assertNotIn(cacheable_from_p2, cacheables)

class AugmentTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ie.new_env()

    def test_error_when_pcolls_from_mixed_pipelines(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        cacheable_from_p = p | beam.Create([1, 2, 3])
        p2 = beam.Pipeline()
        cacheable_from_p2 = p2 | beam.Create([1, 2, 3])
        ib.watch(locals())
        self.assertRaises(AssertionError, lambda : ap.AugmentedPipeline(p, (cacheable_from_p, cacheable_from_p2)))
if __name__ == '__main__':
    unittest.main()