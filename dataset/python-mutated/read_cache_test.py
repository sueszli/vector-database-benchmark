"""Tests for read_cache."""
import unittest
from unittest.mock import patch
import apache_beam as beam
from apache_beam.runners.interactive import augmented_pipeline as ap
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.caching import read_cache
from apache_beam.runners.interactive.testing.pipeline_assertion import assert_pipeline_proto_equal
from apache_beam.runners.interactive.testing.test_cache_manager import InMemoryCache

class ReadCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ie.new_env()

    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.get_cache_manager')
    def test_read_cache(self, mocked_get_cache_manager):
        if False:
            for i in range(10):
                print('nop')
        p = beam.Pipeline()
        pcoll = p | beam.Create([1, 2, 3])
        consumer_transform = beam.Map(lambda x: x * x)
        _ = pcoll | consumer_transform
        ib.watch(locals())
        cache_manager = InMemoryCache()
        mocked_get_cache_manager.return_value = cache_manager
        aug_p = ap.AugmentedPipeline(p)
        key = repr(aug_p._cacheables[pcoll].to_key())
        cache_manager.write('test', 'full', key)
        pcoll_id = aug_p._context.pcollections.get_id(pcoll)
        consumer_transform_id = None
        pipeline_proto = p.to_runner_api()
        for (transform_id, transform) in pipeline_proto.components.transforms.items():
            if pcoll_id in transform.inputs.values():
                consumer_transform_id = transform_id
                break
        self.assertIsNotNone(consumer_transform_id)
        (_, cache_id) = read_cache.ReadCache(pipeline_proto, aug_p._context, aug_p._cache_manager, aug_p._cacheables[pcoll]).read_cache()
        actual_pipeline = pipeline_proto
        transform = read_cache._ReadCacheTransform(aug_p._cache_manager, key)
        p | 'source_cache_' + key >> transform
        expected_pipeline = p.to_runner_api()
        assert_pipeline_proto_equal(self, expected_pipeline, actual_pipeline)
        inputs = actual_pipeline.components.transforms[consumer_transform_id].inputs
        self.assertIn(cache_id, inputs.values())
        self.assertNotIn(pcoll_id, inputs.values())
if __name__ == '__main__':
    unittest.main()