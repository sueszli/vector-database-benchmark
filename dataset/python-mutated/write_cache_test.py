"""Tests for write_cache."""
import unittest
from unittest.mock import patch
import apache_beam as beam
from apache_beam.runners.interactive import augmented_pipeline as ap
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.caching import write_cache
from apache_beam.runners.interactive.testing.pipeline_assertion import assert_pipeline_proto_equal
from apache_beam.runners.interactive.testing.test_cache_manager import InMemoryCache

class WriteCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ie.new_env()

    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.get_cache_manager')
    def test_write_cache(self, mocked_get_cache_manager):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        pcoll = p | beam.Create([1, 2, 3])
        ib.watch(locals())
        cache_manager = InMemoryCache()
        mocked_get_cache_manager.return_value = cache_manager
        aug_p = ap.AugmentedPipeline(p)
        key = repr(aug_p._cacheables[pcoll].to_key())
        pipeline_proto = p.to_runner_api()
        write_cache.WriteCache(pipeline_proto, aug_p._context, aug_p._cache_manager, aug_p._cacheables[pcoll]).write_cache()
        actual_pipeline = pipeline_proto
        transform = write_cache._WriteCacheTransform(aug_p._cache_manager, key)
        _ = pcoll | 'sink_cache_' + key >> transform
        expected_pipeline = p.to_runner_api()
        assert_pipeline_proto_equal(self, expected_pipeline, actual_pipeline)
        pcoll_id = aug_p._context.pcollections.get_id(pcoll)
        write_transform_id = None
        for (transform_id, transform) in actual_pipeline.components.transforms.items():
            if pcoll_id in transform.inputs.values():
                write_transform_id = transform_id
                break
        self.assertIsNotNone(write_transform_id)
        self.assertIn('sink', actual_pipeline.components.transforms[write_transform_id].unique_name)
if __name__ == '__main__':
    unittest.main()