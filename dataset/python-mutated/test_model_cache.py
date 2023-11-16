import unittest
from uuid import uuid4
from pyspark.ml.model_cache import ModelCache
from pyspark.testing.mlutils import SparkSessionTestCase

class ModelCacheTests(SparkSessionTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ModelCacheTests, self).setUp()

    def test_cache(self):
        if False:
            for i in range(10):
                print('nop')

        def predict_fn(inputs):
            if False:
                return 10
            return inputs
        uuids = [uuid4() for i in range(10)]
        for uuid in uuids:
            ModelCache.add(uuid, predict_fn)
        self.assertTrue(len(ModelCache._models) == 3)
        self.assertTrue(list(ModelCache._models.keys()) == uuids[7:10])
        _ = ModelCache.get(uuids[8])
        expected_uuids = uuids[7:8] + uuids[9:10] + [uuids[8]]
        self.assertTrue(list(ModelCache._models.keys()) == expected_uuids)
if __name__ == '__main__':
    from pyspark.ml.tests.test_model_cache import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)