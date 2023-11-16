"""Tests for TPU Initialization."""
from absl.testing import parameterized
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import test

class TPUInitializationTest(parameterized.TestCase, test.TestCase):

    def test_tpu_initialization(self):
        if False:
            return 10
        resolver = tpu_cluster_resolver.TPUClusterResolver('')
        tpu_cluster_resolver.initialize_tpu_system(resolver)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()