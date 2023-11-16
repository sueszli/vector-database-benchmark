"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
from tensorflow.python.compat import v2_compat
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_v2_correctness_base_test

class TPUEmbeddingCorrectnessTest(tpu_embedding_v2_correctness_base_test.TPUEmbeddingCorrectnessBaseTest):

    @parameterized.parameters(['sgd', 'adagrad', 'adam', 'ftrl', 'adagrad_momentum'])
    def test_embedding(self, optimizer_name):
        if False:
            return 10
        if optimizer_name != 'sgd':
            self.skip_if_oss()
        self._test_embedding(optimizer_name, training=True, sparse=True, is_high_dimensional=False)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()