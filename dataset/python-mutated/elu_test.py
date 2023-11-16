import numpy as np
import pytest
from keras import testing
from keras.layers.activations import elu

class ELUTest(testing.TestCase):

    def test_config(self):
        if False:
            return 10
        elu_layer = elu.ELU()
        self.run_class_serialization_test(elu_layer)

    @pytest.mark.requires_trainable_backend
    def test_elu(self):
        if False:
            return 10
        self.run_layer_test(elu.ELU, init_kwargs={}, input_shape=(2, 3, 4), supports_masking=True)

    def test_correctness(self):
        if False:
            print('Hello World!')

        def np_elu(x, alpha=1.0):
            if False:
                while True:
                    i = 10
            return (x > 0) * x + (x <= 0) * alpha * (np.exp(x) - 1)
        x = np.random.random((2, 2, 5))
        elu_layer = elu.ELU()
        self.assertAllClose(elu_layer(x), np_elu(x))
        elu_layer = elu.ELU(alpha=0.7)
        self.assertAllClose(elu_layer(x), np_elu(x, alpha=0.7))