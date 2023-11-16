import pytest
from bigdl.dllib.feature.image import *
from bigdl.orca.test_zoo_utils import ZooTestCase
import tensorflow as tf
import numpy as np
import os
from bigdl.orca.tfpark import KerasModel
resource_path = os.path.join(os.path.split(__file__)[0], '../resources')

class TestTFParkModel(ZooTestCase):

    def setup_method(self, method):
        if False:
            return 10
        tf.keras.backend.clear_session()
        super(TestTFParkModel, self).setup_method(method)

    def create_multi_input_output_model(self):
        if False:
            return 10
        data1 = tf.keras.layers.Input(shape=[10])
        data2 = tf.keras.layers.Input(shape=[10])
        x1 = tf.keras.layers.Flatten()(data1)
        x1 = tf.keras.layers.Dense(10, activation='relu')(x1)
        pred1 = tf.keras.layers.Dense(2, activation='softmax')(x1)
        x2 = tf.keras.layers.Flatten()(data2)
        x2 = tf.keras.layers.Dense(10, activation='relu')(x2)
        pred2 = tf.keras.layers.Dense(2)(x2)
        model = tf.keras.models.Model(inputs=[data1, data2], outputs=[pred1, pred2])
        model.compile(optimizer='rmsprop', loss=['sparse_categorical_crossentropy', 'mse'])
        return model

    def create_training_data(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, 20)
        return (x, y)

    def test_training_with_validation_data_distributed_multi_heads(self):
        if False:
            print('Hello World!')
        keras_model = self.create_multi_input_output_model()
        model = KerasModel(keras_model)
        (x, y) = self.create_training_data()
        (val_x, val_y) = self.create_training_data()
        model.fit([x, x], [y, y], validation_data=([val_x, val_x], [val_y, val_y]), batch_size=4, distributed=True)

    def test_invalid_data_handling(self):
        if False:
            i = 10
            return i + 15
        keras_model = self.create_multi_input_output_model()
        model = KerasModel(keras_model)
        (x, y) = self.create_training_data()
        (val_x, val_y) = self.create_training_data()
        with pytest.raises(RuntimeError) as excinfo:
            model.fit([x, x], [y, y, y], batch_size=4, distributed=True)
        assert 'model_target number does not match data number' in str(excinfo.value)
        with pytest.raises(RuntimeError) as excinfo:
            model.fit({'input_1': x}, [y, y], batch_size=4, distributed=True)
        assert 'all model_input names should exist in data' in str(excinfo.value)
if __name__ == '__main__':
    pytest.main([__file__])