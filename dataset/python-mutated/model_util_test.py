"""Test utility functions for manipulating Keras models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from object_detection.utils import model_util

class ExtractSubmodelUtilTest(tf.test.TestCase):

    def test_simple_model(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = tf.keras.Input(shape=(256,))
        x = tf.keras.layers.Dense(128, activation='relu', name='a')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu', name='b')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='c')(x)
        x = tf.keras.layers.Dense(16, activation='relu', name='d')(x)
        x = tf.keras.layers.Dense(8, activation='relu', name='e')(x)
        predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=predictions)
        new_in = model.get_layer(name='b').input
        new_out = model.get_layer(name='d').output
        new_model = model_util.extract_submodel(model=model, inputs=new_in, outputs=new_out)
        batch_size = 3
        ones = tf.ones((batch_size, 128))
        final_out = new_model(ones)
        self.assertAllEqual(final_out.shape, (batch_size, 16))
if __name__ == '__main__':
    tf.test.main()