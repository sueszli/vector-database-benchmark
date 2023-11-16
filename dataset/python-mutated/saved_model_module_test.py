"""Tests for tensorflow_hub.saved_model."""
import os
import tensorflow as tf
import tensorflow_hub as hub

def _double(input_):
    if False:
        print('Hello World!')
    return input_ * 2

class MyModel(tf.Module):

    @tf.function
    def __call__(self, input_):
        if False:
            while True:
                i = 10
        return _double(input_)

class SavedModelTest(tf.test.TestCase):

    def _create_tf2_saved_model(self):
        if False:
            print('Hello World!')
        model_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        model = MyModel()

        @tf.function
        def serving_default(input_):
            if False:
                return 10
            return {'output': model(input_)}
        signature_function = serving_default.get_concrete_function(tf.TensorSpec(shape=[3], dtype=tf.float32))
        tf.saved_model.save(model, model_dir, signatures={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_function})
        return model_dir

    def testLoadSavedModel(self):
        if False:
            i = 10
            return i + 15
        saved_model_path = self._create_tf2_saved_model()
        loaded = hub.load(saved_model_path)
        self.assertAllClose(loaded.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](tf.constant([2.0, 4.0, 5.0]))['output'], tf.constant([4.0, 8.0, 10.0]))
if __name__ == '__main__':
    tf.test.main()