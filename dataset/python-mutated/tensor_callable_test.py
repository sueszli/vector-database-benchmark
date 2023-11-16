"""Tests for SaveableObject compatibility."""
import os
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.eager import test
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import base

class IncrementWhenSave(base.Trackable):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.read_counter = variables.Variable(0)

    def _serialize_to_tensors(self):
        if False:
            while True:
                i = 10

        def _get_and_increment_counter():
            if False:
                return 10
            value = self.read_counter.read_value()
            self.read_counter.assign_add(1)
            return value
        return {'read_counter': tensor_callable.Callable(_get_and_increment_counter, self.read_counter.dtype, self.read_counter.device)}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            i = 10
            return i + 15
        self.read_counter.assign(restored_tensors['read_counter'])

class CallableTest(test.TestCase):

    def test_callable(self):
        if False:
            return 10
        trackable = IncrementWhenSave()
        ckpt = checkpoint.Checkpoint(attr=trackable)
        prefix = os.path.join(self.get_temp_dir(), 'ckpt')
        save_path = ckpt.save(prefix)
        self.assertEqual(1, self.evaluate(trackable.read_counter))
        ckpt.save(prefix)
        self.assertEqual(2, self.evaluate(trackable.read_counter))
        ckpt.restore(save_path)
        self.assertEqual(0, self.evaluate(trackable.read_counter))

    def test_callable_saved_model_compatibility(self):
        if False:
            print('Hello World!')
        trackable = IncrementWhenSave()
        trackable.read_counter.assign(15)
        save_path = os.path.join(self.get_temp_dir(), 'saved_model')
        with self.assertRaisesRegex(NotImplementedError, 'returns a Callable'):
            saved_model_save.save(trackable, save_path)
if __name__ == '__main__':
    test.main()