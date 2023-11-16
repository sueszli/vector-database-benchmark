"""Test .py file for pybind11 files for SavedModelImpl functions LoadSvaedModel & Run."""
from tensorflow.core.tfrt.saved_model.python import _pywrap_saved_model
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

class SavedModelLoadSavedModelRunTest(test.TestCase):

    def test_give_me_a_name(self):
        if False:
            print('Hello World!')
        with context.eager_mode(), ops.device('CPU'):
            inputs = [constant_op.constant([0, 1, 2, 3, 4, 5, 6, 7]), constant_op.constant([1, 5, 8, 9, 21, 54, 67]), constant_op.constant([90, 81, 32, 13, 24, 55, 46, 67])]
        cpp_tensor = _pywrap_saved_model.RunConvertor(inputs)
        return cpp_tensor
if __name__ == '__main__':
    test.main()