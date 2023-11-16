import unittest
import warnings
import paddle
from paddle import base

class TestSaveModelWithoutVar(unittest.TestCase):

    def test_no_var_save(self):
        if False:
            for i in range(10):
                print('nop')
        data = paddle.static.data(name='data', shape=[-1, 1], dtype='float32')
        data_plus = data + 1
        if base.core.is_compiled_with_cuda():
            place = base.core.CUDAPlace(0)
        else:
            place = base.core.CPUPlace()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            paddle.static.io.save_inference_model('test', data, [data_plus], exe)
            expected_warn = 'no variable in your model, please ensure there are any variables in your model to save'
            self.assertTrue(len(w) > 0)
            self.assertTrue(expected_warn == str(w[-1].message))
if __name__ == '__main__':
    unittest.main()