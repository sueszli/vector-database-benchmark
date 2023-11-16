import unittest
import warnings
import numpy as np
import paddle
from paddle import base

class TestBackwardInferVarDataTypeShape(unittest.TestCase):

    def test_backward_infer_var_data_type_shape(self):
        if False:
            return 10
        paddle.enable_static()
        program = base.default_main_program()
        dy = program.global_block().create_var(name='Tmp@GRAD', shape=[1, 1], dtype=np.float32, persistable=True)
        base.backward._infer_var_data_type_shape_('Tmp@GRAD', program.global_block())
        res = False
        with warnings.catch_warnings():
            res = True
        self.assertTrue(res)
if __name__ == '__main__':
    unittest.main()