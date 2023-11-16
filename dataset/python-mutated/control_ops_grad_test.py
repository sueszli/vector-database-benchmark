import unittest
from caffe2.python import core, test_util, workspace
from caffe2.python.control_ops_grad import disambiguate_grad_if_op_output
from caffe2.python.model_helper import ModelHelper
import numpy as np

class TestControl(test_util.TestCase):

    def test_disambiguate_grad_if_op_output(self):
        if False:
            while True:
                i = 10
        workspace.FeedBlob('cond', np.array(True))
        workspace.FeedBlob('then_grad', np.array(1))
        workspace.FeedBlob('else_grad', np.array(2))
        then_model = ModelHelper(name='then_test_model')
        then_model.net.Copy('then_grad', 'input_grad')
        else_model = ModelHelper(name='else_test_model')
        else_model.net.Copy('else_grad', 'else_temp_grad')
        else_model.net.Copy('else_temp', 'input_grad')
        grad_op = core.CreateOperator('If', ['cond', 'then_grad', 'else_grad'], ['input_grad', 'else_temp_grad'], then_net=then_model.net.Proto(), else_net=else_model.net.Proto())
        new_grad_output = 'input_grad' + '_autosplit_' + '0'
        disambiguate_grad_if_op_output(grad_op, 0, new_grad_output)
        self.assertEqual(grad_op.output[0], new_grad_output)
        for arg in grad_op.arg:
            if arg.name == 'else_net':
                self.assertEqual(arg.n.op[1].output[0], new_grad_output)
            else:
                self.assertEqual(arg.name, 'then_net')
if __name__ == '__main__':
    unittest.main()