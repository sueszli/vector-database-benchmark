"""
Test the tensor attribute grad_fn and the properties of the reverse node grad_node, such as next_function
"""
import unittest
import paddle
from paddle import nn

class Testmodel(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x):
        if False:
            print('Hello World!')
        y = x ** 2
        y = x + y
        return y

class TestAnonmousSurvey(unittest.TestCase):
    """
    Test the tensor attribute grad_fn and the properties of the reverse node grad_node, such as next_function

    """

    def init_graph(self):
        if False:
            i = 10
            return i + 15
        'define reversed graph\n\n        func_name [str]: represents the name of the operator node\n        next_funcs [dict]: represents the operator node\n        '
        self.grad_fn_1 = {'func_name': 'GradNodeAccumulation', 'next_funcs': {}}
        self.grad_fn_2 = {'func_name': 'PowGradNode', 'next_funcs': {'GradNodeAccumulation': self.grad_fn_1}}
        self.grad_fn_3 = {'func_name': 'AddGradNode', 'next_funcs': {'GradNodeAccumulation': self.grad_fn_1, 'PowGradNode': self.grad_fn_2}}
        self.output_grad_fn = {'grad_fn': self.grad_fn_3}

    def init_data(self):
        if False:
            i = 10
            return i + 15
        'define output of model\n\n        the final output will be saved self.output\n        '
        model = Testmodel()
        x = paddle.randn([1, 3, 24, 24])
        x.stop_gradient = False
        self.output = model(x)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.init_graph()
        self.init_data()

    def test_grad_fn_and_next_funs(self):
        if False:
            return 10
        self.check_func(self.output.grad_fn, self.output_grad_fn['grad_fn'])

    def check_func(self, grad_fn, grad_fn_json) -> None:
        if False:
            while True:
                i = 10
        '\n        Check each node, grad_fn is tensor attribute. grad_fn_json is structure of next_node.\n\n        Args:\n            grad_fn (grad_fn): grad_fn of node\n            grad_fn_json (dict): grad_node_json of node\n        '
        self.assertEqual(grad_fn.name(), grad_fn_json['func_name'])
        if hasattr(grad_fn, 'next_functions') and grad_fn.next_functions[0]:
            next_funcs_json = grad_fn_json['next_funcs']
            for u in grad_fn.next_functions:
                self.check_func(u, next_funcs_json[u.name()])
if __name__ == '__main__':
    unittest.main()