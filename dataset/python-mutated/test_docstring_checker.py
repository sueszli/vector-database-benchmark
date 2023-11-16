import astroid
import docstring_checker
import pylint.testutils

class TestDocstring(pylint.testutils.CheckerTestCase):
    CHECKER_CLASS = docstring_checker.DocstringChecker

    def test_one_line(self):
        if False:
            for i in range(10):
                print('nop')
        func_node = astroid.extract_node('\n        def test():\n            """get\n            news.\n            """\n            if True:\n                return 5\n            return 5\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9001' == got[0][0]

    def test_one_line_1(self):
        if False:
            for i in range(10):
                print('nop')
        func_node = astroid.extract_node('\n        def test():\n            """get news"""\n            if True:\n                return 5\n            return 5\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9002' == got[0][0]

    def test_args(self):
        if False:
            print('Hello World!')
        func_node = astroid.extract_node('\n        def test(scale, mean):\n            """get news.\n            Args:\n                scale (int): scale is the number.\n            """\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9003' == got[0][0]

    def test_missing(self):
        if False:
            i = 10
            return i + 15
        func_node = astroid.extract_node('\n        def test():\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9005' == got[0][0]

    def test_indent(self):
        if False:
            return 10
        func_node = astroid.extract_node('\n        def test():\n            """ get get get get get get get get\n              get get get get get get get get.\n            """\n            pass\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9006' == got[0][0]

    def test_with_resturns(self):
        if False:
            print('Hello World!')
        func_node = astroid.extract_node('\n        def test():\n            """get news.\n            Args:\n                scale (int): scale is the number.\n            """\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            return mean\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9007' == got[0][0]

    def test_with_raises(self):
        if False:
            i = 10
            return i + 15
        func_node = astroid.extract_node('\n        def test():\n            """get news.\n            Args:\n                scale (int): scale is the number.\n            """\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            mean=scale\n            raise ValueError(\'A very specific bad thing happened.\')\n        ')
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 1
        assert 'W9008' == got[0][0]

    def test_no_message(self):
        if False:
            print('Hello World!')
        p = '\ndef fc(input,\n       size,\n       num_flatten_dims=1,\n       param_attr=None,\n       bias_attr=None,\n       act=None,\n       name=None):\n    """\n    **Fully Connected Layer**\n    The fully connected layer can take multiple tensors as its inputs. It\n    creates a variable called weights for each input tensor, which represents\n    a fully connected weight matrix from each input unit to each output unit.\n    The fully connected layer multiplies each input tensor with its coresponding\n    weight to produce an output Tensor. If multiple input tensors are given,\n    the results of multiple multiplications will be sumed up. If bias_attr is\n    not None, a bias variable will be created and added to the output. Finally,\n    if activation is not None, it will be applied to the output as well.\n    This process can be formulated as follows:\n\n    Args:\n        input (Variable|list of Variable): The input tensor(s) of this layer, and the dimension of\n            the input tensor(s) is at least 2.\n        size(int): The number of output units in this layer.\n        num_flatten_dims (int, default 1): The fc layer can accept an input tensor with more than\n            two dimensions. If this happens, the multidimensional tensor will first be flattened\n            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input\n            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)\n            dimensions will be flatten to form the first dimension of the final matrix (height of\n            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to\n            form the second dimension of the final matrix (width of the matrix). For example, suppose\n            `X` is a 6-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.\n            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30].\n        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable\n            parameters/weights of this layer.\n        bias_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for the bias\n            of this layer. If it is set to None, no bias will be added to the output units.\n        act (str, default None): Activation to be applied to the output of this layer.\n        name (str, default None): The name of this layer.\n    Returns:\n        A tensor variable storing the transformation result.\n    Raises:\n        ValueError: If rank of the input tensor is less than 2.\n    Examples:\n        .. code-block:: python\n            data = paddle.static.data(name="data", shape=[-1, 32, 32], dtype="float32")\n            fc = paddle.static.nn.fc(x=data, size=1000, activation="tanh")\n    """\n    raise ValueError(\'A very specific bad thing happened.\')\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    size = 1\n    return size\n    '
        func_node = astroid.extract_node(p)
        self.checker.visit_functiondef(func_node)
        got = self.linter.release_messages()
        assert len(got) == 0