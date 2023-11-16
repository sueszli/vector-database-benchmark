import unittest
import numpy as np
import paddle
from paddle import nn

class MyModel(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = nn.Linear(100, 300)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.linear(x)

    @paddle.no_grad()
    def state_dict(self, destination=None, include_sublayers=True, structured_name_prefix='', use_hook=True):
        if False:
            i = 10
            return i + 15
        st = super().state_dict(destination=destination, include_sublayers=include_sublayers, structured_name_prefix=structured_name_prefix, use_hook=use_hook)
        st['linear.new_weight'] = paddle.transpose(st.pop('linear.weight'), [1, 0])
        return st

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        if False:
            while True:
                i = 10
        state_dict['linear.weight'] = paddle.transpose(state_dict.pop('linear.new_weight'), [1, 0])
        return super().set_state_dict(state_dict)

class MyModel2(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = nn.Linear(100, 300)

    def forward(self, x):
        if False:
            return 10
        return self.linear(x)

def is_state_dict_equal(model1, model2):
    if False:
        i = 10
        return i + 15
    st1 = model1.state_dict()
    st2 = model2.state_dict()
    assert set(st1.keys()) == set(st2.keys())
    for (k, v1) in st1.items():
        v2 = st2[k]
        if not np.array_equal(v1.numpy(), v2.numpy()):
            return False
    return True

class TestStateDictConvert(unittest.TestCase):

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        model1 = MyModel()
        model2 = MyModel()
        self.assertFalse(is_state_dict_equal(model1, model2))
        model2.set_state_dict(model1.state_dict())
        self.assertTrue(is_state_dict_equal(model1, model2))

class TestStateDictReturn(unittest.TestCase):

    def test_missing_keys_and_unexpected_keys(self):
        if False:
            i = 10
            return i + 15
        model1 = MyModel2()
        tmp_dict = {}
        tmp_dict['unexpected_keys'] = paddle.to_tensor([1])
        (missing_keys, unexpected_keys) = model1.set_state_dict(tmp_dict)
        self.assertEqual(len(missing_keys), 2)
        self.assertEqual(missing_keys[0], 'linear.weight')
        self.assertEqual(missing_keys[1], 'linear.bias')
        self.assertEqual(len(unexpected_keys), 1)
        self.assertEqual(unexpected_keys[0], 'unexpected_keys')
if __name__ == '__main__':
    unittest.main()