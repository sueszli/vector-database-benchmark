import unittest
import numpy as np
from test_imperative_lod_tensor_to_selected_rows import SimpleNet
from paddle import base
from paddle.base import core
from paddle.base.dygraph import base as imperative_base
call_forward_post_hook = False
call_forward_pre_hook = False

def forward_post_hook(layer, input, output):
    if False:
        i = 10
        return i + 15
    global call_forward_post_hook
    call_forward_post_hook = True

def forward_pre_hook(layer, input):
    if False:
        while True:
            i = 10
    global call_forward_pre_hook
    call_forward_pre_hook = True

def forward_post_hook1(layer, input, output):
    if False:
        for i in range(10):
            print('nop')
    return output * 2

def forward_pre_hook1(layer, input):
    if False:
        return 10
    input_return = (input[0] * 2, input[1])
    return input_return

class Test_Forward_Hook(unittest.TestCase):

    def test_forward_hook_return_value(self):
        if False:
            i = 10
            return i + 15
        seed = 90
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            with base.dygraph.guard(place):
                base.default_startup_program().random_seed = seed
                base.default_main_program().random_seed = seed
                base.set_flags({'FLAGS_sort_sum_gradient': True})
                input_word = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(6, 3).astype('int64')
                input_word1 = input_word * 2
                input_word = input_word.reshape((-1, 3, 1))
                input_word1 = input_word1.reshape((-1, 3, 1))
                y_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(6, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                input = imperative_base.to_variable(input_word)
                input1 = imperative_base.to_variable(input_word1)
                y = imperative_base.to_variable(y_data)
                simplenet = SimpleNet(hidden_size=20, vocab_size=32, num_steps=3, init_scale=0.1, is_sparse=False, dtype='float32')
                outs_origin = simplenet(input, y)
                outs_origin1 = simplenet(input1, y)
                forward_pre_hook_handle1 = simplenet.register_forward_pre_hook(forward_pre_hook1)
                outs_pre_hook = simplenet(input, y)
                np.testing.assert_array_equal(outs_pre_hook.numpy(), outs_origin1.numpy())
                forward_pre_hook_handle1.remove()
                outs_pre_hook = simplenet(input, y)
                np.testing.assert_array_equal(outs_pre_hook.numpy(), outs_origin.numpy())
                forward_post_hook_handle1 = simplenet.register_forward_post_hook(forward_post_hook1)
                outs_forward_hook = simplenet(input, y)
                np.testing.assert_array_equal(outs_forward_hook.numpy(), outs_origin.numpy() * 2)
                forward_post_hook_handle1.remove()
                outs_forward_hook = simplenet(input, y)
                np.testing.assert_array_equal(outs_forward_hook.numpy(), outs_origin.numpy())

    def test_forward_hook(self):
        if False:
            i = 10
            return i + 15
        seed = 90
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            with base.dygraph.guard(place):
                base.default_startup_program().random_seed = seed
                base.default_main_program().random_seed = seed
                base.set_flags({'FLAGS_sort_sum_gradient': True})
                global call_forward_post_hook
                global call_forward_pre_hook
                input_word = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(6, 3).astype('int64')
                input_word = input_word.reshape((-1, 3, 1))
                y_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(6, 3).astype('int64')
                y_data = y_data.reshape((-1, 1))
                input = imperative_base.to_variable(input_word)
                y = imperative_base.to_variable(y_data)
                simplenet = SimpleNet(hidden_size=20, vocab_size=32, num_steps=3, init_scale=0.1, is_sparse=False, dtype='float32')
                outs_origin = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertFalse(call_forward_pre_hook)
                forward_post_hook_handle = simplenet.register_forward_post_hook(forward_post_hook)
                forward_pre_hook_handle = simplenet.register_forward_pre_hook(forward_pre_hook)
                outs_hook = simplenet(input, y)
                self.assertTrue(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)
                outs_hook = simplenet(input, y)
                self.assertTrue(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)
                forward_post_hook_handle.remove()
                call_forward_post_hook = False
                call_forward_pre_hook = False
                outs_remove_forward_hook = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)
                forward_pre_hook_handle.remove()
                call_forward_post_hook = False
                call_forward_pre_hook = False
                outs_remove_hook = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertFalse(call_forward_pre_hook)
if __name__ == '__main__':
    unittest.main()