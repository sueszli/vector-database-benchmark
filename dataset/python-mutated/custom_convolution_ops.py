from cntk import *
from cntk.ops.functions import UserFunction
import numpy as np
import cntk as C

class Sign(UserFunction):

    def __init__(self, arg, name='Sign'):
        if False:
            print('Hello World!')
        super(Sign, self).__init__([arg], as_numpy=False, name=name)
        (self.action, self.actionArg) = self.signFunc(arg)
        (self.grad, self.gradArg, self.gradRoot) = self.gradFunc(arg)

    def signFunc(self, arg):
        if False:
            i = 10
            return i + 15
        signIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        actionfunc = greater(signIn, 0)
        return (element_select(actionfunc, actionfunc, -1), signIn)

    def gradFunc(self, arg):
        if False:
            return 10
        gradIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        gradRoot = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        signGrad = abs(gradIn)
        signGrad = less_equal(signGrad, 1)
        return (element_times(gradRoot, signGrad), gradIn, gradRoot)

    def forward(self, argument, device, outputs_to_retain):
        if False:
            for i in range(10):
                print('nop')
        (_, output_values) = self.action.forward({self.actionArg: argument}, self.action.outputs, device=device, as_numpy=False)
        return (argument.deep_clone(), output_values[self.action.output])

    def backward(self, state, root_gradients):
        if False:
            print('Hello World!')
        val = state
        (_, output_values) = self.grad.forward({self.gradArg: val, self.gradRoot: root_gradients}, self.grad.outputs, device=state.device(), as_numpy=False)
        return output_values[self.grad.output]

    def infer_outputs(self):
        if False:
            i = 10
            return i + 15
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

class pySign(UserFunction):

    def __init__(self, arg, name='pySign'):
        if False:
            return 10
        super(pySign, self).__init__([arg], name=name)

    def forward(self, argument, device=None, outputs_to_retain=None):
        if False:
            return 10
        sign = np.sign(argument)
        np.place(sign, sign == 0, -1)
        return (argument, sign)

    def backward(self, state, root_gradients):
        if False:
            i = 10
            return i + 15
        input = state
        grad = np.abs(input)
        grad = np.less_equal(grad, 1)
        return grad * root_gradients

    def infer_outputs(self):
        if False:
            i = 10
            return i + 15
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

class MultibitKernel(UserFunction):

    def __init__(self, arg1, arg2, name='MultibitKernel'):
        if False:
            for i in range(10):
                print('nop')
        super(MultibitKernel, self).__init__([arg1], as_numpy=False, name=name)
        self.bit_map = arg2
        (self.action, self.actionArg) = self.multiFunc(arg1)
        (self.grad, self.gradArg, self.gradRoot) = self.gradFunc(arg1)

    def multiFunc(self, arg1):
        if False:
            i = 10
            return i + 15
        multiIn = C.input(shape=arg1.shape, dynamic_axes=arg1.dynamic_axes)
        bit_map = C.constant(self.bit_map)
        max_bits = self.bit_map.max()
        shape = multiIn.shape
        reformed = C.reshape(multiIn, (-1,))
        carry_over = multiIn
        approx = C.element_times(multiIn, 0)
        for i in range(max_bits):
            hot_vals = C.greater(bit_map, i)
            valid_vals = C.element_select(hot_vals, carry_over, 0)
            mean = C.element_divide(C.reduce_sum(C.reshape(C.abs(valid_vals), (valid_vals.shape[0], -1)), axis=1), C.reduce_sum(C.reshape(hot_vals, (hot_vals.shape[0], -1)), axis=1))
            mean = C.reshape(mean, (mean.shape[0], mean.shape[1], 1, 1))
            bits = C.greater(carry_over, 0)
            bits = C.element_select(bits, bits, -1)
            bits = C.element_select(hot_vals, bits, 0)
            approx = C.plus(approx, C.element_times(mean, bits))
            carry_over = C.plus(C.element_times(C.element_times(-1, bits), mean), carry_over)
        return (approx, multiIn)

    def gradFunc(self, arg):
        if False:
            for i in range(10):
                print('nop')
        gradIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        gradRoot = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        signGrad = C.abs(gradIn)
        bit_map = C.constant(self.bit_map)
        signGrad = C.less_equal(signGrad, bit_map)
        outGrad = signGrad
        outGrad = element_times(gradRoot, outGrad)
        return (outGrad, gradIn, gradRoot)

    def forward(self, argument, device, outputs_to_retain):
        if False:
            for i in range(10):
                print('nop')
        (_, output_values) = self.action.forward({self.actionArg: argument}, self.action.outputs, device=device, as_numpy=False)
        return (argument.deep_clone(), output_values[self.action.output])

    def backward(self, state, root_gradients):
        if False:
            return 10
        val = state
        (_, output_values) = self.grad.forward({self.gradArg: val, self.gradRoot: root_gradients}, self.grad.outputs, device=state.device(), as_numpy=False)
        return output_values[self.grad.output]

    def infer_outputs(self):
        if False:
            i = 10
            return i + 15
        return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

class Multibit(UserFunction):

    def __init__(self, arg1, arg2, name='Multibit'):
        if False:
            for i in range(10):
                print('nop')
        super(Multibit, self).__init__([arg1], as_numpy=False, name=name)
        self.bit_map = arg2

    def multiFunc(self, arg1):
        if False:
            for i in range(10):
                print('nop')
        multiIn = C.input(shape=arg1.shape, dynamic_axes=arg1.dynamic_axes)
        bit_map = C.constant(self.bit_map)
        max_bits = self.bit_map.max()
        shape = multiIn.shape
        reformed = C.reshape(multiIn, (-1,))
        carry_over = multiIn
        approx = C.element_times(multiIn, 0)
        for i in range(max_bits):
            hot_vals = C.greater(bit_map, i)
            valid_vals = C.element_select(hot_vals, carry_over, 0)
            mean = C.element_divide(C.reduce_sum(C.abs(valid_vals)), C.reduce_sum(hot_vals))
            bits = C.greater(carry_over, 0)
            bits = C.element_select(bits, bits, -1)
            bits = C.element_select(hot_vals, bits, 0)
            approx = C.plus(approx, C.element_times(mean, bits))
            carry_over = C.plus(C.element_times(C.element_times(-1, bits), mean), carry_over)
        return (approx, multiIn)

    def gradFunc(self, arg):
        if False:
            while True:
                i = 10
        gradIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        gradRoot = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        signGrad = C.abs(gradIn)
        bit_map = C.constant(self.bit_map)
        signGrad = C.less_equal(signGrad, bit_map)
        outGrad = signGrad
        outGrad = element_times(gradRoot, outGrad)
        return (outGrad, gradIn, gradRoot)

    def forward(self, argument, device, outputs_to_retain):
        if False:
            print('Hello World!')
        (_, output_values) = self.action.forward({self.actionArg: argument}, self.action.outputs, device=device, as_numpy=False)
        return (argument.deep_clone(), output_values[self.action.output])

    def backward(self, state, root_gradients):
        if False:
            i = 10
            return i + 15
        val = state
        (_, output_values) = self.grad.forward({self.gradArg: val, self.gradRoot: root_gradients}, self.grad.outputs, device=state.device(), as_numpy=False)
        return output_values[self.grad.output]

    def infer_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        output_vars = [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]
        (self.action, self.actionArg) = self.multiFunc(self.inputs[0])
        (self.grad, self.gradArg, self.gradRoot) = self.gradFunc(self.inputs[0])
        return output_vars

    def serialize(self):
        if False:
            for i in range(10):
                print('nop')
        return {'bit_map': np.asarray(self.bit_map, dtype=np.float32)}

    @staticmethod
    def deserialize(inputs, name, state):
        if False:
            while True:
                i = 10
        return Multibit(inputs[0], np.asarray(state['bit_map'], dtype=np.int32), name)

    def clone(self, cloned_inputs):
        if False:
            for i in range(10):
                print('nop')
        cloned_inputs[0].__class__ = C.Variable
        return Multibit(cloned_inputs[0], self.bit_map, self.name)

def CustomSign(input):
    if False:
        i = 10
        return i + 15
    return user_function(Sign(input))

def CustomPySign(input):
    if False:
        for i in range(10):
            print('nop')
    return user_function(pySign(input))

def CustomMultibit(input, bit_map, mean_bits=None):
    if False:
        i = 10
        return i + 15
    if mean_bits:
        bit_map = np.asarray(np.maximum(np.round(np.random.normal(mean_bits, 1, input.shape)), 1), dtype=np.int32)
        print('Mean Bits: ', np.mean(bit_map))
    elif type(bit_map) == int:
        length = C.reshape(input, -1)
        bit_map = [bit_map] * length.shape[0]
        bit_map = np.asarray(bit_map)
        bit_map = bit_map.reshape(input.shape)
    else:
        bit_map = np.asarray(bit_map)
    assert bit_map.shape == input.shape
    return user_function(Multibit(input, bit_map))

def CustomMultibitKernel(input, bit_map, mean_bits=None):
    if False:
        i = 10
        return i + 15
    if mean_bits:
        bit_map = np.asarray(np.maximum(np.round(np.random.normal(mean_bits, 1, input.shape)), 1), dtype=np.int32)
        print('Mean Bits: ', np.mean(bit_map))
    elif type(bit_map) == int:
        length = C.reshape(input, -1)
        bit_map = [bit_map] * length.shape[0]
        bit_map = np.asarray(bit_map)
        bit_map = bit_map.reshape(input.shape)
    else:
        bit_map = np.asarray(bit_map)
    assert bit_map.shape == input.shape
    return user_function(MultibitKernel(input, bit_map))