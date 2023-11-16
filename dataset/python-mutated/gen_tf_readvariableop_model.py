"""Saves a TensorFlow model containing ReadVariableOp nodes.

   The saved model is loaded and executed by tests to check that TF-TRT can
   successfully convert and execute models with variables without freezing.
"""
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save

class MyModel(module.Module):
    """Simple model with two variables."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.var1 = variables.Variable(np.array([[[13.0]]], dtype=np.float32), name='var1')
        self.var2 = variables.Variable(np.array([[[37.0]]], dtype=np.float32), name='var2')

    @def_function.function
    def __call__(self, input1, input2):
        if False:
            while True:
                i = 10
        mul1 = input1 * self.var1
        mul2 = input2 * self.var2
        add = mul1 + mul2
        sub = add - 45.0
        return array_ops.identity(sub, name='output')

def GenerateModelWithReadVariableOp(tf_saved_model_dir):
    if False:
        i = 10
        return i + 15
    'Generate a model with ReadVariableOp nodes.'
    my_model = MyModel()
    cfunc = my_model.__call__.get_concrete_function(tensor_spec.TensorSpec([None, 1, 1], dtypes.float32), tensor_spec.TensorSpec([None, 1, 1], dtypes.float32))
    save(my_model, tf_saved_model_dir, signatures=cfunc)
if __name__ == '__main__':
    GenerateModelWithReadVariableOp(tf_saved_model_dir='tf_readvariableop_saved_model')