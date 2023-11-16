import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.python.framework import function

@function.Defun(tf.float32, tf.float32)
def plus(a, b):
    if False:
        return 10
    return a + b

def test_defun():
    if False:
        print('Hello World!')
    x = tf.constant([[1.0], [1.0], [1.0]])
    y = tf.constant([[2.0], [2.0], [2.0]])
    z = plus(x, y)
    tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)
    tensor_info_z = tf.compat.v1.saved_model.utils.build_tensor_info(z)
    return ({'key': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x, 'y': tensor_info_y}, outputs={'z': tensor_info_z}, method_name='test_function')}, None, None)
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(test_defun)