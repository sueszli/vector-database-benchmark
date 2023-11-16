import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

def Test():
    if False:
        for i in range(10):
            print('nop')
    x = tf.compat.v1.get_variable(name='x', shape=(5, 3), initializer=tf.random_normal_initializer(), trainable=True)
    y = tf.compat.v1.get_variable(name='y', shape=(3, 5), initializer=tf.random_normal_initializer(), trainable=True)
    z = tf.matmul(x, y)
    tensor_info_z = tf.compat.v1.saved_model.utils.build_tensor_info(z)
    return ({'key': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs=None, outputs={'z': tensor_info_z}, method_name='some_function')}, None, None)
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(Test)