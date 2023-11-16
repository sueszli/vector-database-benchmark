import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

def Test():
    if False:
        for i in range(10):
            print('nop')
    x = tf.constant([[1.0], [1.0], [1.0]])
    y = tf.get_variable(name='y', shape=(1, 3), initializer=tf.random_normal_initializer(), trainable=True)
    r = tf.matmul(x, y)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_r = tf.saved_model.utils.build_tensor_info(r)
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'r': tensor_info_r}, method_name='some_function')
    signature_def2 = tf.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'r': tensor_info_r}, method_name='some_other_function')
    return ({'key': signature_def, 'key2': signature_def2}, None, None)
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(Test)