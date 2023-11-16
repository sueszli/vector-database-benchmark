import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

def Test():
    if False:
        for i in range(10):
            print('nop')
    x = tf.constant(1.0, shape=(3, 3))
    y = tf.constant(1.0, shape=(3, 3))
    s = tf.transpose(x)
    t = tf.transpose(y)
    tensor_info_s = tf.compat.v1.saved_model.utils.build_tensor_info(s)
    tensor_info_t = tf.compat.v1.saved_model.utils.build_tensor_info(t)
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(inputs=None, outputs={'s': tensor_info_s}, method_name='some_function')
    signature_def2 = tf.saved_model.signature_def_utils.build_signature_def(inputs=None, outputs={'t': tensor_info_t}, method_name='some_function')
    return ({'key': signature_def, 'key2': signature_def2}, None, None)
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(Test)