import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.python.ops import control_flow_ops

def Test():
    if False:
        for i in range(10):
            print('nop')
    data = tf.constant([1, 2, 3, 4, 5, 6])
    x_op = tf.placeholder(dtype=tf.int32)
    y_op = tf.placeholder(dtype=tf.int32)
    less_op = tf.less(x_op, y_op)
    switch_op = control_flow_ops.switch(data, less_op)
    merge_op = control_flow_ops.merge(switch_op)[0]
    result = tf.transpose(merge_op)
    tensor_info_result = tf.compat.v1.saved_model.utils.build_tensor_info(result)
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(inputs=None, outputs={'result': tensor_info_result}, method_name='some_function')
    return ({'key': signature_def}, None, None)
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(Test)