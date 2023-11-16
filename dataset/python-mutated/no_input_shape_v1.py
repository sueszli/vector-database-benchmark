import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.core.protobuf import meta_graph_pb2

def Test():
    if False:
        i = 10
        return i + 15
    x = tf.placeholder(dtype=tf.float32, shape=[None])
    batch_size = tf.shape(x)[0]
    r = tf.convert_to_tensor([batch_size, 1])
    tensor_info_x = meta_graph_pb2.TensorInfo(name=x.name, dtype=tf.as_dtype(x.dtype).as_datatype_enum)
    tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)
    return ({'key': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'r': tensor_info_r}, method_name='some_function')}, None, None)
if __name__ == '__main__':
    common_v1.set_tf_options()
    common_v1.do_test(Test)