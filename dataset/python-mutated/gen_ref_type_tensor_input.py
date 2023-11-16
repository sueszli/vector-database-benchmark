"""Generates a saved model with tf.Pow to trigger placer and grappler."""
import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS

def main(argv):
    if False:
        while True:
            i = 10
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    shutil.rmtree(FLAGS.saved_model_path)
    x = variable_scope.get_variable(name='x', initializer=[[1], [2], [3]])
    r = math_ops.add(x, 1)
    x1 = array_ops.placeholder(dtype=dtypes.int32, shape=(1, 3), name='input1')
    r1 = math_ops.add(x1, 1)
    sess = session.Session()
    sess.run(variables.global_variables_initializer())
    sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_r = utils.build_tensor_info(r)
    tensor_info_x1 = utils.build_tensor_info(x1)
    tensor_info_r1 = utils.build_tensor_info(r1)
    ref_signature = signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'r': tensor_info_r}, method_name=signature_constants.PREDICT_METHOD_NAME)
    non_ref_signature = signature_def_utils.build_signature_def(inputs={'x1': tensor_info_x1}, outputs={'r1': tensor_info_r1}, method_name=signature_constants.PREDICT_METHOD_NAME)
    sm_builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map={'ref': ref_signature, 'non_ref': non_ref_signature}, strip_default_attrs=True)
    sm_builder.save()
if __name__ == '__main__':
    app.run(main)