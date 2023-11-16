"""Generates a toy v1 saved model for testing."""
import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
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
        return 10
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    shutil.rmtree(FLAGS.saved_model_path)
    x = array_ops.placeholder(dtypes.int32, shape=None, name='input')
    y = variables.Variable(name='y', dtype=dtypes.int32, initial_value=[[1, 2], [3, 4]])
    r = y.sparse_read(x, name='result')
    sess = session.Session()
    sess.run(variables.global_variables_initializer())
    sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_r = utils.build_tensor_info(r)
    toy_signature = signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'r': tensor_info_r}, method_name=signature_constants.PREDICT_METHOD_NAME)
    sm_builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: toy_signature}, strip_default_attrs=True)
    sm_builder.save()
if __name__ == '__main__':
    app.run(main)