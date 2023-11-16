"""SavedModel simple save functionality."""
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['saved_model.simple_save'])
@deprecation.deprecated(None, 'This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructions on how to migrate your code to TensorFlow v2.')
def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
    if False:
        print('Hello World!')
    'Convenience function to build a SavedModel suitable for serving.\n\n  In many common cases, saving models for serving will be as simple as:\n\n      simple_save(session,\n                  export_dir,\n                  inputs={"x": x, "y": y},\n                  outputs={"z": z})\n\n  Although in many cases it\'s not necessary to understand all of the many ways\n      to configure a SavedModel, this method has a few practical implications:\n    - It will be treated as a graph for inference / serving (i.e. uses the tag\n      `saved_model.SERVING`)\n    - The SavedModel will load in TensorFlow Serving and supports the\n      [Predict\n      API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto).\n      To use the Classify, Regress, or MultiInference APIs, please\n      use either\n      [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)\n      or the lower level\n      [SavedModel\n      APIs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).\n    - Some TensorFlow ops depend on information on disk or other information\n      called "assets". These are generally handled automatically by adding the\n      assets to the `GraphKeys.ASSET_FILEPATHS` collection. Only assets in that\n      collection are exported; if you need more custom behavior, you\'ll need to\n      use the\n      [SavedModelBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py).\n\n  More information about SavedModel and signatures can be found here:\n  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md.\n\n  Args:\n    session: The TensorFlow session from which to save the meta graph and\n        variables.\n    export_dir: The path to which the SavedModel will be stored.\n    inputs: dict mapping string input names to tensors. These are added\n        to the SignatureDef as the inputs.\n    outputs:  dict mapping string output names to tensors. These are added\n        to the SignatureDef as the outputs.\n    legacy_init_op: Legacy support for op or group of ops to execute after the\n        restore op upon a load.\n  '
    signature_def_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def_utils.predict_signature_def(inputs, outputs)}
    b = builder.SavedModelBuilder(export_dir)
    b.add_meta_graph_and_variables(session, tags=[tag_constants.SERVING], signature_def_map=signature_def_map, assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS), main_op=legacy_init_op, clear_devices=True)
    b.save()