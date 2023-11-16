"""Serves as a common "main" function for all the SavedModel tests.

There is a fair amount of setup needed to initialize tensorflow and get it
into a proper TF2 execution mode. This hides that boilerplate.
"""
import tempfile
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.python import pywrap_mlir
flags.DEFINE_string('save_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS

def set_tf_options():
    if False:
        print('Hello World!')
    tf.enable_resource_variables()
    tf.compat.v1.disable_eager_execution()

def do_test(create_signature, canonicalize=False, show_debug_info=False, use_lite=False, lift_variables=True, include_variables_in_initializers=False):
    if False:
        while True:
            i = 10
    'Runs test.\n\n  1. Performs absl and tf "main"-like initialization that must run before almost\n     anything else.\n  2. Converts signature_def_map to SavedModel V1\n  3. Converts SavedModel V1 to MLIR\n  4. Prints the textual MLIR to stdout (it is expected that the caller will have\n     FileCheck checks in its file to check this output).\n\n  This is only for use by the MLIR SavedModel importer tests.\n\n  Args:\n    create_signature: A functor that return signature_def_map, init_op and\n      assets_collection. signature_def_map is a map from string key to\n      signature_def. The key will be used as function name in the resulting\n      MLIR.\n    canonicalize: If true, canonicalizer will be run on the resulting MLIR.\n    show_debug_info: If true, shows debug locations in the resulting MLIR.\n    use_lite: If true, importer will not do any graph transformation such as\n      lift variables.\n    lift_variables: If false, no variable lifting will be done on the graph.\n    include_variables_in_initializers: If false, removes variables in\n      initializer functions before lifting variables or adding new variable\n      initialization patterns in the initializer function.\n  '
    logging.set_stderrthreshold('error')

    def app_main(argv):
        if False:
            while True:
                i = 10
        'Function passed to absl.app.run.'
        if len(argv) > 1:
            raise app.UsageError('Too many command-line arguments.')
        if FLAGS.save_model_path:
            save_model_path = FLAGS.save_model_path
        else:
            save_model_path = tempfile.mktemp(suffix='.saved_model')
        (signature_def_map, init_op, assets_collection) = create_signature()
        sess = tf.Session()
        sess.run(tf.initializers.global_variables())
        builder = tf.saved_model.builder.SavedModelBuilder(save_model_path)
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map, main_op=init_op, assets_collection=assets_collection, strip_default_attrs=True)
        builder.save()
        logging.info('Saved model to: %s', save_model_path)
        exported_names = ''
        upgrade_legacy = True
        if use_lite:
            mlir = pywrap_mlir.experimental_convert_saved_model_v1_to_mlir_lite(save_model_path, exported_names, ','.join([tf.saved_model.tag_constants.SERVING]), upgrade_legacy, show_debug_info)
            mlir = pywrap_mlir.experimental_run_pass_pipeline(mlir, 'tf-standard-pipeline', show_debug_info)
        else:
            mlir = pywrap_mlir.experimental_convert_saved_model_v1_to_mlir(save_model_path, exported_names, ','.join([tf.saved_model.tag_constants.SERVING]), lift_variables, include_variables_in_initializers, upgrade_legacy, show_debug_info)
        if canonicalize:
            mlir = pywrap_mlir.experimental_run_pass_pipeline(mlir, 'canonicalize', show_debug_info)
        print(mlir)
    app.run(app_main)