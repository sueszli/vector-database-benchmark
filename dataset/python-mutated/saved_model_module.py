"""Implementation of deprecated hub.Module that loads raw TF1 SavedModels."""
import tensorflow as tf
from tensorflow_hub import native_module
from tensorflow_hub import saved_model_lib
_ALWAYS_DROPPED_COLLECTIONS = [tf.compat.v1.GraphKeys.GLOBAL_STEP, tf.compat.v1.saved_model.constants.LEGACY_INIT_OP_KEY, tf.compat.v1.saved_model.constants.MAIN_OP_KEY]

def _drop_collections(saved_model_handler, collections):
    if False:
        while True:
            i = 10
    for meta_graph in saved_model_handler.meta_graphs:
        for collection in collections:
            if collection in meta_graph.collection_def:
                del meta_graph.collection_def[collection]

def create_module_spec_from_saved_model(saved_model_path, drop_collections=None):
    if False:
        for i in range(10):
            print('nop')
    'Experimental: Create a ModuleSpec out of a SavedModel from TF1.\n\n  Warning: Deprecated. This belongs to the hub.Module API and TF1 Hub format.\n  For TF2, TensorFlow Hub ships plain SavedModels, removing the need for\n  conversions like this.\n\n  Define a ModuleSpec from a SavedModel. Note that this is not guaranteed to\n  work in all cases and it assumes the SavedModel has followed some conventions:\n\n  - The serialized SaverDef can be ignored and instead can be reconstructed.\n  - The init op and main op can be ignored and instead the module can be\n    initialized by using the conventions followed by\n    `tf.train.MonitoredSession`.\n\n  Note that the set of features supported can increase over time and have side\n  effects that were not previously visible. The pattern followed to avoid\n  surprises is forcing users to declare which features to ignore (even\n  if they are not supported).\n\n  Note that this function creates a ModuleSpec that when exported exports a\n  Module (based on a modified copy of the original SavedModel) and not a\n  SavedModel.\n\n  THIS FUNCTION IS DEPRECATED.\n\n  Args:\n    saved_model_path: Directory with the SavedModel to use.\n    drop_collections: Additionally list of collection to drop.\n\n  Returns:\n    A ModuleSpec.\n  '
    saved_model_handler = saved_model_lib.load(saved_model_path)
    checkpoint_filename = saved_model_lib.get_variables_path(saved_model_path)
    drop_collections = set(_ALWAYS_DROPPED_COLLECTIONS) | (set(drop_collections) if drop_collections else set())
    _drop_collections(saved_model_handler, drop_collections)
    return native_module._ModuleSpec(saved_model_handler, checkpoint_filename)