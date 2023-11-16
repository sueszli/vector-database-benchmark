"""TensorFlow Hub Module API for Tensorflow 2.0."""
import os
import tensorflow as tf
from tensorflow_hub import native_module
from tensorflow_hub import registry

def resolve(handle):
    if False:
        return 10
    'Resolves a module handle into a path.\n\n  This function works both for plain TF2 SavedModels and the legacy TF1 Hub\n  format.\n\n  Resolves a module handle into a path by downloading and caching in\n  location specified by TFHUB_CACHE_DIR if needed.\n\n  Currently, three types of module handles are supported:\n    1) Smart URL resolvers such as tfhub.dev, e.g.:\n       https://tfhub.dev/google/nnlm-en-dim128/1.\n    2) A directory on a file system supported by Tensorflow containing module\n       files. This may include a local directory (e.g. /usr/local/mymodule) or a\n       Google Cloud Storage bucket (gs://mymodule).\n    3) A URL pointing to a TGZ archive of a module, e.g.\n       https://example.com/mymodule.tar.gz.\n\n  Args:\n    handle: (string) the Module handle to resolve.\n\n  Returns:\n    A string representing the Module path.\n  '
    return registry.resolver(handle)

def load(handle, tags=None, options=None):
    if False:
        i = 10
        return i + 15
    "Resolves a handle and loads the resulting module.\n\n  This is the preferred API to load a Hub module in low-level TensorFlow 2.\n  Users of higher-level frameworks like Keras should use the framework's\n  corresponding wrapper, like hub.KerasLayer.\n\n  This function is roughly equivalent to the TF2 function\n  `tf.saved_model.load()` on the result of `hub.resolve(handle)`. Calling this\n  function requires TF 1.14 or newer. It can be called both in eager and graph\n  mode.\n\n  Note: Using in a tf.compat.v1.Session with variables placed on parameter\n  servers requires setting `experimental.share_cluster_devices_in_session`\n  within the `tf.compat.v1.ConfigProto`. (It becomes non-experimental in TF2.2.)\n\n  This function can handle the deprecated TF1 Hub format to the extent\n  that `tf.saved_model.load()` in TF2 does. In particular, the returned object\n  has attributes\n    * `.variables`: a list of variables from the loaded object;\n    * `.signatures`: a dict of TF2 ConcreteFunctions, keyed by signature names,\n      that take tensor kwargs and return a tensor dict.\n  However, the information imported by hub.Module into the collections of a\n  tf.Graph is lost (e.g., regularization losses and update ops).\n\n  Args:\n    handle: (string) the Module handle to resolve; see hub.resolve().\n    tags: A set of strings specifying the graph variant to use, if loading from\n      a v1 module.\n    options: Optional, `tf.saved_model.LoadOptions` object that specifies\n      options for loading. This argument can only be used from TensorFlow 2.3\n      onwards.\n\n  Returns:\n    A trackable object (see tf.saved_model.load() documentation for details).\n\n  Raises:\n    NotImplementedError: If the code is running against incompatible (1.x)\n                         version of TF.\n  "
    if not isinstance(handle, str):
        raise ValueError('Expected a string, got %s' % handle)
    module_path = resolve(handle)
    is_hub_module_v1 = tf.io.gfile.exists(native_module.get_module_proto_path(module_path))
    if tags is None and is_hub_module_v1:
        tags = []
    saved_model_path = os.path.join(tf.compat.as_bytes(module_path), tf.compat.as_bytes(tf.saved_model.SAVED_MODEL_FILENAME_PB))
    saved_model_pbtxt_path = os.path.join(tf.compat.as_bytes(module_path), tf.compat.as_bytes(tf.saved_model.SAVED_MODEL_FILENAME_PBTXT))
    if not tf.io.gfile.exists(saved_model_path) and (not tf.io.gfile.exists(saved_model_pbtxt_path)):
        raise ValueError("Trying to load a model of incompatible/unknown type. '%s' contains neither '%s' nor '%s'." % (module_path, tf.saved_model.SAVED_MODEL_FILENAME_PB, tf.saved_model.SAVED_MODEL_FILENAME_PBTXT))
    if options:
        if not hasattr(getattr(tf, 'saved_model', None), 'LoadOptions'):
            raise NotImplementedError('options are not supported for TF < 2.3.x, Current version: %s' % tf.__version__)
        obj = tf.compat.v1.saved_model.load_v2(module_path, tags=tags, options=options)
    else:
        obj = tf.compat.v1.saved_model.load_v2(module_path, tags=tags)
    obj._is_hub_module_v1 = is_hub_module_v1
    return obj