import tensorflow as tf
import os
import glob
import re
_dali_tf_module = None

def load_dali_tf_plugin():
    if False:
        while True:
            i = 10
    global _dali_tf_module
    if _dali_tf_module is not None:
        return _dali_tf_module
    import nvidia.dali as dali
    assert dali
    tf_plugins = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'libdali_tf*.so'))
    tf_version = re.search('(\\d+.\\d+).\\d+', tf.__version__).group(1)
    tf_version_underscore = tf_version.replace('.', '_')
    dali_tf_current = list(filter(lambda x: 'current' in x, tf_plugins))
    dali_tf_prebuilt_tf_ver = list(filter(lambda x: tf_version_underscore in x, tf_plugins))
    dali_tf_prebuilt_others = list(filter(lambda x: 'current' not in x and tf_version_underscore not in x, tf_plugins))
    processed_tf_plugins = dali_tf_current + dali_tf_prebuilt_tf_ver + dali_tf_prebuilt_others
    first_error = None
    for libdali_tf in processed_tf_plugins:
        try:
            _dali_tf_module = tf.load_op_library(libdali_tf)
            break
        except tf.errors.NotFoundError as error:
            if first_error is None:
                first_error = error
    else:
        raise first_error or Exception('No matching DALI plugin found for installed TensorFlow version')
    return _dali_tf_module