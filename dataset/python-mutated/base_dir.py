"""Opensource base_dir configuration for tensorflow doc-generator."""
import pathlib
import keras
from packaging import version
import tensorboard
import tensorflow as tf
from tensorflow_docs.api_generator import public_api
import tensorflow_estimator

def get_base_dirs_and_prefixes(code_url_prefix):
    if False:
        for i in range(10):
            print('nop')
    'Returns the base_dirs and code_prefixes for OSS TensorFlow api gen.'
    base_dir = pathlib.Path(tf.__file__).parent
    if 'dev' in tf.__version__:
        keras_url_prefix = 'https://github.com/keras-team/keras/tree/master/keras'
    else:
        keras_url_prefix = f'https://github.com/keras-team/keras/tree/v{keras.__version__}/keras'
    if version.parse(tf.__version__) >= version.parse('2.13'):
        base_dirs = [pathlib.Path(keras.__file__).parent / 'src', pathlib.Path(tf.keras.__file__).parent, pathlib.Path(keras.__file__).parent, pathlib.Path(tensorboard.__file__).parent, pathlib.Path(tensorflow_estimator.__file__).parent, base_dir]
        code_url_prefixes = (keras_url_prefix, None, None, f'https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard', 'https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator', code_url_prefix)
    elif version.parse(tf.__version__) >= version.parse('2.9'):
        base_dirs = [base_dir, pathlib.Path(keras.__file__).parent, pathlib.Path(tensorboard.__file__).parent, pathlib.Path(tensorflow_estimator.__file__).parent]
        code_url_prefixes = (code_url_prefix, keras_url_prefix, f'https://github.com/tensorflow/tensorboard/tree/{tensorboard.__version__}/tensorboard', 'https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator')
    else:
        raise ValueError('Unsupported: version < 2.9')
    return (base_dirs, code_url_prefixes)

def explicit_filter_keep_keras(parent_path, parent, children):
    if False:
        print('Hello World!')
    'Like explicit_package_contents_filter, but keeps keras.'
    new_children = public_api.explicit_package_contents_filter(parent_path, parent, children)
    if parent_path[-1] not in ['tf', 'v1', 'v2']:
        return new_children
    had_keras = any((name == 'keras' for (name, child) in children))
    has_keras = any((name == 'keras' for (name, child) in new_children))
    if had_keras and (not has_keras):
        new_children.append(('keras', parent.keras))
    return sorted(new_children, key=lambda x: x[0])

def get_callbacks():
    if False:
        for i in range(10):
            print('nop')
    return [explicit_filter_keep_keras]