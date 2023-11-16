"""Utilities for unit-testing Keras."""
import collections
import functools
import itertools
import unittest
from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest
try:
    import h5py
except ImportError:
    h5py = None

class TestCase(test.TestCase, parameterized.TestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        keras.backend.clear_session()
        super(TestCase, self).tearDown()

def run_with_all_saved_model_formats(test_or_class=None, exclude_formats=None):
    if False:
        i = 10
        return i + 15
    'Execute the decorated test with all Keras saved model formats).\n\n  This decorator is intended to be applied either to individual test methods in\n  a `keras_parameterized.TestCase` class, or directly to a test class that\n  extends it. Doing so will cause the contents of the individual test\n  method (or all test methods in the class) to be executed multiple times - once\n  for each Keras saved model format.\n\n  The Keras saved model formats include:\n  1. HDF5: \'h5\'\n  2. SavedModel: \'tf\'\n\n  Note: if stacking this decorator with absl.testing\'s parameterized decorators,\n  those should be at the bottom of the stack.\n\n  Various methods in `testing_utils` to get file path for saved models will\n  auto-generate a string of the two saved model formats. This allows unittests\n  to confirm the equivalence between the two Keras saved model formats.\n\n  For example, consider the following unittest:\n\n  ```python\n  class MyTests(testing_utils.KerasTestCase):\n\n    @testing_utils.run_with_all_saved_model_formats\n    def test_foo(self):\n      save_format = testing_utils.get_save_format()\n      saved_model_dir = \'/tmp/saved_model/\'\n      model = keras.models.Sequential()\n      model.add(keras.layers.Dense(2, input_shape=(3,)))\n      model.add(keras.layers.Dense(3))\n      model.compile(loss=\'mse\', optimizer=\'sgd\', metrics=[\'acc\'])\n\n      keras.models.save_model(model, saved_model_dir, save_format=save_format)\n      model = keras.models.load_model(saved_model_dir)\n\n  if __name__ == "__main__":\n    tf.test.main()\n  ```\n\n  This test tries to save the model into the formats of \'hdf5\', \'h5\', \'keras\',\n  \'tensorflow\', and \'tf\'.\n\n  We can also annotate the whole class if we want this to apply to all tests in\n  the class:\n  ```python\n  @testing_utils.run_with_all_saved_model_formats\n  class MyTests(testing_utils.KerasTestCase):\n\n    def test_foo(self):\n      save_format = testing_utils.get_save_format()\n      saved_model_dir = \'/tmp/saved_model/\'\n      model = keras.models.Sequential()\n      model.add(keras.layers.Dense(2, input_shape=(3,)))\n      model.add(keras.layers.Dense(3))\n      model.compile(loss=\'mse\', optimizer=\'sgd\', metrics=[\'acc\'])\n\n      keras.models.save_model(model, saved_model_dir, save_format=save_format)\n      model = tf.keras.models.load_model(saved_model_dir)\n\n  if __name__ == "__main__":\n    tf.test.main()\n  ```\n\n  Args:\n    test_or_class: test method or class to be annotated. If None,\n      this method returns a decorator that can be applied to a test method or\n      test class. If it is not None this returns the decorator applied to the\n      test or class.\n    exclude_formats: A collection of Keras saved model formats to not run.\n      (May also be a single format not wrapped in a collection).\n      Defaults to None.\n\n  Returns:\n    Returns a decorator that will run the decorated test method multiple times:\n    once for each desired Keras saved model format.\n\n  Raises:\n    ImportError: If abseil parameterized is not installed or not included as\n      a target dependency.\n  '
    if h5py is None:
        exclude_formats.append(['h5'])
    saved_model_formats = ['h5', 'tf', 'tf_no_traces']
    params = [('_%s' % saved_format, saved_format) for saved_format in saved_model_formats if saved_format not in nest.flatten(exclude_formats)]

    def single_method_decorator(f):
        if False:
            while True:
                i = 10
        'Decorator that constructs the test cases.'

        @parameterized.named_parameters(*params)
        @functools.wraps(f)
        def decorated(self, saved_format, *args, **kwargs):
            if False:
                while True:
                    i = 10
            'A run of a single test case w/ the specified model type.'
            if saved_format == 'h5':
                _test_h5_saved_model_format(f, self, *args, **kwargs)
            elif saved_format == 'tf':
                _test_tf_saved_model_format(f, self, *args, **kwargs)
            elif saved_format == 'tf_no_traces':
                _test_tf_saved_model_format_no_traces(f, self, *args, **kwargs)
            else:
                raise ValueError('Unknown model type: %s' % (saved_format,))
        return decorated
    return _test_or_class_decorator(test_or_class, single_method_decorator)

def _test_h5_saved_model_format(f, test_or_class, *args, **kwargs):
    if False:
        while True:
            i = 10
    with testing_utils.saved_model_format_scope('h5'):
        f(test_or_class, *args, **kwargs)

def _test_tf_saved_model_format(f, test_or_class, *args, **kwargs):
    if False:
        print('Hello World!')
    with testing_utils.saved_model_format_scope('tf'):
        f(test_or_class, *args, **kwargs)

def _test_tf_saved_model_format_no_traces(f, test_or_class, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    with testing_utils.saved_model_format_scope('tf', save_traces=False):
        f(test_or_class, *args, **kwargs)

def run_with_all_weight_formats(test_or_class=None, exclude_formats=None):
    if False:
        i = 10
        return i + 15
    'Runs all tests with the supported formats for saving weights.'
    exclude_formats = exclude_formats or []
    exclude_formats.append('tf_no_traces')
    return run_with_all_saved_model_formats(test_or_class, exclude_formats)

def run_with_all_model_types(test_or_class=None, exclude_models=None):
    if False:
        i = 10
        return i + 15
    'Execute the decorated test with all Keras model types.\n\n  This decorator is intended to be applied either to individual test methods in\n  a `keras_parameterized.TestCase` class, or directly to a test class that\n  extends it. Doing so will cause the contents of the individual test\n  method (or all test methods in the class) to be executed multiple times - once\n  for each Keras model type.\n\n  The Keras model types are: [\'functional\', \'subclass\', \'sequential\']\n\n  Note: if stacking this decorator with absl.testing\'s parameterized decorators,\n  those should be at the bottom of the stack.\n\n  Various methods in `testing_utils` to get models will auto-generate a model\n  of the currently active Keras model type. This allows unittests to confirm\n  the equivalence between different Keras models.\n\n  For example, consider the following unittest:\n\n  ```python\n  class MyTests(testing_utils.KerasTestCase):\n\n    @testing_utils.run_with_all_model_types(\n      exclude_models = [\'sequential\'])\n    def test_foo(self):\n      model = testing_utils.get_small_mlp(1, 4, input_dim=3)\n      optimizer = RMSPropOptimizer(learning_rate=0.001)\n      loss = \'mse\'\n      metrics = [\'mae\']\n      model.compile(optimizer, loss, metrics=metrics)\n\n      inputs = np.zeros((10, 3))\n      targets = np.zeros((10, 4))\n      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))\n      dataset = dataset.repeat(100)\n      dataset = dataset.batch(10)\n\n      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)\n\n  if __name__ == "__main__":\n    tf.test.main()\n  ```\n\n  This test tries building a small mlp as both a functional model and as a\n  subclass model.\n\n  We can also annotate the whole class if we want this to apply to all tests in\n  the class:\n  ```python\n  @testing_utils.run_with_all_model_types(exclude_models = [\'sequential\'])\n  class MyTests(testing_utils.KerasTestCase):\n\n    def test_foo(self):\n      model = testing_utils.get_small_mlp(1, 4, input_dim=3)\n      optimizer = RMSPropOptimizer(learning_rate=0.001)\n      loss = \'mse\'\n      metrics = [\'mae\']\n      model.compile(optimizer, loss, metrics=metrics)\n\n      inputs = np.zeros((10, 3))\n      targets = np.zeros((10, 4))\n      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))\n      dataset = dataset.repeat(100)\n      dataset = dataset.batch(10)\n\n      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)\n\n  if __name__ == "__main__":\n    tf.test.main()\n  ```\n\n\n  Args:\n    test_or_class: test method or class to be annotated. If None,\n      this method returns a decorator that can be applied to a test method or\n      test class. If it is not None this returns the decorator applied to the\n      test or class.\n    exclude_models: A collection of Keras model types to not run.\n      (May also be a single model type not wrapped in a collection).\n      Defaults to None.\n\n  Returns:\n    Returns a decorator that will run the decorated test method multiple times:\n    once for each desired Keras model type.\n\n  Raises:\n    ImportError: If abseil parameterized is not installed or not included as\n      a target dependency.\n  '
    model_types = ['functional', 'subclass', 'sequential']
    params = [('_%s' % model, model) for model in model_types if model not in nest.flatten(exclude_models)]

    def single_method_decorator(f):
        if False:
            return 10
        'Decorator that constructs the test cases.'

        @parameterized.named_parameters(*params)
        @functools.wraps(f)
        def decorated(self, model_type, *args, **kwargs):
            if False:
                print('Hello World!')
            'A run of a single test case w/ the specified model type.'
            if model_type == 'functional':
                _test_functional_model_type(f, self, *args, **kwargs)
            elif model_type == 'subclass':
                _test_subclass_model_type(f, self, *args, **kwargs)
            elif model_type == 'sequential':
                _test_sequential_model_type(f, self, *args, **kwargs)
            else:
                raise ValueError('Unknown model type: %s' % (model_type,))
        return decorated
    return _test_or_class_decorator(test_or_class, single_method_decorator)

def _test_functional_model_type(f, test_or_class, *args, **kwargs):
    if False:
        while True:
            i = 10
    with testing_utils.model_type_scope('functional'):
        f(test_or_class, *args, **kwargs)

def _test_subclass_model_type(f, test_or_class, *args, **kwargs):
    if False:
        return 10
    with testing_utils.model_type_scope('subclass'):
        f(test_or_class, *args, **kwargs)

def _test_sequential_model_type(f, test_or_class, *args, **kwargs):
    if False:
        return 10
    with testing_utils.model_type_scope('sequential'):
        f(test_or_class, *args, **kwargs)

def run_all_keras_modes(test_or_class=None, config=None, always_skip_v1=False, always_skip_eager=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Execute the decorated test with all keras execution modes.\n\n  This decorator is intended to be applied either to individual test methods in\n  a `keras_parameterized.TestCase` class, or directly to a test class that\n  extends it. Doing so will cause the contents of the individual test\n  method (or all test methods in the class) to be executed multiple times -\n  once executing in legacy graph mode, once running eagerly and with\n  `should_run_eagerly` returning True, and once running eagerly with\n  `should_run_eagerly` returning False.\n\n  If Tensorflow v2 behavior is enabled, legacy graph mode will be skipped, and\n  the test will only run twice.\n\n  Note: if stacking this decorator with absl.testing\'s parameterized decorators,\n  those should be at the bottom of the stack.\n\n  For example, consider the following unittest:\n\n  ```python\n  class MyTests(testing_utils.KerasTestCase):\n\n    @testing_utils.run_all_keras_modes\n    def test_foo(self):\n      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)\n      optimizer = RMSPropOptimizer(learning_rate=0.001)\n      loss = \'mse\'\n      metrics = [\'mae\']\n      model.compile(\n          optimizer, loss, metrics=metrics,\n          run_eagerly=testing_utils.should_run_eagerly())\n\n      inputs = np.zeros((10, 3))\n      targets = np.zeros((10, 4))\n      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))\n      dataset = dataset.repeat(100)\n      dataset = dataset.batch(10)\n\n      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)\n\n  if __name__ == "__main__":\n    tf.test.main()\n  ```\n\n  This test will try compiling & fitting the small functional mlp using all\n  three Keras execution modes.\n\n  Args:\n    test_or_class: test method or class to be annotated. If None,\n      this method returns a decorator that can be applied to a test method or\n      test class. If it is not None this returns the decorator applied to the\n      test or class.\n    config: An optional config_pb2.ConfigProto to use to configure the\n      session when executing graphs.\n    always_skip_v1: If True, does not try running the legacy graph mode even\n      when Tensorflow v2 behavior is not enabled.\n    always_skip_eager: If True, does not execute the decorated test\n      with eager execution modes.\n    **kwargs: Additional kwargs for configuring tests for\n     in-progress Keras behaviors/ refactorings that we haven\'t fully\n     rolled out yet\n\n  Returns:\n    Returns a decorator that will run the decorated test method multiple times.\n\n  Raises:\n    ImportError: If abseil parameterized is not installed or not included as\n      a target dependency.\n  '
    if kwargs:
        raise ValueError('Unrecognized keyword args: {}'.format(kwargs))
    params = [('_v2_function', 'v2_function')]
    if not always_skip_eager:
        params.append(('_v2_eager', 'v2_eager'))
    if not (always_skip_v1 or tf2.enabled()):
        params.append(('_v1_session', 'v1_session'))

    def single_method_decorator(f):
        if False:
            print('Hello World!')
        'Decorator that constructs the test cases.'

        @parameterized.named_parameters(*params)
        @functools.wraps(f)
        def decorated(self, run_mode, *args, **kwargs):
            if False:
                return 10
            'A run of a single test case w/ specified run mode.'
            if run_mode == 'v1_session':
                _v1_session_test(f, self, config, *args, **kwargs)
            elif run_mode == 'v2_eager':
                _v2_eager_test(f, self, *args, **kwargs)
            elif run_mode == 'v2_function':
                _v2_function_test(f, self, *args, **kwargs)
            else:
                return ValueError('Unknown run mode %s' % run_mode)
        return decorated
    return _test_or_class_decorator(test_or_class, single_method_decorator)

def _v1_session_test(f, test_or_class, config, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    with ops.get_default_graph().as_default():
        with testing_utils.run_eagerly_scope(False):
            with test_or_class.test_session(config=config):
                f(test_or_class, *args, **kwargs)

def _v2_eager_test(f, test_or_class, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    with context.eager_mode():
        with testing_utils.run_eagerly_scope(True):
            f(test_or_class, *args, **kwargs)

def _v2_function_test(f, test_or_class, *args, **kwargs):
    if False:
        return 10
    with context.eager_mode():
        with testing_utils.run_eagerly_scope(False):
            f(test_or_class, *args, **kwargs)

def _test_or_class_decorator(test_or_class, single_method_decorator):
    if False:
        while True:
            i = 10
    'Decorate a test or class with a decorator intended for one method.\n\n  If the test_or_class is a class:\n    This will apply the decorator to all test methods in the class.\n\n  If the test_or_class is an iterable of already-parameterized test cases:\n    This will apply the decorator to all the cases, and then flatten the\n    resulting cross-product of test cases. This allows stacking the Keras\n    parameterized decorators w/ each other, and to apply them to test methods\n    that have already been marked with an absl parameterized decorator.\n\n  Otherwise, treat the obj as a single method and apply the decorator directly.\n\n  Args:\n    test_or_class: A test method (that may have already been decorated with a\n      parameterized decorator, or a test class that extends\n      keras_parameterized.TestCase\n    single_method_decorator:\n      A parameterized decorator intended for a single test method.\n  Returns:\n    The decorated result.\n  '

    def _decorate_test_or_class(obj):
        if False:
            i = 10
            return i + 15
        if isinstance(obj, collections.abc.Iterable):
            return itertools.chain.from_iterable((single_method_decorator(method) for method in obj))
        if isinstance(obj, type):
            cls = obj
            for (name, value) in cls.__dict__.copy().items():
                if callable(value) and name.startswith(unittest.TestLoader.testMethodPrefix):
                    setattr(cls, name, single_method_decorator(value))
            cls = type(cls).__new__(type(cls), cls.__name__, cls.__bases__, cls.__dict__.copy())
            return cls
        return single_method_decorator(obj)
    if test_or_class is not None:
        return _decorate_test_or_class(test_or_class)
    return _decorate_test_or_class