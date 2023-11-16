"""Run doctests for tensorflow."""
import importlib
import os
import pkgutil
import sys
from absl import flags
from absl.testing import absltest
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.eager import context
from tensorflow.python.ops import logging_ops
from tensorflow.tools.docs import tf_doctest_lib
import doctest
tf.compat.v1.enable_v2_behavior()
logging_ops.enable_interactive_logging()
FLAGS = flags.FLAGS
flags.DEFINE_list('module', [], 'A list of specific module to run doctest on.')
flags.DEFINE_list('module_prefix_skip', [], 'A list of modules to ignore when resolving modules.')
flags.DEFINE_boolean('list', None, 'List all the modules in the core package imported.')
flags.DEFINE_integer('required_gpus', 0, 'The number of GPUs required for the tests.')
PACKAGES = ['tensorflow.python.', 'tensorflow.lite.python.']

def recursive_import(root):
    if False:
        while True:
            i = 10
    'Recursively imports all the sub-modules under a root package.\n\n  Args:\n    root: A python package.\n  '
    for (_, name, _) in pkgutil.walk_packages(root.__path__, prefix=root.__name__ + '.'):
        try:
            importlib.import_module(name)
        except (AttributeError, ImportError):
            pass

def find_modules():
    if False:
        i = 10
        return i + 15
    'Finds all the modules in the core package imported.\n\n  Returns:\n    A list containing all the modules in tensorflow.python.\n  '
    tf_modules = []
    for (name, module) in sys.modules.items():
        for package in PACKAGES:
            if name.startswith(package):
                tf_modules.append(module)
    return tf_modules

def filter_on_submodules(all_modules, submodules):
    if False:
        for i in range(10):
            print('nop')
    'Filters all the modules based on the modules flag.\n\n  The module flag has to be relative to the core package imported.\n  For example, if `module=keras.layers` then, this function will return\n  all the modules in the submodule.\n\n  Args:\n    all_modules: All the modules in the core package.\n    submodules: Submodules to filter from all the modules.\n\n  Returns:\n    All the modules in the submodule.\n  '
    filtered_modules = []
    for mod in all_modules:
        for submodule in submodules:
            for package in PACKAGES:
                if package + submodule in mod.__name__:
                    filtered_modules.append(mod)
    return filtered_modules

def setup_gpu(required_gpus):
    if False:
        for i in range(10):
            print('nop')
    "Sets up the GPU devices.\n\n  If there're more available GPUs than needed, it hides the additional ones. If\n  there're less, it creates logical devices. This is to make sure the tests see\n  a fixed number of GPUs regardless of the environment.\n\n  Args:\n    required_gpus: an integer. The number of GPUs required.\n\n  Raises:\n    ValueError: if num_gpus is larger than zero but no GPU is available.\n  "
    if required_gpus == 0:
        return
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    if not available_gpus:
        raise ValueError('requires at least one physical GPU')
    if len(available_gpus) >= required_gpus:
        tf.config.set_visible_devices(available_gpus[:required_gpus])
    else:
        num_logical_gpus = required_gpus - len(available_gpus) + 1
        logical_gpus = [tf.config.LogicalDeviceConfiguration(memory_limit=256) for _ in range(num_logical_gpus)]
        tf.config.set_logical_device_configuration(available_gpus[0], logical_gpus)

class TfTestCase(tf.test.TestCase):

    def set_up(self, test):
        if False:
            while True:
                i = 10
        tf.config.set_soft_device_placement(True)
        self.setUp()
        context.async_wait()

    def tear_down(self, test):
        if False:
            for i in range(10):
                print('nop')
        self.tearDown()

def load_tests(unused_loader, tests, unused_ignore):
    if False:
        for i in range(10):
            print('nop')
    'Loads all the tests in the docstrings and runs them.'
    tf_modules = find_modules()
    if FLAGS.module:
        tf_modules = filter_on_submodules(tf_modules, FLAGS.module)
    if FLAGS.list:
        print('**************************************************')
        for mod in tf_modules:
            print(mod.__name__)
        print('**************************************************')
        return tests
    test_shard_index = int(os.environ.get('TEST_SHARD_INDEX', '0'))
    total_test_shards = int(os.environ.get('TEST_TOTAL_SHARDS', '1'))
    tf_modules = sorted(tf_modules, key=lambda mod: mod.__name__)
    for (n, module) in enumerate(tf_modules):
        if n % total_test_shards != test_shard_index:
            continue
        if any((module.__name__.startswith(package + prefix) for prefix in FLAGS.module_prefix_skip for package in PACKAGES)):
            continue
        testcase = TfTestCase()
        tests.addTests(doctest.DocTestSuite(module, test_finder=doctest.DocTestFinder(exclude_empty=False), extraglobs={'tf': tf, 'np': np, 'os': os}, setUp=testcase.set_up, tearDown=testcase.tear_down, checker=tf_doctest_lib.TfDoctestOutputChecker(), optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL | doctest.DONT_ACCEPT_BLANKLINE))
    return tests

def setUpModule():
    if False:
        i = 10
        return i + 15
    setup_gpu(FLAGS.required_gpus)
if __name__ == '__main__':
    for pkg in PACKAGES:
        recursive_import(importlib.import_module(pkg[:-1]))
    absltest.main()