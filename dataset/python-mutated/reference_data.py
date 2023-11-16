"""TensorFlow testing subclass to automate numerical testing.

Reference tests determine when behavior deviates from some "gold standard," and
are useful for determining when layer definitions have changed without
performing full regression testing, which is generally prohibitive. This class
handles the symbolic graph comparison as well as loading weights to avoid
relying on random number generation, which can change.

The tests performed by this class are:

1) Compare a generated graph against a reference graph. Differences are not
   necessarily fatal.
2) Attempt to load known weights for the graph. If this step succeeds but
   changes are present in the graph, a warning is issued but does not raise
   an exception.
3) Perform a calculation and compare the result to a reference value.

This class also provides a method to generate reference data.

Note:
  The test class is responsible for fixing the random seed during graph
  definition. A convenience method name_to_seed() is provided to make this
  process easier.

The test class should also define a .regenerate() class method which (usually)
just calls the op definition function with test=False for all relevant tests.

A concise example of this class in action is provided in:
  official/utils/testing/reference_data_test.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import hashlib
import json
import os
import shutil
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

class BaseTest(tf.test.TestCase):
    """TestCase subclass for performing reference data tests."""

    def regenerate(self):
        if False:
            i = 10
            return i + 15
        'Subclasses should override this function to generate a new reference.'
        raise NotImplementedError

    @property
    def test_name(self):
        if False:
            return 10
        'Subclass should define its own name.'
        raise NotImplementedError

    @property
    def data_root(self):
        if False:
            i = 10
            return i + 15
        'Use the subclass directory rather than the parent directory.\n\n    Returns:\n      The path prefix for reference data.\n    '
        return os.path.join(os.path.split(os.path.abspath(__file__))[0], 'reference_data', self.test_name)
    ckpt_prefix = 'model.ckpt'

    @staticmethod
    def name_to_seed(name):
        if False:
            print('Hello World!')
        'Convert a string into a 32 bit integer.\n\n    This function allows test cases to easily generate random fixed seeds by\n    hashing the name of the test. The hash string is in hex rather than base 10\n    which is why there is a 16 in the int call, and the modulo projects the\n    seed from a 128 bit int to 32 bits for readability.\n\n    Args:\n      name: A string containing the name of a test.\n\n    Returns:\n      A pseudo-random 32 bit integer derived from name.\n    '
        seed = hashlib.md5(name.encode('utf-8')).hexdigest()
        return int(seed, 16) % (2 ** 32 - 1)

    @staticmethod
    def common_tensor_properties(input_array):
        if False:
            i = 10
            return i + 15
        'Convenience function for matrix testing.\n\n    In tests we wish to determine whether a result has changed. However storing\n    an entire n-dimensional array is impractical. A better approach is to\n    calculate several values from that array and test that those derived values\n    are unchanged. The properties themselves are arbitrary and should be chosen\n    to be good proxies for a full equality test.\n\n    Args:\n      input_array: A numpy array from which key values are extracted.\n\n    Returns:\n      A list of values derived from the input_array for equality tests.\n    '
        output = list(input_array.shape)
        flat_array = input_array.flatten()
        output.extend([float(i) for i in [flat_array[0], flat_array[-1], np.sum(flat_array)]])
        return output

    def default_correctness_function(self, *args):
        if False:
            return 10
        'Returns a vector with the concatenation of common properties.\n\n    This function simply calls common_tensor_properties() for every element.\n    It is useful as it allows one to easily construct tests of layers without\n    having to worry about the details of result checking.\n\n    Args:\n      *args: A list of numpy arrays corresponding to tensors which have been\n        evaluated.\n\n    Returns:\n      A list of values containing properties for every element in args.\n    '
        output = []
        for arg in args:
            output.extend(self.common_tensor_properties(arg))
        return output

    def _construct_and_save_reference_files(self, name, graph, ops_to_eval, correctness_function):
        if False:
            for i in range(10):
                print('nop')
        'Save reference data files.\n\n    Constructs a serialized graph_def, layer weights, and computation results.\n    It then saves them to files which are read at test time.\n\n    Args:\n      name: String defining the run. This will be used to define folder names\n        and will be used for random seed construction.\n      graph: The graph in which the test is conducted.\n      ops_to_eval: Ops which the user wishes to be evaluated under a controlled\n        session.\n      correctness_function: This function accepts the evaluated results of\n        ops_to_eval, and returns a list of values. This list must be JSON\n        serializable; in particular it is up to the user to convert numpy\n        dtypes into builtin dtypes.\n    '
        data_dir = os.path.join(self.data_root, name)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir)
        graph_bytes = graph.as_graph_def().SerializeToString()
        expected_file = os.path.join(data_dir, 'expected_graph')
        with tf.io.gfile.GFile(expected_file, 'wb') as f:
            f.write(graph_bytes)
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            saver = tf.compat.v1.train.Saver()
        with self.session(graph=graph) as sess:
            sess.run(init)
            saver.save(sess=sess, save_path=os.path.join(data_dir, self.ckpt_prefix))
            os.remove(os.path.join(data_dir, 'checkpoint'))
            os.remove(os.path.join(data_dir, self.ckpt_prefix + '.meta'))
            eval_results = [op.eval() for op in ops_to_eval]
            if correctness_function is not None:
                results = correctness_function(*eval_results)
                result_json = os.path.join(data_dir, 'results.json')
                with tf.io.gfile.GFile(result_json, 'w') as f:
                    json.dump(results, f)
            tf_version_json = os.path.join(data_dir, 'tf_version.json')
            with tf.io.gfile.GFile(tf_version_json, 'w') as f:
                json.dump([tf.version.VERSION, tf.version.GIT_VERSION], f)

    def _evaluate_test_case(self, name, graph, ops_to_eval, correctness_function):
        if False:
            i = 10
            return i + 15
        'Determine if a graph agrees with the reference data.\n\n    Args:\n      name: String defining the run. This will be used to define folder names\n        and will be used for random seed construction.\n      graph: The graph in which the test is conducted.\n      ops_to_eval: Ops which the user wishes to be evaluated under a controlled\n        session.\n      correctness_function: This function accepts the evaluated results of\n        ops_to_eval, and returns a list of values. This list must be JSON\n        serializable; in particular it is up to the user to convert numpy\n        dtypes into builtin dtypes.\n    '
        data_dir = os.path.join(self.data_root, name)
        graph_bytes = graph.as_graph_def().SerializeToString()
        expected_file = os.path.join(data_dir, 'expected_graph')
        with tf.io.gfile.GFile(expected_file, 'rb') as f:
            expected_graph_bytes = f.read()
        differences = pywrap_tensorflow.EqualGraphDefWrapper(graph_bytes, expected_graph_bytes).decode('utf-8')
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            saver = tf.compat.v1.train.Saver()
        with tf.io.gfile.GFile(os.path.join(data_dir, 'tf_version.json'), 'r') as f:
            (tf_version_reference, tf_git_version_reference) = json.load(f)
        tf_version_comparison = ''
        if tf.version.GIT_VERSION != tf_git_version_reference:
            tf_version_comparison = 'Test was built using:     {} (git = {})\nLocal TensorFlow version: {} (git = {})'.format(tf_version_reference, tf_git_version_reference, tf.version.VERSION, tf.version.GIT_VERSION)
        with self.session(graph=graph) as sess:
            sess.run(init)
            try:
                saver.restore(sess=sess, save_path=os.path.join(data_dir, self.ckpt_prefix))
                if differences:
                    tf.compat.v1.logging.warn('The provided graph is different than expected:\n  {}\nHowever the weights were still able to be loaded.\n{}'.format(differences, tf_version_comparison))
            except:
                raise self.failureException('Weight load failed. Graph comparison:\n  {}{}'.format(differences, tf_version_comparison))
            eval_results = [op.eval() for op in ops_to_eval]
            if correctness_function is not None:
                results = correctness_function(*eval_results)
                result_json = os.path.join(data_dir, 'results.json')
                with tf.io.gfile.GFile(result_json, 'r') as f:
                    expected_results = json.load(f)
                self.assertAllClose(results, expected_results)

    def _save_or_test_ops(self, name, graph, ops_to_eval=None, test=True, correctness_function=None):
        if False:
            print('Hello World!')
        'Utility function to automate repeated work of graph checking and saving.\n\n    The philosophy of this function is that the user need only define ops on\n    a graph and specify which results should be validated. The actual work of\n    managing snapshots and calculating results should be automated away.\n\n    Args:\n      name: String defining the run. This will be used to define folder names\n        and will be used for random seed construction.\n      graph: The graph in which the test is conducted.\n      ops_to_eval: Ops which the user wishes to be evaluated under a controlled\n        session.\n      test: Boolean. If True this function will test graph correctness, load\n        weights, and compute numerical values. If False the necessary test data\n        will be generated and saved.\n      correctness_function: This function accepts the evaluated results of\n        ops_to_eval, and returns a list of values. This list must be JSON\n        serializable; in particular it is up to the user to convert numpy\n        dtypes into builtin dtypes.\n    '
        ops_to_eval = ops_to_eval or []
        if test:
            try:
                self._evaluate_test_case(name=name, graph=graph, ops_to_eval=ops_to_eval, correctness_function=correctness_function)
            except:
                tf.compat.v1.logging.error('Failed unittest {}'.format(name))
                raise
        else:
            self._construct_and_save_reference_files(name=name, graph=graph, ops_to_eval=ops_to_eval, correctness_function=correctness_function)

class ReferenceDataActionParser(argparse.ArgumentParser):
    """Minimal arg parser so that test regeneration can be called from the CLI."""

    def __init__(self):
        if False:
            return 10
        super(ReferenceDataActionParser, self).__init__()
        self.add_argument('--regenerate', '-regen', action='store_true', help='Enable this flag to regenerate test data. If not set unit testswill be run.')

def main(argv, test_class):
    if False:
        while True:
            i = 10
    'Simple switch function to allow test regeneration from the CLI.'
    flags = ReferenceDataActionParser().parse_args(argv[1:])
    if flags.regenerate:
        if sys.version_info[0] == 2:
            raise NameError('\nPython2 unittest does not support being run as a standalone class.\nAs a result tests must be regenerated using Python3.\nTests can be run under 2 or 3.')
        test_class().regenerate()
    else:
        tf.test.main()