"""Unit tests for source_utils."""
import ast
import os
import sys
import tempfile
import zipfile
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect

def line_number_above():
    if False:
        while True:
            i = 10
    "Get lineno of the AST node immediately above this function's call site.\n\n  It is assumed that there is no empty line(s) between the call site and the\n  preceding AST node.\n\n  Returns:\n    The lineno of the preceding AST node, at the same level of the AST.\n    If the preceding AST spans multiple lines:\n      - In Python 3.8+, the lineno of the first line is returned.\n      - In older Python versions, the lineno of the last line is returned.\n  "
    call_site_lineno = tf_inspect.stack()[1][2]
    if sys.version_info < (3, 8):
        return call_site_lineno - 1
    else:
        with open(__file__, 'rb') as f:
            source_text = f.read().decode('utf-8')
        source_tree = ast.parse(source_text)
        prev_node = _find_preceding_ast_node(source_tree, call_site_lineno)
        return prev_node.lineno

def _find_preceding_ast_node(node, lineno):
    if False:
        i = 10
        return i + 15
    'Find the ast node immediately before and not including lineno.'
    for (i, child_node) in enumerate(node.body):
        if child_node.lineno == lineno:
            return node.body[i - 1]
        if hasattr(child_node, 'body'):
            found_node = _find_preceding_ast_node(child_node, lineno)
            if found_node:
                return found_node

class GuessIsTensorFlowLibraryTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.curr_file_path = os.path.normpath(os.path.abspath(__file__))

    def tearDown(self):
        if False:
            print('Hello World!')
        ops.reset_default_graph()

    def testGuessedBaseDirIsProbablyCorrect(self):
        if False:
            print('Hello World!')
        self.assertIn(os.path.basename(source_utils._TENSORFLOW_BASEDIR), ['tensorflow', 'tensorflow_core'])

    def testUnitTestFileReturnsFalse(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(self.curr_file_path))

    def testSourceUtilModuleReturnsTrue(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(source_utils.guess_is_tensorflow_py_library(source_utils.__file__))

    @test_util.run_v1_only('Tensor.op is not available in TF 2.x')
    def testFileInPythonKernelsPathReturnsTrue(self):
        if False:
            print('Hello World!')
        x = constant_op.constant(42.0, name='x')
        self.assertTrue(source_utils.guess_is_tensorflow_py_library(x.op.traceback[-1][0]))

    def testDebuggerExampleFilePathReturnsFalse(self):
        if False:
            print('Hello World!')
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(os.path.normpath('site-packages/tensorflow/python/debug/examples/debug_mnist.py')))
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(os.path.normpath('site-packages/tensorflow/python/debug/examples/v1/example_v1.py')))
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(os.path.normpath('site-packages/tensorflow/python/debug/examples/v2/example_v2.py')))
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(os.path.normpath('site-packages/tensorflow/python/debug/examples/v3/example_v3.py')))

    def testReturnsFalseForNonPythonFile(self):
        if False:
            return 10
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(os.path.join(os.path.dirname(self.curr_file_path), 'foo.cc')))

    def testReturnsFalseForStdin(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(source_utils.guess_is_tensorflow_py_library('<stdin>'))

    def testReturnsFalseForEmptyFileName(self):
        if False:
            print('Hello World!')
        self.assertFalse(source_utils.guess_is_tensorflow_py_library(''))

class SourceHelperTest(test_util.TensorFlowTestCase):

    def createAndRunGraphHelper(self):
        if False:
            return 10
        'Create and run a TensorFlow Graph to generate debug dumps.\n\n    This is intentionally done in separate method, to make it easier to test\n    the stack-top mode of source annotation.\n    '
        self.dump_root = self.get_temp_dir()
        self.curr_file_path = os.path.abspath(tf_inspect.getfile(tf_inspect.currentframe()))
        with session.Session() as sess:
            self.u_init = constant_op.constant(np.array([[5.0, 3.0], [-1.0, 0.0]]), shape=[2, 2], name='u_init')
            self.u_init_line_number = line_number_above()
            self.u = variables.Variable(self.u_init, name='u')
            self.u_line_number = line_number_above()
            self.v_init = constant_op.constant(np.array([[2.0], [-1.0]]), shape=[2, 1], name='v_init')
            self.v_init_line_number = line_number_above()
            self.v = variables.Variable(self.v_init, name='v')
            self.v_line_number = line_number_above()
            self.w = math_ops.matmul(self.u, self.v, name='w')
            self.w_line_number = line_number_above()
            self.evaluate(self.u.initializer)
            self.evaluate(self.v.initializer)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_urls=['file://%s' % self.dump_root])
            run_metadata = config_pb2.RunMetadata()
            sess.run(self.w, options=run_options, run_metadata=run_metadata)
            self.dump = debug_data.DebugDumpDir(self.dump_root, partition_graphs=run_metadata.partition_graphs)
            self.dump.set_python_graph(sess.graph)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.createAndRunGraphHelper()
        self.helper_line_number = line_number_above()

    def tearDown(self):
        if False:
            return 10
        if os.path.isdir(self.dump_root):
            file_io.delete_recursively(self.dump_root)
        ops.reset_default_graph()

    def testAnnotateWholeValidSourceFileGivesCorrectResult(self):
        if False:
            return 10
        source_annotation = source_utils.annotate_source(self.dump, self.curr_file_path)
        self.assertIn(self.u_init.op.name, source_annotation[self.u_init_line_number])
        self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
        self.assertIn(self.v_init.op.name, source_annotation[self.v_init_line_number])
        self.assertIn(self.v.op.name, source_annotation[self.v_line_number])
        self.assertIn(self.w.op.name, source_annotation[self.w_line_number])
        self.assertIn(self.u_init.op.name, source_annotation[self.helper_line_number])
        self.assertIn(self.u.op.name, source_annotation[self.helper_line_number])
        self.assertIn(self.v_init.op.name, source_annotation[self.helper_line_number])
        self.assertIn(self.v.op.name, source_annotation[self.helper_line_number])
        self.assertIn(self.w.op.name, source_annotation[self.helper_line_number])

    def testAnnotateWithStackTopGivesCorrectResult(self):
        if False:
            print('Hello World!')
        source_annotation = source_utils.annotate_source(self.dump, self.curr_file_path, file_stack_top=True)
        self.assertIn(self.u_init.op.name, source_annotation[self.u_init_line_number])
        self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
        self.assertIn(self.v_init.op.name, source_annotation[self.v_init_line_number])
        self.assertIn(self.v.op.name, source_annotation[self.v_line_number])
        self.assertIn(self.w.op.name, source_annotation[self.w_line_number])
        self.assertNotIn(self.helper_line_number, source_annotation)

    def testAnnotateSubsetOfLinesGivesCorrectResult(self):
        if False:
            return 10
        source_annotation = source_utils.annotate_source(self.dump, self.curr_file_path, min_line=self.u_line_number, max_line=self.u_line_number + 1)
        self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
        self.assertNotIn(self.v_line_number, source_annotation)

    def testAnnotateDumpedTensorsGivesCorrectResult(self):
        if False:
            i = 10
            return i + 15
        source_annotation = source_utils.annotate_source(self.dump, self.curr_file_path, do_dumped_tensors=True)
        self.assertIn(self.u.name, source_annotation[self.u_line_number])
        self.assertIn(self.v.name, source_annotation[self.v_line_number])
        self.assertIn(self.w.name, source_annotation[self.w_line_number])
        self.assertNotIn(self.u.op.name, source_annotation[self.u_line_number])
        self.assertNotIn(self.v.op.name, source_annotation[self.v_line_number])
        self.assertNotIn(self.w.op.name, source_annotation[self.w_line_number])
        self.assertIn(self.u.name, source_annotation[self.helper_line_number])
        self.assertIn(self.v.name, source_annotation[self.helper_line_number])
        self.assertIn(self.w.name, source_annotation[self.helper_line_number])

    def testCallingAnnotateSourceWithoutPythonGraphRaisesException(self):
        if False:
            return 10
        self.dump.set_python_graph(None)
        with self.assertRaises(ValueError):
            source_utils.annotate_source(self.dump, self.curr_file_path)

    def testCallingAnnotateSourceOnUnrelatedSourceFileDoesNotError(self):
        if False:
            print('Hello World!')
        (fd, unrelated_source_path) = tempfile.mkstemp()
        with open(fd, 'wt') as source_file:
            source_file.write("print('hello, world')\n")
        self.assertEqual({}, source_utils.annotate_source(self.dump, unrelated_source_path))
        os.remove(unrelated_source_path)

    def testLoadingPythonSourceFileWithNonAsciiChars(self):
        if False:
            for i in range(10):
                print('nop')
        (fd, source_path) = tempfile.mkstemp()
        with open(fd, 'wb') as source_file:
            source_file.write(u"print('🙂')\n".encode('utf-8'))
        (source_lines, _) = source_utils.load_source(source_path)
        self.assertEqual(source_lines, [u"print('🙂')", u''])
        os.remove(source_path)

    def testLoadNonexistentNonParPathFailsWithIOError(self):
        if False:
            return 10
        bad_path = os.path.join(self.get_temp_dir(), 'nonexistent.py')
        with self.assertRaisesRegex(IOError, 'neither exists nor can be loaded.*par.*'):
            source_utils.load_source(bad_path)

    def testLoadingPythonSourceFileInParFileSucceeds(self):
        if False:
            return 10
        temp_file_path = os.path.join(self.get_temp_dir(), 'model.py')
        with open(temp_file_path, 'wb') as f:
            f.write(b'import tensorflow as tf\nx = tf.constant(42.0)\n')
        par_path = os.path.join(self.get_temp_dir(), 'train_model.par')
        with zipfile.ZipFile(par_path, 'w') as zf:
            zf.write(temp_file_path, os.path.join('tensorflow_models', 'model.py'))
        source_path = os.path.join(par_path, 'tensorflow_models', 'model.py')
        (source_lines, _) = source_utils.load_source(source_path)
        self.assertEqual(source_lines, ['import tensorflow as tf', 'x = tf.constant(42.0)', ''])

    def testLoadingPythonSourceFileInParFileFailsRaisingIOError(self):
        if False:
            for i in range(10):
                print('nop')
        temp_file_path = os.path.join(self.get_temp_dir(), 'model.py')
        with open(temp_file_path, 'wb') as f:
            f.write(b'import tensorflow as tf\nx = tf.constant(42.0)\n')
        par_path = os.path.join(self.get_temp_dir(), 'train_model.par')
        with zipfile.ZipFile(par_path, 'w') as zf:
            zf.write(temp_file_path, os.path.join('tensorflow_models', 'model.py'))
        source_path = os.path.join(par_path, 'tensorflow_models', 'nonexistent.py')
        with self.assertRaisesRegex(IOError, 'neither exists nor can be loaded.*par.*'):
            source_utils.load_source(source_path)

@test_util.run_v1_only('Sessions are not available in TF 2.x')
class ListSourceAgainstDumpTest(test_util.TensorFlowTestCase):

    def createAndRunGraphWithWhileLoop(self):
        if False:
            while True:
                i = 10
        'Create and run a TensorFlow Graph with a while loop to generate dumps.'
        self.dump_root = self.get_temp_dir()
        self.curr_file_path = os.path.abspath(tf_inspect.getfile(tf_inspect.currentframe()))
        with session.Session() as sess:
            loop_body = lambda i: math_ops.add(i, 2)
            self.traceback_first_line = line_number_above()
            loop_cond = lambda i: math_ops.less(i, 16)
            i = constant_op.constant(10, name='i')
            loop = while_loop.while_loop(loop_cond, loop_body, [i])
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_urls=['file://%s' % self.dump_root])
            run_metadata = config_pb2.RunMetadata()
            sess.run(loop, options=run_options, run_metadata=run_metadata)
            self.dump = debug_data.DebugDumpDir(self.dump_root, partition_graphs=run_metadata.partition_graphs)
            self.dump.set_python_graph(sess.graph)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.createAndRunGraphWithWhileLoop()

    def tearDown(self):
        if False:
            while True:
                i = 10
        if os.path.isdir(self.dump_root):
            file_io.delete_recursively(self.dump_root)
        ops.reset_default_graph()

    def testGenerateSourceList(self):
        if False:
            while True:
                i = 10
        source_list = source_utils.list_source_files_against_dump(self.dump)
        file_paths = [item[0] for item in source_list]
        self.assertEqual(sorted(file_paths), file_paths)
        self.assertEqual(len(set(file_paths)), len(file_paths))
        for item in source_list:
            self.assertTrue(isinstance(item, tuple))
            self.assertEqual(6, len(item))
        (_, is_tf_py_library, num_nodes, num_tensors, num_dumps, first_line) = source_list[file_paths.index(self.curr_file_path)]
        self.assertFalse(is_tf_py_library)
        self.assertEqual(12, num_nodes)
        self.assertEqual(14, num_tensors)
        self.assertEqual(39, num_dumps)
        self.assertEqual(self.traceback_first_line, first_line)

    def testGenerateSourceListWithNodeNameFilter(self):
        if False:
            print('Hello World!')
        source_list = source_utils.list_source_files_against_dump(self.dump, node_name_regex_allowlist='while/Add.*')
        file_paths = [item[0] for item in source_list]
        self.assertEqual(sorted(file_paths), file_paths)
        self.assertEqual(len(set(file_paths)), len(file_paths))
        for item in source_list:
            self.assertTrue(isinstance(item, tuple))
            self.assertEqual(6, len(item))
        (_, is_tf_py_library, num_nodes, num_tensors, num_dumps, _) = source_list[file_paths.index(self.curr_file_path)]
        self.assertFalse(is_tf_py_library)
        self.assertEqual(2, num_nodes)
        self.assertEqual(2, num_tensors)
        self.assertEqual(6, num_dumps)

    def testGenerateSourceListWithPathRegexFilter(self):
        if False:
            print('Hello World!')
        curr_file_basename = os.path.basename(self.curr_file_path)
        source_list = source_utils.list_source_files_against_dump(self.dump, path_regex_allowlist='.*' + curr_file_basename.replace('.', '\\.') + '$')
        self.assertEqual(1, len(source_list))
        (file_path, is_tf_py_library, num_nodes, num_tensors, num_dumps, first_line) = source_list[0]
        self.assertEqual(self.curr_file_path, file_path)
        self.assertFalse(is_tf_py_library)
        self.assertEqual(12, num_nodes)
        self.assertEqual(14, num_tensors)
        self.assertEqual(39, num_dumps)
        self.assertEqual(self.traceback_first_line, first_line)
if __name__ == '__main__':
    googletest.main()