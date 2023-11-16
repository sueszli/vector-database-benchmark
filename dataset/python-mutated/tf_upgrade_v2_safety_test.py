"""Tests for tf 2.0 upgrader in safety mode."""
import io
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade_v2_safety

class TfUpgradeV2SafetyTest(test_util.TensorFlowTestCase):

    def _upgrade(self, old_file_text):
        if False:
            print('Hello World!')
        in_file = io.StringIO(old_file_text)
        out_file = io.StringIO()
        upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2_safety.TFAPIChangeSpec())
        (count, report, errors) = upgrader.process_opened_file('test.py', in_file, 'test_out.py', out_file)
        return (count, report, errors, out_file.getvalue())

    def testContribWarning(self):
        if False:
            print('Hello World!')
        text = 'tf.contrib.foo()'
        (_, report, _, _) = self._upgrade(text)
        expected_info = 'tf.contrib will not be distributed'
        self.assertIn(expected_info, report)

    def testTensorFlowImport(self):
        if False:
            i = 10
            return i + 15
        text = 'import tensorflow as tf'
        expected_text = 'import tensorflow.compat.v1 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'import tensorflow as tf, other_import as y'
        expected_text = 'import tensorflow.compat.v1 as tf, other_import as y'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'import tensorflow'
        expected_text = 'import tensorflow.compat.v1 as tensorflow'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'import tensorflow.foo'
        expected_text = 'import tensorflow.compat.v1.foo'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'import tensorflow.foo as bar'
        expected_text = 'import tensorflow.compat.v1.foo as bar'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)

    def testTensorFlowGoogleImport(self):
        if False:
            i = 10
            return i + 15
        text = 'import tensorflow.google as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'import tensorflow.google'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'import tensorflow.google.compat.v1 as tf'
        expected_text = 'import tensorflow.google.compat.v1 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'import tensorflow.google.compat.v2 as tf'
        expected_text = 'import tensorflow.google.compat.v2 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)

    def testTensorFlowImportInIndent(self):
        if False:
            for i in range(10):
                print('nop')
        text = '\ntry:\n  import tensorflow as tf  # import line\n\n  tf.ones([4, 5])\nexcept AttributeError:\n  pass\n'
        expected_text = '\ntry:\n  import tensorflow.compat.v1 as tf  # import line\n\n  tf.ones([4, 5])\nexcept AttributeError:\n  pass\n'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)

    def testTensorFlowFromImport(self):
        if False:
            i = 10
            return i + 15
        text = 'from tensorflow import foo'
        expected_text = 'from tensorflow.compat.v1 import foo'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'from tensorflow.foo import bar'
        expected_text = 'from tensorflow.compat.v1.foo import bar'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)
        text = 'from tensorflow import *'
        expected_text = 'from tensorflow.compat.v1 import *'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(expected_text, new_text)

    def testTensorFlowImportAlreadyHasCompat(self):
        if False:
            for i in range(10):
                print('nop')
        text = 'import tensorflow.compat.v1 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'import tensorflow.compat.v2 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'from tensorflow.compat import v2 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)

    def testTensorFlowGoogleFromImport(self):
        if False:
            while True:
                i = 10
        text = 'from tensorflow.google.compat import v1 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'from tensorflow.google.compat import v2 as tf'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)

    def testTensorFlowDontChangeContrib(self):
        if False:
            print('Hello World!')
        text = 'import tensorflow.contrib as foo'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'from tensorflow import contrib'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)

    def test_contrib_to_addons_move(self):
        if False:
            i = 10
            return i + 15
        small_mapping = {'tf.contrib.layers.poincare_normalize': 'tfa.layers.PoincareNormalize', 'tf.contrib.layers.maxout': 'tfa.layers.Maxout', 'tf.contrib.layers.group_norm': 'tfa.layers.GroupNormalization', 'tf.contrib.layers.instance_norm': 'tfa.layers.InstanceNormalization'}
        for (symbol, replacement) in small_mapping.items():
            text = "{}('stuff', *args, **kwargs)".format(symbol)
            (_, report, _, _) = self._upgrade(text)
            self.assertIn(replacement, report)
if __name__ == '__main__':
    test_lib.main()

    def testTensorFlowDontChangeContrib(self):
        if False:
            print('Hello World!')
        text = 'import tensorflow.contrib as foo'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)
        text = 'from tensorflow import contrib'
        (_, _, _, new_text) = self._upgrade(text)
        self.assertEqual(text, new_text)

    def test_contrib_to_addons_move(self):
        if False:
            while True:
                i = 10
        small_mapping = {'tf.contrib.layers.poincare_normalize': 'tfa.layers.PoincareNormalize', 'tf.contrib.layers.maxout': 'tfa.layers.Maxout', 'tf.contrib.layers.group_norm': 'tfa.layers.GroupNormalization', 'tf.contrib.layers.instance_norm': 'tfa.layers.InstanceNormalization'}
        for (symbol, replacement) in small_mapping.items():
            text = "{}('stuff', *args, **kwargs)".format(symbol)
            (_, report, _, _) = self._upgrade(text)
            self.assertIn(replacement, report)
if __name__ == '__main__':
    test_lib.main()