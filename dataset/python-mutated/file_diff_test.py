"""Diff test that compares two files are identical."""
from absl import flags
import tensorflow as tf
FLAGS = flags.FLAGS
flags.DEFINE_string('actual_file', None, 'File to test.')
flags.DEFINE_string('expected_file', None, 'File with expected contents.')

class DiffTest(tf.test.TestCase):

    def testEqualFiles(self):
        if False:
            print('Hello World!')
        content_actual = None
        content_expected = None
        try:
            with open(FLAGS.actual_file) as actual:
                content_actual = actual.read()
        except IOError as e:
            self.fail("Error opening '%s': %s" % (FLAGS.actual_file, e.strerror))
        try:
            with open(FLAGS.expected_file) as expected:
                content_expected = expected.read()
        except IOError as e:
            self.fail("Error opening '%s': %s" % (FLAGS.expected_file, e.strerror))
        self.assertTrue(content_actual == content_expected)
if __name__ == '__main__':
    tf.test.main()