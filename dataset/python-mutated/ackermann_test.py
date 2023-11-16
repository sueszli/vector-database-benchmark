"""Tests for custom user ops."""
import os
from tensorflow.python.framework import load_library
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test

class AckermannTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testBasic(self):
        if False:
            return 10
        library_filename = os.path.join(resource_loader.get_data_files_path(), 'ackermann_op.so')
        ackermann = load_library.load_op_library(library_filename)
        with self.cached_session():
            self.assertEqual(ackermann.ackermann().eval(), b'A(m, 0) == A(m-1, 1)')
if __name__ == '__main__':
    test.main()