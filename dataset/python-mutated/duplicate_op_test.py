"""Tests for custom user ops."""
import os
from tensorflow.python.framework import load_library
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test

class DuplicateOpTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testBasic(self):
        if False:
            while True:
                i = 10
        library_filename = os.path.join(resource_loader.get_data_files_path(), 'duplicate_op.so')
        load_library.load_op_library(library_filename)
        with self.cached_session():
            self.assertEqual(math_ops.add(1, 41).eval(), 42)
if __name__ == '__main__':
    test.main()