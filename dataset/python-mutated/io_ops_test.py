"""Tests for tensorflow.python.ops.io_ops."""
import os
import shutil
import tempfile
from tensorflow.python.framework import test_util
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class IoOpsTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testReadFile(self):
        if False:
            for i in range(10):
                print('nop')
        cases = ['', 'Some contents', 'Неки садржаји на српском']
        for contents in cases:
            contents = compat.as_bytes(contents)
            with tempfile.NamedTemporaryFile(prefix='ReadFileTest', dir=self.get_temp_dir(), delete=False) as temp:
                temp.write(contents)
            with self.cached_session():
                read = io_ops.read_file(temp.name)
                self.assertEqual([], read.get_shape())
                self.assertEqual(self.evaluate(read), contents)
            os.remove(temp.name)

    def testWriteFile(self):
        if False:
            i = 10
            return i + 15
        cases = ['', 'Some contents']
        for contents in cases:
            contents = compat.as_bytes(contents)
            with tempfile.NamedTemporaryFile(prefix='WriteFileTest', dir=self.get_temp_dir(), delete=False) as temp:
                pass
            with self.cached_session() as sess:
                w = io_ops.write_file(temp.name, contents)
                self.evaluate(w)
                with open(temp.name, 'rb') as f:
                    file_contents = f.read()
                self.assertEqual(file_contents, contents)
            os.remove(temp.name)

    def testWriteFileCreateDir(self):
        if False:
            print('Hello World!')
        cases = ['', 'Some contents']
        for contents in cases:
            contents = compat.as_bytes(contents)
            subdir = os.path.join(self.get_temp_dir(), 'subdir1')
            filepath = os.path.join(subdir, 'subdir2', 'filename')
            with self.cached_session() as sess:
                w = io_ops.write_file(filepath, contents)
                self.evaluate(w)
                with open(filepath, 'rb') as f:
                    file_contents = f.read()
                self.assertEqual(file_contents, contents)
            shutil.rmtree(subdir)

    def _subset(self, files, indices):
        if False:
            i = 10
            return i + 15
        return set((compat.as_bytes(files[i].name) for i in range(len(files)) if i in indices))

    @test_util.run_deprecated_v1
    def testMatchingFiles(self):
        if False:
            while True:
                i = 10
        cases = ['ABcDEF.GH', 'ABzDEF.GH', 'ABasdfjklDEF.GH', 'AB3DEF.GH', 'AB4DEF.GH', 'ABDEF.GH', 'XYZ']
        files = [tempfile.NamedTemporaryFile(prefix=c, dir=self.get_temp_dir(), delete=True) for c in cases]
        with self.cached_session():
            for f in files:
                self.assertEqual(io_ops.matching_files(f.name).eval(), compat.as_bytes(f.name))
            directory_path = files[0].name[:files[0].name.find(cases[0])]
            pattern = directory_path + 'AB%sDEF.GH*'
            self.assertEqual(set(io_ops.matching_files(pattern % 'z').eval()), self._subset(files, [1]))
            self.assertEqual(set(io_ops.matching_files(pattern % '?').eval()), self._subset(files, [0, 1, 3, 4]))
            self.assertEqual(set(io_ops.matching_files(pattern % '*').eval()), self._subset(files, [0, 1, 2, 3, 4, 5]))
            if os.name != 'nt':
                self.assertEqual(set(io_ops.matching_files(pattern % '[cxz]').eval()), self._subset(files, [0, 1]))
                self.assertEqual(set(io_ops.matching_files(pattern % '[0-9]').eval()), self._subset(files, [3, 4]))
            self.assertItemsEqual(io_ops.matching_files([]).eval(), [])
            self.assertItemsEqual(io_ops.matching_files([files[0].name, files[1].name, files[2].name]).eval(), self._subset(files, [0, 1, 2]))
            self.assertItemsEqual(io_ops.matching_files([pattern % '?', directory_path + 'X?Z*']).eval(), self._subset(files, [0, 1, 3, 4, 6]))
        for f in files:
            f.close()
if __name__ == '__main__':
    test.main()