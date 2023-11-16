"""Tests for the private `MatchingFilesDataset`."""
import os
import shutil
import tempfile
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import matching_files
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class MatchingFilesDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MatchingFilesDatasetTest, self).setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            return 10
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super(MatchingFilesDatasetTest, self).tearDown()

    def _touchTempFiles(self, filenames):
        if False:
            print('Hello World!')
        for filename in filenames:
            open(os.path.join(self.tmp_dir, filename), 'a').close()

    @combinations.generate(test_base.default_test_combinations())
    def testNonExistingDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the MatchingFiles dataset with a non-existing directory.'
        self.tmp_dir = os.path.join(self.tmp_dir, 'nonexistingdir')
        dataset = matching_files.MatchingFilesDataset(os.path.join(self.tmp_dir, '*'))
        self.assertDatasetProduces(dataset, expected_error=(errors.NotFoundError, ''))

    @combinations.generate(test_base.default_test_combinations())
    def testEmptyDirectory(self):
        if False:
            print('Hello World!')
        'Test the MatchingFiles dataset with an empty directory.'
        dataset = matching_files.MatchingFilesDataset(os.path.join(self.tmp_dir, '*'))
        self.assertDatasetProduces(dataset, expected_error=(errors.NotFoundError, ''))

    @combinations.generate(test_base.default_test_combinations())
    def testSimpleDirectory(self):
        if False:
            print('Hello World!')
        'Test the MatchingFiles dataset with a simple directory.'
        filenames = ['a', 'b', 'c']
        self._touchTempFiles(filenames)
        dataset = matching_files.MatchingFilesDataset(os.path.join(self.tmp_dir, '*'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(os.path.join(self.tmp_dir, filename)) for filename in filenames], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testFileSuffixes(self):
        if False:
            return 10
        'Test the MatchingFiles dataset using the suffixes of filename.'
        filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
        self._touchTempFiles(filenames)
        dataset = matching_files.MatchingFilesDataset(os.path.join(self.tmp_dir, '*.py'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(os.path.join(self.tmp_dir, filename)) for filename in filenames[1:-1]], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testFileMiddles(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the MatchingFiles dataset using the middles of filename.'
        filenames = ['aa.txt', 'bb.py', 'bbc.pyc', 'cc.pyc']
        self._touchTempFiles(filenames)
        dataset = matching_files.MatchingFilesDataset(os.path.join(self.tmp_dir, 'b*.py*'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(os.path.join(self.tmp_dir, filename)) for filename in filenames[1:3]], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testNestedDirectories(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the MatchingFiles dataset with nested directories.'
        filenames = []
        width = 8
        depth = 4
        for i in range(width):
            for j in range(depth):
                new_base = os.path.join(self.tmp_dir, str(i), *[str(dir_name) for dir_name in range(j)])
                os.makedirs(new_base)
                child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
                for f in child_files:
                    filename = os.path.join(new_base, f)
                    filenames.append(filename)
                    open(filename, 'w').close()
        patterns = [os.path.join(self.tmp_dir, os.path.join(*['**' for _ in range(depth)]), suffix) for suffix in ['*.txt', '*.log']]
        dataset = matching_files.MatchingFilesDataset(patterns)
        next_element = self.getNext(dataset)
        expected_filenames = [compat.as_bytes(filename) for filename in filenames if filename.endswith('.txt') or filename.endswith('.log')]
        actual_filenames = []
        while True:
            try:
                actual_filenames.append(compat.as_bytes(self.evaluate(next_element())))
            except errors.OutOfRangeError:
                break
        self.assertCountEqual(expected_filenames, actual_filenames)

class MatchingFilesDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_iterator_graph(self, test_patterns):
        if False:
            return 10
        return matching_files.MatchingFilesDataset(test_patterns)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            print('Hello World!')
        tmp_dir = tempfile.mkdtemp()
        width = 16
        depth = 8
        for i in range(width):
            for j in range(depth):
                new_base = os.path.join(tmp_dir, str(i), *[str(dir_name) for dir_name in range(j)])
                if not os.path.exists(new_base):
                    os.makedirs(new_base)
                child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
                for f in child_files:
                    filename = os.path.join(new_base, f)
                    open(filename, 'w').close()
        patterns = [os.path.join(tmp_dir, os.path.join(*['**' for _ in range(depth)]), suffix) for suffix in ['*.txt', '*.log']]
        num_outputs = width * len(patterns)
        verify_fn(self, lambda : self._build_iterator_graph(patterns), num_outputs)
        shutil.rmtree(tmp_dir, ignore_errors=True)
if __name__ == '__main__':
    test.main()