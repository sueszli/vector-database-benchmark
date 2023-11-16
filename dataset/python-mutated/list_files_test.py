"""Tests for `tf.data.Dataset.list_files()`."""
from os import path
import shutil
import tempfile
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.util import compat

class ListFilesTest(test_base.DatasetTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(ListFilesTest, self).setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super(ListFilesTest, self).tearDown()

    def _touchTempFiles(self, filenames):
        if False:
            print('Hello World!')
        for filename in filenames:
            open(path.join(self.tmp_dir, filename), 'a').close()

    @combinations.generate(test_base.default_test_combinations())
    def testEmptyDirectory(self):
        if False:
            print('Hello World!')
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, 'No files matched'):
            dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
            self.getNext(dataset, requires_initialization=True)

    @combinations.generate(test_base.default_test_combinations())
    def testSimpleDirectory(self):
        if False:
            while True:
                i = 10
        filenames = ['a', 'b', 'c']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testSimpleDirectoryNotShuffled(self):
        if False:
            while True:
                i = 10
        filenames = ['b', 'c', 'a']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'), shuffle=False)
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in sorted(filenames)])

    def testFixedSeedResultsInRepeatableOrder(self):
        if False:
            i = 10
            return i + 15
        filenames = ['a', 'b', 'c']
        self._touchTempFiles(filenames)

        def dataset_fn():
            if False:
                i = 10
                return i + 15
            return dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'), shuffle=True, seed=37)
        expected_filenames = [compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames]
        all_actual_filenames = []
        for _ in range(3):
            actual_filenames = []
            next_element = self.getNext(dataset_fn(), requires_initialization=True)
            try:
                while True:
                    actual_filenames.append(self.evaluate(next_element()))
            except errors.OutOfRangeError:
                pass
            all_actual_filenames.append(actual_filenames)
        self.assertCountEqual(expected_filenames, all_actual_filenames[0])
        self.assertEqual(all_actual_filenames[0], all_actual_filenames[1])
        self.assertEqual(all_actual_filenames[0], all_actual_filenames[2])

    @combinations.generate(test_base.default_test_combinations())
    def tesEmptyDirectoryInitializer(self):
        if False:
            return 10

        def dataset_fn():
            if False:
                return 10
            return dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
        self.assertDatasetProduces(dataset_fn(), expected_error=(errors.InvalidArgumentError, 'No files matched pattern'), requires_initialization=True)

    @combinations.generate(test_base.default_test_combinations())
    def testSimpleDirectoryInitializer(self):
        if False:
            print('Hello World!')
        filenames = ['a', 'b', 'c']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testFileSuffixes(self):
        if False:
            print('Hello World!')
        filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*.py'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames[1:-1]], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testFileMiddles(self):
        if False:
            print('Hello World!')
        filenames = ['a.txt', 'b.py', 'c.pyc']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*.py*'))
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames[1:]], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testNoShuffle(self):
        if False:
            print('Hello World!')
        filenames = ['a', 'b', 'c']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files(path.join(self.tmp_dir, '*'), shuffle=False).repeat(2)
        next_element = self.getNext(dataset)
        expected_filenames = []
        actual_filenames = []
        for filename in filenames * 2:
            expected_filenames.append(compat.as_bytes(path.join(self.tmp_dir, filename)))
            actual_filenames.append(compat.as_bytes(self.evaluate(next_element())))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(next_element())
        self.assertCountEqual(expected_filenames, actual_filenames)
        self.assertEqual(actual_filenames[:len(filenames)], actual_filenames[len(filenames):])

    @combinations.generate(test_base.default_test_combinations())
    def testMultiplePatternsAsList(self):
        if False:
            for i in range(10):
                print('nop')
        filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
        self._touchTempFiles(filenames)
        patterns = [path.join(self.tmp_dir, pat) for pat in ['*.py', '*.txt']]
        dataset = dataset_ops.Dataset.list_files(patterns)
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames[:-1]], assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testMultiplePatternsAsTensor(self):
        if False:
            while True:
                i = 10
        filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
        self._touchTempFiles(filenames)
        dataset = dataset_ops.Dataset.list_files([path.join(self.tmp_dir, pat) for pat in ['*.py', '*.txt']])
        self.assertDatasetProduces(dataset, expected_output=[compat.as_bytes(path.join(self.tmp_dir, filename)) for filename in filenames[:-1]], assert_items_equal=True)
if __name__ == '__main__':
    test.main()