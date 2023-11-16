"""Tests for the `SnapshotDataset` transformation."""
import multiprocessing
import os
import shutil
import time
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import snapshot
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.kernel_tests import tf_record_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

def is_graphdef_file(filename):
    if False:
        i = 10
        return i + 15
    return filename.endswith('-graph.pbtxt')

def is_temp_file(filename):
    if False:
        return 10
    return '-tmp-' in filename

def listdir_and_filter(dirname, filter_fn):
    if False:
        print('Hello World!')
    return [path for path in sorted(os.listdir(dirname)) if filter_fn(path)]

class SnapshotTest(tf_record_test_base.TFRecordTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(SnapshotTest, self).setUp()
        tmpdir = self.get_temp_dir()
        tmpdir = os.path.join(tmpdir, 'snapshot')
        os.mkdir(tmpdir)
        self._snapshot_dir = tmpdir

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super(SnapshotTest, self).tearDown()
        shutil.rmtree(self._snapshot_dir)

    def createTFRecords(self, num_files=10, num_records=100):
        if False:
            i = 10
            return i + 15
        self._num_files = num_files
        self._num_records = num_records
        self._filenames = self._createFiles()

    def removeTFRecords(self):
        if False:
            print('Hello World!')
        for filename in self._filenames:
            os.remove(filename)
        self._filenames = []
        self._num_files = None
        self._num_records = None

    def assertDatasetProducesSet(self, dataset, expected):
        if False:
            print('Hello World!')
        actual = []
        next_fn = self.getNext(dataset)
        for _ in range(len(expected)):
            elem = self.evaluate(next_fn())
            actual.append(elem)
        self.assertCountEqual(actual, expected)
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(next_fn())

    def assertSnapshotDirectoryContains(self, directory, num_fingerprints, num_runs_per_fingerprint, num_snapshot_shards_per_run):
        if False:
            i = 10
            return i + 15
        dirlist = listdir_and_filter(directory, lambda p: not (is_graphdef_file(p) or is_temp_file(p)))
        self.assertLen(dirlist, num_fingerprints)
        for i in range(num_fingerprints):
            fingerprint_dir = os.path.join(directory, dirlist[i])
            fingerprint_dir_list = listdir_and_filter(fingerprint_dir, lambda p: not is_temp_file(p))
            self.assertLen(fingerprint_dir_list, num_runs_per_fingerprint + 1)
            self.assertEqual(fingerprint_dir_list[num_runs_per_fingerprint], 'snapshot.metadata')
            for j in range(num_runs_per_fingerprint):
                run_dir = os.path.join(fingerprint_dir, fingerprint_dir_list[j])
                run_dirlist = sorted(os.listdir(run_dir))
                for k in range(10):
                    if len(run_dirlist) == num_snapshot_shards_per_run:
                        break
                    time.sleep(1)
                    run_dirlist = sorted(os.listdir(run_dir))
                self.assertLen(run_dirlist, num_snapshot_shards_per_run)
                file_counter = 0
                for filename in run_dirlist:
                    self.assertEqual(filename, '%08d.shard' % file_counter)
                    file_counter += 1

    @combinations.generate(test_base.default_test_combinations())
    def testCreateSnapshotDataset(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.from_tensors([1, 2, 3])
        dataset.snapshot(path=self._snapshot_dir)

    @combinations.generate(test_base.default_test_combinations())
    def testReadSnapshotDatasetDefault(self):
        if False:
            while True:
                i = 10
        self.createTFRecords()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 100)]
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset, expected)
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset2, expected)

    @combinations.generate(test_base.default_test_combinations())
    def testReadSnapshotDatasetAutoWriteSnappyRead(self):
        if False:
            while True:
                i = 10
        self.createTFRecords()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 100)]
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.snapshot(path=self._snapshot_dir, compression='AUTO')
        self.assertDatasetProduces(dataset, expected)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.snapshot(path=self._snapshot_dir, compression='SNAPPY')
        self.assertDatasetProduces(dataset2, expected)

    @combinations.generate(test_base.default_test_combinations())
    def testReadSnapshotDatasetCustomShardFn(self):
        if False:
            for i in range(10):
                print('nop')
        self.createTFRecords()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 100)]
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.snapshot(path=self._snapshot_dir, shard_func=lambda _: np.int64(0))
        self.assertDatasetProduces(dataset, expected)
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=1)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.snapshot(path=self._snapshot_dir, shard_func=lambda _: 0)
        self.assertDatasetProduces(dataset2, expected)

    @combinations.generate(test_base.default_test_combinations())
    def testReadSnapshotDatasetCustomReaderFn(self):
        if False:
            i = 10
            return i + 15
        self.createTFRecords()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 100)]
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.snapshot(path=self._snapshot_dir, reader_func=lambda ds: ds.interleave(lambda x: x, cycle_length=4, num_parallel_calls=4))
        self.assertDatasetProduces(dataset, expected)
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.snapshot(self._snapshot_dir, reader_func=lambda ds: ds.interleave(lambda x: x, cycle_length=4, num_parallel_calls=4))
        self.assertDatasetProducesSet(dataset2, expected)

    @combinations.generate(test_base.default_test_combinations())
    def testSnapshotDatasetInvalidShardFn(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(1000)
        with self.assertRaises(TypeError):
            dataset = dataset.snapshot(path=self._snapshot_dir, shard_func=lambda _: 'invalid_fn')
            next_fn = self.getNext(dataset)
            self.evaluate(next_fn())

    @combinations.generate(test_base.default_test_combinations())
    def testSnapshotDatasetInvalidReaderFn(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(1000)
        with self.assertRaises(TypeError):
            dataset = dataset.snapshot(path=self._snapshot_dir, reader_func=lambda x: x + 1)
            next_fn = self.getNext(dataset)
            self.evaluate(next_fn())

    @combinations.generate(test_base.default_test_combinations())
    def testRoundtripEmptySnapshot(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(0)
        dataset = dataset.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset, [])
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=0)
        dataset2 = dataset_ops.Dataset.range(0)
        dataset2 = dataset.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset2, [])

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotDatasetSimple(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset, list(range(1000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotDatasetMultipleFingerprints(self):
        if False:
            i = 10
            return i + 15
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset1, list(range(1000)))
        dataset2 = dataset_ops.Dataset.range(2000)
        dataset2 = dataset2.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset2, list(range(2000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=2, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotDatasetSameFingerprintMultipleCompleteRuns(self):
        if False:
            print('Hello World!')
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset1, list(range(1000)))
        dataset2 = dataset_ops.Dataset.range(1000)
        dataset2 = dataset2.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset2, list(range(1000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotDatasetSameFingerprintIncompleteRunRestart(self):
        if False:
            for i in range(10):
                print('nop')
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.snapshot(path=self._snapshot_dir)
        next1 = self.getNext(dataset1)
        for i in range(500):
            self.assertEqual(i, self.evaluate(next1()))
        dataset2 = dataset_ops.Dataset.range(1000)
        dataset2 = dataset2.snapshot(path=self._snapshot_dir)
        next2 = self.getNext(dataset2)
        for i in range(500):
            self.assertEqual(i, self.evaluate(next2()))
        for i in range(500, 1000):
            self.assertEqual(i, self.evaluate(next1()))
            self.assertEqual(i, self.evaluate(next2()))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=2, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotCustomShardFunction(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.enumerate()
        dataset = dataset.snapshot(path=self._snapshot_dir, shard_func=lambda i, _: i % 2)
        dataset = dataset.map(lambda _, elem: elem)
        self.assertDatasetProduces(dataset, list(range(1000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=2)

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotDatasetWithTuples(self):
        if False:
            for i in range(10):
                print('nop')
        dataset1 = dataset_ops.Dataset.range(0, 1000)
        dataset2 = dataset_ops.Dataset.range(1000, 2000)
        dataset3 = dataset_ops.Dataset.range(2000, 3000)
        dataset4 = dataset_ops.Dataset.range(3000, 4000)
        dataset = dataset_ops.Dataset.zip((dataset1, dataset2, dataset3, dataset4))
        dataset = dataset.snapshot(path=self._snapshot_dir)
        expected = list(zip(range(0, 1000), range(1000, 2000), range(2000, 3000), range(3000, 4000)))
        self.assertDatasetProduces(dataset, expected)
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotShuffleSameFingerprint(self):
        if False:
            for i in range(10):
                print('nop')

        def make_dataset():
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(1000)
            dataset = dataset.shuffle(1000)
            dataset = dataset.snapshot(path=self._snapshot_dir)
            return dataset
        dataset1 = make_dataset()
        self.assertDatasetProducesSet(dataset1, list(range(1000)))
        dataset2 = make_dataset()
        self.assertDatasetProducesSet(dataset2, list(range(1000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testReadUsingFlatMap(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.snapshot(path=self._snapshot_dir)
        self.assertDatasetProduces(dataset, list(range(1000)))
        flat_map = dataset_ops.Dataset.from_tensors(dataset).flat_map(lambda x: x)
        self.assertDatasetProduces(flat_map, list(range(1000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testReadOptimizableUsingFlatMap(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.shuffle(10)
        dataset = dataset.repeat(2)
        dataset = dataset.snapshot(path=self._snapshot_dir)
        self.assertDatasetProducesSet(dataset, 2 * list(range(1000)))
        flat_map = dataset_ops.Dataset.from_tensors(dataset).flat_map(lambda x: x)
        self.assertDatasetProducesSet(flat_map, 2 * list(range(1000)))
        self.assertSnapshotDirectoryContains(self._snapshot_dir, num_fingerprints=1, num_runs_per_fingerprint=1, num_snapshot_shards_per_run=multiprocessing.cpu_count())

    @combinations.generate(test_base.default_test_combinations())
    def testRepeatAndPrefetch(self):
        if False:
            i = 10
            return i + 15
        'This test reproduces github.com/tensorflow/tensorflow/issues/48903.'
        dataset = dataset_ops.Dataset.from_tensor_slices(np.random.rand(16, 32))
        dataset = dataset.snapshot(path=self._snapshot_dir)
        dataset = dataset.shuffle(buffer_size=16)
        dataset = dataset.batch(16)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)
        next_element = self.getNext(dataset)
        for _ in range(30):
            self.evaluate(next_element())

    def testName(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.from_tensors(42)
        dataset = dataset.snapshot(path=self._snapshot_dir, name='snapshot')
        self.assertDatasetProduces(dataset, [42])

class LegacySnapshotTest(tf_record_test_base.TFRecordTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(LegacySnapshotTest, self).setUp()
        self.removeTFRecords()
        tmpdir = self.get_temp_dir()
        tmpdir = os.path.join(tmpdir, 'snapshot')
        os.mkdir(tmpdir)
        self.snapshot_dir = tmpdir

    def tearDown(self):
        if False:
            while True:
                i = 10
        super(LegacySnapshotTest, self).tearDown()
        shutil.rmtree(self.snapshot_dir)

    def removeTFRecords(self):
        if False:
            i = 10
            return i + 15
        for filename in self._filenames:
            os.remove(filename)
        self._filenames = []

    def setUpTFRecord(self, num_files=10, num_records=10):
        if False:
            for i in range(10):
                print('nop')
        self._num_files = num_files
        self._num_records = num_records
        self._filenames = self._createFiles()

    def makeSnapshotDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        return self.snapshot_dir

    def assertSnapshotDirectoryContains(self, directory, num_fingerprints, num_runs_per_fp, num_snapshot_files):
        if False:
            i = 10
            return i + 15
        dirlist = listdir_and_filter(directory, lambda p: not (is_graphdef_file(p) or is_temp_file(p)))
        self.assertLen(dirlist, num_fingerprints)
        for i in range(num_fingerprints):
            fingerprint_dir = os.path.join(directory, dirlist[i])
            fingerprint_dir_list = listdir_and_filter(fingerprint_dir, lambda p: not is_temp_file(p))
            self.assertLen(fingerprint_dir_list, num_runs_per_fp + 1)
            self.assertEqual(fingerprint_dir_list[num_runs_per_fp], 'snapshot.metadata')
            for j in range(num_runs_per_fp):
                run_dir = os.path.join(fingerprint_dir, fingerprint_dir_list[j])
                run_dirlist = sorted(os.listdir(run_dir))
                self.assertLen(run_dirlist, num_snapshot_files)
                file_counter = 0
                for filename in run_dirlist:
                    self.assertEqual(filename, '%08d.snapshot' % file_counter)
                    file_counter += 1

    @combinations.generate(test_base.default_test_combinations())
    def testWriteDifferentPipelinesInOneDirectory(self):
        if False:
            while True:
                i = 10
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
        self.assertDatasetProduces(dataset, list(range(1000)))
        dataset = dataset_ops.Dataset.range(1001)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
        self.assertDatasetProduces(dataset, list(range(1001)))
        self.assertSnapshotDirectoryContains(tmpdir, 2, 1, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testWriteSnapshotMultipleSimultaneous(self):
        if False:
            i = 10
            return i + 15
        tmpdir = self.snapshot_dir
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.apply(snapshot.legacy_snapshot(tmpdir))
        next1 = self.getNext(dataset1)
        dataset2 = dataset_ops.Dataset.range(1000)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir))
        next2 = self.getNext(dataset2)
        for i in range(0, 1000):
            self.assertEqual(i, self.evaluate(next1()))
            self.assertEqual(i, self.evaluate(next2()))
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testGetNextCreatesDir(self):
        if False:
            print('Hello World!')
        tmpdir = self.snapshot_dir
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.apply(snapshot.legacy_snapshot(tmpdir))
        next1 = self.getNext(dataset1)
        dataset2 = dataset_ops.Dataset.range(1001)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir))
        _ = self.getNext(dataset2)
        for _ in range(1000):
            self.evaluate(next1())
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP, snapshot.COMPRESSION_SNAPPY])))
    def testWriteSnapshotSimpleSuccessful(self, compression):
        if False:
            while True:
                i = 10
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(1000)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, compression=compression))
        self.assertDatasetProduces(dataset, list(range(1000)))
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP, snapshot.COMPRESSION_SNAPPY])))
    def testWriteSnapshotRepeatAfterwards(self, compression):
        if False:
            i = 10
            return i + 15
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, compression=compression))
        dataset = dataset.repeat(10)
        self.assertDatasetProduces(dataset, list(range(10)) * 10)
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP, snapshot.COMPRESSION_SNAPPY])))
    def testWriteSnapshotMixTypes(self, compression):
        if False:
            for i in range(10):
                print('nop')
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)

        def map_fn(x):
            if False:
                while True:
                    i = 10
            return (x, string_ops.as_string(x), string_ops.as_string(2 * x), 2 * x)
        dataset = dataset.map(map_fn)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, compression=compression))
        dataset = dataset.repeat(10)
        expected = []
        for i in range(10):
            expected.append((i, str(i), str(2 * i), 2 * i))
        self.assertDatasetProduces(dataset, expected * 10)
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testSpecifySnapshotNameWriteAndRead(self):
        if False:
            i = 10
            return i + 15
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, snapshot_name='my_custom_snapshot'))
        dataset = dataset.repeat(10)
        self.assertDatasetProduces(dataset, list(range(10)) * 10)
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)
        self.assertTrue(os.path.exists(os.path.join(tmpdir, 'custom-my_custom_snapshot')))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, 'custom-my_custom_snapshot', 'custom')))

    @combinations.generate(test_base.default_test_combinations())
    def testForcePassthroughMode(self):
        if False:
            while True:
                i = 10
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode='passthrough'))
        dataset = dataset.repeat(10)
        self.assertDatasetProduces(dataset, list(range(10)) * 10)
        self.assertSnapshotDirectoryContains(tmpdir, 0, 0, 0)

    @combinations.generate(test_base.default_test_combinations())
    def testForceWriteMode(self):
        if False:
            print('Hello World!')
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode='write'))
        dataset = dataset.repeat(10)
        self.assertDatasetProduces(dataset, list(range(10)) * 10)
        self.assertSnapshotDirectoryContains(tmpdir, 1, 10, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testForceReadMode(self):
        if False:
            return 10
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode='write', snapshot_name='my_custom_snapshot'))
        self.assertDatasetProduces(dataset, list(range(10)))
        shutil.move(os.path.join(tmpdir, 'custom-my_custom_snapshot'), os.path.join(tmpdir, 'custom-my_custom_snapshot_2'))
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode='read', snapshot_name='my_custom_snapshot_2'))
        self.assertDatasetProduces(dataset, list(range(10)))
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testForceReadNonexistentSnapshot(self):
        if False:
            return 10
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        with self.assertRaises(errors.NotFoundError):
            dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode='read'))
            get_next = self.getNext(dataset)
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testForceReadNonexistentNamedSnapshot(self):
        if False:
            i = 10
            return i + 15
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.range(10)
        with self.assertRaises(errors.NotFoundError):
            dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, mode='read', snapshot_name='my_nonexistent_snapshot'))
            get_next = self.getNext(dataset)
            self.evaluate(get_next())

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP, snapshot.COMPRESSION_SNAPPY])))
    def testReadSnapshotBackAfterWrite(self, compression):
        if False:
            print('Hello World!')
        self.setUpTFRecord()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 10)]
        tmpdir = self.snapshot_dir
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, compression=compression))
        self.assertDatasetProduces(dataset, expected)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir, compression=compression))
        self.assertDatasetProduces(dataset2, expected)

    @combinations.generate(test_base.default_test_combinations())
    def testReadShuffledSnapshotAfterWrite(self):
        if False:
            i = 10
            return i + 15
        self.setUpTFRecord(num_files=10, num_records=50)
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 50)]
        tmpdir = self.snapshot_dir
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=100))
        self.assertDatasetProduces(dataset, expected)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=100, shuffle_on_read=True))
        shuffled_elements = self.getDatasetOutput(dataset2)
        self.assertNotEqual(shuffled_elements, expected)
        self.assertCountEqual(shuffled_elements, expected)
        dataset3 = core_readers._TFRecordDataset(filenames)
        dataset3 = dataset3.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=100, shuffle_on_read=True))
        self.assertDatasetProduces(dataset3, expected, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testReadShuffledSnapshotWithSeedAfterWrite(self):
        if False:
            print('Hello World!')
        self.setUpTFRecord(num_files=10, num_records=50)
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 50)]
        tmpdir = self.snapshot_dir
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10))
        self.assertDatasetProduces(dataset, expected)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10, shuffle_on_read=True, shuffle_seed=123456))
        next2 = self.getNext(dataset2)
        dataset3 = core_readers._TFRecordDataset(filenames)
        dataset3 = dataset3.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10, shuffle_on_read=True, shuffle_seed=123456))
        next3 = self.getNext(dataset3)
        for _ in range(500):
            res2 = self.evaluate(next2())
            res3 = self.evaluate(next3())
            self.assertEqual(res2, res3)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP, snapshot.COMPRESSION_SNAPPY])))
    def testReadSnapshotParallelAfterWrite(self, compression):
        if False:
            for i in range(10):
                print('nop')
        self.setUpTFRecord(5, 500)
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 5) for r in range(0, 500)]
        tmpdir = self.snapshot_dir
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=1024 * 1024, num_reader_threads=2, reader_buffer_size=10, compression=compression))
        self.assertDatasetProduces(dataset, expected, assert_items_equal=True)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=1024 * 1024, num_reader_threads=2, reader_buffer_size=10, compression=compression))
        self.assertDatasetProduces(dataset2, expected, assert_items_equal=True)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.times(combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP]), combinations.combine(threads=2, size=[1, 2]) + combinations.combine(threads=8, size=[1, 4, 8]))))
    def testReadSnapshotBackAfterMultiThreadedWrite(self, compression, threads, size):
        if False:
            while True:
                i = 10
        self.setUpTFRecord()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 10)]
        tmpdir = self.snapshot_dir
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, compression=compression, num_writer_threads=threads, writer_buffer_size=size))
        self.assertDatasetProduces(dataset, expected)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir, compression=compression))
        self.assertDatasetProduces(dataset2, expected, assert_items_equal=True)

    @combinations.generate(test_base.default_test_combinations())
    def testSameFingerprintWithDifferentInitializationOrder(self):
        if False:
            return 10
        tmpdir = self.snapshot_dir
        dataset1 = dataset_ops.Dataset.range(0, 100)
        dataset2 = dataset_ops.Dataset.range(100, 200)
        dataset3 = dataset_ops.Dataset.range(200, 300)
        dataset = dataset1.concatenate(dataset2).concatenate(dataset3)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
        self.assertDatasetProduces(dataset, list(range(300)))
        dataset4 = dataset_ops.Dataset.range(200, 300)
        dataset5 = dataset_ops.Dataset.range(100, 200)
        dataset6 = dataset_ops.Dataset.range(0, 100)
        dataset = dataset6.concatenate(dataset5).concatenate(dataset4)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
        self.assertDatasetProduces(dataset, list(range(300)))
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testExpiredSnapshotRewrite(self):
        if False:
            print('Hello World!')
        tmpdir = self.snapshot_dir
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.apply(snapshot.legacy_snapshot(tmpdir, pending_snapshot_expiry_seconds=1))
        next1 = self.getNext(dataset1)
        for _ in range(500):
            self.evaluate(next1())
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)
        time.sleep(2)
        dataset2 = dataset_ops.Dataset.range(1000)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir, pending_snapshot_expiry_seconds=1))
        next2 = self.getNext(dataset2)
        for _ in range(500):
            self.evaluate(next2())
        self.assertSnapshotDirectoryContains(tmpdir, 1, 2, 1)

    @combinations.generate(test_base.default_test_combinations())
    def testSnapshotArgsCreateNewSnapshot(self):
        if False:
            for i in range(10):
                print('nop')
        tmpdir = self.snapshot_dir
        dataset1 = dataset_ops.Dataset.range(1000)
        dataset1 = dataset1.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10000))
        next1 = self.getNext(dataset1)
        for _ in range(1000):
            self.evaluate(next1())
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, 1)
        dataset2 = dataset_ops.Dataset.range(1000)
        dataset2 = dataset1.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=20000))
        next2 = self.getNext(dataset2)
        for _ in range(1000):
            self.evaluate(next2())
        self.assertSnapshotDirectoryContains(tmpdir, 2, 1, 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(compression=[snapshot.COMPRESSION_NONE, snapshot.COMPRESSION_GZIP, snapshot.COMPRESSION_SNAPPY])))
    def testSpecifyShardSize(self, compression):
        if False:
            print('Hello World!')
        tmpdir = self.snapshot_dir
        dataset = dataset_ops.Dataset.from_tensor_slices([1.0])
        dataset = dataset.map(lambda x: gen_array_ops.broadcast_to(x, [1024, 1024]))
        dataset = dataset.repeat(10)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir, shard_size_bytes=10 * 1024 * 1024, compression=compression))
        next_fn = self.getNext(dataset)
        for _ in range(10):
            self.evaluate(next_fn())
        num_files = 1
        if compression == snapshot.COMPRESSION_NONE:
            num_files = 3
        self.assertSnapshotDirectoryContains(tmpdir, 1, 1, num_files)

    @combinations.generate(test_base.default_test_combinations())
    def testAdditionalOperationsAfterReadBack(self):
        if False:
            print('Hello World!')
        self.setUpTFRecord()
        filenames = self._filenames
        expected = [b'Record %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 10)]
        tmpdir = self.snapshot_dir
        dataset = core_readers._TFRecordDataset(filenames)
        dataset = dataset.apply(snapshot.legacy_snapshot(tmpdir))
        self.assertDatasetProduces(dataset, expected)
        self.removeTFRecords()
        dataset2 = core_readers._TFRecordDataset(filenames)
        dataset2 = dataset2.apply(snapshot.legacy_snapshot(tmpdir))
        self.assertDatasetProduces(dataset2, expected)
        expected_after = [b'cord %d of file %d' % (r, f) for f in range(0, 10) for r in range(0, 10)]
        dataset3 = core_readers._TFRecordDataset(filenames)
        dataset3 = dataset3.apply(snapshot.legacy_snapshot(tmpdir))
        dataset3 = dataset3.map(lambda x: string_ops.substr_v2(x, 2, 1000))
        self.assertDatasetProduces(dataset3, expected_after)

class SnapshotCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_snapshot_dataset(self, repeat=False):
        if False:
            i = 10
            return i + 15

        def ds_fn():
            if False:
                return 10
            self._snapshot_dir = os.path.join(self.get_temp_dir(), 'snapshot')
            if not os.path.exists(self._snapshot_dir):
                os.mkdir(self._snapshot_dir)
            dataset = dataset_ops.Dataset.range(100)
            dataset = dataset.snapshot(path=self._snapshot_dir)
            if repeat:
                dataset = dataset.repeat(2)
            return dataset
        return ds_fn

    @combinations.generate(test_base.default_test_combinations())
    def testCheckpointBeforeEpochEndNoRepeat(self):
        if False:
            while True:
                i = 10
        ds_fn = self._build_snapshot_dataset(repeat=False)
        outputs = self.gen_outputs(ds_fn, [], 50, verify_exhausted=False)
        self.assertSequenceEqual(outputs, range(50))
        outputs.extend(self.gen_outputs(ds_fn, [], 50, ckpt_saved=True, verify_exhausted=True))
        self.assertSequenceEqual(outputs, range(100))

    @combinations.generate(test_base.default_test_combinations())
    def testCheckpointBeforeOneEpochWithReading(self):
        if False:
            for i in range(10):
                print('nop')
        ds_fn = self._build_snapshot_dataset(repeat=True)
        outputs = self.gen_outputs(ds_fn, [], 50, verify_exhausted=False)
        self.assertSequenceEqual(outputs, list(range(50)))
        t = self.gen_outputs(ds_fn, [], 150, ckpt_saved=True, verify_exhausted=False)
        outputs.extend(t)
        self.assertSequenceEqual(outputs, list(range(50)) + list(range(50, 100)) + list(range(100)))

    @combinations.generate(test_base.default_test_combinations())
    def testCheckpointBeforeOneEpochThenRunAFewSteps(self):
        if False:
            return 10
        ds_fn = self._build_snapshot_dataset(repeat=False)
        outputs = self.gen_outputs(ds_fn, [10], 20, verify_exhausted=False, save_checkpoint_at_end=False)
        self.assertSequenceEqual(outputs, range(20))
        outputs = outputs[:10]
        outputs.extend(self.gen_outputs(ds_fn, [], 90, ckpt_saved=True, verify_exhausted=True))
        self.assertSequenceEqual(outputs, range(100))

    @combinations.generate(test_base.default_test_combinations())
    def testCheckpointAfterOneEpoch(self):
        if False:
            return 10
        ds_fn = self._build_snapshot_dataset(repeat=True)
        outputs = self.gen_outputs(ds_fn, [], 110, verify_exhausted=False)
        self.assertSequenceEqual(outputs, list(range(100)) + list(range(10)))
        t = self.gen_outputs(ds_fn, [], 90, ckpt_saved=True, verify_exhausted=True)
        outputs.extend(t)
        self.assertSequenceEqual(outputs, list(range(100)) + list(range(10)) + list(range(10, 100)))

    @combinations.generate(test_base.default_test_combinations())
    def testCheckpointAfterOneEpochRunFewSteps(self):
        if False:
            return 10
        ds_fn = self._build_snapshot_dataset(repeat=True)
        outputs = self.gen_outputs(ds_fn, [110], 120, verify_exhausted=False, save_checkpoint_at_end=False)
        self.assertSequenceEqual(outputs, list(range(100)) + list(range(20)))
        outputs = outputs[:110]
        t = self.gen_outputs(ds_fn, [], 90, ckpt_saved=True, verify_exhausted=True)
        outputs.extend(t)
        self.assertSequenceEqual(outputs, list(range(100)) + list(range(10)) + list(range(10, 100)))

class LegacySnapshotCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_snapshot_dataset(self, num_threads=1, repeat=False, pending_snapshot_expiry_seconds=-1, shard_size_bytes=None):
        if False:
            return 10

        def ds_fn():
            if False:
                print('Hello World!')
            self.snapshot_dir = os.path.join(self.get_temp_dir(), 'snapshot')
            if not os.path.exists(self.snapshot_dir):
                os.mkdir(self.snapshot_dir)
            dataset = dataset_ops.Dataset.range(1000)
            dataset = dataset.apply(snapshot.legacy_snapshot(self.snapshot_dir, num_writer_threads=num_threads, writer_buffer_size=2 * num_threads, num_reader_threads=num_threads, reader_buffer_size=2 * num_threads, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds, shard_size_bytes=shard_size_bytes))
            if repeat:
                dataset = dataset.repeat(2)
            options = options_lib.Options()
            options.experimental_optimization.inject_prefetch = False
            dataset = dataset.with_options(options)
            return dataset
        return ds_fn

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
    def testSnapshotBeforeEpochEnd(self, pending_snapshot_expiry_seconds):
        if False:
            print('Hello World!')
        ds_fn = self._build_snapshot_dataset(pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
        outputs = self.gen_outputs(ds_fn, [], 100, verify_exhausted=False)
        self.assertSequenceEqual(outputs, range(100))
        outputs.extend(self.gen_outputs(ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
        self.assertSequenceEqual(outputs, range(1000))

    @combinations.generate(combinations.times(test_base.graph_only_combinations(), combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
    def testCheckpointBeforeOneEpochThenRunFewStepsSmallShardMultiThread(self, pending_snapshot_expiry_seconds):
        if False:
            i = 10
            return i + 15
        ds_fn = self._build_snapshot_dataset(pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds, shard_size_bytes=100)
        outputs = []
        with ops.Graph().as_default() as g:
            (init_op, get_next_op, saver) = self._build_graph(ds_fn)
            with self.session(graph=g) as sess:
                self._initialize(init_op, sess)
                start = 0
                end = 100
                num_iters = end - start
                for _ in range(num_iters):
                    outputs.append(sess.run(get_next_op))
                self._save(sess, saver)
                start = 100
                end = 400
                num_iters = end - start
                for _ in range(num_iters):
                    outputs.append(sess.run(get_next_op))
        self.assertSequenceEqual(outputs, range(400))
        outputs = outputs[:100]
        outputs.extend(self.gen_outputs(ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
        self.assertSequenceEqual(outputs, range(1000))
        fp_dir_list = os.listdir(self.snapshot_dir)
        self.assertLen(list(fp_dir_list), 2)
        for d in fp_dir_list:
            if not d.endswith('-graph.pbtxt'):
                fp_dir = os.path.join(self.snapshot_dir, d)
                run_dir_list = os.listdir(fp_dir)
                self.assertLen(list(run_dir_list), 2)
                for e in run_dir_list:
                    if e != 'snapshot.metadata':
                        run_dir = os.path.join(fp_dir, e)
                        self.assertLen(list(os.listdir(run_dir)), 258)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
    def testCheckpointBeforeOneEpochThenRunFewSteps(self, pending_snapshot_expiry_seconds):
        if False:
            return 10
        ds_fn = self._build_snapshot_dataset(pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
        outputs = self.gen_outputs(ds_fn, [100], 200, verify_exhausted=False, save_checkpoint_at_end=False)
        self.assertSequenceEqual(outputs, range(200))
        outputs = outputs[:100]
        outputs.extend(self.gen_outputs(ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
        self.assertSequenceEqual(outputs, range(1000))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
    def testCheckpointBeforeOneEpochThenRunFewStepsMultipleThreads(self, pending_snapshot_expiry_seconds):
        if False:
            i = 10
            return i + 15
        ds_fn = self._build_snapshot_dataset(num_threads=2, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
        outputs = self.gen_outputs(ds_fn, [100], 200, verify_exhausted=False, save_checkpoint_at_end=False)
        self.assertSequenceEqual(outputs, range(200))
        outputs = outputs[:100]
        outputs.extend(self.gen_outputs(ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False))
        self.assertSequenceEqual(outputs, range(1000))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
    def testCheckpointAfterOneEpoch(self, pending_snapshot_expiry_seconds):
        if False:
            print('Hello World!')
        ds_fn = self._build_snapshot_dataset(repeat=True, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
        outputs = self.gen_outputs(ds_fn, [], 1100, verify_exhausted=False)
        self.assertSequenceEqual(outputs, list(range(1000)) + list(range(100)))
        t = self.gen_outputs(ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False)
        outputs.extend(t)
        self.assertSequenceEqual(outputs, list(range(1000)) + list(range(100)) + list(range(900)))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(pending_snapshot_expiry_seconds=[None, 1])))
    def testCheckpointAfterOneEpochThenRunFewSteps(self, pending_snapshot_expiry_seconds):
        if False:
            return 10
        ds_fn = self._build_snapshot_dataset(repeat=True, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds)
        outputs = self.gen_outputs(ds_fn, [1100], 1200, verify_exhausted=False, save_checkpoint_at_end=False)
        self.assertSequenceEqual(outputs, list(range(1000)) + list(range(100)) + list(range(100)))
        outputs = outputs[:1100]
        t = self.gen_outputs(ds_fn, [], 900, ckpt_saved=True, verify_exhausted=False)
        outputs.extend(t)
        self.assertSequenceEqual(outputs, list(range(1000)) + list(range(100)) + list(range(900)))
if __name__ == '__main__':
    test.main()