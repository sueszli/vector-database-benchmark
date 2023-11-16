import numpy as np
from keras import testing
from keras.utils import timeseries_dataset_utils

class TimeseriesDatasetTest(testing.TestCase):

    def test_basics(self):
        if False:
            print('Hello World!')
        data = np.arange(100)
        targets = data * 2
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, targets, sequence_length=9, batch_size=5)
        for (i, batch) in enumerate(dataset):
            self.assertLen(batch, 2)
            (inputs, targets) = batch
            if i < 18:
                self.assertEqual(inputs.shape, (5, 9))
            if i == 18:
                self.assertEqual(inputs.shape, (2, 9))
            self.assertAllClose(targets, inputs[:, 0] * 2)
            for j in range(min(5, len(inputs))):
                self.assertAllClose(inputs[j], np.arange(i * 5 + j, i * 5 + j + 9))

    def test_timeseries_regression(self):
        if False:
            while True:
                i = 10
        data = np.arange(10)
        offset = 3
        targets = data[offset:]
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, targets, sequence_length=offset, batch_size=1)
        i = 0
        for batch in dataset:
            self.assertLen(batch, 2)
            (inputs, targets) = batch
            self.assertEqual(inputs.shape, (1, 3))
            self.assertAllClose(targets[0], data[offset + i])
            self.assertAllClose(inputs[0], data[i:i + offset])
            i += 1
        self.assertEqual(i, 7)

    def test_no_targets(self):
        if False:
            i = 10
            return i + 15
        data = np.arange(50)
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, None, sequence_length=10, batch_size=5)
        i = None
        for (i, batch) in enumerate(dataset):
            if i < 8:
                self.assertEqual(batch.shape, (5, 10))
            elif i == 8:
                self.assertEqual(batch.shape, (1, 10))
            for j in range(min(5, len(batch))):
                self.assertAllClose(batch[j], np.arange(i * 5 + j, i * 5 + j + 10))
        self.assertEqual(i, 8)

    def test_shuffle(self):
        if False:
            i = 10
            return i + 15
        data = np.arange(10)
        targets = data * 2
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, targets, sequence_length=5, batch_size=1, shuffle=True, seed=123)
        first_seq = None
        for (x, y) in dataset.take(1):
            self.assertNotAllClose(x, np.arange(0, 5))
            self.assertAllClose(x[:, 0] * 2, y)
            first_seq = x
        for (x, _) in dataset.take(1):
            self.assertNotAllClose(x, first_seq)
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, targets, sequence_length=5, batch_size=1, shuffle=True, seed=123)
        for (x, _) in dataset.take(1):
            self.assertAllClose(x, first_seq)

    def test_sampling_rate(self):
        if False:
            print('Hello World!')
        data = np.arange(100)
        targets = data * 2
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, targets, sequence_length=9, batch_size=5, sampling_rate=2)
        for (i, batch) in enumerate(dataset):
            self.assertLen(batch, 2)
            (inputs, targets) = batch
            if i < 16:
                self.assertEqual(inputs.shape, (5, 9))
            if i == 16:
                self.assertEqual(inputs.shape, (4, 9))
            self.assertAllClose(inputs[:, 0] * 2, targets)
            for j in range(min(5, len(inputs))):
                start_index = i * 5 + j
                end_index = start_index + 9 * 2
                self.assertAllClose(inputs[j], np.arange(start_index, end_index, 2))

    def test_sequence_stride(self):
        if False:
            print('Hello World!')
        data = np.arange(100)
        targets = data * 2
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, targets, sequence_length=9, batch_size=5, sequence_stride=3)
        for (i, batch) in enumerate(dataset):
            self.assertLen(batch, 2)
            (inputs, targets) = batch
            if i < 6:
                self.assertEqual(inputs.shape, (5, 9))
            if i == 6:
                self.assertEqual(inputs.shape, (1, 9))
            self.assertAllClose(inputs[:, 0] * 2, targets)
            for j in range(min(5, len(inputs))):
                start_index = i * 5 * 3 + j * 3
                end_index = start_index + 9
                self.assertAllClose(inputs[j], np.arange(start_index, end_index))

    def test_start_and_end_index(self):
        if False:
            return 10
        data = np.arange(100)
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, None, sequence_length=9, batch_size=5, sequence_stride=3, sampling_rate=2, start_index=10, end_index=90)
        for batch in dataset:
            self.assertLess(np.max(batch[0]), 90)
            self.assertGreater(np.min(batch[0]), 9)

    def test_errors(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, '`start_index` must be '):
            _ = timeseries_dataset_utils.timeseries_dataset_from_array(np.arange(10), None, 3, start_index=-1)
        with self.assertRaisesRegex(ValueError, '`start_index` must be '):
            _ = timeseries_dataset_utils.timeseries_dataset_from_array(np.arange(10), None, 3, start_index=11)
        with self.assertRaisesRegex(ValueError, '`end_index` must be '):
            _ = timeseries_dataset_utils.timeseries_dataset_from_array(np.arange(10), None, 3, end_index=-1)
        with self.assertRaisesRegex(ValueError, '`end_index` must be '):
            _ = timeseries_dataset_utils.timeseries_dataset_from_array(np.arange(10), None, 3, end_index=11)
        with self.assertRaisesRegex(ValueError, '`sampling_rate` must be '):
            _ = timeseries_dataset_utils.timeseries_dataset_from_array(np.arange(10), None, 3, sampling_rate=0)
        with self.assertRaisesRegex(ValueError, '`sequence_stride` must be '):
            _ = timeseries_dataset_utils.timeseries_dataset_from_array(np.arange(10), None, 3, sequence_stride=0)

    def test_not_batched(self):
        if False:
            return 10
        data = np.arange(100)
        dataset = timeseries_dataset_utils.timeseries_dataset_from_array(data, None, sequence_length=9, batch_size=None, shuffle=True)
        sample = next(iter(dataset))
        self.assertEqual(len(sample.shape), 1)