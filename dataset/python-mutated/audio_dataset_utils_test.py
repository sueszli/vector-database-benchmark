import os
import numpy as np
from keras import testing
from keras.utils import audio_dataset_utils
from keras.utils.module_utils import tensorflow as tf

class AudioDatasetFromDirectoryTest(testing.TestCase):

    def _get_audio_samples(self, count=16, different_sequence_lengths=False):
        if False:
            print('Hello World!')
        sequence_length = 30
        num_channels = 1
        audio_samples = []
        for _ in range(count):
            if different_sequence_lengths:
                random_sequence_length = np.random.randint(10, sequence_length + 1)
                audio = np.random.random((random_sequence_length, num_channels))
            else:
                audio = np.random.random((sequence_length, num_channels))
            audio_samples.append(tf.audio.encode_wav(audio, 1000))
        return audio_samples

    def _prepare_directory(self, num_classes=2, nested_dirs=False, count=16, different_sequence_lengths=False):
        if False:
            return 10
        temp_dir = self.get_temp_dir()
        paths = []
        for class_index in range(num_classes):
            class_directory = f'class_{class_index}'
            if nested_dirs:
                class_paths = [class_directory, os.path.join(class_directory, 'subfolder_1'), os.path.join(class_directory, 'subfolder_2'), os.path.join(class_directory, 'subfolder_1', 'sub-subfolder')]
            else:
                class_paths = [class_directory]
            for path in class_paths:
                os.mkdir(os.path.join(temp_dir, path))
            paths += class_paths
        i = 0
        for audio in self._get_audio_samples(count=count, different_sequence_lengths=different_sequence_lengths):
            path = paths[i % len(paths)]
            ext = 'wav'
            filename = os.path.join(path, f'audio_{i}.{ext}')
            with open(os.path.join(temp_dir, filename), 'wb') as f:
                f.write(audio.numpy())
            i += 1
        return temp_dir

    def test_audio_dataset_from_directory_standalone(self):
        if False:
            print('Hello World!')
        directory = self._prepare_directory(count=7, num_classes=2)
        for (i, audio) in enumerate(self._get_audio_samples(3)):
            filename = f'audio_{i}.wav'
            with open(os.path.join(directory, filename), 'wb') as f:
                f.write(audio.numpy())
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=5, output_sequence_length=30, labels=None)
        batch = next(iter(dataset))
        self.assertEqual(batch.shape, (5, 30, 1))
        self.assertEqual(batch.dtype.name, 'float32')
        batch_count = 0
        sample_count = 0
        for batch in dataset:
            batch_count += 1
            sample_count += batch.shape[0]
        self.assertEqual(batch_count, 2)
        self.assertEqual(sample_count, 10)

    def test_audio_dataset_from_directory_binary(self):
        if False:
            while True:
                i = 10
        directory = self._prepare_directory(num_classes=2)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode='int')
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (8, 30, 1))
        self.assertEqual(batch[0].dtype.name, 'float32')
        self.assertEqual(batch[1].shape, (8,))
        self.assertEqual(batch[1].dtype.name, 'int32')
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode='binary')
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (8, 30, 1))
        self.assertEqual(batch[0].dtype.name, 'float32')
        self.assertEqual(batch[1].shape, (8, 1))
        self.assertEqual(batch[1].dtype.name, 'float32')
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode='categorical')
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (8, 30, 1))
        self.assertEqual(batch[0].dtype.name, 'float32')
        self.assertEqual(batch[1].shape, (8, 2))
        self.assertEqual(batch[1].dtype.name, 'float32')

    def test_static_shape_in_graph(self):
        if False:
            i = 10
            return i + 15
        directory = self._prepare_directory(num_classes=2)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode='int')
        test_case = self

        @tf.function
        def symbolic_fn(ds):
            if False:
                for i in range(10):
                    print('nop')
            for (x, _) in ds.take(1):
                test_case.assertListEqual(x.shape.as_list(), [None, 30, None])
        symbolic_fn(dataset)

    def test_sample_count(self):
        if False:
            for i in range(10):
                print('nop')
        directory = self._prepare_directory(num_classes=4, count=15)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode=None)
        sample_count = 0
        for batch in dataset:
            sample_count += batch.shape[0]
        self.assertEqual(sample_count, 15)

    def test_audio_dataset_from_directory_multiclass(self):
        if False:
            return 10
        directory = self._prepare_directory(num_classes=4, count=15)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode=None)
        batch = next(iter(dataset))
        self.assertEqual(batch.shape, (8, 30, 1))
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode=None)
        sample_count = 0
        iterator = iter(dataset)
        for batch in dataset:
            sample_count += next(iterator).shape[0]
        self.assertEqual(sample_count, 15)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode='int')
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (8, 30, 1))
        self.assertEqual(batch[0].dtype.name, 'float32')
        self.assertEqual(batch[1].shape, (8,))
        self.assertEqual(batch[1].dtype.name, 'int32')
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode='categorical')
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (8, 30, 1))
        self.assertEqual(batch[0].dtype.name, 'float32')
        self.assertEqual(batch[1].shape, (8, 4))
        self.assertEqual(batch[1].dtype.name, 'float32')

    def test_audio_dataset_from_directory_validation_split(self):
        if False:
            print('Hello World!')
        directory = self._prepare_directory(num_classes=2, count=10)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=10, output_sequence_length=30, validation_split=0.2, subset='training', seed=1337)
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (8, 30, 1))
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=10, output_sequence_length=30, validation_split=0.2, subset='validation', seed=1337)
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(batch[0].shape, (2, 30, 1))

    def test_audio_dataset_from_directory_manual_labels(self):
        if False:
            i = 10
            return i + 15
        directory = self._prepare_directory(num_classes=2, count=2)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, labels=[0, 1], shuffle=False)
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertAllClose(batch[1], [0, 1])

    def test_audio_dataset_from_directory_follow_links(self):
        if False:
            return 10
        directory = self._prepare_directory(num_classes=2, count=25, nested_dirs=True)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=8, output_sequence_length=30, label_mode=None, follow_links=True)
        sample_count = 0
        for batch in dataset:
            sample_count += batch.shape[0]
        self.assertEqual(sample_count, 25)

    def test_audio_dataset_from_directory_no_audio(self):
        if False:
            while True:
                i = 10
        directory = self._prepare_directory(num_classes=2, count=0)
        with self.assertRaisesRegex(ValueError, 'No audio files found in directory'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory)

    def test_audio_dataset_from_directory_ragged(self):
        if False:
            print('Hello World!')
        directory = self._prepare_directory(num_classes=2, count=16, different_sequence_lengths=True)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, ragged=True, batch_size=8)
        batch = next(iter(dataset))
        self.assertEqual(batch[0].shape.as_list(), [8, None, None])

    def test_audio_dataset_from_directory_no_output_sequence_length_no_ragged(self):
        if False:
            print('Hello World!')
        directory = self._prepare_directory(num_classes=2, count=16, different_sequence_lengths=True)
        (min_sequence_length, max_sequence_length) = (10, 30)
        possible_sequence_lengths = [i for i in range(min_sequence_length, max_sequence_length + 1)]
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=2)
        sequence_lengths = list(set([b.shape[1] for (b, _) in dataset]))
        for seq_len in sequence_lengths:
            self.assertIn(seq_len, possible_sequence_lengths)

    def test_audio_dataset_from_directory_no_output_sequence_length_same_lengths(self):
        if False:
            print('Hello World!')
        directory = self._prepare_directory(num_classes=2, count=16, different_sequence_lengths=False)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=2)
        sequence_lengths = list(set([batch[0].shape[1] for batch in dataset]))
        self.assertEqual(len(sequence_lengths), 1)

    def test_audio_dataset_from_directory_errors(self):
        if False:
            print('Hello World!')
        directory = self._prepare_directory(num_classes=3, count=5)
        with self.assertRaisesRegex(ValueError, '`sampling_rate` should be higher than 0. Received:'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, ragged=False, output_sequence_length=10, sampling_rate=-1)
        with self.assertRaisesRegex(ValueError, '`sampling_rate` should have an integer value. Received:'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, ragged=False, output_sequence_length=10, sampling_rate=1.2)
        try:
            import tensorflow_io
        except ImportError:
            with self.assertRaisesRegex(ImportError, 'To use the argument `sampling_rate`.*tensorflow_io.*'):
                _ = audio_dataset_utils.audio_dataset_from_directory(directory, ragged=False, output_sequence_length=10, sampling_rate=44100)
        with self.assertRaisesRegex(ValueError, 'Cannot set both `ragged` and `output_sequence_length`'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, ragged=True, output_sequence_length=30)
        with self.assertRaisesRegex(ValueError, '`labels` argument should be'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, labels='other')
        with self.assertRaisesRegex(ValueError, '`label_mode` argument must be'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, label_mode='other')
        with self.assertRaisesRegex(ValueError, 'only pass `class_names` if `labels="inferred"`'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, labels=[0, 0, 1, 1, 1], class_names=['class_0', 'class_1', 'class_2'])
        with self.assertRaisesRegex(ValueError, 'Expected the lengths of `labels` to match the number of files'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, labels=[0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, '`class_names` passed did not match'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, class_names=['class_0', 'wrong_class'])
        with self.assertRaisesRegex(ValueError, 'there must be exactly 2'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, label_mode='binary')
        with self.assertRaisesRegex(ValueError, '`validation_split` must be between 0 and 1'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, validation_split=2)
        with self.assertRaisesRegex(ValueError, '`subset` must be either "training",'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, validation_split=0.2, subset='other')
        with self.assertRaisesRegex(ValueError, '`validation_split` must be set'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, validation_split=0, subset='training')
        with self.assertRaisesRegex(ValueError, 'must provide a `seed`'):
            _ = audio_dataset_utils.audio_dataset_from_directory(directory, validation_split=0.2, subset='training')

    def test_audio_dataset_from_directory_not_batched(self):
        if False:
            for i in range(10):
                print('nop')
        directory = self._prepare_directory(num_classes=2, count=2)
        dataset = audio_dataset_utils.audio_dataset_from_directory(directory, batch_size=None, output_sequence_length=30, label_mode=None, shuffle=False)
        sample = next(iter(dataset))
        self.assertEqual(len(sample.shape), 2)