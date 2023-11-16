"""Tests for data input for speech commands."""
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.speech_commands import input_data
from tensorflow.examples.speech_commands import models
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class InputDataTest(test.TestCase):

    def _getWavData(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            sample_data = tf.zeros([32000, 2])
            wav_encoder = tf.audio.encode_wav(sample_data, 16000)
            wav_data = self.evaluate(wav_encoder)
        return wav_data

    def _saveTestWavFile(self, filename, wav_data):
        if False:
            while True:
                i = 10
        with open(filename, 'wb') as f:
            f.write(wav_data)

    def _saveWavFolders(self, root_dir, labels, how_many):
        if False:
            return 10
        wav_data = self._getWavData()
        for label in labels:
            dir_name = os.path.join(root_dir, label)
            os.mkdir(dir_name)
            for i in range(how_many):
                file_path = os.path.join(dir_name, 'some_audio_%d.wav' % i)
                self._saveTestWavFile(file_path, wav_data)

    def _model_settings(self):
        if False:
            i = 10
            return i + 15
        return {'desired_samples': 160, 'fingerprint_size': 40, 'label_count': 4, 'window_size_samples': 100, 'window_stride_samples': 100, 'fingerprint_width': 40, 'preprocess': 'mfcc'}

    def _runGetDataTest(self, preprocess, window_length_ms):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, 'wavs')
        os.mkdir(wav_dir)
        self._saveWavFolders(wav_dir, ['a', 'b', 'c'], 100)
        background_dir = os.path.join(wav_dir, '_background_noise_')
        os.mkdir(background_dir)
        wav_data = self._getWavData()
        for i in range(10):
            file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
            self._saveTestWavFile(file_path, wav_data)
        model_settings = models.prepare_model_settings(4, 16000, 1000, window_length_ms, 20, 40, preprocess)
        with self.cached_session() as sess:
            audio_processor = input_data.AudioProcessor('', wav_dir, 10, 10, ['a', 'b'], 10, 10, model_settings, tmp_dir)
            (result_data, result_labels) = audio_processor.get_data(10, 0, model_settings, 0.3, 0.1, 100, 'training', sess)
            self.assertEqual(10, len(result_data))
            self.assertEqual(10, len(result_labels))

    def testPrepareWordsList(self):
        if False:
            while True:
                i = 10
        words_list = ['a', 'b']
        self.assertGreater(len(input_data.prepare_words_list(words_list)), len(words_list))

    def testWhichSet(self):
        if False:
            print('Hello World!')
        self.assertEqual(input_data.which_set('foo.wav', 10, 10), input_data.which_set('foo.wav', 10, 10))
        self.assertEqual(input_data.which_set('foo_nohash_0.wav', 10, 10), input_data.which_set('foo_nohash_1.wav', 10, 10))

    @test_util.run_deprecated_v1
    def testPrepareDataIndex(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_temp_dir()
        self._saveWavFolders(tmp_dir, ['a', 'b', 'c'], 100)
        audio_processor = input_data.AudioProcessor('', tmp_dir, 10, 10, ['a', 'b'], 10, 10, self._model_settings(), tmp_dir)
        self.assertLess(0, audio_processor.set_size('training'))
        self.assertIn('training', audio_processor.data_index)
        self.assertIn('validation', audio_processor.data_index)
        self.assertIn('testing', audio_processor.data_index)
        self.assertEqual(input_data.UNKNOWN_WORD_INDEX, audio_processor.word_to_index['c'])

    def testPrepareDataIndexEmpty(self):
        if False:
            return 10
        tmp_dir = self.get_temp_dir()
        self._saveWavFolders(tmp_dir, ['a', 'b', 'c'], 0)
        with self.assertRaises(Exception) as e:
            _ = input_data.AudioProcessor('', tmp_dir, 10, 10, ['a', 'b'], 10, 10, self._model_settings(), tmp_dir)
        self.assertIn('No .wavs found', str(e.exception))

    def testPrepareDataIndexMissing(self):
        if False:
            i = 10
            return i + 15
        tmp_dir = self.get_temp_dir()
        self._saveWavFolders(tmp_dir, ['a', 'b', 'c'], 100)
        with self.assertRaises(Exception) as e:
            _ = input_data.AudioProcessor('', tmp_dir, 10, 10, ['a', 'b', 'd'], 10, 10, self._model_settings(), tmp_dir)
        self.assertIn('Expected to find', str(e.exception))

    @test_util.run_deprecated_v1
    def testPrepareBackgroundData(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_temp_dir()
        background_dir = os.path.join(tmp_dir, '_background_noise_')
        os.mkdir(background_dir)
        wav_data = self._getWavData()
        for i in range(10):
            file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
            self._saveTestWavFile(file_path, wav_data)
        self._saveWavFolders(tmp_dir, ['a', 'b', 'c'], 100)
        audio_processor = input_data.AudioProcessor('', tmp_dir, 10, 10, ['a', 'b'], 10, 10, self._model_settings(), tmp_dir)
        self.assertEqual(10, len(audio_processor.background_data))

    def testLoadWavFile(self):
        if False:
            while True:
                i = 10
        tmp_dir = self.get_temp_dir()
        file_path = os.path.join(tmp_dir, 'load_test.wav')
        wav_data = self._getWavData()
        self._saveTestWavFile(file_path, wav_data)
        sample_data = input_data.load_wav_file(file_path)
        self.assertIsNotNone(sample_data)

    def testSaveWavFile(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_temp_dir()
        file_path = os.path.join(tmp_dir, 'load_test.wav')
        save_data = np.zeros([16000, 1])
        input_data.save_wav_file(file_path, save_data, 16000)
        loaded_data = input_data.load_wav_file(file_path)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(16000, len(loaded_data))

    @test_util.run_deprecated_v1
    def testPrepareProcessingGraph(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, 'wavs')
        os.mkdir(wav_dir)
        self._saveWavFolders(wav_dir, ['a', 'b', 'c'], 100)
        background_dir = os.path.join(wav_dir, '_background_noise_')
        os.mkdir(background_dir)
        wav_data = self._getWavData()
        for i in range(10):
            file_path = os.path.join(background_dir, 'background_audio_%d.wav' % i)
            self._saveTestWavFile(file_path, wav_data)
        model_settings = {'desired_samples': 160, 'fingerprint_size': 40, 'label_count': 4, 'window_size_samples': 100, 'window_stride_samples': 100, 'fingerprint_width': 40, 'preprocess': 'mfcc'}
        audio_processor = input_data.AudioProcessor('', wav_dir, 10, 10, ['a', 'b'], 10, 10, model_settings, tmp_dir)
        self.assertIsNotNone(audio_processor.wav_filename_placeholder_)
        self.assertIsNotNone(audio_processor.foreground_volume_placeholder_)
        self.assertIsNotNone(audio_processor.time_shift_padding_placeholder_)
        self.assertIsNotNone(audio_processor.time_shift_offset_placeholder_)
        self.assertIsNotNone(audio_processor.background_data_placeholder_)
        self.assertIsNotNone(audio_processor.background_volume_placeholder_)
        self.assertIsNotNone(audio_processor.output_)

    @test_util.run_deprecated_v1
    def testGetDataAverage(self):
        if False:
            while True:
                i = 10
        self._runGetDataTest('average', 10)

    @test_util.run_deprecated_v1
    def testGetDataAverageLongWindow(self):
        if False:
            i = 10
            return i + 15
        self._runGetDataTest('average', 30)

    @test_util.run_deprecated_v1
    def testGetDataMfcc(self):
        if False:
            for i in range(10):
                print('nop')
        self._runGetDataTest('mfcc', 30)

    @test_util.run_deprecated_v1
    def testGetDataMicro(self):
        if False:
            print('Hello World!')
        self._runGetDataTest('micro', 20)

    @test_util.run_deprecated_v1
    def testGetUnprocessedData(self):
        if False:
            return 10
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, 'wavs')
        os.mkdir(wav_dir)
        self._saveWavFolders(wav_dir, ['a', 'b', 'c'], 100)
        model_settings = {'desired_samples': 160, 'fingerprint_size': 40, 'label_count': 4, 'window_size_samples': 100, 'window_stride_samples': 100, 'fingerprint_width': 40, 'preprocess': 'mfcc'}
        audio_processor = input_data.AudioProcessor('', wav_dir, 10, 10, ['a', 'b'], 10, 10, model_settings, tmp_dir)
        (result_data, result_labels) = audio_processor.get_unprocessed_data(10, model_settings, 'training')
        self.assertEqual(10, len(result_data))
        self.assertEqual(10, len(result_labels))

    @test_util.run_deprecated_v1
    def testGetFeaturesForWav(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.get_temp_dir()
        wav_dir = os.path.join(tmp_dir, 'wavs')
        os.mkdir(wav_dir)
        self._saveWavFolders(wav_dir, ['a', 'b', 'c'], 1)
        desired_samples = 1600
        model_settings = {'desired_samples': desired_samples, 'fingerprint_size': 40, 'label_count': 4, 'window_size_samples': 100, 'window_stride_samples': 100, 'fingerprint_width': 40, 'average_window_width': 6, 'preprocess': 'average'}
        with self.cached_session() as sess:
            audio_processor = input_data.AudioProcessor('', wav_dir, 10, 10, ['a', 'b'], 10, 10, model_settings, tmp_dir)
            sample_data = np.zeros([desired_samples, 1])
            for i in range(desired_samples):
                phase = i % 4
                if phase == 0:
                    sample_data[i, 0] = 0
                elif phase == 1:
                    sample_data[i, 0] = -1
                elif phase == 2:
                    sample_data[i, 0] = 0
                elif phase == 3:
                    sample_data[i, 0] = 1
            test_wav_path = os.path.join(tmp_dir, 'test_wav.wav')
            input_data.save_wav_file(test_wav_path, sample_data, 16000)
            results = audio_processor.get_features_for_wav(test_wav_path, model_settings, sess)
            spectrogram = results[0]
            self.assertEqual(1, spectrogram.shape[0])
            self.assertEqual(16, spectrogram.shape[1])
            self.assertEqual(11, spectrogram.shape[2])
            self.assertNear(0, spectrogram[0, 0, 0], 0.1)
            self.assertNear(200, spectrogram[0, 0, 5], 0.1)

    def testGetFeaturesRange(self):
        if False:
            i = 10
            return i + 15
        model_settings = {'preprocess': 'average'}
        (features_min, _) = input_data.get_features_range(model_settings)
        self.assertNear(0.0, features_min, 1e-05)

    def testGetMfccFeaturesRange(self):
        if False:
            print('Hello World!')
        model_settings = {'preprocess': 'mfcc'}
        (features_min, features_max) = input_data.get_features_range(model_settings)
        self.assertLess(features_min, features_max)
if __name__ == '__main__':
    test.main()