"""Preprocesses TIMIT from raw wavfiles to create a set of TFRecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import random
import re
import numpy as np
import tensorflow as tf
tf.app.flags.DEFINE_string('raw_timit_dir', None, 'Directory containing TIMIT files.')
tf.app.flags.DEFINE_string('out_dir', None, 'Output directory for TFRecord files.')
tf.app.flags.DEFINE_float('valid_frac', 0.05, 'Fraction of train set to use as valid set. Must be between 0.0 and 1.0.')
tf.app.flags.mark_flag_as_required('raw_timit_dir')
tf.app.flags.mark_flag_as_required('out_dir')
FLAGS = tf.app.flags.FLAGS
NUM_TRAIN_FILES = 4620
NUM_TEST_FILES = 1680
SAMPLES_PER_TIMESTEP = 200
SAMPLE_COUNT_REGEX = re.compile('sample_count -i (\\d+)')
SAMPLE_MIN_REGEX = re.compile('sample_min -i (-?\\d+)')
SAMPLE_MAX_REGEX = re.compile('sample_max -i (-?\\d+)')

def get_filenames(split):
    if False:
        print('Hello World!')
    'Get all wav filenames from the TIMIT archive.'
    path = os.path.join(FLAGS.raw_timit_dir, 'TIMIT', split, '*', '*', '*.WAV')
    files = sorted(glob.glob(path))
    return files

def load_timit_wav(filename):
    if False:
        i = 10
        return i + 15
    'Loads a TIMIT wavfile into a numpy array.\n\n  TIMIT wavfiles include a SPHERE header, detailed in the TIMIT docs. The first\n  line is the header type and the second is the length of the header in bytes.\n  After the header, the remaining bytes are actual WAV data.\n\n  The header includes information about the WAV data such as the number of\n  samples and minimum and maximum amplitude. This function asserts that the\n  loaded wav data matches the header.\n\n  Args:\n    filename: The name of the TIMIT wavfile to load.\n  Returns:\n    wav: A numpy array containing the loaded wav data.\n  '
    wav_file = open(filename, 'rb')
    header_type = wav_file.readline()
    header_length_str = wav_file.readline()
    header_remaining_bytes = int(header_length_str) - len(header_type) - len(header_length_str)
    header = wav_file.read(header_remaining_bytes)
    sample_count = int(SAMPLE_COUNT_REGEX.search(header).group(1))
    sample_min = int(SAMPLE_MIN_REGEX.search(header).group(1))
    sample_max = int(SAMPLE_MAX_REGEX.search(header).group(1))
    wav = np.fromstring(wav_file.read(), dtype='int16').astype('float32')
    assert len(wav) == sample_count
    assert wav.min() == sample_min
    assert wav.max() == sample_max
    return wav

def preprocess(wavs, block_size, mean, std):
    if False:
        for i in range(10):
            print('nop')
    'Normalize the wav data and reshape it into chunks.'
    processed_wavs = []
    for wav in wavs:
        wav = (wav - mean) / std
        wav_length = wav.shape[0]
        if wav_length % block_size != 0:
            pad_width = block_size - wav_length % block_size
            wav = np.pad(wav, (0, pad_width), 'constant')
        assert wav.shape[0] % block_size == 0
        wav = wav.reshape((-1, block_size))
        processed_wavs.append(wav)
    return processed_wavs

def create_tfrecord_from_wavs(wavs, output_file):
    if False:
        while True:
            i = 10
    'Writes processed wav files to disk as sharded TFRecord files.'
    with tf.python_io.TFRecordWriter(output_file) as builder:
        for wav in wavs:
            builder.write(wav.astype(np.float32).tobytes())

def main(unused_argv):
    if False:
        while True:
            i = 10
    train_filenames = get_filenames('TRAIN')
    test_filenames = get_filenames('TEST')
    num_train_files = len(train_filenames)
    num_test_files = len(test_filenames)
    num_valid_files = int(num_train_files * FLAGS.valid_frac)
    num_train_files -= num_valid_files
    print('%d train / %d valid / %d test' % (num_train_files, num_valid_files, num_test_files))
    random.seed(1234)
    random.shuffle(train_filenames)
    valid_filenames = train_filenames[:num_valid_files]
    train_filenames = train_filenames[num_valid_files:]
    train_s = set(train_filenames)
    test_s = set(test_filenames)
    valid_s = set(valid_filenames)
    assert len(train_s & test_s) == 0
    assert len(train_s & valid_s) == 0
    assert len(valid_s & test_s) == 0
    train_wavs = [load_timit_wav(f) for f in train_filenames]
    valid_wavs = [load_timit_wav(f) for f in valid_filenames]
    test_wavs = [load_timit_wav(f) for f in test_filenames]
    assert len(train_wavs) + len(valid_wavs) == NUM_TRAIN_FILES
    assert len(test_wavs) == NUM_TEST_FILES
    train_stacked = np.hstack(train_wavs)
    train_mean = np.mean(train_stacked)
    train_std = np.std(train_stacked)
    print('train mean: %f  train std: %f' % (train_mean, train_std))
    processed_train_wavs = preprocess(train_wavs, SAMPLES_PER_TIMESTEP, train_mean, train_std)
    processed_valid_wavs = preprocess(valid_wavs, SAMPLES_PER_TIMESTEP, train_mean, train_std)
    processed_test_wavs = preprocess(test_wavs, SAMPLES_PER_TIMESTEP, train_mean, train_std)
    create_tfrecord_from_wavs(processed_train_wavs, os.path.join(FLAGS.out_dir, 'train'))
    create_tfrecord_from_wavs(processed_valid_wavs, os.path.join(FLAGS.out_dir, 'valid'))
    create_tfrecord_from_wavs(processed_test_wavs, os.path.join(FLAGS.out_dir, 'test'))
if __name__ == '__main__':
    tf.app.run()