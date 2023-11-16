"""Model definitions for simple speech recognition.

"""
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import numpy as np
import urllib
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
tf.compat.v1.disable_eager_execution()
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
except ImportError:
    frontend_op = None
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185

def prepare_words_list(wanted_words):
    if False:
        return 10
    'Prepends common tokens to the custom word list.\n\n  Args:\n    wanted_words: List of strings containing the custom words.\n\n  Returns:\n    List with the standard silence and unknown tokens added.\n  '
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words

def which_set(filename, validation_percentage, testing_percentage):
    if False:
        for i in range(10):
            print('nop')
    "Determines which data partition the file should belong to.\n\n  We want to keep files in the same training, validation, or testing sets even\n  if new ones are added over time. This makes it less likely that testing\n  samples will accidentally be reused in training when long runs are restarted\n  for example. To keep this stability, a hash of the filename is taken and used\n  to determine which set it should belong to. This determination only depends on\n  the name and the set proportions, so it won't change as other files are added.\n\n  It's also useful to associate particular files as related (for example words\n  spoken by the same person), so anything after '_nohash_' in a filename is\n  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and\n  'bobby_nohash_1.wav' are always in the same set, for example.\n\n  Args:\n    filename: File path of the data sample.\n    validation_percentage: How much of the data set to use for validation.\n    testing_percentage: How much of the data set to use for testing.\n\n  Returns:\n    String, one of 'training', 'validation', or 'testing'.\n  "
    base_name = os.path.basename(filename)
    hash_name = re.sub('_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1) * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < testing_percentage + validation_percentage:
        result = 'testing'
    else:
        result = 'training'
    return result

def load_wav_file(filename):
    if False:
        print('Hello World!')
    'Loads an audio file and returns a float PCM-encoded array of samples.\n\n  Args:\n    filename: Path to the .wav file to load.\n\n  Returns:\n    Numpy array holding the sample data as floats between -1.0 and 1.0.\n  '
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(wav_decoder, feed_dict={wav_filename_placeholder: filename}).audio.flatten()

def save_wav_file(filename, wav_data, sample_rate):
    if False:
        while True:
            i = 10
    'Saves audio sample data to a .wav audio file.\n\n  Args:\n    filename: Path to save the file to.\n    wav_data: 2D array of float PCM-encoded audio data.\n    sample_rate: Samples per second to encode in the file.\n  '
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
        sample_rate_placeholder = tf.compat.v1.placeholder(tf.int32, [])
        wav_data_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 1])
        wav_encoder = tf.audio.encode_wav(wav_data_placeholder, sample_rate_placeholder)
        wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
        sess.run(wav_saver, feed_dict={wav_filename_placeholder: filename, sample_rate_placeholder: sample_rate, wav_data_placeholder: np.reshape(wav_data, (-1, 1))})

def get_features_range(model_settings):
    if False:
        for i in range(10):
            print('nop')
    "Returns the expected min/max for generated features.\n\n  Args:\n    model_settings: Information about the current model being trained.\n\n  Returns:\n    Min/max float pair holding the range of features.\n\n  Raises:\n    Exception: If preprocessing mode isn't recognized.\n  "
    if model_settings['preprocess'] == 'average':
        features_min = 0.0
        features_max = 127.5
    elif model_settings['preprocess'] == 'mfcc':
        features_min = -247.0
        features_max = 30.0
    elif model_settings['preprocess'] == 'micro':
        features_min = 0.0
        features_max = 26.0
    else:
        raise Exception('Unknown preprocess mode "%s" (should be "mfcc", "average", or "micro")' % model_settings['preprocess'])
    return (features_min, features_max)

class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage, wanted_words, validation_percentage, testing_percentage, model_settings, summaries_dir):
        if False:
            print('Hello World!')
        if data_dir:
            self.data_dir = data_dir
            self.maybe_download_and_extract_dataset(data_url, data_dir)
            self.prepare_data_index(silence_percentage, unknown_percentage, wanted_words, validation_percentage, testing_percentage)
            self.prepare_background_data()
        self.prepare_processing_graph(model_settings, summaries_dir)

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        if False:
            for i in range(10):
                print('nop')
        "Download and extract data set tar file.\n\n    If the data set we're using doesn't already exist, this function\n    downloads it from the TensorFlow.org website and unpacks it into a\n    directory.\n    If the data_url is none, don't download anything and expect the data\n    directory to contain the correct files already.\n\n    Args:\n      data_url: Web location of the tar file containing the data set.\n      dest_directory: File path to extract data to.\n    "
        if not data_url:
            return
        if not gfile.Exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not gfile.Exists(filepath):

            def _progress(count, block_size, total_size):
                if False:
                    while True:
                        i = 10
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            try:
                (filepath, _) = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                tf.compat.v1.logging.error('Failed to download URL: {0} to folder: {1}. Please make sure you have enough free space and an internet connection'.format(data_url, filepath))
                raise
            print()
            statinfo = os.stat(filepath)
            tf.compat.v1.logging.info('Successfully downloaded {0} ({1} bytes)'.format(filename, statinfo.st_size))
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def prepare_data_index(self, silence_percentage, unknown_percentage, wanted_words, validation_percentage, testing_percentage):
        if False:
            return 10
        'Prepares a list of the samples organized by set and label.\n\n    The training loop needs a list of all the available data, organized by\n    which partition it should belong to, and with ground truth labels attached.\n    This function analyzes the folders below the `data_dir`, figures out the\n    right\n    labels for each file based on the name of the subdirectory it belongs to,\n    and uses a stable hash to assign it to a data set partition.\n\n    Args:\n      silence_percentage: How much of the resulting data should be background.\n      unknown_percentage: How much should be audio outside the wanted classes.\n      wanted_words: Labels of the classes we want to be able to recognize.\n      validation_percentage: How much of the data set to use for validation.\n      testing_percentage: How much of the data set to use for testing.\n\n    Returns:\n      Dictionary containing a list of file information for each set partition,\n      and a lookup map for each class to determine its numeric index.\n\n    Raises:\n      Exception: If expected files are not found.\n    '
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for (index, wanted_word) in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            (_, word) = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for (index, wanted_word) in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word + ' in labels but only found ' + ', '.join(all_words.keys()))
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({'label': SILENCE_LABEL, 'file': silence_wav_path})
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        if False:
            print('Hello World!')
        "Searches a folder for background noise audio, and loads it into memory.\n\n    It's expected that the background audio samples will be in a subdirectory\n    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match\n    the sample rate of the training data, but can be much longer in duration.\n\n    If the '_background_noise_' folder doesn't exist at all, this isn't an\n    error, it's just taken to mean that no background noise augmentation should\n    be used. If the folder does exist, but it's empty, that's treated as an\n    error.\n\n    Returns:\n      List of raw PCM-encoded audio samples of background noise.\n\n    Raises:\n      Exception: If files aren't found in the folder.\n    "
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not gfile.Exists(background_dir):
            return self.background_data
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
            for wav_path in gfile.Glob(search_path):
                wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in ' + search_path)

    def prepare_processing_graph(self, model_settings, summaries_dir):
        if False:
            while True:
                i = 10
        "Builds a TensorFlow graph to apply the input distortions.\n\n    Creates a graph that loads a WAVE file, decodes it, scales the volume,\n    shifts it in time, adds in background noise, calculates a spectrogram, and\n    then builds an MFCC fingerprint from that.\n\n    This must be called with an active TensorFlow session running, and it\n    creates multiple placeholder inputs, and one output:\n\n      - wav_filename_placeholder_: Filename of the WAV to load.\n      - foreground_volume_placeholder_: How loud the main clip should be.\n      - time_shift_padding_placeholder_: Where to pad the clip.\n      - time_shift_offset_placeholder_: How much to move the clip in time.\n      - background_data_placeholder_: PCM sample data for background noise.\n      - background_volume_placeholder_: Loudness of mixed-in background.\n      - output_: Output 2D fingerprint of processed audio.\n\n    Args:\n      model_settings: Information about the current model being trained.\n      summaries_dir: Path to save training summary information to.\n\n    Raises:\n      ValueError: If the preprocessing mode isn't recognized.\n      Exception: If the preprocessor wasn't compiled in.\n    "
        with tf.compat.v1.get_default_graph().name_scope('data'):
            desired_samples = model_settings['desired_samples']
            self.wav_filename_placeholder_ = tf.compat.v1.placeholder(tf.string, [], name='wav_filename')
            wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
            self.foreground_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [], name='foreground_volume')
            scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)
            self.time_shift_padding_placeholder_ = tf.compat.v1.placeholder(tf.int32, [2, 2], name='time_shift_padding')
            self.time_shift_offset_placeholder_ = tf.compat.v1.placeholder(tf.int32, [2], name='time_shift_offset')
            padded_foreground = tf.pad(tensor=scaled_foreground, paddings=self.time_shift_padding_placeholder_, mode='CONSTANT')
            sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, [desired_samples, -1])
            self.background_data_placeholder_ = tf.compat.v1.placeholder(tf.float32, [desired_samples, 1], name='background_data')
            self.background_volume_placeholder_ = tf.compat.v1.placeholder(tf.float32, [], name='background_volume')
            background_mul = tf.multiply(self.background_data_placeholder_, self.background_volume_placeholder_)
            background_add = tf.add(background_mul, sliced_foreground)
            background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
            spectrogram = audio_ops.audio_spectrogram(background_clamp, window_size=model_settings['window_size_samples'], stride=model_settings['window_stride_samples'], magnitude_squared=True)
            tf.compat.v1.summary.image('spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
            if model_settings['preprocess'] == 'average':
                self.output_ = tf.nn.pool(input=tf.expand_dims(spectrogram, -1), window_shape=[1, model_settings['average_window_width']], strides=[1, model_settings['average_window_width']], pooling_type='AVG', padding='SAME')
                tf.compat.v1.summary.image('shrunk_spectrogram', self.output_, max_outputs=1)
            elif model_settings['preprocess'] == 'mfcc':
                self.output_ = audio_ops.mfcc(spectrogram, wav_decoder.sample_rate, dct_coefficient_count=model_settings['fingerprint_width'])
                tf.compat.v1.summary.image('mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
            elif model_settings['preprocess'] == 'micro':
                if not frontend_op:
                    raise Exception('Micro frontend op is currently not available when running TensorFlow directly from Python, you need to build and run through Bazel')
                sample_rate = model_settings['sample_rate']
                window_size_ms = model_settings['window_size_samples'] * 1000 / sample_rate
                window_step_ms = model_settings['window_stride_samples'] * 1000 / sample_rate
                int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
                micro_frontend = frontend_op.audio_microfrontend(int16_input, sample_rate=sample_rate, window_size=window_size_ms, window_step=window_step_ms, num_channels=model_settings['fingerprint_width'], out_scale=1, out_type=tf.float32)
                self.output_ = tf.multiply(micro_frontend, 10.0 / 256.0)
                tf.compat.v1.summary.image('micro', tf.expand_dims(tf.expand_dims(self.output_, -1), 0), max_outputs=1)
            else:
                raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",  "average", or "micro")' % model_settings['preprocess'])
            self.merged_summaries_ = tf.compat.v1.summary.merge_all(scope='data')
            if summaries_dir:
                self.summary_writer_ = tf.compat.v1.summary.FileWriter(summaries_dir + '/data', tf.compat.v1.get_default_graph())

    def set_size(self, mode):
        if False:
            while True:
                i = 10
        "Calculates the number of samples in the dataset partition.\n\n    Args:\n      mode: Which partition, must be 'training', 'validation', or 'testing'.\n\n    Returns:\n      Number of samples in the partition.\n    "
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, model_settings, background_frequency, background_volume_range, time_shift, mode, sess):
        if False:
            i = 10
            return i + 15
        "Gather samples from the data set, applying transformations as needed.\n\n    When the mode is 'training', a random selection of samples will be returned,\n    otherwise the first N clips in the partition will be used. This ensures that\n    validation always uses the same samples, reducing noise in the metrics.\n\n    Args:\n      how_many: Desired number of samples to return. -1 means the entire\n        contents of this partition.\n      offset: Where to start when fetching deterministically.\n      model_settings: Information about the current model being trained.\n      background_frequency: How many clips will have background noise, 0.0 to\n        1.0.\n      background_volume_range: How loud the background noise will be.\n      time_shift: How much to randomly shift the clips by in time.\n      mode: Which partition to use, must be 'training', 'validation', or\n        'testing'.\n      sess: TensorFlow session that was active when processor was created.\n\n    Returns:\n      List of sample data for the transformed samples, and list of label indexes\n\n    Raises:\n      ValueError: If background samples are too short.\n    "
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        data = np.zeros((sample_count, model_settings['fingerprint_size']))
        labels = np.zeros(sample_count)
        desired_samples = model_settings['desired_samples']
        use_background = self.background_data and mode == 'training'
        pick_deterministically = mode != 'training'
        for i in range(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            input_dict = {self.wav_filename_placeholder_: sample['file'], self.time_shift_padding_placeholder_: time_shift_padding, self.time_shift_offset_placeholder_: time_shift_offset}
            if use_background or sample['label'] == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= model_settings['desired_samples']:
                    raise ValueError('Background sample is too short! Need more than %d samples but only %d were found' % (model_settings['desired_samples'], len(background_samples)))
                background_offset = np.random.randint(0, len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:background_offset + desired_samples]
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                if sample['label'] == SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume
            if sample['label'] == SILENCE_LABEL:
                input_dict[self.foreground_volume_placeholder_] = 0
            else:
                input_dict[self.foreground_volume_placeholder_] = 1
            (summary, data_tensor) = sess.run([self.merged_summaries_, self.output_], feed_dict=input_dict)
            self.summary_writer_.add_summary(summary)
            data[i - offset, :] = data_tensor.flatten()
            label_index = self.word_to_index[sample['label']]
            labels[i - offset] = label_index
        return (data, labels)

    def get_features_for_wav(self, wav_filename, model_settings, sess):
        if False:
            return 10
        'Applies the feature transformation process to the input_wav.\n\n    Runs the feature generation process (generally producing a spectrogram from\n    the input samples) on the WAV file. This can be useful for testing and\n    verifying implementations being run on other platforms.\n\n    Args:\n      wav_filename: The path to the input audio file.\n      model_settings: Information about the current model being trained.\n      sess: TensorFlow session that was active when processor was created.\n\n    Returns:\n      Numpy data array containing the generated features.\n    '
        desired_samples = model_settings['desired_samples']
        input_dict = {self.wav_filename_placeholder_: wav_filename, self.time_shift_padding_placeholder_: [[0, 0], [0, 0]], self.time_shift_offset_placeholder_: [0, 0], self.background_data_placeholder_: np.zeros([desired_samples, 1]), self.background_volume_placeholder_: 0, self.foreground_volume_placeholder_: 1}
        data_tensor = sess.run([self.output_], feed_dict=input_dict)
        return data_tensor

    def get_unprocessed_data(self, how_many, model_settings, mode):
        if False:
            print('Hello World!')
        "Retrieve sample data for the given partition, with no transformations.\n\n    Args:\n      how_many: Desired number of samples to return. -1 means the entire\n        contents of this partition.\n      model_settings: Information about the current model being trained.\n      mode: Which partition to use, must be 'training', 'validation', or\n        'testing'.\n\n    Returns:\n      List of sample data for the samples, and list of labels in one-hot form.\n    "
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = how_many
        desired_samples = model_settings['desired_samples']
        words_list = self.words_list
        data = np.zeros((sample_count, desired_samples))
        labels = []
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
            foreground_volume_placeholder = tf.compat.v1.placeholder(tf.float32, [])
            scaled_foreground = tf.multiply(wav_decoder.audio, foreground_volume_placeholder)
            for i in range(sample_count):
                if how_many == -1:
                    sample_index = i
                else:
                    sample_index = np.random.randint(len(candidates))
                sample = candidates[sample_index]
                input_dict = {wav_filename_placeholder: sample['file']}
                if sample['label'] == SILENCE_LABEL:
                    input_dict[foreground_volume_placeholder] = 0
                else:
                    input_dict[foreground_volume_placeholder] = 1
                data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
                label_index = self.word_to_index[sample['label']]
                labels.append(words_list[label_index])
        return (data, labels)