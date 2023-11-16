"""Converts WAV audio files into input features for neural networks.

The models used in this example take in two-dimensional spectrograms as the
input to their neural network portions. For testing and porting purposes it's
useful to be able to generate these spectrograms outside of the full model, so
that on-device implementations using their own FFT and streaming code can be
tested against the version used in training for example. The output is as a
C source file, so it can be easily linked into an embedded test application.

To use this, run:

bazel run tensorflow/examples/speech_commands:wav_to_features -- \\
--input_wav=my.wav --output_c_file=my_wav_data.c

"""
import argparse
import os.path
import sys
import tensorflow as tf
import input_data
import models
from tensorflow.python.platform import gfile
FLAGS = None

def wav_to_features(sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, feature_bin_count, quantize, preprocess, input_wav, output_c_file):
    if False:
        return 10
    'Converts an audio file into its corresponding feature map.\n\n  Args:\n    sample_rate: Expected sample rate of the wavs.\n    clip_duration_ms: Expected duration in milliseconds of the wavs.\n    window_size_ms: How long each spectrogram timeslice is.\n    window_stride_ms: How far to move in time between spectrogram timeslices.\n    feature_bin_count: How many bins to use for the feature fingerprint.\n    quantize: Whether to train the model for eight-bit deployment.\n    preprocess: Spectrogram processing mode; "mfcc", "average" or "micro".\n    input_wav: Path to the audio WAV file to read.\n    output_c_file: Where to save the generated C source file.\n  '
    sess = tf.compat.v1.InteractiveSession()
    model_settings = models.prepare_model_settings(0, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, feature_bin_count, preprocess)
    audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0, model_settings, None)
    results = audio_processor.get_features_for_wav(input_wav, model_settings, sess)
    features = results[0]
    variable_base = os.path.splitext(os.path.basename(input_wav).lower())[0]
    with gfile.GFile(output_c_file, 'w') as f:
        f.write('/* File automatically created by\n')
        f.write(' * tensorflow/examples/speech_commands/wav_to_features.py \\\n')
        f.write(' * --sample_rate=%d \\\n' % sample_rate)
        f.write(' * --clip_duration_ms=%d \\\n' % clip_duration_ms)
        f.write(' * --window_size_ms=%d \\\n' % window_size_ms)
        f.write(' * --window_stride_ms=%d \\\n' % window_stride_ms)
        f.write(' * --feature_bin_count=%d \\\n' % feature_bin_count)
        if quantize:
            f.write(' * --quantize=1 \\\n')
        f.write(' * --preprocess="%s" \\\n' % preprocess)
        f.write(' * --input_wav="%s" \\\n' % input_wav)
        f.write(' * --output_c_file="%s" \\\n' % output_c_file)
        f.write(' */\n\n')
        f.write('const int g_%s_width = %d;\n' % (variable_base, model_settings['fingerprint_width']))
        f.write('const int g_%s_height = %d;\n' % (variable_base, model_settings['spectrogram_length']))
        if quantize:
            (features_min, features_max) = input_data.get_features_range(model_settings)
            f.write('const unsigned char g_%s_data[] = {' % variable_base)
            i = 0
            for value in features.flatten():
                quantized_value = int(round(255 * (value - features_min) / (features_max - features_min)))
                if quantized_value < 0:
                    quantized_value = 0
                if quantized_value > 255:
                    quantized_value = 255
                if i == 0:
                    f.write('\n  ')
                f.write('%d, ' % quantized_value)
                i = (i + 1) % 10
        else:
            f.write('const float g_%s_data[] = {\n' % variable_base)
            i = 0
            for value in features.flatten():
                if i == 0:
                    f.write('\n  ')
                f.write('%f, ' % value)
                i = (i + 1) % 10
        f.write('\n};\n')

def main(_):
    if False:
        return 10
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    wav_to_features(FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms, FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.quantize, FLAGS.preprocess, FLAGS.input_wav, FLAGS.output_c_file)
    tf.compat.v1.logging.info('Wrote to "%s"' % FLAGS.output_c_file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=int, default=16000, help='Expected sample rate of the wavs')
    parser.add_argument('--clip_duration_ms', type=int, default=1000, help='Expected duration in milliseconds of the wavs')
    parser.add_argument('--window_size_ms', type=float, default=30.0, help='How long each spectrogram timeslice is.')
    parser.add_argument('--window_stride_ms', type=float, default=10.0, help='How far to move in time between spectrogram timeslices.')
    parser.add_argument('--feature_bin_count', type=int, default=40, help='How many bins to use for the MFCC fingerprint')
    parser.add_argument('--quantize', type=bool, default=False, help='Whether to train the model for eight-bit deployment')
    parser.add_argument('--preprocess', type=str, default='mfcc', help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')
    parser.add_argument('--input_wav', type=str, default=None, help='Path to the audio WAV file to read')
    parser.add_argument('--output_c_file', type=str, default=None, help='Where to save the generated C source file containing the features')
    (FLAGS, unparsed) = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)