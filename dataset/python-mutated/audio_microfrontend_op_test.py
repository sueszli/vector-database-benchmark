"""Tests for AudioMicrofrontend."""
import tensorflow as tf
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow.python.framework import ops
SAMPLE_RATE = 1000
WINDOW_SIZE = 25
WINDOW_STEP = 10
NUM_CHANNELS = 2
UPPER_BAND_LIMIT = 450.0
LOWER_BAND_LIMIT = 8.0
SMOOTHING_BITS = 10

class AudioFeatureGenerationTest(tf.test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(AudioFeatureGenerationTest, self).setUp()
        ops.disable_eager_execution()

    def testSimple(self):
        if False:
            i = 10
            return i + 15
        with self.test_session():
            audio = tf.constant([0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4), tf.int16)
            filterbanks = frontend_op.audio_microfrontend(audio, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, num_channels=NUM_CHANNELS, upper_band_limit=UPPER_BAND_LIMIT, lower_band_limit=LOWER_BAND_LIMIT, smoothing_bits=SMOOTHING_BITS, enable_pcan=True)
            self.assertAllEqual(self.evaluate(filterbanks), [[479, 425], [436, 378], [410, 350], [391, 325]])

    def testSimpleFloatScaled(self):
        if False:
            while True:
                i = 10
        with self.test_session():
            audio = tf.constant([0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4), tf.int16)
            filterbanks = frontend_op.audio_microfrontend(audio, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, num_channels=NUM_CHANNELS, upper_band_limit=UPPER_BAND_LIMIT, lower_band_limit=LOWER_BAND_LIMIT, smoothing_bits=SMOOTHING_BITS, enable_pcan=True, out_scale=64, out_type=tf.float32)
            self.assertAllEqual(self.evaluate(filterbanks), [[7.484375, 6.640625], [6.8125, 5.90625], [6.40625, 5.46875], [6.109375, 5.078125]])

    def testStacking(self):
        if False:
            while True:
                i = 10
        with self.test_session():
            audio = tf.constant([0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4), tf.int16)
            filterbanks = frontend_op.audio_microfrontend(audio, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, num_channels=NUM_CHANNELS, upper_band_limit=UPPER_BAND_LIMIT, lower_band_limit=LOWER_BAND_LIMIT, smoothing_bits=SMOOTHING_BITS, enable_pcan=True, right_context=1, frame_stride=2)
            self.assertAllEqual(self.evaluate(filterbanks), [[479, 425, 436, 378], [410, 350, 391, 325]])

    def testStackingWithOverlap(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session():
            audio = tf.constant([0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4), tf.int16)
            filterbanks = frontend_op.audio_microfrontend(audio, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, num_channels=NUM_CHANNELS, upper_band_limit=UPPER_BAND_LIMIT, lower_band_limit=LOWER_BAND_LIMIT, smoothing_bits=SMOOTHING_BITS, enable_pcan=True, left_context=1, right_context=1)
            self.assertAllEqual(self.evaluate(filterbanks), [[479, 425, 479, 425, 436, 378], [479, 425, 436, 378, 410, 350], [436, 378, 410, 350, 391, 325], [410, 350, 391, 325, 391, 325]])

    def testStackingDropFrame(self):
        if False:
            while True:
                i = 10
        with self.test_session():
            audio = tf.constant([0, 32767, 0, -32768] * ((WINDOW_SIZE + 4 * WINDOW_STEP) // 4), tf.int16)
            filterbanks = frontend_op.audio_microfrontend(audio, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, num_channels=NUM_CHANNELS, upper_band_limit=UPPER_BAND_LIMIT, lower_band_limit=LOWER_BAND_LIMIT, smoothing_bits=SMOOTHING_BITS, enable_pcan=True, left_context=1, frame_stride=2)
            self.assertAllEqual(self.evaluate(filterbanks), [[479, 425, 479, 425], [436, 378, 410, 350]])

    def testZeroPadding(self):
        if False:
            return 10
        with self.test_session():
            audio = tf.constant([0, 32767, 0, -32768] * ((WINDOW_SIZE + 7 * WINDOW_STEP) // 4), tf.int16)
            filterbanks = frontend_op.audio_microfrontend(audio, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, num_channels=NUM_CHANNELS, upper_band_limit=UPPER_BAND_LIMIT, lower_band_limit=LOWER_BAND_LIMIT, smoothing_bits=SMOOTHING_BITS, enable_pcan=True, left_context=2, frame_stride=3, zero_padding=True)
            self.assertAllEqual(self.evaluate(filterbanks), [[0, 0, 0, 0, 479, 425], [436, 378, 410, 350, 391, 325], [374, 308, 362, 292, 352, 275]])
if __name__ == '__main__':
    tf.test.main()