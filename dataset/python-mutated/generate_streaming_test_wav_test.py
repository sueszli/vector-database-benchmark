"""Tests for test file generation for speech commands."""
import numpy as np
from tensorflow.examples.speech_commands import generate_streaming_test_wav
from tensorflow.python.platform import test

class GenerateStreamingTestWavTest(test.TestCase):

    def testMixInAudioSample(self):
        if False:
            for i in range(10):
                print('nop')
        track_data = np.zeros([10000])
        sample_data = np.ones([1000])
        generate_streaming_test_wav.mix_in_audio_sample(track_data, 2000, sample_data, 0, 1000, 1.0, 100, 100)
        self.assertNear(1.0, track_data[2500], 0.0001)
        self.assertNear(0.0, track_data[3500], 0.0001)
if __name__ == '__main__':
    test.main()