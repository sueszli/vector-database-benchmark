"""Tests for summary V1 audio op."""
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.summary import summary

class SummaryV1AudioOpTest(test.TestCase):

    def _AsSummary(self, s):
        if False:
            i = 10
            return i + 15
        summ = summary_pb2.Summary()
        summ.ParseFromString(s)
        return summ

    def _CheckProto(self, audio_summ, sample_rate, num_channels, length_frames):
        if False:
            i = 10
            return i + 15
        'Verify that the non-audio parts of the audio_summ proto match shape.'
        for v in audio_summ.value:
            v.audio.ClearField('encoded_audio_string')
        expected = '\n'.join(('\n        value {\n          tag: "snd/audio/%d"\n          audio { content_type: "audio/wav" sample_rate: %d\n                  num_channels: %d length_frames: %d }\n        }' % (i, sample_rate, num_channels, length_frames) for i in range(3)))
        self.assertProtoEquals(expected, audio_summ)

    def testAudioSummary(self):
        if False:
            return 10
        np.random.seed(7)
        for channels in (1, 2, 5, 8):
            with self.session(graph=ops.Graph()) as sess:
                num_frames = 7
                shape = (4, num_frames, channels)
                const = 2.0 * np.random.random(shape) - 1.0
                sample_rate = 8000
                summ = summary.audio('snd', const, max_outputs=3, sample_rate=sample_rate)
                value = self.evaluate(summ)
                self.assertEqual([], summ.get_shape())
                audio_summ = self._AsSummary(value)
                self._CheckProto(audio_summ, sample_rate, channels, num_frames)
if __name__ == '__main__':
    test.main()