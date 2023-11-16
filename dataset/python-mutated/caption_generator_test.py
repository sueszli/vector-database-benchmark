"""Unit tests for CaptionGenerator."""
import math
import numpy as np
import tensorflow as tf
from im2txt.inference_utils import caption_generator

class FakeVocab(object):
    """Fake Vocabulary for testing purposes."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.start_id = 0
        self.end_id = 1

class FakeModel(object):
    """Fake model for testing purposes."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._vocab_size = 12
        self._state_size = 1
        self._probabilities = {0: {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}, 2: {5: 0.1, 6: 0.9}, 3: {1: 0.1, 7: 0.4, 8: 0.5}, 4: {1: 0.3, 9: 0.3, 10: 0.4}, 5: {1: 1.0}, 6: {1: 1.0}, 7: {1: 1.0}, 8: {1: 1.0}, 9: {1: 0.5, 11: 0.5}, 10: {1: 1.0}, 11: {1: 1.0}}

    def feed_image(self, sess, encoded_image):
        if False:
            while True:
                i = 10
        return np.zeros([1, self._state_size])

    def inference_step(self, sess, input_feed, state_feed):
        if False:
            print('Hello World!')
        batch_size = input_feed.shape[0]
        softmax_output = np.zeros([batch_size, self._vocab_size])
        for (batch_index, word_id) in enumerate(input_feed):
            for (next_word, probability) in self._probabilities[word_id].items():
                softmax_output[batch_index, next_word] = probability
        new_state = np.zeros([batch_size, self._state_size])
        metadata = None
        return (softmax_output, new_state, metadata)

class CaptionGeneratorTest(tf.test.TestCase):

    def _assertExpectedCaptions(self, expected_captions, beam_size=3, max_caption_length=20, length_normalization_factor=0):
        if False:
            for i in range(10):
                print('nop')
        'Tests that beam search generates the expected captions.\n\n    Args:\n      expected_captions: A sequence of pairs (sentence, probability), where\n        sentence is a list of integer ids and probability is a float in [0, 1].\n      beam_size: Parameter passed to beam_search().\n      max_caption_length: Parameter passed to beam_search().\n      length_normalization_factor: Parameter passed to beam_search().\n    '
        expected_sentences = [c[0] for c in expected_captions]
        expected_probabilities = [c[1] for c in expected_captions]
        generator = caption_generator.CaptionGenerator(model=FakeModel(), vocab=FakeVocab(), beam_size=beam_size, max_caption_length=max_caption_length, length_normalization_factor=length_normalization_factor)
        actual_captions = generator.beam_search(sess=None, encoded_image=None)
        actual_sentences = [c.sentence for c in actual_captions]
        actual_probabilities = [math.exp(c.logprob) for c in actual_captions]
        self.assertEqual(expected_sentences, actual_sentences)
        self.assertAllClose(expected_probabilities, actual_probabilities)

    def testBeamSize(self):
        if False:
            while True:
                i = 10
        expected = [([0, 4, 10, 1], 0.16)]
        self._assertExpectedCaptions(expected, beam_size=1)
        expected = [([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)]
        self._assertExpectedCaptions(expected, beam_size=2)
        expected = [([0, 2, 6, 1], 0.18), ([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)]
        self._assertExpectedCaptions(expected, beam_size=3)

    def testMaxLength(self):
        if False:
            for i in range(10):
                print('nop')
        expected = [([0], 1.0)]
        self._assertExpectedCaptions(expected, max_caption_length=1)
        expected = [([0, 4], 0.4), ([0, 3], 0.3), ([0, 2], 0.2)]
        self._assertExpectedCaptions(expected, max_caption_length=2)
        expected = [([0, 4, 1], 0.12), ([0, 3, 1], 0.03)]
        self._assertExpectedCaptions(expected, max_caption_length=3)
        expected = [([0, 2, 6, 1], 0.18), ([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)]
        self._assertExpectedCaptions(expected, max_caption_length=4)

    def testLengthNormalization(self):
        if False:
            for i in range(10):
                print('nop')
        expected = [([0, 4, 9, 11, 1], 0.06), ([0, 2, 6, 1], 0.18), ([0, 4, 10, 1], 0.16), ([0, 3, 8, 1], 0.15)]
        self._assertExpectedCaptions(expected, beam_size=4, length_normalization_factor=3)
if __name__ == '__main__':
    tf.test.main()