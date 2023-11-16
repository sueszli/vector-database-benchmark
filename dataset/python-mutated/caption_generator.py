"""Class for generating captions from an image-to-text model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import heapq
import math
import numpy as np

class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        if False:
            i = 10
            return i + 15
        "Initializes the Caption.\n\n    Args:\n      sentence: List of word ids in the caption.\n      state: Model state after generating the previous word.\n      logprob: Log-probability of the caption.\n      score: Score of the caption.\n      metadata: Optional metadata associated with the partial sentence. If not\n        None, a list of strings with the same length as 'sentence'.\n    "
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Compares Captions by score.'
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        assert isinstance(other, Caption)
        return self.score < other.score

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        assert isinstance(other, Caption)
        return self.score == other.score

class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self._n = n
        self._data = []

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Pushes a new element.'
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        if False:
            for i in range(10):
                print('nop')
        'Extracts all elements from the TopN. This is a destructive operation.\n\n    The only method that can be called immediately after extract() is reset().\n\n    Args:\n      sort: Whether to return the elements in descending sorted order.\n\n    Returns:\n      A list of data; the top n elements provided to the set.\n    '
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        if False:
            while True:
                i = 10
        'Returns the TopN to an empty state.'
        self._data = []

class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self, model, vocab, beam_size=3, max_caption_length=20, length_normalization_factor=0.0):
        if False:
            while True:
                i = 10
        'Initializes the generator.\n\n    Args:\n      model: Object encapsulating a trained image-to-text model. Must have\n        methods feed_image() and inference_step(). For example, an instance of\n        InferenceWrapperBase.\n      vocab: A Vocabulary object.\n      beam_size: Beam size to use when generating captions.\n      max_caption_length: The maximum caption length before stopping the search.\n      length_normalization_factor: If != 0, a number x such that captions are\n        scored by logprob/length^x, rather than logprob. This changes the\n        relative scores of captions depending on their lengths. For example, if\n        x > 0 then longer captions will be favored.\n    '
        self.vocab = vocab
        self.model = model
        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, sess, encoded_image):
        if False:
            while True:
                i = 10
        'Runs beam search caption generation on a single image.\n\n    Args:\n      sess: TensorFlow Session object.\n      encoded_image: An encoded image string.\n\n    Returns:\n      A list of Caption sorted by descending score.\n    '
        initial_state = self.model.feed_image(sess, encoded_image)
        initial_beam = Caption(sentence=[self.vocab.start_id], state=initial_state[0], logprob=0.0, score=0.0, metadata=[''])
        partial_captions = TopN(self.beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(self.beam_size)
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])
            (softmax, new_states, metadata) = self.model.inference_step(sess, input_feed, state_feed)
            for (i, partial_caption) in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                most_likely_words = np.argsort(word_probabilities)[:-self.beam_size][::-1]
                for w in most_likely_words:
                    p = word_probabilities[w]
                    if p < 1e-12:
                        continue
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + [metadata[i]]
                    else:
                        metadata_list = None
                    if w == self.vocab.end_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                break
        if not complete_captions.size():
            complete_captions = partial_captions
        return complete_captions.extract(sort=True)