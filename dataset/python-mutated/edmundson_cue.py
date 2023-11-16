from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from ._summarizer import AbstractSummarizer

class EdmundsonCueMethod(AbstractSummarizer):

    def __init__(self, stemmer, bonus_words, stigma_words):
        if False:
            print('Hello World!')
        super(EdmundsonCueMethod, self).__init__(stemmer)
        self._bonus_words = bonus_words
        self._stigma_words = stigma_words

    def __call__(self, document, sentences_count, bonus_word_weight, stigma_word_weight):
        if False:
            return 10
        return self._get_best_sentences(document.sentences, sentences_count, self._rate_sentence, bonus_word_weight, stigma_word_weight)

    def _rate_sentence(self, sentence, bonus_word_weight, stigma_word_weight):
        if False:
            print('Hello World!')
        words = map(self.stem_word, sentence.words)
        (bonus_words_count, stigma_words_count) = self._count_words(words)
        bonus_rating = bonus_words_count * bonus_word_weight
        stigma_rating = stigma_words_count * stigma_word_weight
        return bonus_rating - stigma_rating

    def _count_words(self, words):
        if False:
            i = 10
            return i + 15
        '\n        Counts number of bonus/stigma words.\n\n        :param iterable words:\n            Collection of words.\n        :returns pair:\n            Tuple with number of words (bonus words, stigma words).\n        '
        bonus_words_count = 0
        stigma_words_count = 0
        for word in words:
            if word in self._bonus_words:
                bonus_words_count += 1
            if word in self._stigma_words:
                stigma_words_count += 1
        return (bonus_words_count, stigma_words_count)

    def rate_sentences(self, document, bonus_word_weight=1, stigma_word_weight=1):
        if False:
            i = 10
            return i + 15
        return {sentence: self._rate_sentence(sentence, bonus_word_weight, stigma_word_weight) for sentence in document.sentences}