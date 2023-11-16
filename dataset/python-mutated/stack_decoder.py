"""
A decoder that uses stacks to implement phrase-based translation.

In phrase-based translation, the source sentence is segmented into
phrases of one or more words, and translations for those phrases are
used to build the target sentence.

Hypothesis data structures are used to keep track of the source words
translated so far and the partial output. A hypothesis can be expanded
by selecting an untranslated phrase, looking up its translation in a
phrase table, and appending that translation to the partial output.
Translation is complete when a hypothesis covers all source words.

The search space is huge because the source sentence can be segmented
in different ways, the source phrases can be selected in any order,
and there could be multiple translations for the same source phrase in
the phrase table. To make decoding tractable, stacks are used to limit
the number of candidate hypotheses by doing histogram and/or threshold
pruning.

Hypotheses with the same number of words translated are placed in the
same stack. In histogram pruning, each stack has a size limit, and
the hypothesis with the lowest score is removed when the stack is full.
In threshold pruning, hypotheses that score below a certain threshold
of the best hypothesis in that stack are removed.

Hypothesis scoring can include various factors such as phrase
translation probability, language model probability, length of
translation, cost of remaining words to be translated, and so on.


References:
Philipp Koehn. 2010. Statistical Machine Translation.
Cambridge University Press, New York.
"""
import warnings
from collections import defaultdict
from math import log

class StackDecoder:
    """
    Phrase-based stack decoder for machine translation

    >>> from nltk.translate import PhraseTable
    >>> phrase_table = PhraseTable()
    >>> phrase_table.add(('niemand',), ('nobody',), log(0.8))
    >>> phrase_table.add(('niemand',), ('no', 'one'), log(0.2))
    >>> phrase_table.add(('erwartet',), ('expects',), log(0.8))
    >>> phrase_table.add(('erwartet',), ('expecting',), log(0.2))
    >>> phrase_table.add(('niemand', 'erwartet'), ('one', 'does', 'not', 'expect'), log(0.1))
    >>> phrase_table.add(('die', 'spanische', 'inquisition'), ('the', 'spanish', 'inquisition'), log(0.8))
    >>> phrase_table.add(('!',), ('!',), log(0.8))

    >>> #  nltk.model should be used here once it is implemented
    >>> from collections import defaultdict
    >>> language_prob = defaultdict(lambda: -999.0)
    >>> language_prob[('nobody',)] = log(0.5)
    >>> language_prob[('expects',)] = log(0.4)
    >>> language_prob[('the', 'spanish', 'inquisition')] = log(0.2)
    >>> language_prob[('!',)] = log(0.1)
    >>> language_model = type('',(object,),{'probability_change': lambda self, context, phrase: language_prob[phrase], 'probability': lambda self, phrase: language_prob[phrase]})()

    >>> stack_decoder = StackDecoder(phrase_table, language_model)

    >>> stack_decoder.translate(['niemand', 'erwartet', 'die', 'spanische', 'inquisition', '!'])
    ['nobody', 'expects', 'the', 'spanish', 'inquisition', '!']

    """

    def __init__(self, phrase_table, language_model):
        if False:
            print('Hello World!')
        '\n        :param phrase_table: Table of translations for source language\n            phrases and the log probabilities for those translations.\n        :type phrase_table: PhraseTable\n\n        :param language_model: Target language model. Must define a\n            ``probability_change`` method that calculates the change in\n            log probability of a sentence, if a given string is appended\n            to it.\n            This interface is experimental and will likely be replaced\n            with nltk.model once it is implemented.\n        :type language_model: object\n        '
        self.phrase_table = phrase_table
        self.language_model = language_model
        self.word_penalty = 0.0
        '\n        float: Influences the translation length exponentially.\n            If positive, shorter translations are preferred.\n            If negative, longer translations are preferred.\n            If zero, no penalty is applied.\n        '
        self.beam_threshold = 0.0
        '\n        float: Hypotheses that score below this factor of the best\n            hypothesis in a stack are dropped from consideration.\n            Value between 0.0 and 1.0.\n        '
        self.stack_size = 100
        '\n        int: Maximum number of hypotheses to consider in a stack.\n            Higher values increase the likelihood of a good translation,\n            but increases processing time.\n        '
        self.__distortion_factor = 0.5
        self.__compute_log_distortion()

    @property
    def distortion_factor(self):
        if False:
            while True:
                i = 10
        '\n        float: Amount of reordering of source phrases.\n            Lower values favour monotone translation, suitable when\n            word order is similar for both source and target languages.\n            Value between 0.0 and 1.0. Default 0.5.\n        '
        return self.__distortion_factor

    @distortion_factor.setter
    def distortion_factor(self, d):
        if False:
            print('Hello World!')
        self.__distortion_factor = d
        self.__compute_log_distortion()

    def __compute_log_distortion(self):
        if False:
            return 10
        if self.__distortion_factor == 0.0:
            self.__log_distortion_factor = log(1e-09)
        else:
            self.__log_distortion_factor = log(self.__distortion_factor)

    def translate(self, src_sentence):
        if False:
            i = 10
            return i + 15
        '\n        :param src_sentence: Sentence to be translated\n        :type src_sentence: list(str)\n\n        :return: Translated sentence\n        :rtype: list(str)\n        '
        sentence = tuple(src_sentence)
        sentence_length = len(sentence)
        stacks = [_Stack(self.stack_size, self.beam_threshold) for _ in range(0, sentence_length + 1)]
        empty_hypothesis = _Hypothesis()
        stacks[0].push(empty_hypothesis)
        all_phrases = self.find_all_src_phrases(sentence)
        future_score_table = self.compute_future_scores(sentence)
        for stack in stacks:
            for hypothesis in stack:
                possible_expansions = StackDecoder.valid_phrases(all_phrases, hypothesis)
                for src_phrase_span in possible_expansions:
                    src_phrase = sentence[src_phrase_span[0]:src_phrase_span[1]]
                    for translation_option in self.phrase_table.translations_for(src_phrase):
                        raw_score = self.expansion_score(hypothesis, translation_option, src_phrase_span)
                        new_hypothesis = _Hypothesis(raw_score=raw_score, src_phrase_span=src_phrase_span, trg_phrase=translation_option.trg_phrase, previous=hypothesis)
                        new_hypothesis.future_score = self.future_score(new_hypothesis, future_score_table, sentence_length)
                        total_words = new_hypothesis.total_translated_words()
                        stacks[total_words].push(new_hypothesis)
        if not stacks[sentence_length]:
            warnings.warn('Unable to translate all words. The source sentence contains words not in the phrase table')
            return []
        best_hypothesis = stacks[sentence_length].best()
        return best_hypothesis.translation_so_far()

    def find_all_src_phrases(self, src_sentence):
        if False:
            return 10
        '\n        Finds all subsequences in src_sentence that have a phrase\n        translation in the translation table\n\n        :type src_sentence: tuple(str)\n\n        :return: Subsequences that have a phrase translation,\n            represented as a table of lists of end positions.\n            For example, if result[2] is [5, 6, 9], then there are\n            three phrases starting from position 2 in ``src_sentence``,\n            ending at positions 5, 6, and 9 exclusive. The list of\n            ending positions are in ascending order.\n        :rtype: list(list(int))\n        '
        sentence_length = len(src_sentence)
        phrase_indices = [[] for _ in src_sentence]
        for start in range(0, sentence_length):
            for end in range(start + 1, sentence_length + 1):
                potential_phrase = src_sentence[start:end]
                if potential_phrase in self.phrase_table:
                    phrase_indices[start].append(end)
        return phrase_indices

    def compute_future_scores(self, src_sentence):
        if False:
            i = 10
            return i + 15
        '\n        Determines the approximate scores for translating every\n        subsequence in ``src_sentence``\n\n        Future scores can be used a look-ahead to determine the\n        difficulty of translating the remaining parts of a src_sentence.\n\n        :type src_sentence: tuple(str)\n\n        :return: Scores of subsequences referenced by their start and\n            end positions. For example, result[2][5] is the score of the\n            subsequence covering positions 2, 3, and 4.\n        :rtype: dict(int: (dict(int): float))\n        '
        scores = defaultdict(lambda : defaultdict(lambda : float('-inf')))
        for seq_length in range(1, len(src_sentence) + 1):
            for start in range(0, len(src_sentence) - seq_length + 1):
                end = start + seq_length
                phrase = src_sentence[start:end]
                if phrase in self.phrase_table:
                    score = self.phrase_table.translations_for(phrase)[0].log_prob
                    score += self.language_model.probability(phrase)
                    scores[start][end] = score
                for mid in range(start + 1, end):
                    combined_score = scores[start][mid] + scores[mid][end]
                    if combined_score > scores[start][end]:
                        scores[start][end] = combined_score
        return scores

    def future_score(self, hypothesis, future_score_table, sentence_length):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines the approximate score for translating the\n        untranslated words in ``hypothesis``\n        '
        score = 0.0
        for span in hypothesis.untranslated_spans(sentence_length):
            score += future_score_table[span[0]][span[1]]
        return score

    def expansion_score(self, hypothesis, translation_option, src_phrase_span):
        if False:
            return 10
        '\n        Calculate the score of expanding ``hypothesis`` with\n        ``translation_option``\n\n        :param hypothesis: Hypothesis being expanded\n        :type hypothesis: _Hypothesis\n\n        :param translation_option: Information about the proposed expansion\n        :type translation_option: PhraseTableEntry\n\n        :param src_phrase_span: Word position span of the source phrase\n        :type src_phrase_span: tuple(int, int)\n        '
        score = hypothesis.raw_score
        score += translation_option.log_prob
        score += self.language_model.probability_change(hypothesis, translation_option.trg_phrase)
        score += self.distortion_score(hypothesis, src_phrase_span)
        score -= self.word_penalty * len(translation_option.trg_phrase)
        return score

    def distortion_score(self, hypothesis, next_src_phrase_span):
        if False:
            print('Hello World!')
        if not hypothesis.src_phrase_span:
            return 0.0
        next_src_phrase_start = next_src_phrase_span[0]
        prev_src_phrase_end = hypothesis.src_phrase_span[1]
        distortion_distance = next_src_phrase_start - prev_src_phrase_end
        return abs(distortion_distance) * self.__log_distortion_factor

    @staticmethod
    def valid_phrases(all_phrases_from, hypothesis):
        if False:
            i = 10
            return i + 15
        '\n        Extract phrases from ``all_phrases_from`` that contains words\n        that have not been translated by ``hypothesis``\n\n        :param all_phrases_from: Phrases represented by their spans, in\n            the same format as the return value of\n            ``find_all_src_phrases``\n        :type all_phrases_from: list(list(int))\n\n        :type hypothesis: _Hypothesis\n\n        :return: A list of phrases, represented by their spans, that\n            cover untranslated positions.\n        :rtype: list(tuple(int, int))\n        '
        untranslated_spans = hypothesis.untranslated_spans(len(all_phrases_from))
        valid_phrases = []
        for available_span in untranslated_spans:
            start = available_span[0]
            available_end = available_span[1]
            while start < available_end:
                for phrase_end in all_phrases_from[start]:
                    if phrase_end > available_end:
                        break
                    valid_phrases.append((start, phrase_end))
                start += 1
        return valid_phrases

class _Hypothesis:
    """
    Partial solution to a translation.

    Records the word positions of the phrase being translated, its
    translation, raw score, and the cost of the untranslated parts of
    the sentence. When the next phrase is selected to build upon the
    partial solution, a new _Hypothesis object is created, with a back
    pointer to the previous hypothesis.

    To find out which words have been translated so far, look at the
    ``src_phrase_span`` in the hypothesis chain. Similarly, the
    translation output can be found by traversing up the chain.
    """

    def __init__(self, raw_score=0.0, src_phrase_span=(), trg_phrase=(), previous=None, future_score=0.0):
        if False:
            while True:
                i = 10
        '\n        :param raw_score: Likelihood of hypothesis so far.\n            Higher is better. Does not account for untranslated words.\n        :type raw_score: float\n\n        :param src_phrase_span: Span of word positions covered by the\n            source phrase in this hypothesis expansion. For example,\n            (2, 5) means that the phrase is from the second word up to,\n            but not including the fifth word in the source sentence.\n        :type src_phrase_span: tuple(int)\n\n        :param trg_phrase: Translation of the source phrase in this\n            hypothesis expansion\n        :type trg_phrase: tuple(str)\n\n        :param previous: Previous hypothesis before expansion to this one\n        :type previous: _Hypothesis\n\n        :param future_score: Approximate score for translating the\n            remaining words not covered by this hypothesis. Higher means\n            that the remaining words are easier to translate.\n        :type future_score: float\n        '
        self.raw_score = raw_score
        self.src_phrase_span = src_phrase_span
        self.trg_phrase = trg_phrase
        self.previous = previous
        self.future_score = future_score

    def score(self):
        if False:
            return 10
        '\n        Overall score of hypothesis after accounting for local and\n        global features\n        '
        return self.raw_score + self.future_score

    def untranslated_spans(self, sentence_length):
        if False:
            while True:
                i = 10
        '\n        Starting from each untranslated word, find the longest\n        continuous span of untranslated positions\n\n        :param sentence_length: Length of source sentence being\n            translated by the hypothesis\n        :type sentence_length: int\n\n        :rtype: list(tuple(int, int))\n        '
        translated_positions = self.translated_positions()
        translated_positions.sort()
        translated_positions.append(sentence_length)
        untranslated_spans = []
        start = 0
        for end in translated_positions:
            if start < end:
                untranslated_spans.append((start, end))
            start = end + 1
        return untranslated_spans

    def translated_positions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        List of positions in the source sentence of words already\n        translated. The list is not sorted.\n\n        :rtype: list(int)\n        '
        translated_positions = []
        current_hypothesis = self
        while current_hypothesis.previous is not None:
            translated_span = current_hypothesis.src_phrase_span
            translated_positions.extend(range(translated_span[0], translated_span[1]))
            current_hypothesis = current_hypothesis.previous
        return translated_positions

    def total_translated_words(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.translated_positions())

    def translation_so_far(self):
        if False:
            while True:
                i = 10
        translation = []
        self.__build_translation(self, translation)
        return translation

    def __build_translation(self, hypothesis, output):
        if False:
            print('Hello World!')
        if hypothesis.previous is None:
            return
        self.__build_translation(hypothesis.previous, output)
        output.extend(hypothesis.trg_phrase)

class _Stack:
    """
    Collection of _Hypothesis objects
    """

    def __init__(self, max_size=100, beam_threshold=0.0):
        if False:
            print('Hello World!')
        '\n        :param beam_threshold: Hypotheses that score less than this\n            factor of the best hypothesis are discarded from the stack.\n            Value must be between 0.0 and 1.0.\n        :type beam_threshold: float\n        '
        self.max_size = max_size
        self.items = []
        if beam_threshold == 0.0:
            self.__log_beam_threshold = float('-inf')
        else:
            self.__log_beam_threshold = log(beam_threshold)

    def push(self, hypothesis):
        if False:
            while True:
                i = 10
        '\n        Add ``hypothesis`` to the stack.\n        Removes lowest scoring hypothesis if the stack is full.\n        After insertion, hypotheses that score less than\n        ``beam_threshold`` times the score of the best hypothesis\n        are removed.\n        '
        self.items.append(hypothesis)
        self.items.sort(key=lambda h: h.score(), reverse=True)
        while len(self.items) > self.max_size:
            self.items.pop()
        self.threshold_prune()

    def threshold_prune(self):
        if False:
            while True:
                i = 10
        if not self.items:
            return
        threshold = self.items[0].score() + self.__log_beam_threshold
        for hypothesis in reversed(self.items):
            if hypothesis.score() < threshold:
                self.items.pop()
            else:
                break

    def best(self):
        if False:
            print('Hello World!')
        '\n        :return: Hypothesis with the highest score in the stack\n        :rtype: _Hypothesis\n        '
        if self.items:
            return self.items[0]
        return None

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.items)

    def __contains__(self, hypothesis):
        if False:
            i = 10
            return i + 15
        return hypothesis in self.items

    def __bool__(self):
        if False:
            return 10
        return len(self.items) != 0
    __nonzero__ = __bool__