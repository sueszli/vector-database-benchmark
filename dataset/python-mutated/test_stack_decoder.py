"""
Tests for stack decoder
"""
import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack

class TestStackDecoder(unittest.TestCase):

    def test_find_all_src_phrases(self):
        if False:
            print('Hello World!')
        phrase_table = TestStackDecoder.create_fake_phrase_table()
        stack_decoder = StackDecoder(phrase_table, None)
        sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
        src_phrase_spans = stack_decoder.find_all_src_phrases(sentence)
        self.assertEqual(src_phrase_spans[0], [2])
        self.assertEqual(src_phrase_spans[1], [2])
        self.assertEqual(src_phrase_spans[2], [3])
        self.assertEqual(src_phrase_spans[3], [5, 6])
        self.assertFalse(src_phrase_spans[4])
        self.assertEqual(src_phrase_spans[5], [6])

    def test_distortion_score(self):
        if False:
            print('Hello World!')
        stack_decoder = StackDecoder(None, None)
        stack_decoder.distortion_factor = 0.5
        hypothesis = _Hypothesis()
        hypothesis.src_phrase_span = (3, 5)
        score = stack_decoder.distortion_score(hypothesis, (8, 10))
        expected_score = log(stack_decoder.distortion_factor) * (8 - 5)
        self.assertEqual(score, expected_score)

    def test_distortion_score_of_first_expansion(self):
        if False:
            i = 10
            return i + 15
        stack_decoder = StackDecoder(None, None)
        stack_decoder.distortion_factor = 0.5
        hypothesis = _Hypothesis()
        score = stack_decoder.distortion_score(hypothesis, (8, 10))
        self.assertEqual(score, 0.0)

    def test_compute_future_costs(self):
        if False:
            while True:
                i = 10
        phrase_table = TestStackDecoder.create_fake_phrase_table()
        language_model = TestStackDecoder.create_fake_language_model()
        stack_decoder = StackDecoder(phrase_table, language_model)
        sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
        future_scores = stack_decoder.compute_future_scores(sentence)
        self.assertEqual(future_scores[1][2], phrase_table.translations_for(('hovercraft',))[0].log_prob + language_model.probability(('hovercraft',)))
        self.assertEqual(future_scores[0][2], phrase_table.translations_for(('my', 'hovercraft'))[0].log_prob + language_model.probability(('my', 'hovercraft')))

    def test_compute_future_costs_for_phrases_not_in_phrase_table(self):
        if False:
            while True:
                i = 10
        phrase_table = TestStackDecoder.create_fake_phrase_table()
        language_model = TestStackDecoder.create_fake_language_model()
        stack_decoder = StackDecoder(phrase_table, language_model)
        sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
        future_scores = stack_decoder.compute_future_scores(sentence)
        self.assertEqual(future_scores[1][3], future_scores[1][2] + future_scores[2][3])

    def test_future_score(self):
        if False:
            for i in range(10):
                print('nop')
        hypothesis = _Hypothesis()
        hypothesis.untranslated_spans = lambda _: [(0, 2), (5, 8)]
        future_score_table = defaultdict(lambda : defaultdict(float))
        future_score_table[0][2] = 0.4
        future_score_table[5][8] = 0.5
        stack_decoder = StackDecoder(None, None)
        future_score = stack_decoder.future_score(hypothesis, future_score_table, 8)
        self.assertEqual(future_score, 0.4 + 0.5)

    def test_valid_phrases(self):
        if False:
            return 10
        hypothesis = _Hypothesis()
        hypothesis.untranslated_spans = lambda _: [(0, 2), (3, 6)]
        all_phrases_from = [[1, 4], [2], [], [5], [5, 6, 7], [], [7]]
        phrase_spans = StackDecoder.valid_phrases(all_phrases_from, hypothesis)
        self.assertEqual(phrase_spans, [(0, 1), (1, 2), (3, 5), (4, 5), (4, 6)])

    @staticmethod
    def create_fake_phrase_table():
        if False:
            for i in range(10):
                print('nop')
        phrase_table = PhraseTable()
        phrase_table.add(('hovercraft',), ('',), 0.8)
        phrase_table.add(('my', 'hovercraft'), ('', ''), 0.7)
        phrase_table.add(('my', 'cheese'), ('', ''), 0.7)
        phrase_table.add(('is',), ('',), 0.8)
        phrase_table.add(('is',), ('',), 0.5)
        phrase_table.add(('full', 'of'), ('', ''), 0.01)
        phrase_table.add(('full', 'of', 'eels'), ('', '', ''), 0.5)
        phrase_table.add(('full', 'of', 'spam'), ('', ''), 0.5)
        phrase_table.add(('eels',), ('',), 0.5)
        phrase_table.add(('spam',), ('',), 0.5)
        return phrase_table

    @staticmethod
    def create_fake_language_model():
        if False:
            return 10
        language_prob = defaultdict(lambda : -999.0)
        language_prob['my',] = log(0.1)
        language_prob['hovercraft',] = log(0.1)
        language_prob['is',] = log(0.1)
        language_prob['full',] = log(0.1)
        language_prob['of',] = log(0.1)
        language_prob['eels',] = log(0.1)
        language_prob['my', 'hovercraft'] = log(0.3)
        language_model = type('', (object,), {'probability': lambda _, phrase: language_prob[phrase]})()
        return language_model

class TestHypothesis(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        root = _Hypothesis()
        child = _Hypothesis(raw_score=0.5, src_phrase_span=(3, 7), trg_phrase=('hello', 'world'), previous=root)
        grandchild = _Hypothesis(raw_score=0.4, src_phrase_span=(1, 2), trg_phrase=('and', 'goodbye'), previous=child)
        self.hypothesis_chain = grandchild

    def test_translation_so_far(self):
        if False:
            return 10
        translation = self.hypothesis_chain.translation_so_far()
        self.assertEqual(translation, ['hello', 'world', 'and', 'goodbye'])

    def test_translation_so_far_for_empty_hypothesis(self):
        if False:
            while True:
                i = 10
        hypothesis = _Hypothesis()
        translation = hypothesis.translation_so_far()
        self.assertEqual(translation, [])

    def test_total_translated_words(self):
        if False:
            while True:
                i = 10
        total_translated_words = self.hypothesis_chain.total_translated_words()
        self.assertEqual(total_translated_words, 5)

    def test_translated_positions(self):
        if False:
            for i in range(10):
                print('nop')
        translated_positions = self.hypothesis_chain.translated_positions()
        translated_positions.sort()
        self.assertEqual(translated_positions, [1, 3, 4, 5, 6])

    def test_untranslated_spans(self):
        if False:
            return 10
        untranslated_spans = self.hypothesis_chain.untranslated_spans(10)
        self.assertEqual(untranslated_spans, [(0, 1), (2, 3), (7, 10)])

    def test_untranslated_spans_for_empty_hypothesis(self):
        if False:
            for i in range(10):
                print('nop')
        hypothesis = _Hypothesis()
        untranslated_spans = hypothesis.untranslated_spans(10)
        self.assertEqual(untranslated_spans, [(0, 10)])

class TestStack(unittest.TestCase):

    def test_push_bumps_off_worst_hypothesis_when_stack_is_full(self):
        if False:
            i = 10
            return i + 15
        stack = _Stack(3)
        poor_hypothesis = _Hypothesis(0.01)
        stack.push(_Hypothesis(0.2))
        stack.push(poor_hypothesis)
        stack.push(_Hypothesis(0.1))
        stack.push(_Hypothesis(0.3))
        self.assertFalse(poor_hypothesis in stack)

    def test_push_removes_hypotheses_that_fall_below_beam_threshold(self):
        if False:
            return 10
        stack = _Stack(3, 0.5)
        poor_hypothesis = _Hypothesis(0.01)
        worse_hypothesis = _Hypothesis(0.009)
        stack.push(poor_hypothesis)
        stack.push(worse_hypothesis)
        stack.push(_Hypothesis(0.9))
        self.assertFalse(poor_hypothesis in stack)
        self.assertFalse(worse_hypothesis in stack)

    def test_push_does_not_add_hypothesis_that_falls_below_beam_threshold(self):
        if False:
            while True:
                i = 10
        stack = _Stack(3, 0.5)
        poor_hypothesis = _Hypothesis(0.01)
        stack.push(_Hypothesis(0.9))
        stack.push(poor_hypothesis)
        self.assertFalse(poor_hypothesis in stack)

    def test_best_returns_the_best_hypothesis(self):
        if False:
            return 10
        stack = _Stack(3)
        best_hypothesis = _Hypothesis(0.99)
        stack.push(_Hypothesis(0.0))
        stack.push(best_hypothesis)
        stack.push(_Hypothesis(0.5))
        self.assertEqual(stack.best(), best_hypothesis)

    def test_best_returns_none_when_stack_is_empty(self):
        if False:
            for i in range(10):
                print('nop')
        stack = _Stack(3)
        self.assertEqual(stack.best(), None)