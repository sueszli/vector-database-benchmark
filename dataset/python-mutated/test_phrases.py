"""
Automated tests for the phrase detection module.
"""
import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath

class TestPhraseAnalysis(unittest.TestCase):

    class AnalysisTester(_PhrasesTransformation):

        def __init__(self, scores, threshold):
            if False:
                return 10
            super().__init__(connector_words={'a', 'the', 'with', 'of'})
            self.scores = scores
            self.threshold = threshold

        def score_candidate(self, word_a, word_b, in_between):
            if False:
                i = 10
                return i + 15
            phrase = '_'.join([word_a] + in_between + [word_b])
            score = self.scores.get(phrase, -1)
            if score > self.threshold:
                return (phrase, score)
            return (None, None)

    def test_simple_analysis(self):
        if False:
            print('Hello World!')
        'Test transformation with no phrases.'
        sentence = ['simple', 'sentence', 'should', 'pass']
        result = self.AnalysisTester({}, threshold=1)[sentence]
        self.assertEqual(result, sentence)
        sentence = ['a', 'simple', 'sentence', 'with', 'no', 'bigram', 'but', 'common', 'terms']
        result = self.AnalysisTester({}, threshold=1)[sentence]
        self.assertEqual(result, sentence)

    def test_analysis_bigrams(self):
        if False:
            return 10
        scores = {'simple_sentence': 2, 'sentence_many': 2, 'many_possible': 2, 'possible_bigrams': 2}
        sentence = ['simple', 'sentence', 'many', 'possible', 'bigrams']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, ['simple_sentence', 'many_possible', 'bigrams'])
        sentence = ['some', 'simple', 'sentence', 'many', 'bigrams']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, ['some', 'simple_sentence', 'many', 'bigrams'])
        sentence = ['some', 'unrelated', 'simple', 'words']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, sentence)

    def test_analysis_connector_words(self):
        if False:
            print('Hello World!')
        scores = {'simple_sentence': 2, 'sentence_many': 2, 'many_possible': 2, 'possible_bigrams': 2}
        sentence = ['a', 'simple', 'sentence', 'many', 'the', 'possible', 'bigrams']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, ['a', 'simple_sentence', 'many', 'the', 'possible_bigrams'])
        sentence = ['simple', 'the', 'sentence', 'and', 'many', 'possible', 'bigrams', 'with', 'a']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, ['simple', 'the', 'sentence', 'and', 'many_possible', 'bigrams', 'with', 'a'])

    def test_analysis_connector_words_in_between(self):
        if False:
            print('Hello World!')
        scores = {'simple_sentence': 2, 'sentence_with_many': 2, 'many_possible': 2, 'many_of_the_possible': 2, 'possible_bigrams': 2}
        sentence = ['sentence', 'with', 'many', 'possible', 'bigrams']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, ['sentence_with_many', 'possible_bigrams'])
        sentence = ['a', 'simple', 'sentence', 'with', 'many', 'of', 'the', 'possible', 'bigrams', 'with']
        result = self.AnalysisTester(scores, threshold=1)[sentence]
        self.assertEqual(result, ['a', 'simple_sentence', 'with', 'many_of_the_possible', 'bigrams', 'with'])

class PhrasesData:
    sentences = common_texts + [['graph', 'minors', 'survey', 'human', 'interface']]
    connector_words = frozenset()
    bigram1 = u'response_time'
    bigram2 = u'graph_minors'
    bigram3 = u'human_interface'

    def gen_sentences(self):
        if False:
            return 10
        return ((w for w in sentence) for sentence in self.sentences)

class PhrasesCommon(PhrasesData):
    """Tests for both Phrases and FrozenPhrases classes."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        self.bigram_default = Phrases(self.sentences, connector_words=self.connector_words)

    def test_empty_phrasified_sentences_iterator(self):
        if False:
            print('Hello World!')
        bigram_phrases = Phrases(self.sentences)
        bigram_phraser = FrozenPhrases(bigram_phrases)
        trigram_phrases = Phrases(bigram_phraser[self.sentences])
        trigram_phraser = FrozenPhrases(trigram_phrases)
        trigrams = trigram_phraser[bigram_phraser[self.sentences]]
        (fst, snd) = (list(trigrams), list(trigrams))
        self.assertEqual(fst, snd)
        self.assertNotEqual(snd, [])

    def test_empty_inputs_on_bigram_construction(self):
        if False:
            print('Hello World!')
        "Test that empty inputs don't throw errors and return the expected result."
        self.assertEqual(list(self.bigram_default[[]]), [])
        self.assertEqual(list(self.bigram_default[iter(())]), [])
        self.assertEqual(list(self.bigram_default[[[], []]]), [[], []])
        self.assertEqual(list(self.bigram_default[iter([[], []])]), [[], []])
        self.assertEqual(list(self.bigram_default[(iter(()) for i in range(2))]), [[], []])

    def test_sentence_generation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test basic bigram using a dummy corpus.'
        self.assertEqual(len(self.sentences), len(list(self.bigram_default[self.sentences])))

    def test_sentence_generation_with_generator(self):
        if False:
            for i in range(10):
                print('nop')
        'Test basic bigram production when corpus is a generator.'
        self.assertEqual(len(list(self.gen_sentences())), len(list(self.bigram_default[self.gen_sentences()])))

    def test_bigram_construction(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Phrases bigram construction.'
        bigram1_seen = False
        bigram2_seen = False
        for sentence in self.bigram[self.sentences]:
            if not bigram1_seen and self.bigram1 in sentence:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in sentence:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[1]])
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[4]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-2]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-1]])
        self.assertTrue(self.bigram3 in self.bigram[self.sentences[-1]])

    def test_bigram_construction_from_generator(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Phrases bigram construction building when corpus is a generator.'
        bigram1_seen = False
        bigram2_seen = False
        for s in self.bigram[self.gen_sentences()]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)

    def test_bigram_construction_from_array(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Phrases bigram construction building when corpus is a numpy array.'
        bigram1_seen = False
        bigram2_seen = False
        for s in self.bigram[np.array(self.sentences, dtype=object)]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)

def dumb_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    if False:
        return 10
    return 1

class TestPhrasesModel(PhrasesCommon, unittest.TestCase):

    def test_export_phrases(self):
        if False:
            while True:
                i = 10
        'Test Phrases bigram and trigram export phrases.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        trigram = Phrases(bigram[self.sentences], min_count=1, threshold=1, delimiter=' ')
        seen_bigrams = set(bigram.export_phrases().keys())
        seen_trigrams = set(trigram.export_phrases().keys())
        assert seen_bigrams == set(['human interface', 'response time', 'graph minors', 'minors survey'])
        assert seen_trigrams == set(['human interface', 'graph minors survey'])

    def test_find_phrases(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Phrases bigram find phrases.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        seen_bigrams = set(bigram.find_phrases(self.sentences).keys())
        assert seen_bigrams == set(['response time', 'graph minors', 'human interface'])

    def test_multiple_bigrams_single_entry(self):
        if False:
            return 10
        'Test a single entry produces multiple bigrams.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        seen_bigrams = set(bigram.find_phrases(test_sentences).keys())
        assert seen_bigrams == {'graph minors', 'human interface'}

    def test_scoring_default(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the default scoring, from the mikolov word2vec paper.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        assert seen_scores == {5.167, 3.444}

    def test__getitem__(self):
        if False:
            i = 10
            return i + 15
        'Test Phrases[sentences] with a single sentence.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        phrased_sentence = next(bigram[test_sentences].__iter__())
        assert phrased_sentence == ['graph_minors', 'survey', 'human_interface']

    def test_scoring_npmi(self):
        if False:
            print('Hello World!')
        'Test normalized pointwise mutual information scoring.'
        bigram = Phrases(self.sentences, min_count=1, threshold=0.5, scoring='npmi')
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        assert seen_scores == {0.882, 0.714}

    def test_custom_scorer(self):
        if False:
            while True:
                i = 10
        'Test using a custom scoring function.'
        bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = list(bigram.find_phrases(test_sentences).values())
        assert all((score == 1 for score in seen_scores))
        assert len(seen_scores) == 3

    def test_bad_parameters(self):
        if False:
            print('Hello World!')
        'Test the phrases module with bad parameters.'
        self.assertRaises(ValueError, Phrases, self.sentences, min_count=0)
        self.assertRaises(ValueError, Phrases, self.sentences, threshold=-1)

    def test_pruning(self):
        if False:
            i = 10
            return i + 15
        'Test that max_vocab_size parameter is respected.'
        bigram = Phrases(self.sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)

class TestPhrasesPersistence(PhrasesData, unittest.TestCase):

    def test_save_load_custom_scorer(self):
        if False:
            i = 10
            return i + 15
        'Test saving and loading a Phrases object with a custom scorer.'
        bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer)
        with temporary_file('test.pkl') as fpath:
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = list(bigram_loaded.find_phrases(test_sentences).values())
        assert all((score == 1 for score in seen_scores))
        assert len(seen_scores) == 3

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        'Test saving and loading a Phrases object.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        with temporary_file('test.pkl') as fpath:
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = set((round(score, 3) for score in bigram_loaded.find_phrases(test_sentences).values()))
        assert seen_scores == set([5.167, 3.444])

    def test_save_load_with_connector_words(self):
        if False:
            for i in range(10):
                print('nop')
        'Test saving and loading a Phrases object.'
        connector_words = frozenset({'of'})
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=connector_words)
        with temporary_file('test.pkl') as fpath:
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
        assert bigram_loaded.connector_words == connector_words

    def test_save_load_string_scoring(self):
        if False:
            print('Hello World!')
        'Test backwards compatibility with a previous version of Phrases with custom scoring.'
        bigram_loaded = Phrases.load(datapath('phrases-scoring-str.pkl'))
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = set((round(score, 3) for score in bigram_loaded.find_phrases(test_sentences).values()))
        assert seen_scores == set([5.167, 3.444])

    def test_save_load_no_scoring(self):
        if False:
            print('Hello World!')
        'Test backwards compatibility with old versions of Phrases with no scoring parameter.'
        bigram_loaded = Phrases.load(datapath('phrases-no-scoring.pkl'))
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        seen_scores = set((round(score, 3) for score in bigram_loaded.find_phrases(test_sentences).values()))
        assert seen_scores == set([5.167, 3.444])

    def test_save_load_no_common_terms(self):
        if False:
            print('Hello World!')
        'Ensure backwards compatibility with old versions of Phrases, before connector_words.'
        bigram_loaded = Phrases.load(datapath('phrases-no-common-terms.pkl'))
        self.assertEqual(bigram_loaded.connector_words, frozenset())
        phraser = FrozenPhrases(bigram_loaded)
        phraser[['human', 'interface', 'survey']]

class TestFrozenPhrasesPersistence(PhrasesData, unittest.TestCase):

    def test_save_load_custom_scorer(self):
        if False:
            i = 10
            return i + 15
        'Test saving and loading a FrozenPhrases object with a custom scorer.'
        with temporary_file('test.pkl') as fpath:
            bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer))
            bigram.save(fpath)
            bigram_loaded = FrozenPhrases.load(fpath)
            self.assertEqual(bigram_loaded.scoring, dumb_scorer)

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        'Test saving and loading a FrozenPhrases object.'
        with temporary_file('test.pkl') as fpath:
            bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=1))
            bigram.save(fpath)
            bigram_loaded = FrozenPhrases.load(fpath)
            self.assertEqual(bigram_loaded[['graph', 'minors', 'survey', 'human', 'interface', 'system']], ['graph_minors', 'survey', 'human_interface', 'system'])

    def test_save_load_with_connector_words(self):
        if False:
            return 10
        'Test saving and loading a FrozenPhrases object.'
        connector_words = frozenset({'of'})
        with temporary_file('test.pkl') as fpath:
            bigram = FrozenPhrases(Phrases(self.sentences, min_count=1, threshold=1, connector_words=connector_words))
            bigram.save(fpath)
            bigram_loaded = FrozenPhrases.load(fpath)
            self.assertEqual(bigram_loaded.connector_words, connector_words)

    def test_save_load_string_scoring(self):
        if False:
            print('Hello World!')
        'Test saving and loading a FrozenPhrases object with a string scoring parameter.\n        This should ensure backwards compatibility with the previous version of FrozenPhrases'
        bigram_loaded = FrozenPhrases.load(datapath('phraser-scoring-str.pkl'))
        self.assertEqual(bigram_loaded.scoring, original_scorer)

    def test_save_load_no_scoring(self):
        if False:
            return 10
        'Test saving and loading a FrozenPhrases object with no scoring parameter.\n        This should ensure backwards compatibility with old versions of FrozenPhrases'
        bigram_loaded = FrozenPhrases.load(datapath('phraser-no-scoring.pkl'))
        self.assertEqual(bigram_loaded.scoring, original_scorer)

    def test_save_load_no_common_terms(self):
        if False:
            while True:
                i = 10
        'Ensure backwards compatibility with old versions of FrozenPhrases, before connector_words.'
        bigram_loaded = FrozenPhrases.load(datapath('phraser-no-common-terms.pkl'))
        self.assertEqual(bigram_loaded.connector_words, frozenset())

class TestFrozenPhrasesModel(PhrasesCommon, unittest.TestCase):
    """Test FrozenPhrases models."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up FrozenPhrases models for the tests.'
        bigram_phrases = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        self.bigram = FrozenPhrases(bigram_phrases)
        bigram_default_phrases = Phrases(self.sentences, connector_words=self.connector_words)
        self.bigram_default = FrozenPhrases(bigram_default_phrases)

class CommonTermsPhrasesData:
    """This mixin permits to reuse tests with the connector_words option."""
    sentences = [['human', 'interface', 'with', 'computer'], ['survey', 'of', 'user', 'computer', 'system', 'lack', 'of', 'interest'], ['eps', 'user', 'interface', 'system'], ['system', 'and', 'human', 'system', 'eps'], ['user', 'lack', 'of', 'interest'], ['trees'], ['graph', 'of', 'trees'], ['data', 'and', 'graph', 'of', 'trees'], ['data', 'and', 'graph', 'survey'], ['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
    connector_words = ['of', 'and', 'for']
    bigram1 = u'lack_of_interest'
    bigram2 = u'data_and_graph'
    bigram3 = u'human_interface'
    expression1 = u'lack of interest'
    expression2 = u'data and graph'
    expression3 = u'human interface'

    def gen_sentences(self):
        if False:
            while True:
                i = 10
        return ((w for w in sentence) for sentence in self.sentences)

class TestPhrasesModelCommonTerms(CommonTermsPhrasesData, TestPhrasesModel):
    """Test Phrases models with connector words."""

    def test_multiple_bigrams_single_entry(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a single entry produces multiple bigrams.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words, delimiter=' ')
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_bigrams = set(bigram.find_phrases(test_sentences).keys())
        assert seen_bigrams == set(['data and graph', 'human interface'])

    def test_find_phrases(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Phrases bigram export phrases.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words, delimiter=' ')
        seen_bigrams = set(bigram.find_phrases(self.sentences).keys())
        assert seen_bigrams == set(['human interface', 'graph of trees', 'data and graph', 'lack of interest'])

    def test_export_phrases(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Phrases bigram export phrases.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, delimiter=' ')
        seen_bigrams = set(bigram.export_phrases().keys())
        assert seen_bigrams == set(['and graph', 'data and', 'graph of', 'graph survey', 'human interface', 'lack of', 'of interest', 'of trees'])

    def test_scoring_default(self):
        if False:
            return 10
        ' test the default scoring, from the mikolov word2vec paper '
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        min_count = float(bigram.min_count)
        len_vocab = float(len(bigram.vocab))
        graph = float(bigram.vocab['graph'])
        data = float(bigram.vocab['data'])
        data_and_graph = float(bigram.vocab['data_and_graph'])
        human = float(bigram.vocab['human'])
        interface = float(bigram.vocab['interface'])
        human_interface = float(bigram.vocab['human_interface'])
        assert seen_scores == set([round((data_and_graph - min_count) / data / graph * len_vocab, 3), round((human_interface - min_count) / human / interface * len_vocab, 3)])

    def test_scoring_npmi(self):
        if False:
            i = 10
            return i + 15
        'Test normalized pointwise mutual information scoring.'
        bigram = Phrases(self.sentences, min_count=1, threshold=0.5, scoring='npmi', connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_scores = set((round(score, 3) for score in bigram.find_phrases(test_sentences).values()))
        assert seen_scores == set([0.74, 0.894])

    def test_custom_scorer(self):
        if False:
            i = 10
            return i + 15
        'Test using a custom scoring function.'
        bigram = Phrases(self.sentences, min_count=1, threshold=0.001, scoring=dumb_scorer, connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        seen_scores = list(bigram.find_phrases(test_sentences).values())
        assert all(seen_scores)
        assert len(seen_scores) == 2

    def test__getitem__(self):
        if False:
            while True:
                i = 10
        'Test Phrases[sentences] with a single sentence.'
        bigram = Phrases(self.sentences, min_count=1, threshold=1, connector_words=self.connector_words)
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        phrased_sentence = next(bigram[test_sentences].__iter__())
        assert phrased_sentence == ['data_and_graph', 'survey', 'for', 'human_interface']

class TestFrozenPhrasesModelCompatibility(unittest.TestCase):

    def test_compatibility(self):
        if False:
            print('Hello World!')
        phrases = Phrases.load(datapath('phrases-3.6.0.model'))
        phraser = FrozenPhrases.load(datapath('phraser-3.6.0.model'))
        test_sentences = ['trees', 'graph', 'minors']
        self.assertEqual(phrases[test_sentences], ['trees', 'graph_minors'])
        self.assertEqual(phraser[test_sentences], ['trees', 'graph_minors'])
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()