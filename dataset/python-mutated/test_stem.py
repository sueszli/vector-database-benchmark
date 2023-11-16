import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

class SnowballTest(unittest.TestCase):

    def test_arabic(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        this unit testing for test the snowball arabic light stemmer\n        this stemmer deals with prefixes and suffixes\n        '
        ar_stemmer = SnowballStemmer('arabic', True)
        assert ar_stemmer.stem('الْعَرَبِــــــيَّة') == 'عرب'
        assert ar_stemmer.stem('العربية') == 'عرب'
        assert ar_stemmer.stem('فقالوا') == 'قال'
        assert ar_stemmer.stem('الطالبات') == 'طالب'
        assert ar_stemmer.stem('فالطالبات') == 'طالب'
        assert ar_stemmer.stem('والطالبات') == 'طالب'
        assert ar_stemmer.stem('الطالبون') == 'طالب'
        assert ar_stemmer.stem('اللذان') == 'اللذان'
        assert ar_stemmer.stem('من') == 'من'
        ar_stemmer = SnowballStemmer('arabic', False)
        assert ar_stemmer.stem('اللذان') == 'اللذ'
        assert ar_stemmer.stem('الطالبات') == 'طالب'
        assert ar_stemmer.stem('الكلمات') == 'كلم'
        ar_stemmer = SnowballStemmer('arabic')
        assert ar_stemmer.stem('الْعَرَبِــــــيَّة') == 'عرب'
        assert ar_stemmer.stem('العربية') == 'عرب'
        assert ar_stemmer.stem('فقالوا') == 'قال'
        assert ar_stemmer.stem('الطالبات') == 'طالب'
        assert ar_stemmer.stem('الكلمات') == 'كلم'

    def test_russian(self):
        if False:
            for i in range(10):
                print('nop')
        stemmer_russian = SnowballStemmer('russian')
        assert stemmer_russian.stem('авантненькая') == 'авантненьк'

    def test_german(self):
        if False:
            for i in range(10):
                print('nop')
        stemmer_german = SnowballStemmer('german')
        stemmer_german2 = SnowballStemmer('german', ignore_stopwords=True)
        assert stemmer_german.stem('Schränke') == 'schrank'
        assert stemmer_german2.stem('Schränke') == 'schrank'
        assert stemmer_german.stem('keinen') == 'kein'
        assert stemmer_german2.stem('keinen') == 'keinen'

    def test_spanish(self):
        if False:
            i = 10
            return i + 15
        stemmer = SnowballStemmer('spanish')
        assert stemmer.stem('Visionado') == 'vision'
        assert stemmer.stem('algue') == 'algu'

    def test_short_strings_bug(self):
        if False:
            for i in range(10):
                print('nop')
        stemmer = SnowballStemmer('english')
        assert stemmer.stem("y's") == 'y'

class PorterTest(unittest.TestCase):

    def _vocabulary(self):
        if False:
            print('Hello World!')
        with closing(data.find('stemmers/porter_test/porter_vocabulary.txt').open(encoding='utf-8')) as fp:
            return fp.read().splitlines()

    def _test_against_expected_output(self, stemmer_mode, expected_stems):
        if False:
            return 10
        stemmer = PorterStemmer(mode=stemmer_mode)
        for (word, true_stem) in zip(self._vocabulary(), expected_stems):
            our_stem = stemmer.stem(word)
            assert our_stem == true_stem, '{} should stem to {} in {} mode but got {}'.format(word, true_stem, stemmer_mode, our_stem)

    def test_vocabulary_martin_mode(self):
        if False:
            while True:
                i = 10
        "Tests all words from the test vocabulary provided by M Porter\n\n        The sample vocabulary and output were sourced from\n        https://tartarus.org/martin/PorterStemmer/voc.txt and\n        https://tartarus.org/martin/PorterStemmer/output.txt\n        and are linked to from the Porter Stemmer algorithm's homepage\n        at https://tartarus.org/martin/PorterStemmer/\n        "
        with closing(data.find('stemmers/porter_test/porter_martin_output.txt').open(encoding='utf-8')) as fp:
            self._test_against_expected_output(PorterStemmer.MARTIN_EXTENSIONS, fp.read().splitlines())

    def test_vocabulary_nltk_mode(self):
        if False:
            return 10
        with closing(data.find('stemmers/porter_test/porter_nltk_output.txt').open(encoding='utf-8')) as fp:
            self._test_against_expected_output(PorterStemmer.NLTK_EXTENSIONS, fp.read().splitlines())

    def test_vocabulary_original_mode(self):
        if False:
            for i in range(10):
                print('nop')
        with closing(data.find('stemmers/porter_test/porter_original_output.txt').open(encoding='utf-8')) as fp:
            self._test_against_expected_output(PorterStemmer.ORIGINAL_ALGORITHM, fp.read().splitlines())
        self._test_against_expected_output(PorterStemmer.ORIGINAL_ALGORITHM, data.find('stemmers/porter_test/porter_original_output.txt').open(encoding='utf-8').read().splitlines())

    def test_oed_bug(self):
        if False:
            print('Hello World!')
        "Test for bug https://github.com/nltk/nltk/issues/1581\n\n        Ensures that 'oed' can be stemmed without throwing an error.\n        "
        assert PorterStemmer().stem('oed') == 'o'

    def test_lowercase_option(self):
        if False:
            i = 10
            return i + 15
        'Test for improvement on https://github.com/nltk/nltk/issues/2507\n\n        Ensures that stems are lowercased when `to_lowercase=True`\n        '
        porter = PorterStemmer()
        assert porter.stem('On') == 'on'
        assert porter.stem('I') == 'i'
        assert porter.stem('I', to_lowercase=False) == 'I'
        assert porter.stem('Github') == 'github'
        assert porter.stem('Github', to_lowercase=False) == 'Github'