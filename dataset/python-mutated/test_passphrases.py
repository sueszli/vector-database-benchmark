from unittest import mock
import pytest
from passphrases import InvalidWordListError, PassphraseGenerator

class TestPassphrasesGenerator:

    def test_default_generator(self):
        if False:
            print('Hello World!')
        generator = PassphraseGenerator.get_default()
        assert generator.available_languages == {'en', 'fr'}
        passphrase = generator.generate_passphrase()
        assert passphrase
        assert len(passphrase) >= 20
        assert len(passphrase.split(' ')) >= 7

    def test_default_generator_passphrases_are_random(self):
        if False:
            i = 10
            return i + 15
        generator = PassphraseGenerator.get_default()
        passphrase1 = generator.generate_passphrase()
        passphrase2 = generator.generate_passphrase()
        assert passphrase1 != passphrase2

    @mock.patch.object(PassphraseGenerator, '_WORD_LIST_MINIMUM_SIZE', 1)
    def test_generate_passphrase_with_specific_language(self):
        if False:
            print('Hello World!')
        generator = PassphraseGenerator(language_to_words={'en': ['boat'], 'fr': ['bateau']})
        assert generator.available_languages == {'en', 'fr'}
        passphrase = generator.generate_passphrase(preferred_language='fr')
        assert 'bateau' in passphrase
        assert 'boat' not in passphrase

    @mock.patch.object(PassphraseGenerator, '_WORD_LIST_MINIMUM_SIZE', 1)
    def test_generate_passphrase_with_specific_language_that_is_not_available(self):
        if False:
            i = 10
            return i + 15
        generator = PassphraseGenerator(language_to_words={'en': ['boat'], 'fr': ['bateau']}, fallback_language='en')
        assert generator.available_languages == {'en', 'fr'}
        passphrase = generator.generate_passphrase(preferred_language='es')
        assert 'boat' in passphrase
        assert 'bateau' not in passphrase

    def test_word_list_does_not_have_enough_words(self):
        if False:
            while True:
                i = 10
        with pytest.raises(InvalidWordListError, match='long-enough words'):
            PassphraseGenerator(language_to_words={'en': ['only', 'three', 'words']})

    @mock.patch.object(PassphraseGenerator, '_WORD_LIST_MINIMUM_SIZE', 1)
    def test_word_list_will_generate_overly_long_passphrase(self):
        if False:
            while True:
                i = 10
        with pytest.raises(InvalidWordListError, match='over the maximum length'):
            PassphraseGenerator(language_to_words={'en': ['overlylongwordtogetoverthelimit']})

    @mock.patch.object(PassphraseGenerator, '_WORD_LIST_MINIMUM_SIZE', 1)
    def test_word_list_will_generate_overly_short_passphrase(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(InvalidWordListError, match='under the minimum length'):
            PassphraseGenerator(language_to_words={'en': ['b', 'a']})

    @mock.patch.object(PassphraseGenerator, '_WORD_LIST_MINIMUM_SIZE', 1)
    def test_word_list_has_non_ascii_string(self):
        if False:
            print('Hello World!')
        with pytest.raises(InvalidWordListError, match='non-ASCII words'):
            PassphraseGenerator(language_to_words={'en': ['word', 'éoèô']})