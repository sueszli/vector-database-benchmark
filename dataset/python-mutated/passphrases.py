from pathlib import Path
from secrets import SystemRandom
from typing import Dict, List, NewType, Optional, Set
from sdconfig import SecureDropConfig
DicewarePassphrase = NewType('DicewarePassphrase', str)
_default_generator: Optional['PassphraseGenerator'] = None

class InvalidWordListError(Exception):
    pass

class PassphraseGenerator:
    PASSPHRASE_WORDS_COUNT = 7
    MAX_PASSPHRASE_LENGTH = 128
    MIN_PASSPHRASE_LENGTH = 20
    _WORD_LIST_MINIMUM_SIZE = 7300

    def __init__(self, language_to_words: Dict[str, List[str]], fallback_language: str='en') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._random_generator = SystemRandom()
        self._fallback_language = fallback_language
        self._language_to_words = language_to_words
        if self._fallback_language not in self._language_to_words:
            raise InvalidWordListError(f"Missing words list for fallback language '{self._fallback_language}'")
        for (language, word_list) in self._language_to_words.items():
            word_list_size = len(word_list)
            if word_list_size < self._WORD_LIST_MINIMUM_SIZE:
                raise InvalidWordListError("The word list for language '{}' only contains {} long-enough words; minimum required is {} words.".format(language, word_list_size, self._WORD_LIST_MINIMUM_SIZE))
            try:
                ' '.join(word_list).encode('ascii')
            except UnicodeEncodeError:
                raise InvalidWordListError("The word list for language '{}' contains non-ASCII words.")
            longest_word = max(word_list, key=len)
            longest_passphrase_length = len(longest_word) * self.PASSPHRASE_WORDS_COUNT
            longest_passphrase_length += self.PASSPHRASE_WORDS_COUNT
            if longest_passphrase_length >= self.MAX_PASSPHRASE_LENGTH:
                raise InvalidWordListError("Passphrases over the maximum length ({}) may be generated: longest word in word list for language '{}' is '{}' and number of words per passphrase is {}".format(self.MAX_PASSPHRASE_LENGTH, language, longest_word, self.PASSPHRASE_WORDS_COUNT))
            shortest_word = min(word_list, key=len)
            shortest_passphrase_length = len(shortest_word) * self.PASSPHRASE_WORDS_COUNT
            shortest_passphrase_length += self.PASSPHRASE_WORDS_COUNT
            if shortest_passphrase_length <= self.MIN_PASSPHRASE_LENGTH:
                raise InvalidWordListError("Passphrases under the minimum length ({}) may be generated: shortest word in word list for language '{}' is '{}' and number of words per passphrase is {}".format(self.MIN_PASSPHRASE_LENGTH, language, shortest_word, self.PASSPHRASE_WORDS_COUNT))

    @classmethod
    def get_default(cls) -> 'PassphraseGenerator':
        if False:
            print('Hello World!')
        global _default_generator
        if _default_generator is None:
            config = SecureDropConfig.get_current()
            language_to_words = _parse_available_words_list(config.SECUREDROP_ROOT)
            _default_generator = cls(language_to_words)
        return _default_generator

    @property
    def available_languages(self) -> Set[str]:
        if False:
            return 10
        return set(self._language_to_words.keys())

    def generate_passphrase(self, preferred_language: Optional[str]=None) -> DicewarePassphrase:
        if False:
            while True:
                i = 10
        final_language = preferred_language if preferred_language else self._fallback_language
        try:
            words_list = self._language_to_words[final_language]
        except KeyError:
            words_list = self._language_to_words[self._fallback_language]
        words: List[str] = [self._random_generator.choice(words_list) for _ in range(self.PASSPHRASE_WORDS_COUNT)]
        return DicewarePassphrase(' '.join(words))

def _parse_available_words_list(securedrop_root: Path) -> Dict[str, List[str]]:
    if False:
        i = 10
        return i + 15
    'Find all .txt files in the wordlists folder and parse them as words lists.\n\n    This will also ignore words that are too short.\n    '
    language_to_words = {}
    words_lists_folder = securedrop_root / 'wordlists'
    for words_file in words_lists_folder.glob('*.txt'):
        language = words_file.stem
        all_words = words_file.read_text().strip().splitlines()
        words_that_are_long_enough = [word for word in all_words if len(word) >= 2]
        language_to_words[language] = words_that_are_long_enough
    return language_to_words