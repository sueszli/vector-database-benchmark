"""
Reto #9: HETEROGRAMA, ISOGRAMA Y PANGRAMA

Crea 3 funciones, cada una encargada de detectar si una cadena de
texto es un heterograma, un isograma o un pangrama.

Debes buscar la definición de cada uno de estos términos.
"""
from collections import Counter
from re import sub
from string import ascii_lowercase
from unicodedata import normalize

class TextAnalyzer:
    """
    A class that contains methods for text analysis.
    """

    def is_heterogram(self, text: str) -> bool:
        if False:
            print('Hello World!')
        '\n        Determines whether a string is a heterogram, meaning that it does not\n        contain any repeated letters.\n\n        Args:\n            text (str): The word or phrase to check.\n\n        Returns:\n            bool: True if the word or phrase is a heterogram, False otherwise.\n        '
        normalized_text = self.normalize_string(text)
        char_counter = Counter(normalized_text)
        return len(char_counter) == len(normalized_text)

    def is_isogram(self, text: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if a given word or phrase is an isogram, i.e., a word or phrase\n        in which each letter appears the same number of times.\n\n        Args:\n            text (str): The word or phrase to check.\n        Returns:\n            bool: True if the word or phrase is an isogram, False otherwise.\n        '
        normalized_text = self.normalize_string(text)
        char_counter = Counter(normalized_text)
        char_freq = [value for (char, value) in char_counter.most_common()]
        return min(char_freq) == max(char_freq)

    def is_pangram(self, text: str) -> bool:
        if False:
            return 10
        '\n        Check if a given text is a pangram. A pangram is a sentence or phrase\n        that contains all the letters of the alphabet at least once. \n\n        Args:\n            text (str): The text to check.\n\n        Returns:\n            bool: True if the text is a pangram, False otherwise.\n        '
        normalized_text = self.normalize_string(text)
        return set(normalized_text) == set(ascii_lowercase)

    def normalize_string(self, text: str) -> str:
        if False:
            return 10
        '\n        Normalizes a string by removing diacritical marks (e.g. accents,\n        umlauts), spaces and non-alphanumeric characters.\n        \n        Args:\n            text (str): A string to normalize.\n\n        Returns:\n            str: The normalized string.\n        '
        text_without_spaces = text.lower().replace(' ', '')
        canonical_unicode_text = normalize('NFKD', text_without_spaces)
        ascii_bytes_string = canonical_unicode_text.encode('ASCII', 'ignore')
        text_without_diacritics = ascii_bytes_string.decode('ASCII')
        text_without_symbols = sub(pattern='[^a-z\\s]', repl='', string=text_without_diacritics)
        return text_without_symbols
if __name__ == '__main__':
    analyzer = TextAnalyzer()
    print(analyzer.is_heterogram('El Piso!'))
    print(analyzer.is_heterogram('perro'))
    print(analyzer.is_isogram('papa'))
    print(analyzer.is_isogram('para'))
    pangram = 'Un jugoso zumo de piña y kiwi bien frío es exquisito y no lleva alcohol.'
    not_pangram = 'perro'
    print(analyzer.is_pangram(pangram))
    print(analyzer.is_pangram(not_pangram))