"""
The Sonority Sequencing Principle (SSP) is a language agnostic algorithm proposed
by Otto Jesperson in 1904. The sonorous quality of a phoneme is judged by the
openness of the lips. Syllable breaks occur before troughs in sonority. For more
on the SSP see Selkirk (1984).

The default implementation uses the English alphabet, but the `sonority_hiearchy`
can be modified to IPA or any other alphabet for the use-case. The SSP is a
universal syllabification algorithm, but that does not mean it performs equally
across languages. Bartlett et al. (2009) is a good benchmark for English accuracy
if utilizing IPA (pg. 311).

Importantly, if a custom hierarchy is supplied and vowels span across more than
one level, they should be given separately to the `vowels` class attribute.

References:

- Otto Jespersen. 1904. Lehrbuch der Phonetik.
  Leipzig, Teubner. Chapter 13, Silbe, pp. 185-203.
- Elisabeth Selkirk. 1984. On the major class features and syllable theory.
  In Aronoff & Oehrle (eds.) Language Sound Structure: Studies in Phonology.
  Cambridge, MIT Press. pp. 107-136.
- Susan Bartlett, et al. 2009. On the Syllabification of Phonemes.
  In HLT-NAACL. pp. 308-316.
"""
import re
import warnings
from string import punctuation
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams

class SyllableTokenizer(TokenizerI):
    """
    Syllabifies words based on the Sonority Sequencing Principle (SSP).

        >>> from nltk.tokenize import SyllableTokenizer
        >>> from nltk import word_tokenize
        >>> SSP = SyllableTokenizer()
        >>> SSP.tokenize('justification')
        ['jus', 'ti', 'fi', 'ca', 'tion']
        >>> text = "This is a foobar-like sentence."
        >>> [SSP.tokenize(token) for token in word_tokenize(text)]
        [['This'], ['is'], ['a'], ['foo', 'bar', '-', 'li', 'ke'], ['sen', 'ten', 'ce'], ['.']]
    """

    def __init__(self, lang='en', sonority_hierarchy=False):
        if False:
            return 10
        "\n        :param lang: Language parameter, default is English, 'en'\n        :type lang: str\n        :param sonority_hierarchy: Sonority hierarchy according to the\n                                   Sonority Sequencing Principle.\n        :type sonority_hierarchy: list(str)\n        "
        if not sonority_hierarchy and lang == 'en':
            sonority_hierarchy = ['aeiouy', 'lmnrw', 'zvsf', 'bcdgtkpqxhj']
        self.vowels = sonority_hierarchy[0]
        self.phoneme_map = {}
        for (i, level) in enumerate(sonority_hierarchy):
            for c in level:
                sonority_level = len(sonority_hierarchy) - i
                self.phoneme_map[c] = sonority_level
                self.phoneme_map[c.upper()] = sonority_level

    def assign_values(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assigns each phoneme its value from the sonority hierarchy.\n        Note: Sentence/text has to be tokenized first.\n\n        :param token: Single word or token\n        :type token: str\n        :return: List of tuples, first element is character/phoneme and\n                 second is the soronity value.\n        :rtype: list(tuple(str, int))\n        '
        syllables_values = []
        for c in token:
            try:
                syllables_values.append((c, self.phoneme_map[c]))
            except KeyError:
                if c not in '0123456789' and c not in punctuation:
                    warnings.warn("Character not defined in sonority_hierarchy, assigning as vowel: '{}'".format(c))
                    syllables_values.append((c, max(self.phoneme_map.values())))
                    if c not in self.vowels:
                        self.vowels += c
                else:
                    syllables_values.append((c, -1))
        return syllables_values

    def validate_syllables(self, syllable_list):
        if False:
            for i in range(10):
                print('nop')
        "\n        Ensures each syllable has at least one vowel.\n        If the following syllable doesn't have vowel, add it to the current one.\n\n        :param syllable_list: Single word or token broken up into syllables.\n        :type syllable_list: list(str)\n        :return: Single word or token broken up into syllables\n                 (with added syllables if necessary)\n        :rtype: list(str)\n        "
        valid_syllables = []
        front = ''
        vowel_pattern = re.compile('|'.join(self.vowels))
        for (i, syllable) in enumerate(syllable_list):
            if syllable in punctuation:
                valid_syllables.append(syllable)
                continue
            if not vowel_pattern.search(syllable):
                if len(valid_syllables) == 0:
                    front += syllable
                else:
                    valid_syllables = valid_syllables[:-1] + [valid_syllables[-1] + syllable]
            elif len(valid_syllables) == 0:
                valid_syllables.append(front + syllable)
            else:
                valid_syllables.append(syllable)
        return valid_syllables

    def tokenize(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply the SSP to return a list of syllables.\n        Note: Sentence/text has to be tokenized first.\n\n        :param token: Single word or token\n        :type token: str\n        :return syllable_list: Single word or token broken up into syllables.\n        :rtype: list(str)\n        '
        syllables_values = self.assign_values(token)
        if sum((token.count(x) for x in self.vowels)) <= 1:
            return [token]
        syllable_list = []
        syllable = syllables_values[0][0]
        for trigram in ngrams(syllables_values, n=3):
            (phonemes, values) = zip(*trigram)
            (prev_value, focal_value, next_value) = values
            focal_phoneme = phonemes[1]
            if focal_value == -1:
                syllable_list.append(syllable)
                syllable_list.append(focal_phoneme)
                syllable = ''
            elif prev_value >= focal_value == next_value:
                syllable += focal_phoneme
                syllable_list.append(syllable)
                syllable = ''
            elif prev_value > focal_value < next_value:
                syllable_list.append(syllable)
                syllable = ''
                syllable += focal_phoneme
            else:
                syllable += focal_phoneme
        syllable += syllables_values[-1][0]
        syllable_list.append(syllable)
        return self.validate_syllables(syllable_list)