"""
The Legality Principle is a language agnostic principle maintaining that syllable
onsets and codas (the beginning and ends of syllables not including the vowel)
are only legal if they are found as word onsets or codas in the language. The English
word ''admit'' must then be syllabified as ''ad-mit'' since ''dm'' is not found
word-initially in the English language (Bartlett et al.). This principle was first proposed
in Daniel Kahn's 1976 dissertation, ''Syllable-based generalizations in English phonology''.

Kahn further argues that there is a ''strong tendency to syllabify in such a way that
initial clusters are of maximal length, consistent with the general constraints on
word-initial consonant clusters.'' Consequently, in addition to being legal onsets,
the longest legal onset is preferable---''Onset Maximization''.

The default implementation assumes an English vowel set, but the `vowels` attribute
can be set to IPA or any other alphabet's vowel set for the use-case.
Both a valid set of vowels as well as a text corpus of words in the language
are necessary to determine legal onsets and subsequently syllabify words.

The legality principle with onset maximization is a universal syllabification algorithm,
but that does not mean it performs equally across languages. Bartlett et al. (2009)
is a good benchmark for English accuracy if utilizing IPA (pg. 311).

References:

- Otto Jespersen. 1904. Lehrbuch der Phonetik.
  Leipzig, Teubner. Chapter 13, Silbe, pp. 185-203.
- Theo Vennemann, ''On the Theory of Syllabic Phonology,'' 1972, p. 11.
- Daniel Kahn, ''Syllable-based generalizations in English phonology'', (PhD diss., MIT, 1976).
- Elisabeth Selkirk. 1984. On the major class features and syllable theory.
  In Aronoff & Oehrle (eds.) Language Sound Structure: Studies in Phonology.
  Cambridge, MIT Press. pp. 107-136.
- Jeremy Goslin and Ulrich Frauenfelder. 2001. A comparison of theoretical and human syllabification. Language and Speech, 44:409–436.
- Susan Bartlett, et al. 2009. On the Syllabification of Phonemes.
  In HLT-NAACL. pp. 308-316.
- Christopher Hench. 2017. Resonances in Middle High German: New Methodologies in Prosody. UC Berkeley.
"""
from collections import Counter
from nltk.tokenize.api import TokenizerI

class LegalitySyllableTokenizer(TokenizerI):
    """
    Syllabifies words based on the Legality Principle and Onset Maximization.

        >>> from nltk.tokenize import LegalitySyllableTokenizer
        >>> from nltk import word_tokenize
        >>> from nltk.corpus import words
        >>> text = "This is a wonderful sentence."
        >>> text_words = word_tokenize(text)
        >>> LP = LegalitySyllableTokenizer(words.words())
        >>> [LP.tokenize(word) for word in text_words]
        [['This'], ['is'], ['a'], ['won', 'der', 'ful'], ['sen', 'ten', 'ce'], ['.']]
    """

    def __init__(self, tokenized_source_text, vowels='aeiouy', legal_frequency_threshold=0.001):
        if False:
            print('Hello World!')
        '\n        :param tokenized_source_text: List of valid tokens in the language\n        :type tokenized_source_text: list(str)\n        :param vowels: Valid vowels in language or IPA representation\n        :type vowels: str\n        :param legal_frequency_threshold: Lowest frequency of all onsets to be considered a legal onset\n        :type legal_frequency_threshold: float\n        '
        self.legal_frequency_threshold = legal_frequency_threshold
        self.vowels = vowels
        self.legal_onsets = self.find_legal_onsets(tokenized_source_text)

    def find_legal_onsets(self, words):
        if False:
            return 10
        '\n        Gathers all onsets and then return only those above the frequency threshold\n\n        :param words: List of words in a language\n        :type words: list(str)\n        :return: Set of legal onsets\n        :rtype: set(str)\n        '
        onsets = [self.onset(word) for word in words]
        legal_onsets = [k for (k, v) in Counter(onsets).items() if v / len(onsets) > self.legal_frequency_threshold]
        return set(legal_onsets)

    def onset(self, word):
        if False:
            return 10
        '\n        Returns consonant cluster of word, i.e. all characters until the first vowel.\n\n        :param word: Single word or token\n        :type word: str\n        :return: String of characters of onset\n        :rtype: str\n        '
        onset = ''
        for c in word.lower():
            if c in self.vowels:
                return onset
            else:
                onset += c
        return onset

    def tokenize(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply the Legality Principle in combination with\n        Onset Maximization to return a list of syllables.\n\n        :param token: Single word or token\n        :type token: str\n        :return syllable_list: Single word or token broken up into syllables.\n        :rtype: list(str)\n        '
        syllables = []
        (syllable, current_onset) = ('', '')
        (vowel, onset) = (False, False)
        for char in token[::-1]:
            char_lower = char.lower()
            if not vowel:
                syllable += char
                vowel = bool(char_lower in self.vowels)
            elif char_lower + current_onset[::-1] in self.legal_onsets:
                syllable += char
                current_onset += char_lower
                onset = True
            elif char_lower in self.vowels and (not onset):
                syllable += char
                current_onset += char_lower
            else:
                syllables.append(syllable)
                syllable = char
                current_onset = ''
                vowel = bool(char_lower in self.vowels)
        syllables.append(syllable)
        syllables_ordered = [syllable[::-1] for syllable in syllables][::-1]
        return syllables_ordered