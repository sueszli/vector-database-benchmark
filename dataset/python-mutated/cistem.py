import re
from typing import Tuple
from nltk.stem.api import StemmerI

class Cistem(StemmerI):
    """
    CISTEM Stemmer for German

    This is the official Python implementation of the CISTEM stemmer.
    It is based on the paper
    Leonie Weissweiler, Alexander Fraser (2017). Developing a Stemmer for German
    Based on a Comparative Analysis of Publicly Available Stemmers.
    In Proceedings of the German Society for Computational Linguistics and Language
    Technology (GSCL)
    which can be read here:
    https://www.cis.lmu.de/~weissweiler/cistem/

    In the paper, we conducted an analysis of publicly available stemmers,
    developed two gold standards for German stemming and evaluated the stemmers
    based on the two gold standards. We then proposed the stemmer implemented here
    and show that it achieves slightly better f-measure than the other stemmers and
    is thrice as fast as the Snowball stemmer for German while being about as fast
    as most other stemmers.

    case_insensitive is a a boolean specifying if case-insensitive stemming
    should be used. Case insensitivity improves performance only if words in the
    text may be incorrectly upper case. For all-lowercase and correctly cased
    text, best performance is achieved by setting case_insensitive for false.

    :param case_insensitive: if True, the stemming is case insensitive. False by default.
    :type case_insensitive: bool
    """
    strip_ge = re.compile('^ge(.{4,})')
    repl_xx = re.compile('(.)\\1')
    strip_emr = re.compile('e[mr]$')
    strip_nd = re.compile('nd$')
    strip_t = re.compile('t$')
    strip_esn = re.compile('[esn]$')
    repl_xx_back = re.compile('(.)\\*')

    def __init__(self, case_insensitive: bool=False):
        if False:
            while True:
                i = 10
        self._case_insensitive = case_insensitive

    @staticmethod
    def replace_to(word: str) -> str:
        if False:
            i = 10
            return i + 15
        word = word.replace('sch', '$')
        word = word.replace('ei', '%')
        word = word.replace('ie', '&')
        word = Cistem.repl_xx.sub('\\1*', word)
        return word

    @staticmethod
    def replace_back(word: str) -> str:
        if False:
            return 10
        word = Cistem.repl_xx_back.sub('\\1\\1', word)
        word = word.replace('%', 'ei')
        word = word.replace('&', 'ie')
        word = word.replace('$', 'sch')
        return word

    def stem(self, word: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Stems the input word.\n\n        :param word: The word that is to be stemmed.\n        :type word: str\n        :return: The stemmed word.\n        :rtype: str\n\n        >>> from nltk.stem.cistem import Cistem\n        >>> stemmer = Cistem()\n        >>> s1 = "Speicherbehältern"\n        >>> stemmer.stem(s1)\n        \'speicherbehalt\'\n        >>> s2 = "Grenzpostens"\n        >>> stemmer.stem(s2)\n        \'grenzpost\'\n        >>> s3 = "Ausgefeiltere"\n        >>> stemmer.stem(s3)\n        \'ausgefeilt\'\n        >>> stemmer = Cistem(True)\n        >>> stemmer.stem(s1)\n        \'speicherbehal\'\n        >>> stemmer.stem(s2)\n        \'grenzpo\'\n        >>> stemmer.stem(s3)\n        \'ausgefeil\'\n        '
        if len(word) == 0:
            return word
        upper = word[0].isupper()
        word = word.lower()
        word = word.replace('ü', 'u')
        word = word.replace('ö', 'o')
        word = word.replace('ä', 'a')
        word = word.replace('ß', 'ss')
        word = Cistem.strip_ge.sub('\\1', word)
        return self._segment_inner(word, upper)[0]

    def segment(self, word: str) -> Tuple[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        This method works very similarly to stem (:func:\'cistem.stem\'). The difference is that in\n        addition to returning the stem, it also returns the rest that was removed at\n        the end. To be able to return the stem unchanged so the stem and the rest\n        can be concatenated to form the original word, all subsitutions that altered\n        the stem in any other way than by removing letters at the end were left out.\n\n        :param word: The word that is to be stemmed.\n        :type word: str\n        :return: A tuple of the stemmed word and the removed suffix.\n        :rtype: Tuple[str, str]\n\n        >>> from nltk.stem.cistem import Cistem\n        >>> stemmer = Cistem()\n        >>> s1 = "Speicherbehältern"\n        >>> stemmer.segment(s1)\n        (\'speicherbehält\', \'ern\')\n        >>> s2 = "Grenzpostens"\n        >>> stemmer.segment(s2)\n        (\'grenzpost\', \'ens\')\n        >>> s3 = "Ausgefeiltere"\n        >>> stemmer.segment(s3)\n        (\'ausgefeilt\', \'ere\')\n        >>> stemmer = Cistem(True)\n        >>> stemmer.segment(s1)\n        (\'speicherbehäl\', \'tern\')\n        >>> stemmer.segment(s2)\n        (\'grenzpo\', \'stens\')\n        >>> stemmer.segment(s3)\n        (\'ausgefeil\', \'tere\')\n        '
        if len(word) == 0:
            return ('', '')
        upper = word[0].isupper()
        word = word.lower()
        return self._segment_inner(word, upper)

    def _segment_inner(self, word: str, upper: bool):
        if False:
            while True:
                i = 10
        'Inner method for iteratively applying the code stemming regexes.\n        This method receives a pre-processed variant of the word to be stemmed,\n        or the word to be segmented, and returns a tuple of the word and the\n        removed suffix.\n\n        :param word: A pre-processed variant of the word that is to be stemmed.\n        :type word: str\n        :param upper: Whether the original word started with a capital letter.\n        :type upper: bool\n        :return: A tuple of the stemmed word and the removed suffix.\n        :rtype: Tuple[str, str]\n        '
        rest_length = 0
        word_copy = word[:]
        word = Cistem.replace_to(word)
        rest = ''
        while len(word) > 3:
            if len(word) > 5:
                (word, n) = Cistem.strip_emr.subn('', word)
                if n != 0:
                    rest_length += 2
                    continue
                (word, n) = Cistem.strip_nd.subn('', word)
                if n != 0:
                    rest_length += 2
                    continue
            if not upper or self._case_insensitive:
                (word, n) = Cistem.strip_t.subn('', word)
                if n != 0:
                    rest_length += 1
                    continue
            (word, n) = Cistem.strip_esn.subn('', word)
            if n != 0:
                rest_length += 1
                continue
            else:
                break
        word = Cistem.replace_back(word)
        if rest_length:
            rest = word_copy[-rest_length:]
        return (word, rest)