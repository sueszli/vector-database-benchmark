"""
Multi-Word Expression Tokenizer

A ``MWETokenizer`` takes a string which has already been divided into tokens and
retokenizes it, merging multi-word expressions into single tokens, using a lexicon
of MWEs:


    >>> from nltk.tokenize import MWETokenizer

    >>> tokenizer = MWETokenizer([('a', 'little'), ('a', 'little', 'bit'), ('a', 'lot')])
    >>> tokenizer.add_mwe(('in', 'spite', 'of'))

    >>> tokenizer.tokenize('Testing testing testing one two three'.split())
    ['Testing', 'testing', 'testing', 'one', 'two', 'three']

    >>> tokenizer.tokenize('This is a test in spite'.split())
    ['This', 'is', 'a', 'test', 'in', 'spite']

    >>> tokenizer.tokenize('In a little or a little bit or a lot in spite of'.split())
    ['In', 'a_little', 'or', 'a_little_bit', 'or', 'a_lot', 'in_spite_of']

"""
from nltk.tokenize.api import TokenizerI
from nltk.util import Trie

class MWETokenizer(TokenizerI):
    """A tokenizer that processes tokenized text and merges multi-word expressions
    into single tokens.
    """

    def __init__(self, mwes=None, separator='_'):
        if False:
            for i in range(10):
                print('nop')
        "Initialize the multi-word tokenizer with a list of expressions and a\n        separator\n\n        :type mwes: list(list(str))\n        :param mwes: A sequence of multi-word expressions to be merged, where\n            each MWE is a sequence of strings.\n        :type separator: str\n        :param separator: String that should be inserted between words in a multi-word\n            expression token. (Default is '_')\n\n        "
        if not mwes:
            mwes = []
        self._mwes = Trie(mwes)
        self._separator = separator

    def add_mwe(self, mwe):
        if False:
            i = 10
            return i + 15
        "Add a multi-word expression to the lexicon (stored as a word trie)\n\n        We use ``util.Trie`` to represent the trie. Its form is a dict of dicts.\n        The key True marks the end of a valid MWE.\n\n        :param mwe: The multi-word expression we're adding into the word trie\n        :type mwe: tuple(str) or list(str)\n\n        :Example:\n\n        >>> tokenizer = MWETokenizer()\n        >>> tokenizer.add_mwe(('a', 'b'))\n        >>> tokenizer.add_mwe(('a', 'b', 'c'))\n        >>> tokenizer.add_mwe(('a', 'x'))\n        >>> expected = {'a': {'x': {True: None}, 'b': {True: None, 'c': {True: None}}}}\n        >>> tokenizer._mwes == expected\n        True\n\n        "
        self._mwes.insert(mwe)

    def tokenize(self, text):
        if False:
            while True:
                i = 10
        '\n\n        :param text: A list containing tokenized text\n        :type text: list(str)\n        :return: A list of the tokenized text with multi-words merged together\n        :rtype: list(str)\n\n        :Example:\n\n        >>> tokenizer = MWETokenizer([(\'hors\', "d\'oeuvre")], separator=\'+\')\n        >>> tokenizer.tokenize("An hors d\'oeuvre tonight, sir?".split())\n        [\'An\', "hors+d\'oeuvre", \'tonight,\', \'sir?\']\n\n        '
        i = 0
        n = len(text)
        result = []
        while i < n:
            if text[i] in self._mwes:
                j = i
                trie = self._mwes
                last_match = -1
                while j < n and text[j] in trie:
                    trie = trie[text[j]]
                    j = j + 1
                    if Trie.LEAF in trie:
                        last_match = j
                else:
                    if last_match > -1:
                        j = last_match
                    if Trie.LEAF in trie or last_match > -1:
                        result.append(self._separator.join(text[i:j]))
                        i = j
                    else:
                        result.append(text[i])
                        i += 1
            else:
                result.append(text[i])
                i += 1
        return result