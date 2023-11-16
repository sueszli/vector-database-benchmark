"""

Penn Treebank Tokenizer

The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
This implementation is a port of the tokenizer sed script written by Robert McIntyre
and available at http://www.cis.upenn.edu/~treebank/tokenizer.sed.
"""
import re
import warnings
from typing import Iterator, List, Tuple
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.destructive import MacIntyreContractions
from nltk.tokenize.util import align_tokens

class TreebankWordTokenizer(TokenizerI):
    """
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.

    This tokenizer performs the following steps:

    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line

    >>> from nltk.tokenize import TreebankWordTokenizer
    >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
    >>> TreebankWordTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
    >>> s = "They'll save and invest more."
    >>> TreebankWordTokenizer().tokenize(s)
    ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
    >>> s = "hi, my name can't hello,"
    >>> TreebankWordTokenizer().tokenize(s)
    ['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']
    """
    STARTING_QUOTES = [(re.compile('^\\"'), '``'), (re.compile('(``)'), ' \\1 '), (re.compile('([ \\(\\[{<])(\\"|\\\'{2})'), '\\1 `` ')]
    PUNCTUATION = [(re.compile('([:,])([^\\d])'), ' \\1 \\2'), (re.compile('([:,])$'), ' \\1 '), (re.compile('\\.\\.\\.'), ' ... '), (re.compile('[;@#$%&]'), ' \\g<0> '), (re.compile('([^\\.])(\\.)([\\]\\)}>"\\\']*)\\s*$'), '\\1 \\2\\3 '), (re.compile('[?!]'), ' \\g<0> '), (re.compile("([^'])' "), "\\1 ' ")]
    PARENS_BRACKETS = (re.compile('[\\]\\[\\(\\)\\{\\}\\<\\>]'), ' \\g<0> ')
    CONVERT_PARENTHESES = [(re.compile('\\('), '-LRB-'), (re.compile('\\)'), '-RRB-'), (re.compile('\\['), '-LSB-'), (re.compile('\\]'), '-RSB-'), (re.compile('\\{'), '-LCB-'), (re.compile('\\}'), '-RCB-')]
    DOUBLE_DASHES = (re.compile('--'), ' -- ')
    ENDING_QUOTES = [(re.compile("''"), " '' "), (re.compile('"'), " '' "), (re.compile("([^' ])('[sS]|'[mM]|'[dD]|') "), '\\1 \\2 '), (re.compile("([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), '\\1 \\2 ')]
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text: str, convert_parentheses: bool=False, return_str: bool=False) -> List[str]:
        if False:
            while True:
                i = 10
        "Return a tokenized copy of `text`.\n\n        >>> from nltk.tokenize import TreebankWordTokenizer\n        >>> s = '''Good muffins cost $3.88 (roughly 3,36 euros)\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''\n        >>> TreebankWordTokenizer().tokenize(s) # doctest: +NORMALIZE_WHITESPACE\n        ['Good', 'muffins', 'cost', '$', '3.88', '(', 'roughly', '3,36',\n        'euros', ')', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',\n        'of', 'them.', 'Thanks', '.']\n        >>> TreebankWordTokenizer().tokenize(s, convert_parentheses=True) # doctest: +NORMALIZE_WHITESPACE\n        ['Good', 'muffins', 'cost', '$', '3.88', '-LRB-', 'roughly', '3,36',\n        'euros', '-RRB-', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',\n        'of', 'them.', 'Thanks', '.']\n\n        :param text: A string with a sentence or sentences.\n        :type text: str\n        :param convert_parentheses: if True, replace parentheses to PTB symbols,\n            e.g. `(` to `-LRB-`. Defaults to False.\n        :type convert_parentheses: bool, optional\n        :param return_str: If True, return tokens as space-separated string,\n            defaults to False.\n        :type return_str: bool, optional\n        :return: List of tokens from `text`.\n        :rtype: List[str]\n        "
        if return_str is not False:
            warnings.warn("Parameter 'return_str' has been deprecated and should no longer be used.", category=DeprecationWarning, stacklevel=2)
        for (regexp, substitution) in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)
        for (regexp, substitution) in self.PUNCTUATION:
            text = regexp.sub(substitution, text)
        (regexp, substitution) = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        if convert_parentheses:
            for (regexp, substitution) in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)
        (regexp, substitution) = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)
        text = ' ' + text + ' '
        for (regexp, substitution) in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(' \\1 \\2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(' \\1 \\2 ', text)
        return text.split()

    def span_tokenize(self, text: str) -> Iterator[Tuple[int, int]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the spans of the tokens in ``text``.\n        Uses the post-hoc nltk.tokens.align_tokens to return the offset spans.\n\n            >>> from nltk.tokenize import TreebankWordTokenizer\n            >>> s = '''Good muffins cost $3.88\\nin New (York).  Please (buy) me\\ntwo of them.\\n(Thanks).'''\n            >>> expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23),\n            ... (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38),\n            ... (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59),\n            ... (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]\n            >>> list(TreebankWordTokenizer().span_tokenize(s)) == expected\n            True\n            >>> expected = ['Good', 'muffins', 'cost', '$', '3.88', 'in',\n            ... 'New', '(', 'York', ')', '.', 'Please', '(', 'buy', ')',\n            ... 'me', 'two', 'of', 'them.', '(', 'Thanks', ')', '.']\n            >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected\n            True\n\n        :param text: A string with a sentence or sentences.\n        :type text: str\n        :yield: Tuple[int, int]\n        "
        raw_tokens = self.tokenize(text)
        if '"' in text or "''" in text:
            matched = [m.group() for m in re.finditer('``|\'{2}|\\"', text)]
            tokens = [matched.pop(0) if tok in ['"', '``', "''"] else tok for tok in raw_tokens]
        else:
            tokens = raw_tokens
        yield from align_tokens(tokens, text)

class TreebankWordDetokenizer(TokenizerI):
    """
    The Treebank detokenizer uses the reverse regex operations corresponding to
    the Treebank tokenizer's regexes.

    Note:

    - There're additional assumption mades when undoing the padding of ``[;@#$%&]``
      punctuation symbols that isn't presupposed in the TreebankTokenizer.
    - There're additional regexes added in reversing the parentheses tokenization,
       such as the ``r'([\\]\\)\\}\\>])\\s([:;,.])'``, which removes the additional right
       padding added to the closing parentheses precedding ``[:;,.]``.
    - It's not possible to return the original whitespaces as they were because
      there wasn't explicit records of where `'\\n'`, `'\\t'` or `'\\s'` were removed at
      the text.split() operation.

    >>> from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
    >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
    >>> d = TreebankWordDetokenizer()
    >>> t = TreebankWordTokenizer()
    >>> toks = t.tokenize(s)
    >>> d.detokenize(toks)
    'Good muffins cost $3.88 in New York. Please buy me two of them. Thanks.'

    The MXPOST parentheses substitution can be undone using the ``convert_parentheses``
    parameter:

    >>> s = '''Good muffins cost $3.88\\nin New (York).  Please (buy) me\\ntwo of them.\\n(Thanks).'''
    >>> expected_tokens = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
    ... 'New', '-LRB-', 'York', '-RRB-', '.', 'Please', '-LRB-', 'buy',
    ... '-RRB-', 'me', 'two', 'of', 'them.', '-LRB-', 'Thanks', '-RRB-', '.']
    >>> expected_tokens == t.tokenize(s, convert_parentheses=True)
    True
    >>> expected_detoken = 'Good muffins cost $3.88 in New (York). Please (buy) me two of them. (Thanks).'
    >>> expected_detoken == d.detokenize(t.tokenize(s, convert_parentheses=True), convert_parentheses=True)
    True

    During tokenization it's safe to add more spaces but during detokenization,
    simply undoing the padding doesn't really help.

    - During tokenization, left and right pad is added to ``[!?]``, when
      detokenizing, only left shift the ``[!?]`` is needed.
      Thus ``(re.compile(r'\\s([?!])'), r'\\g<1>')``.

    - During tokenization ``[:,]`` are left and right padded but when detokenizing,
      only left shift is necessary and we keep right pad after comma/colon
      if the string after is a non-digit.
      Thus ``(re.compile(r'\\s([:,])\\s([^\\d])'), r'\\1 \\2')``.

    >>> from nltk.tokenize.treebank import TreebankWordDetokenizer
    >>> toks = ['hello', ',', 'i', 'ca', "n't", 'feel', 'my', 'feet', '!', 'Help', '!', '!']
    >>> twd = TreebankWordDetokenizer()
    >>> twd.detokenize(toks)
    "hello, i can't feel my feet! Help!!"

    >>> toks = ['hello', ',', 'i', "can't", 'feel', ';', 'my', 'feet', '!',
    ... 'Help', '!', '!', 'He', 'said', ':', 'Help', ',', 'help', '?', '!']
    >>> twd.detokenize(toks)
    "hello, i can't feel; my feet! Help!! He said: Help, help?!"
    """
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = [re.compile(pattern.replace('(?#X)', '\\s')) for pattern in _contractions.CONTRACTIONS2]
    CONTRACTIONS3 = [re.compile(pattern.replace('(?#X)', '\\s')) for pattern in _contractions.CONTRACTIONS3]
    ENDING_QUOTES = [(re.compile("([^' ])\\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), '\\1\\2 '), (re.compile("([^' ])\\s('[sS]|'[mM]|'[dD]|') "), '\\1\\2 '), (re.compile("(\\S)\\s(\\'\\')"), '\\1\\2'), (re.compile("(\\'\\')\\s([.,:)\\]>};%])"), '\\1\\2'), (re.compile("''"), '"')]
    DOUBLE_DASHES = (re.compile(' -- '), '--')
    CONVERT_PARENTHESES = [(re.compile('-LRB-'), '('), (re.compile('-RRB-'), ')'), (re.compile('-LSB-'), '['), (re.compile('-RSB-'), ']'), (re.compile('-LCB-'), '{'), (re.compile('-RCB-'), '}')]
    PARENS_BRACKETS = [(re.compile('([\\[\\(\\{\\<])\\s'), '\\g<1>'), (re.compile('\\s([\\]\\)\\}\\>])'), '\\g<1>'), (re.compile('([\\]\\)\\}\\>])\\s([:;,.])'), '\\1\\2')]
    PUNCTUATION = [(re.compile("([^'])\\s'\\s"), "\\1' "), (re.compile('\\s([?!])'), '\\g<1>'), (re.compile('([^\\.])\\s(\\.)([\\]\\)}>"\\\']*)\\s*$'), '\\1\\2\\3'), (re.compile('([#$])\\s'), '\\g<1>'), (re.compile('\\s([;%])'), '\\g<1>'), (re.compile('\\s\\.\\.\\.\\s'), '...'), (re.compile('\\s([:,])'), '\\1')]
    STARTING_QUOTES = [(re.compile('([ (\\[{<])\\s``'), '\\1``'), (re.compile('(``)\\s'), '\\1'), (re.compile('``'), '"')]

    def tokenize(self, tokens: List[str], convert_parentheses: bool=False) -> str:
        if False:
            while True:
                i = 10
        '\n        Treebank detokenizer, created by undoing the regexes from\n        the TreebankWordTokenizer.tokenize.\n\n        :param tokens: A list of strings, i.e. tokenized text.\n        :type tokens: List[str]\n        :param convert_parentheses: if True, replace PTB symbols with parentheses,\n            e.g. `-LRB-` to `(`. Defaults to False.\n        :type convert_parentheses: bool, optional\n        :return: str\n        '
        text = ' '.join(tokens)
        text = ' ' + text + ' '
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub('\\1\\2', text)
        for regexp in self.CONTRACTIONS2:
            text = regexp.sub('\\1\\2', text)
        for (regexp, substitution) in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)
        text = text.strip()
        (regexp, substitution) = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)
        if convert_parentheses:
            for (regexp, substitution) in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)
        for (regexp, substitution) in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)
        for (regexp, substitution) in self.PUNCTUATION:
            text = regexp.sub(substitution, text)
        for (regexp, substitution) in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)
        return text.strip()

    def detokenize(self, tokens: List[str], convert_parentheses: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Duck-typing the abstract *tokenize()*.'
        return self.tokenize(tokens, convert_parentheses)