import re
import warnings
from typing import Iterator, List, Tuple
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import align_tokens

class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """
    CONTRACTIONS2 = ['(?i)\\b(can)(?#X)(not)\\b', "(?i)\\b(d)(?#X)('ye)\\b", '(?i)\\b(gim)(?#X)(me)\\b', '(?i)\\b(gon)(?#X)(na)\\b', '(?i)\\b(got)(?#X)(ta)\\b', '(?i)\\b(lem)(?#X)(me)\\b', "(?i)\\b(more)(?#X)('n)\\b", '(?i)\\b(wan)(?#X)(na)(?=\\s)']
    CONTRACTIONS3 = ["(?i) ('t)(?#X)(is)\\b", "(?i) ('t)(?#X)(was)\\b"]
    CONTRACTIONS4 = ['(?i)\\b(whad)(dd)(ya)\\b', '(?i)\\b(wha)(t)(cha)\\b']

class NLTKWordTokenizer(TokenizerI):
    """
    The NLTK tokenizer that has improved upon the TreebankWordTokenizer.

    This is the method that is invoked by ``word_tokenize()``.  It assumes that the
    text has already been segmented into sentences, e.g. using ``sent_tokenize()``.

    The tokenizer is "destructive" such that the regexes applied will munge the
    input string to a state beyond re-construction. It is possible to apply
    `TreebankWordDetokenizer.detokenize` to the tokenized outputs of
    `NLTKDestructiveWordTokenizer.tokenize` but there's no guarantees to
    revert to the original string.
    """
    STARTING_QUOTES = [(re.compile('([«“‘„]|[`]+)', re.U), ' \\1 '), (re.compile('^\\"'), '``'), (re.compile('(``)'), ' \\1 '), (re.compile('([ \\(\\[{<])(\\"|\\\'{2})'), '\\1 `` '), (re.compile("(?i)(\\')(?!re|ve|ll|m|t|s|d|n)(\\w)\\b", re.U), '\\1 \\2')]
    ENDING_QUOTES = [(re.compile('([»”’])', re.U), ' \\1 '), (re.compile("''"), " '' "), (re.compile('"'), " '' "), (re.compile("([^' ])('[sS]|'[mM]|'[dD]|') "), '\\1 \\2 '), (re.compile("([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), '\\1 \\2 ')]
    PUNCTUATION = [(re.compile('([^\\.])(\\.)([\\]\\)}>"\\\'»”’ ]*)\\s*$', re.U), '\\1 \\2 \\3 '), (re.compile('([:,])([^\\d])'), ' \\1 \\2'), (re.compile('([:,])$'), ' \\1 '), (re.compile('\\.{2,}', re.U), ' \\g<0> '), (re.compile('[;@#$%&]'), ' \\g<0> '), (re.compile('([^\\.])(\\.)([\\]\\)}>"\\\']*)\\s*$'), '\\1 \\2\\3 '), (re.compile('[?!]'), ' \\g<0> '), (re.compile("([^'])' "), "\\1 ' "), (re.compile('[*]', re.U), ' \\g<0> ')]
    PARENS_BRACKETS = (re.compile('[\\]\\[\\(\\)\\{\\}\\<\\>]'), ' \\g<0> ')
    CONVERT_PARENTHESES = [(re.compile('\\('), '-LRB-'), (re.compile('\\)'), '-RRB-'), (re.compile('\\['), '-LSB-'), (re.compile('\\]'), '-RSB-'), (re.compile('\\{'), '-LCB-'), (re.compile('\\}'), '-RCB-')]
    DOUBLE_DASHES = (re.compile('--'), ' -- ')
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text: str, convert_parentheses: bool=False, return_str: bool=False) -> List[str]:
        if False:
            return 10
        "Return a tokenized copy of `text`.\n\n        >>> from nltk.tokenize import NLTKWordTokenizer\n        >>> s = '''Good muffins cost $3.88 (roughly 3,36 euros)\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''\n        >>> NLTKWordTokenizer().tokenize(s) # doctest: +NORMALIZE_WHITESPACE\n        ['Good', 'muffins', 'cost', '$', '3.88', '(', 'roughly', '3,36',\n        'euros', ')', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',\n        'of', 'them.', 'Thanks', '.']\n        >>> NLTKWordTokenizer().tokenize(s, convert_parentheses=True) # doctest: +NORMALIZE_WHITESPACE\n        ['Good', 'muffins', 'cost', '$', '3.88', '-LRB-', 'roughly', '3,36',\n        'euros', '-RRB-', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',\n        'of', 'them.', 'Thanks', '.']\n\n\n        :param text: A string with a sentence or sentences.\n        :type text: str\n        :param convert_parentheses: if True, replace parentheses to PTB symbols,\n            e.g. `(` to `-LRB-`. Defaults to False.\n        :type convert_parentheses: bool, optional\n        :param return_str: If True, return tokens as space-separated string,\n            defaults to False.\n        :type return_str: bool, optional\n        :return: List of tokens from `text`.\n        :rtype: List[str]\n        "
        if return_str:
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
            i = 10
            return i + 15
        "\n        Returns the spans of the tokens in ``text``.\n        Uses the post-hoc nltk.tokens.align_tokens to return the offset spans.\n\n            >>> from nltk.tokenize import NLTKWordTokenizer\n            >>> s = '''Good muffins cost $3.88\\nin New (York).  Please (buy) me\\ntwo of them.\\n(Thanks).'''\n            >>> expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23),\n            ... (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38),\n            ... (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59),\n            ... (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]\n            >>> list(NLTKWordTokenizer().span_tokenize(s)) == expected\n            True\n            >>> expected = ['Good', 'muffins', 'cost', '$', '3.88', 'in',\n            ... 'New', '(', 'York', ')', '.', 'Please', '(', 'buy', ')',\n            ... 'me', 'two', 'of', 'them.', '(', 'Thanks', ')', '.']\n            >>> [s[start:end] for start, end in NLTKWordTokenizer().span_tokenize(s)] == expected\n            True\n\n        :param text: A string with a sentence or sentences.\n        :type text: str\n        :yield: Tuple[int, int]\n        "
        raw_tokens = self.tokenize(text)
        if '"' in text or "''" in text:
            matched = [m.group() for m in re.finditer('``|\'{2}|\\"', text)]
            tokens = [matched.pop(0) if tok in ['"', '``', "''"] else tok for tok in raw_tokens]
        else:
            tokens = raw_tokens
        yield from align_tokens(tokens, text)