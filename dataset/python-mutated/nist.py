"""
This is a NLTK port of the tokenizer used in the NIST BLEU evaluation script,
https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L926
which was also ported into Python in
https://github.com/lium-lst/nmtpy/blob/master/nmtpy/metrics/mtevalbleu.py#L162
"""
import io
import re
from nltk.corpus import perluniprops
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import xml_unescape

class NISTTokenizer(TokenizerI):
    """
    This NIST tokenizer is sentence-based instead of the original
    paragraph-based tokenization from mteval-14.pl; The sentence-based
    tokenization is consistent with the other tokenizers available in NLTK.

    >>> from nltk.tokenize.nist import NISTTokenizer
    >>> nist = NISTTokenizer()
    >>> s = "Good muffins cost $3.88 in New York."
    >>> expected_lower = [u'good', u'muffins', u'cost', u'$', u'3.88', u'in', u'new', u'york', u'.']
    >>> expected_cased = [u'Good', u'muffins', u'cost', u'$', u'3.88', u'in', u'New', u'York', u'.']
    >>> nist.tokenize(s, lowercase=False) == expected_cased
    True
    >>> nist.tokenize(s, lowercase=True) == expected_lower  # Lowercased.
    True

    The international_tokenize() is the preferred function when tokenizing
    non-european text, e.g.

    >>> from nltk.tokenize.nist import NISTTokenizer
    >>> nist = NISTTokenizer()

    # Input strings.
    >>> albb = u'Alibaba Group Holding Limited (Chinese: 阿里巴巴集团控股 有限公司) us a Chinese e-commerce company...'
    >>> amz = u'Amazon.com, Inc. (/ˈæməzɒn/) is an American electronic commerce...'
    >>> rkt = u'Rakuten, Inc. (楽天株式会社 Rakuten Kabushiki-gaisha) is a Japanese electronic commerce and Internet company based in Tokyo.'

    # Expected tokens.
    >>> expected_albb = [u'Alibaba', u'Group', u'Holding', u'Limited', u'(', u'Chinese', u':', u'阿里巴巴集团控股', u'有限公司', u')']
    >>> expected_amz = [u'Amazon', u'.', u'com', u',', u'Inc', u'.', u'(', u'/', u'ˈæ', u'm']
    >>> expected_rkt = [u'Rakuten', u',', u'Inc', u'.', u'(', u'楽天株式会社', u'Rakuten', u'Kabushiki', u'-', u'gaisha']

    >>> nist.international_tokenize(albb)[:10] == expected_albb
    True
    >>> nist.international_tokenize(amz)[:10] == expected_amz
    True
    >>> nist.international_tokenize(rkt)[:10] == expected_rkt
    True

    # Doctest for patching issue #1926
    >>> sent = u'this is a foo☄sentence.'
    >>> expected_sent = [u'this', u'is', u'a', u'foo', u'☄', u'sentence', u'.']
    >>> nist.international_tokenize(sent) == expected_sent
    True
    """
    STRIP_SKIP = (re.compile('<skipped>'), '')
    STRIP_EOL_HYPHEN = (re.compile('\u2028'), ' ')
    PUNCT = (re.compile('([\\{-\\~\\[-\\` -\\&\\(-\\+\\:-\\@\\/])'), ' \\1 ')
    PERIOD_COMMA_PRECEED = (re.compile('([^0-9])([\\.,])'), '\\1 \\2 ')
    PERIOD_COMMA_FOLLOW = (re.compile('([\\.,])([^0-9])'), ' \\1 \\2')
    DASH_PRECEED_DIGIT = (re.compile('([0-9])(-)'), '\\1 \\2 ')
    LANG_DEPENDENT_REGEXES = [PUNCT, PERIOD_COMMA_PRECEED, PERIOD_COMMA_FOLLOW, DASH_PRECEED_DIGIT]
    pup_number = str(''.join(set(perluniprops.chars('Number'))))
    pup_punct = str(''.join(set(perluniprops.chars('Punctuation'))))
    pup_symbol = str(''.join(set(perluniprops.chars('Symbol'))))
    number_regex = re.sub('[]^\\\\-]', '\\\\\\g<0>', pup_number)
    punct_regex = re.sub('[]^\\\\-]', '\\\\\\g<0>', pup_punct)
    symbol_regex = re.sub('[]^\\\\-]', '\\\\\\g<0>', pup_symbol)
    NONASCII = (re.compile('([\x00-\x7f]+)'), ' \\1 ')
    PUNCT_1 = (re.compile(f'([{number_regex}])([{punct_regex}])'), '\\1 \\2 ')
    PUNCT_2 = (re.compile(f'([{punct_regex}])([{number_regex}])'), ' \\1 \\2')
    SYMBOLS = (re.compile(f'([{symbol_regex}])'), ' \\1 ')
    INTERNATIONAL_REGEXES = [NONASCII, PUNCT_1, PUNCT_2, SYMBOLS]

    def lang_independent_sub(self, text):
        if False:
            i = 10
            return i + 15
        'Performs the language independent string substituitions.'
        (regexp, substitution) = self.STRIP_SKIP
        text = regexp.sub(substitution, text)
        text = xml_unescape(text)
        (regexp, substitution) = self.STRIP_EOL_HYPHEN
        text = regexp.sub(substitution, text)
        return text

    def tokenize(self, text, lowercase=False, western_lang=True, return_str=False):
        if False:
            print('Hello World!')
        text = str(text)
        text = self.lang_independent_sub(text)
        if western_lang:
            text = ' ' + text + ' '
            if lowercase:
                text = text.lower()
            for (regexp, substitution) in self.LANG_DEPENDENT_REGEXES:
                text = regexp.sub(substitution, text)
        text = ' '.join(text.split())
        text = str(text.strip())
        return text if return_str else text.split()

    def international_tokenize(self, text, lowercase=False, split_non_ascii=True, return_str=False):
        if False:
            return 10
        text = str(text)
        (regexp, substitution) = self.STRIP_SKIP
        text = regexp.sub(substitution, text)
        (regexp, substitution) = self.STRIP_EOL_HYPHEN
        text = regexp.sub(substitution, text)
        text = xml_unescape(text)
        if lowercase:
            text = text.lower()
        for (regexp, substitution) in self.INTERNATIONAL_REGEXES:
            text = regexp.sub(substitution, text)
        text = ' '.join(text.strip().split())
        return text if return_str else text.split()