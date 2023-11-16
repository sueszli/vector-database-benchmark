"""
The tok-tok tokenizer is a simple, general tokenizer, where the input has one
sentence per line; thus only final period is tokenized.

Tok-tok has been tested on, and gives reasonably good results for English,
Persian, Russian, Czech, French, German, Vietnamese, Tajik, and a few others.
The input should be in UTF-8 encoding.

Reference:
Jon Dehdari. 2014. A Neurophysiologically-Inspired Statistical Language
Model (Doctoral dissertation). Columbus, OH, USA: The Ohio State University.
"""
import re
from nltk.tokenize.api import TokenizerI

class ToktokTokenizer(TokenizerI):
    """
    This is a Python port of the tok-tok.pl from
    https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl

    >>> toktok = ToktokTokenizer()
    >>> text = u'Is 9.5 or 525,600 my favorite number?'
    >>> print(toktok.tokenize(text, return_str=True))
    Is 9.5 or 525,600 my favorite number ?
    >>> text = u'The https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl is a website with/and/or slashes and sort of weird : things'
    >>> print(toktok.tokenize(text, return_str=True))
    The https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl is a website with/and/or slashes and sort of weird : things
    >>> text = u'¡This, is a sentence with weird» symbols… appearing everywhere¿'
    >>> expected = u'¡ This , is a sentence with weird » symbols … appearing everywhere ¿'
    >>> assert toktok.tokenize(text, return_str=True) == expected
    >>> toktok.tokenize(text) == [u'¡', u'This', u',', u'is', u'a', u'sentence', u'with', u'weird', u'»', u'symbols', u'…', u'appearing', u'everywhere', u'¿']
    True
    """
    NON_BREAKING = (re.compile('\xa0'), ' ')
    FUNKY_PUNCT_1 = (re.compile('([،;؛¿!"\\])}»›”؟¡%٪°±©®।॥…])'), ' \\1 ')
    FUNKY_PUNCT_2 = (re.compile('([({\\[“‘„‚«‹「『])'), ' \\1 ')
    EN_EM_DASHES = (re.compile('([–—])'), ' \\1 ')
    AMPERCENT = (re.compile('& '), '&amp; ')
    TAB = (re.compile('\t'), ' &#9; ')
    PIPE = (re.compile('\\|'), ' &#124; ')
    COMMA_IN_NUM = (re.compile('(?<!,)([,،])(?![,\\d])'), ' \\1 ')
    PROB_SINGLE_QUOTES = (re.compile("(['’`])"), ' \\1 ')
    STUPID_QUOTES_1 = (re.compile(' ` ` '), ' `` ')
    STUPID_QUOTES_2 = (re.compile(" ' ' "), " '' ")
    FINAL_PERIOD_1 = (re.compile('(?<!\\.)\\.$'), ' .')
    FINAL_PERIOD_2 = (re.compile('(?<!\\.)\\.\\s*(["\'’»›”]) *$'), ' . \\1')
    MULTI_COMMAS = (re.compile('(,{2,})'), ' \\1 ')
    MULTI_DASHES = (re.compile('(-{2,})'), ' \\1 ')
    MULTI_DOTS = (re.compile('(\\.{2,})'), ' \\1 ')
    OPEN_PUNCT = str('([{༺༼᚛‚„⁅⁽₍〈❨❪❬❮❰❲❴⟅⟦⟨⟪⟬⟮⦃⦅⦇⦉⦋⦍⦏⦑⦓⦕⦗⧘⧚⧼⸢⸤⸦⸨〈《「『【〔〖〘〚〝﴾︗︵︷︹︻︽︿﹁﹃﹇﹙﹛﹝（［｛｟｢')
    CLOSE_PUNCT = str(')]}༻༽᚜⁆⁾₎〉❩❫❭❯❱❳❵⟆⟧⟩⟫⟭⟯⦄⦆⦈⦊⦌⦎⦐⦒⦔⦖⦘⧙⧛⧽⸣⸥⸧⸩〉》」』】〕〗〙〛〞〟﴿︘︶︸︺︼︾﹀﹂﹄﹈﹚﹜﹞）］｝｠｣')
    CURRENCY_SYM = str('$¢£¤¥֏؋৲৳৻૱௹฿៛₠₡₢₣₤₥₦₧₨₩₪₫€₭₮₯₰₱₲₳₴₵₶₷₸₹₺꠸﷼﹩＄￠￡￥￦')
    OPEN_PUNCT_RE = (re.compile(f'([{OPEN_PUNCT}])'), '\\1 ')
    CLOSE_PUNCT_RE = (re.compile(f'([{CLOSE_PUNCT}])'), '\\1 ')
    CURRENCY_SYM_RE = (re.compile(f'([{CURRENCY_SYM}])'), '\\1 ')
    URL_FOE_1 = (re.compile(':(?!//)'), ' : ')
    URL_FOE_2 = (re.compile('\\?(?!\\S)'), ' ? ')
    URL_FOE_3 = (re.compile('(:\\/\\/)[\\S+\\.\\S+\\/\\S+][\\/]'), ' / ')
    URL_FOE_4 = (re.compile(' /'), ' / ')
    LSTRIP = (re.compile('^ +'), '')
    RSTRIP = (re.compile('\\s+$'), '\n')
    ONE_SPACE = (re.compile(' {2,}'), ' ')
    TOKTOK_REGEXES = [NON_BREAKING, FUNKY_PUNCT_1, FUNKY_PUNCT_2, URL_FOE_1, URL_FOE_2, URL_FOE_3, URL_FOE_4, AMPERCENT, TAB, PIPE, OPEN_PUNCT_RE, CLOSE_PUNCT_RE, MULTI_COMMAS, COMMA_IN_NUM, PROB_SINGLE_QUOTES, STUPID_QUOTES_1, STUPID_QUOTES_2, CURRENCY_SYM_RE, EN_EM_DASHES, MULTI_DASHES, MULTI_DOTS, FINAL_PERIOD_1, FINAL_PERIOD_2, ONE_SPACE]

    def tokenize(self, text, return_str=False):
        if False:
            print('Hello World!')
        text = str(text)
        for (regexp, substitution) in self.TOKTOK_REGEXES:
            text = regexp.sub(substitution, text)
        text = str(text.strip())
        return text if return_str else text.split()