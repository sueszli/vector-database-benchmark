"""BOPOMOFO 相关的几个拼音风格实现:

Style.BOPOMOFO
Style.BOPOMOFO_FIRST
"""
from __future__ import unicode_literals
import re
from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._constants import RE_TONE3
from pypinyin.style._utils import replace_symbol_to_number
BOPOMOFO_REPLACE = ((re.compile('^m(\\d)$'), 'mu\\1'), (re.compile('^n(\\d)$'), 'N\\1'), (re.compile('^r5$'), 'er5'), (re.compile('iu'), 'iou'), (re.compile('ui'), 'uei'), (re.compile('ong'), 'ung'), (re.compile('^yi?'), 'i'), (re.compile('^wu?'), 'u'), (re.compile('iu'), 'v'), (re.compile('^([jqx])u'), '\\1v'), (re.compile('([iuv])n'), '\\1en'), (re.compile('^zhi?'), 'Z'), (re.compile('^chi?'), 'C'), (re.compile('^shi?'), 'S'), (re.compile('^([zcsr])i'), '\\1'), (re.compile('ai'), 'A'), (re.compile('ei'), 'I'), (re.compile('ao'), 'O'), (re.compile('ou'), 'U'), (re.compile('ang'), 'K'), (re.compile('eng'), 'G'), (re.compile('an'), 'M'), (re.compile('en'), 'N'), (re.compile('er'), 'R'), (re.compile('eh'), 'E'), (re.compile('([iv])e'), '\\1E'), (re.compile('([^0-4])$'), '\\g<1>0'), (re.compile('1$'), ''))
BOPOMOFO_TABLE = dict(zip('bpmfdtnlgkhjqxZCSrzcsiuvaoeEAIOUMNKGR2340ê', 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦˊˇˋ˙ㄝ'))

class BopomofoConverter(object):

    def to_bopomofo(self, pinyin, **kwargs):
        if False:
            return 10
        pinyin = self._pre_convert(pinyin)
        for (find_re, replace) in BOPOMOFO_REPLACE:
            pinyin = find_re.sub(replace, pinyin)
        pinyin = ''.join((BOPOMOFO_TABLE.get(x, x) for x in pinyin))
        return pinyin

    def to_bopomofo_first(self, pinyin, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pinyin = self.to_bopomofo(pinyin, **kwargs)
        return pinyin[0]

    def _pre_convert(self, pinyin):
        if False:
            return 10
        pinyin = replace_symbol_to_number(pinyin)
        return RE_TONE3.sub('\\1\\3\\2', pinyin)
converter = BopomofoConverter()
register(Style.BOPOMOFO, func=converter.to_bopomofo)
register(Style.BOPOMOFO_FIRST, func=converter.to_bopomofo_first)