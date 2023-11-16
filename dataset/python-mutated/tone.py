"""TONE 相关的几个拼音风格实现:

Style.TONE
Style.TONE2
Style.TONE3
"""
from __future__ import unicode_literals
from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._constants import RE_TONE3
from pypinyin.style._utils import replace_symbol_to_number

class ToneConverter(object):

    def to_tone(self, pinyin, **kwargs):
        if False:
            i = 10
            return i + 15
        return pinyin

    def to_tone2(self, pinyin, **kwargs):
        if False:
            return 10
        pinyin = replace_symbol_to_number(pinyin)
        return pinyin

    def to_tone3(self, pinyin, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pinyin = self.to_tone2(pinyin, **kwargs)
        return RE_TONE3.sub('\\1\\3\\2', pinyin)
converter = ToneConverter()
register(Style.TONE, func=converter.to_tone)
register(Style.TONE2, func=converter.to_tone2)
register(Style.TONE3, func=converter.to_tone3)