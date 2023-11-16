"""Style.INITIALS 风格"""
from __future__ import unicode_literals
from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._utils import get_initials

@register(Style.INITIALS)
def convert(pinyin, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    strict = kwargs.get('strict', True)
    return get_initials(pinyin, strict)