from __future__ import unicode_literals
from itertools import chain
from pypinyin.compat import text_type, bytes_type
from pypinyin.constants import RE_HANS, PHRASES_DICT
from pypinyin.seg import mmseg

def seg(hans):
    if False:
        return 10
    hans = simple_seg(hans)
    ret = []
    for x in hans:
        if not RE_HANS.match(x):
            ret.append(x)
        elif PHRASES_DICT:
            ret.extend(list(mmseg.seg.cut(x)))
        else:
            ret.append(x)
    return ret

def simple_seg(hans):
    if False:
        print('Hello World!')
    '将传入的字符串按是否是汉字来分割'
    assert not isinstance(hans, bytes_type), 'must be unicode string or [unicode, ...] list'
    if isinstance(hans, text_type):
        return _seg(hans)
    else:
        hans = list(hans)
        if len(hans) == 1:
            return simple_seg(hans[0])
        return list(chain(*[simple_seg(x) for x in hans]))

def _seg(chars):
    if False:
        i = 10
        return i + 15
    '按是否是汉字进行分词'
    s = ''
    ret = []
    flag = 0
    for (n, c) in enumerate(chars):
        if RE_HANS.match(c):
            if n == 0:
                flag = 0
            if flag == 0:
                s += c
            else:
                ret.append(s)
                flag = 0
                s = c
        else:
            if n == 0:
                flag = 1
            if flag == 1:
                s += c
            else:
                ret.append(s)
                flag = 1
                s = c
    ret.append(s)
    return ret