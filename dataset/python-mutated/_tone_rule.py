from __future__ import unicode_literals
'\n标调位置\n\n    有 ɑ 不放过，\n\n\u3000\u3000没 ɑ 找 o、e；\n\n\u3000\u3000ɑ、o、e、i、u、ü\n\n\u3000\u3000标调就按这顺序；\n\n\u3000\u3000i、u 若是连在一起，\n\n\u3000\u3000谁在后面就标谁。\n\nhttp://www.hwjyw.com/resource/content/2010/06/04/8183.shtml\nhttps://www.zhihu.com/question/23655297\nhttps://github.com/mozillazg/python-pinyin/issues/160\nhttp://www.pinyin.info/rules/where.html\n'

def right_mark_index(pinyin_no_tone):
    if False:
        while True:
            i = 10
    if 'iou' in pinyin_no_tone:
        return pinyin_no_tone.index('u')
    if 'uei' in pinyin_no_tone:
        return pinyin_no_tone.index('i')
    if 'uen' in pinyin_no_tone:
        return pinyin_no_tone.index('u')
    for c in ['a', 'o', 'e']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1
    for c in ['iu', 'ui']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1
    for c in ['i', 'u', 'v', 'ü']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1
    for c in ['n', 'm', 'ê']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + len(c) - 1