import re
_PERFECTIVE_GROUND = '(ив|ивши|ившись|ыв|ывши|ывшись((?<=[ая])(в|вши|вшись)))$'
_REFLEXIVE = '(с[яьи])$'
_ADJECTIVE = '(ими|ій|ий|а|е|ова|ове|ів|є|їй|єє|еє|я|ім|ем|им|ім|их|іх|ою|йми|іми|у|ю|ого|ому|ої)$'
_PARTICIPLE = '(ий|ого|ому|им|ім|а|ій|у|ою|ій|і|их|йми|их)$'
_VERB = '(сь|ся|ив|ать|ять|у|ю|ав|али|учи|ячи|вши|ши|е|ме|ати|яти|є)$'
_NOUN = '(а|ев|ов|е|ями|ами|еи|и|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я|і|ові|ї|ею|єю|ою|є|еві|ем|єм|ів|їв|ю)$'
_RVRE = '[аеиоуюяіїє]'
_DERIVATIONAL = '[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(?<=о)сть?$'

def stem_word(word):
    if False:
        return 10
    '\n    Based on https://drupal.org/project/ukstemmer and ported to Python https://github.com/Amice13/ukr_stemmer\n    '
    word = _preprocess(word)
    if not re.search('[аеиоуюяіїє]', word):
        return word
    p = re.search(_RVRE, word)
    start = word[0:p.span()[1]]
    suffix = word[p.span()[1]:]
    (updated, suffix) = _update_suffix(suffix, _PERFECTIVE_GROUND, '')
    if not updated:
        (_, suffix) = _update_suffix(suffix, _REFLEXIVE, '')
        (updated, suffix) = _update_suffix(suffix, _ADJECTIVE, '')
        if updated:
            (updated, suffix) = _update_suffix(suffix, _PARTICIPLE, '')
        else:
            (updated, suffix) = _update_suffix(suffix, _VERB, '')
            if not updated:
                (_, suffix) = _update_suffix(suffix, _NOUN, '')
    (updated, suffix) = _update_suffix(suffix, 'и$', '')
    if re.search(_DERIVATIONAL, suffix):
        (updated, suffix) = _update_suffix(suffix, 'ость$', '')
    (updated, suffix) = _update_suffix(suffix, 'ь$', '')
    if updated:
        (_, suffix) = _update_suffix(suffix, 'ейше?$', '')
        (_, suffix) = _update_suffix(suffix, 'нн$', u'н')
    return start + suffix

def _preprocess(word):
    if False:
        for i in range(10):
            print('nop')
    return word.lower().replace("'", '').replace('ё', 'е').replace('ъ', 'ї')

def _update_suffix(suffix, pattern, replacement):
    if False:
        for i in range(10):
            print('nop')
    result = re.sub(pattern, replacement, suffix)
    return (suffix != result, result)