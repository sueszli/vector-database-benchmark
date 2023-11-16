import unicodedata
from unicodedata import normalize

def basic_aurebesh_translator(text: str, aurebesh: bool) -> str:
    if False:
        i = 10
        return i + 15
    basic_dict = {'a': 'aurek', 'b': 'besh', 'c': 'cresh', 'ch': 'cherek', 'd': 'dorn', 'e': 'esk', 'eo': 'onith', 'f': 'forn', 'g': 'grek', 'h': 'herf', 'i': 'isk', 'j': 'jenth', 'k': 'krill', 'kh': 'krenth', 'l': 'leth', 'm': 'mern', 'n': 'nern', 'ng': 'nen', 'o': 'osk', 'oo': 'orenth', 'p': 'peth', 'q': 'qek', 'r': 'resh', 's': 'senth', 'sh': 'shen', 't': 'trill', 'th': 'thesh', 'u': 'usk', 'v': 'vev', 'w': 'wesk', 'x': 'xesh', 'y': 'yirt', 'z': 'zerek', 'ae': 'enth'}
    aurebesh_alphabet = dict()
    for (key, value) in basic_dict.items():
        aurebesh_alphabet[value] = key
    text = text.lower()
    text = normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    translated_text = ''
    if aurebesh:
        translated_text = text
        for (key, value) in aurebesh_alphabet.items():
            translated_text = translated_text.replace(key, value)
    else:
        character_index = 0
        while character_index < len(text):
            if text[character_index] in basic_dict:
                translated_text += basic_dict[text[character_index]]
                character_index += 1
            elif text[character_index:character_index + 2] in basic_dict:
                translated_text += basic_dict[text[character_index:character_index + 2]]
                character_index += 2
            else:
                translated_text += text[character_index]
                character_index += 1
    return translated_text
aurebesh = basic_aurebesh_translator('I am Jedi', False)
print(aurebesh)
basic = basic_aurebesh_translator(aurebesh, True)
print(basic)