letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def heterograma(text):
    if False:
        for i in range(10):
            print('nop')
    text = text.upper()
    text_set = set(text)
    if len(text) == len(text_set):
        return 'Es un heterograma'

def isograma(text):
    if False:
        i = 10
        return i + 15
    counter = {}
    for each in text:
        if each in counter:
            counter[each] += 1
        else:
            counter[each] = 1
    if len(set(counter.values())) == 1:
        return 'Es un isograma'

def pangrama(text):
    if False:
        return 10
    text = text.upper().replace(' ', '')
    counter = {}
    for each in text:
        if each in counter:
            counter[each] += 1
        else:
            counter[each] = 1
    if len(letras) == len(counter):
        return 'Es un pangrama'
print(heterograma('holi'))
print(isograma('sestettes'))
print(pangrama('El veloz murcielago hindu comia feliz cardillo y kiwi La ciguena tocaba el saxofon detras del palenque de paja'))