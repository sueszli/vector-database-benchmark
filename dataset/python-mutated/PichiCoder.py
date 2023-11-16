import string

def heterograma(palabra) -> bool:
    if False:
        return 10
    '\n    Args:\n        palabra (string)\n    Use:\n    Detecta si la cadena es heterograma. Esto es una palabra o \n    frase que no contiene ninguna letra repetida.\n    '
    test = ''
    for i in palabra:
        if i == ' ':
            continue
        if i not in test:
            test += i
        else:
            return False
    return True

def isograma(palabra) -> bool:
    if False:
        return 10
    '\n    Args:\n        palabra (string)\n    Use:\n    Detecta si la cadena es isograma. Esto es una palabra o frase \n    en la que cada letra aparece el mismo número de veces. \n    '
    test = ''
    for i in palabra:
        if i == ' ':
            continue
        if i not in test:
            test += i
    counts = palabra.count(test[0])
    for l in test:
        if palabra.count(l) != counts:
            return False
    return True

def pangrama(frase) -> bool:
    if False:
        while True:
            i = 10
    '\n    Args:\n        frase (string)\n    Use:\n    Detecta si la cadena es un pangrama. Esto es una frase \n    en la que aparecen todas las letras del abecedario.\n    '
    for i in string.ascii_lowercase:
        if i not in frase.lower():
            return False
    return True