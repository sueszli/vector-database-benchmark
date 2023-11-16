import re

def heterograma(texto):
    if False:
        for i in range(10):
            print('nop')
    repeticion = True
    letras = []
    for letra in texto:
        if letra not in letras:
            letras.append(letra)
        else:
            repeticion = False
            break
    return repeticion

def isograma(texto):
    if False:
        while True:
            i = 10
    repeticion = {}
    for letra in texto:
        if not letra in repeticion:
            repeticion[letra] = 1
        else:
            contador = repeticion[letra]
            repeticion[letra] = contador + 1
    for contador in repeticion:
        if repeticion[contador] != repeticion[next(iter(repeticion))]:
            return False
    return True

def pangrama(texto):
    if False:
        for i in range(10):
            print('nop')
    letras = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    aparicion = [False] * len(letras)
    for letra in texto:
        if aparicion[letras.index(letra)] is False:
            aparicion[letras.index(letra)] = True
    return all(aparicion)
texto = input('Introduce el texto a analizar: ')
if heterograma(re.sub('[^a-zA-ZñÑ]', '', texto).lower()):
    print('Si, es un heterograma')
else:
    print('No es un heterograma')
if isograma(re.sub('[^a-zA-ZñÑ]', '', texto).lower()):
    print('Si, es un isograma')
else:
    print('No es un isograma')
if pangrama(re.sub('[^a-zA-ZñÑ]', '', texto).lower()):
    print('Si, es un pangrama')
else:
    print('No es un pangrama')