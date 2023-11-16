"""
HETEROGRAMA:
    Un heterograma (del griego héteros, 'diferente' y gramma, 'letra') es una
    palabra o frase que no contiene ninguna letra repetida.

ISOGRAMA:
    En un isograma de primer orden, cada letra aparece solo una vez: dialogue es un
    ejemplo. En un isograma de segundo orden, cada letra aparece dos veces: deed
    es un ejemplo.
    Los ejemplos más largos son difíciles de encontrar: incluyen Vivienne,
    Caucasus, intestines,
    y (importante para un fonetista saber esto) bilabial.
    En un isograma de tercer orden, cada letra aparece tres veces. Estas son
    palabras muy raras e inusuales como deeded ('transmitido por hecho'), sestettes
    (una variante ortográfica de sextetos), y geggee ('víctima de un engaño').
    No conozco ningún isograma de cuarto orden.

PANGRAMA:
    Se denomina así a la frase mínima que utiliza todas las letras del alfabeto
    de un determinado idioma.
"""
from unicodedata import normalize

def heterogram(text):
    if False:
        print('Hello World!')
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    text = list(normalize('NFD', text.upper()))
    array = []
    for letter in text:
        if letter in abc:
            array.append(letter)
    letters_once = []
    condition = 0
    for letter in array:
        if letter in letters_once:
            condition = 1
        else:
            letters_once.append(letter)
    if condition == 1:
        p = 'no es Heterograma'
    else:
        p = 'es un Heterograma'
    return f'El texto {p}.'

def isogram(text):
    if False:
        i = 10
        return i + 15
    text = list(normalize('NFD', text.upper()))
    lista = list(set(text))
    times = []
    for letter in lista:
        times.append(text.count(letter))
    if min(times) == max(times):
        p = f'es un Isograma de grado {max(times)}'
    elif min(times) != max(times):
        p = 'no es un Isograma'
    return f'El texto {p}.'

def pangram(text):
    if False:
        while True:
            i = 10
    abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    text = list(normalize('NFD', text.upper()))
    array = []
    for letter in text:
        if letter in abc:
            array.append(letter)
    array_once = set(array)
    abc = set(abc)
    if array_once == abc:
        p = 'es Pangrama'
    else:
        p = 'no es Pangrama'
    return f'El texto {p}.'
if __name__ == '__main__':
    texto_prueba = 'abcd'
    texto_prueba2 = 'sestettes'
    texto_prueba3 = 'El veloz murciélago hindú comía feliz cardillo y kiwi. La cigüeña tocaba el saxofón detrás del palenque de paja%'
    print(heterogram(texto_prueba))
    print(isogram(texto_prueba2))
    print(pangram(texto_prueba3))