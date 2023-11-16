import string

def removeDiacritics(cadena):
    if False:
        i = 10
        return i + 15
    diacriticos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u', 'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u', 'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u', 'ã': 'a', 'ñ': 'n', 'õ': 'o', 'ç': 'c'}
    cadena_sin_diacriticos = ''
    for caracter in cadena:
        if caracter in diacriticos:
            cadena_sin_diacriticos += diacriticos[caracter]
        else:
            cadena_sin_diacriticos += caracter
    return cadena_sin_diacriticos

def isHeterogram(cadena):
    if False:
        while True:
            i = 10
    return len(cadena) == len(set(removeDiacritics(cadena)))

def isIsogram(cadena):
    if False:
        print('Hello World!')
    letras_vistas = set()
    for letra in removeDiacritics(cadena):
        if letra in letras_vistas:
            return False
        letras_vistas.add(letra)
    return True

def isPangram(cadena):
    if False:
        return 10
    alfabeto = set(string.ascii_lowercase)
    cadena = removeDiacritics(cadena.lower())
    for letra in cadena:
        if letra in alfabeto:
            alfabeto.remove(letra)
        if not alfabeto:
            return True
    return False
string1 = 'murcielago'
string2 = 'esdrújula'
string3 = 'El veloz murciélago hindú comía feliz cardillo y kiwi. La cigüeña tocaba el saxofón detrás del palenque de paja'
print(isHeterogram(string1))
print(isHeterogram(string2))
print(isIsogram(string1))
print(isIsogram(string2))
print(isPangram(string3))