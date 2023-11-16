def heterograma(palabra: str) -> str:
    if False:
        while True:
            i = 10
    letras_unicas = set(palabra)
    if len(palabra) == len(letras_unicas):
        return f'"{palabra}" Es un heterograma ✅'
    else:
        return f'"{palabra}" No es un heterograma ❌'

def isograma(palabra: str) -> str:
    if False:
        i = 10
        return i + 15
    letras = {}
    for letra in palabra:
        if letra in letras:
            letras[letra] += 1
        else:
            letras[letra] = 1
    if len(set(letras.values())) == 1:
        return f'"{palabra}" Es un isograma ✅'
    else:
        return f'"{palabra}" No es un isograma ❌'

def pangrama(frase: str):
    if False:
        print('Hello World!')
    letras = set('abcdefghijklmnopqrstuvwxyz')
    letras_faltantes = letras.difference(set(frase.lower()))
    if letras_faltantes:
        return f'"{frase}" -> No es un pangrama ❌'
    else:
        return f'"{frase}" -> Es un pangrama ✅'
if __name__ == '__main__':
    print(heterograma('Perfectamente'))
    print(heterograma('Marte'))
    print()
    print(isograma('Automatizado'))
    print(isograma('Compra'))
    print()
    print(pangrama('Mouredev reto de programación'))
    print(pangrama('The quick brown fox jumps over the lazy dog.'))