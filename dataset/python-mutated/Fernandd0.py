from string import ascii_lowercase
'\nCrea 3 funciones, cada una encargada de detectar si una cadena de\ntexto es un heterograma, un isograma o un pangrama.\nDebes buscar la definición de cada uno de estos términos.\n'

class AnalisaPalabras:

    def __init__(self, palabra):
        if False:
            while True:
                i = 10
        self.palabra = palabra

    def heterograma(self):
        if False:
            i = 10
            return i + 15
        palabra = self.palabra.lower()
        letras_unicas = set(palabra)
        if len(letras_unicas) == len(palabra):
            return print('Heterograma ->', True)
        else:
            return print('Heterograma ->', False)

    def isograma(self):
        if False:
            while True:
                i = 10
        palabra = self.palabra.lower()
        letras_unicas = set(palabra)
        if len(letras_unicas) != len(palabra):
            return print('Isograma ->', True)
        else:
            return print('Isograma ->', False)

    def pangrama(self, texto):
        if False:
            i = 10
            return i + 15
        texto = set(''.join(texto.lower().split()))
        abc = list(ascii_lowercase)
        if 'ñ' in texto:
            abc.append('ñ')
        if any((f not in texto for f in abc)):
            return print('Pangrama ->', False)
        else:
            return print('Pangrama ->', True)
palabra01 = AnalisaPalabras('Fernando')
palabra01.heterograma()
palabra01.isograma()
palabra01.pangrama('The quick brown fox jumps over the lazy dogñ')