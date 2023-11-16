import string
import re

class Reto:

    def __diccionario_texto(self) -> dict[str, int]:
        if False:
            while True:
                i = 10
        diccionario = {}
        for letra in self.__texto:
            if letra not in self.__letras:
                continue
            elif letra in diccionario.keys():
                diccionario[letra] += 1
            else:
                diccionario[letra] = 1
        return diccionario

    def __heterograma(self):
        if False:
            while True:
                i = 10
        valores = sorted(list(set(list(self.__diccionario_texto().values()))))
        if len(valores) == 1 and valores[0] == 1:
            self.__es_heterograma = True
        else:
            self.__es_heterograma = False

    def __isograma(self):
        if False:
            i = 10
            return i + 15
        valores = list(set(list(self.__diccionario_texto().values())))
        self.__es_isograma = False if len(valores) > 1 else True

    def __pangrama(self):
        if False:
            while True:
                i = 10
        valores = list(self.__diccionario_texto().keys())
        for letra in self.__letras:
            if letra not in valores:
                self.__es_pangrama = False
                return
        self.__es_pangrama = True

    def __obtener_input(self) -> str:
        if False:
            return 10
        while True:
            texto = input('Ingresa el texto a evaluar (sin numeros): ')
            if re.search('\\d', texto) or len(texto) == 0:
                continue
            else:
                return texto

    def proceso(self):
        if False:
            i = 10
            return i + 15
        self.__texto = self.__obtener_input().lower()
        self.__heterograma()
        self.__isograma()
        self.__pangrama()

    def __init__(self):
        if False:
            return 10
        self.__letras = string.ascii_lowercase
        self.proceso()

    @property
    def es_heterograma(self):
        if False:
            print('Hello World!')
        return self.__es_heterograma

    @property
    def es_isograma(self):
        if False:
            return 10
        return self.__es_isograma

    @property
    def es_pangrama(self):
        if False:
            print('Hello World!')
        return self.__es_pangrama

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        texto = ''
        texto += 'Heterograma: '
        texto += 'Si, ' if self.es_heterograma else 'No, '
        texto += 'Isograma: '
        texto += 'Si, ' if self.es_isograma else 'No, '
        texto += 'Pangrama: '
        texto += 'Si' if self.es_pangrama else 'No'
        return texto

def run():
    if False:
        print('Hello World!')
    reto = Reto()
    print(reto)
if __name__ == '__main__':
    run()