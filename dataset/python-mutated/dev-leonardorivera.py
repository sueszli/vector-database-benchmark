from os import system
abc = []
ABC = []
for i in range(26):
    abc.append(chr(ord('a') + i))
    abc.append('ñ')
for i in abc:
    ABC.append(i.upper())
    ABC.append('Ñ')

def heterograma(texto):
    if False:
        return 10
    texto = texto.lower()
    contador = 0
    respuesta = True
    if len(texto) < 1:
        print('El exto introducido esta vacio')
        system('PAUSE')
        system('CLS')
        respuesta = False
    else:
        for letra1 in texto:
            for letra2 in texto:
                if letra1 == letra2:
                    contador = contador + 1
                    if contador > 1:
                        respuesta = False
                        break
            contador = 0
    return respuesta

def isograma(texto):
    if False:
        while True:
            i = 10
    texto = texto.lower()
    letras = {}
    respuesta = False
    if len(texto) < 1:
        print('El exto introducido esta vacio')
        system('PAUSE')
        system('CLS')
    else:
        for letra in texto:
            if letra.isalpha():
                if letra in letras:
                    letras[letra] += 1
                else:
                    letras[letra] = 1
        if all((valor == list(letras.values())[0] for valor in letras.values())):
            respuesta = True
        else:
            respuesta = False
    return respuesta

def pangrama(texto):
    if False:
        while True:
            i = 10
    count = 0
    countMax = len(abc)
    porcentaje = 0
    respuesta = False
    if len(texto) < 1:
        print('El exto introducido esta vacio')
        system('PAUSE')
        system('CLS')
    else:
        for caracter in abc:
            if caracter in texto:
                count = count + 1
        for caracter in ABC:
            if caracter in texto:
                count = count + 1
        porcentaje = 100 * count / countMax
        if porcentaje > 95:
            respuesta = True
    return respuesta

def main():
    if False:
        for i in range(10):
            print('nop')
    opcion = ''
    while True:
        print('  Reto #9: HETEROGRAMA, ISOGRAMA Y PANGRAMA  ')
        print('-------------------- MENU --------------------')
        print('- 1.- Validar HETEROGRAMA                    -')
        print('- 2.- Validar ISOGRAMA                       -')
        print('- 3.- Validar PANGRAMA                       -')
        print('- 0.- SALIR                                  -')
        opcion = input('opcion: ')
        if opcion == '1':
            system('cls')
            print('Heterograma es una palabra o frase que no contiene ninguna letra repetida')
            texto = input('ingrese texto:')
            if heterograma(texto):
                print('Heterograma: Si')
            else:
                print('Heterograma: No')
            system('PAUSE')
            system('cls')
        elif opcion == '2':
            system('cls')
            print('Una palabra isograma es donde todas las letras deben contener la misma cantidad en la palabra')
            texto = input('ingrese texto:')
            if isograma(texto):
                print('Isograma: Si')
            else:
                print('Isograma: No')
            system('PAUSE')
            system('cls')
        elif opcion == '3':
            system('cls')
            print('PANGRAMA es un texto que usa todas las letras posibles del alfabeto de un idioma.')
            texto = input('ingrese texto:')
            if pangrama(texto):
                print('Pangrama: Si')
            else:
                print('Pangrama: No')
            system('PAUSE')
            system('cls')
        elif opcion == '0':
            break
        else:
            system('cls')
            print('Opcion no encontrada, intente de nuevo')
            system('PAUSE')
            system('cls')
system('cls')
main()