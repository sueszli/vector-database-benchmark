def heterograma(cadena):
    if False:
        while True:
            i = 10
    cadena = cadena.lower()
    for i in cadena:
        if cadena.count(i) > 1:
            return False
    return True

def isograma(cadena):
    if False:
        i = 10
        return i + 15
    cadena = cadena.lower()
    for i in cadena:
        if cadena.count(i) > 1:
            return False
    return True

def pangrama(cadena):
    if False:
        print('Hello World!')
    cadena = cadena.lower()
    for i in 'abcdefghijklmnopqrstuvwxyz':
        if i not in cadena:
            return False
    return True
print(heterograma('luteranismo'))
print(isograma('papelera'))
print(pangrama('Benjamín pidió una bebida de kiwi y fresa. Noé, sin vergüenza, la más exquisita champaña del menú.'))