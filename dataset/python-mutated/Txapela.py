"""
 * Escribe un programa que, dado un número, compruebe y muestre si es primo, fibonacci y par.
 * Ejemplos:
 * - Con el número 2, nos dirá: "2 es primo, fibonacci y es par"
 * - Con el número 7, nos dirá: "7 es primo, no es fibonacci y es impar"
"""
number = int(input('Por favor, introduzca un número entero a analizar: '))

def fibonacci():
    if False:
        while True:
            i = 10
    a = 0
    b = 1
    for i in range(0, number):
        c = a + b
        a = b
        b = c
        if b == number:
            return True
fibonacci()
print(fibonacci())

def par():
    if False:
        for i in range(10):
            print('nop')
    if number % 2 == 0:
        return True
par()
print(par())

def primo():
    if False:
        return 10
    for i in range(2, number):
        if number % i == 0:
            return
    return True
primo()
print(primo())
salida = 'El número ' + str(number)
if primo() == True:
    salida += ' es primo,'
else:
    salida += ' NO es primo,'
if fibonacci() == True:
    salida += ' es fibonacci,'
else:
    salida += ' NO es fibonacci,'
if par() == True:
    salida += ' es par.'
else:
    salida += ' NO es par.'
print(salida)