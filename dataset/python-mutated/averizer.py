"""
/*
 * Escribe un programa que, dado un número, compruebe y muestre si es primo, fibonacci y par.
 * Ejemplos:
 * - Con el número 2, nos dirá: "2 es primo, fibonacci y es par"
 * - Con el número 7, nos dirá: "7 es primo, no es fibonacci y es impar"
 */
"""
import math

def comprobarPar(numero):
    if False:
        print('Hello World!')
    return numero % 2 == 0

def comprobarPrimo(numero):
    if False:
        print('Hello World!')
    if numero <= 2:
        return True
    for x in range(2, math.ceil(numero / 2)):
        if numero % x == 0:
            return False
    return True

def comprobarFibonacci(numero):
    if False:
        while True:
            i = 10
    a = 1
    b = 1
    totales = []
    if numero <= 1:
        return True
    else:
        total = 0
        while total <= numero:
            total = a + b
            b = a
            a = total
            totales.append(total)
        return numero in totales
aEvaluar = 8
esPar = comprobarPar(aEvaluar)
esPrimo = comprobarPrimo(aEvaluar)
esFibo = comprobarFibonacci(aEvaluar)
print('Es par: ', esPar, ' Es primo: ', esPrimo, ' Es fibo: ', esFibo)