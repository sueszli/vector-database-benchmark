def tabla_multiplicar(numero):
    if False:
        for i in range(10):
            print('nop')
    multiplicacion = []
    for i in range(1, 11):
        multiplicacion.append(numero * i)
    return multiplicacion

def imprimir(multiplicacion, numero):
    if False:
        print('Hello World!')
    for i in range(len(multiplicacion)):
        print(numero, 'x', i + 1, '=', multiplicacion[i])
numero = int(input('Ingresa un numero: '))
multiplicacion = tabla_multiplicar(numero)
imprimir(multiplicacion, numero)