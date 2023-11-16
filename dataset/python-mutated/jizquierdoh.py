def es_primo(n):
    if False:
        while True:
            i = 10
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def es_fibonacci(n):
    if False:
        print('Hello World!')
    (a, b) = (0, 1)
    while b < n:
        (a, b) = (b, a + b)
    return b == n

def es_par(n):
    if False:
        return 10
    return n % 2 == 0

def resultado(n):
    if False:
        print('Hello World!')
    resultado = []
    if es_primo(n):
        resultado.append('primo')
    if es_fibonacci(n):
        resultado.append('fibonacci')
    if es_par(n):
        resultado.append('par')
    else:
        resultado.append('impar')
    return resultado
numero = int(input('Introduce un número: '))
resultado = resultado(numero)
if 'primo' in resultado:
    print('{} es primo'.format(numero), end=', ')
else:
    print('{} no es primo'.format(numero), end=', ')
if 'fibonacci' in resultado:
    print('es fibonacci', end=', ')
else:
    print('no es fibonacci', end=', ')
if 'par' in resultado:
    print('es par.')
else:
    print('es impar.')