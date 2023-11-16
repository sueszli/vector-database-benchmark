def es_par(p):
    if False:
        for i in range(10):
            print('nop')
    if p % 2 == 0:
        prop1 = ' es par,'
    else:
        prop1 = ' es impar,'
    return prop1

def es_fibo(n):
    if False:
        while True:
            i = 10
    (a, b) = (0, 1)
    while a < n:
        if a == n:
            prop2 = ' pertenece a fibonacci,'
        else:
            prop2 = ' no pertenece a fibonacci,'
        (a, b) = (b, a + b)
    return prop2

def es_primo(u):
    if False:
        print('Hello World!')
    for i in range(2, u):
        if u % i == 0:
            prop3 = ' no es primo,'
            break
        else:
            prop3 = ' es primo,'
    return prop3
x = 's'
while x == 's' or x == 'S':
    n = int(input('Introduce un número entero mayor que 1: '))
    s = 'El número ' + str(n) + ',' + es_par(n) + es_fibo(n) + es_primo(n)
    print(s)
    x = input('¿Otro cálculo? s/n: ')