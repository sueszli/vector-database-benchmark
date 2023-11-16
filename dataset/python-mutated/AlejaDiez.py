def esPar(num) -> bool:
    if False:
        i = 10
        return i + 15
    return num % 2 == 0

def esPrimo(num) -> bool:
    if False:
        print('Hello World!')
    if num < 2:
        return False
    elif num != 2 and esPar(num):
        return False
    else:
        for x in range(3, num):
            if num % x == 0:
                return False
        return True

def esFibonacci(num) -> bool:
    if False:
        for i in range(10):
            print('nop')
    fibonacci = [0, 1]

    def sumaUltimosElementos() -> int:
        if False:
            for i in range(10):
                print('nop')
        return fibonacci[-2] + fibonacci[-1]
    while fibonacci.count(num) == 0:
        if sumaUltimosElementos() > num:
            return False
        else:
            fibonacci.append(sumaUltimosElementos())
    return True
try:
    numero = int(input('Introduce un número: '))
    print(f'  - Es par: {esPar(numero)}')
    print(f'  - Es primo: {esPrimo(numero)}')
    print(f'  - Es Fibonacci: {esFibonacci(numero)}')
except ValueError:
    print('Tienes que introducir un número entero.')