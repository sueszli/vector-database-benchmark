"""
RETO #0
Escribe un programa que muestre por consola (con un print) los
números de 1 a 100 (ambos incluidos y con un salto de línea entre
cada impresión), sustituyendo los siguientes:
- Múltiplos de 3 por la palabra "fizz".
- Múltiplos de 5 por la palabra "buzz".
- Múltiplos de 3 y de 5 a la vez por la palabra "fizzbuzz".
"""

def fizzBuzz():
    if False:
        i = 10
        return i + 15
    for i in range(1, 101):
        if i % 5 == 0 and i % 3 == 0:
            print(f'{i} es FizzBuzz')
        if i % 3 == 0:
            print(f'{i} es Fizz! ')
        if i % 5 == 0:
            print(f'{i} es Buzz! ')
        else:
            print(i)
fizzBuzz()