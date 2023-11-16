"""
Escribe un programa que muestre por consola (con un print) los
números de 1 a 100 (ambos incluidos y con un salto de línea entre
cada impresión), sustituyendo los siguientes:
Múltiplos de 3 por la palabra "fizz".
Múltiplos de 5 por la palabra "buzz".
Múltiplos de 3 y de 5 a la vez por la palabra "fizzbuzz"
"""

def fizz_buzz_junior():
    if False:
        for i in range(10):
            print('nop')
    for number in range(1, 101):
        if number % 3 == 0 and number % 5 == 0:
            print('fizzbuzz')
        elif number % 3 == 0:
            print('fizz')
        elif number % 5 == 0:
            print('buzz')
        else:
            print(number)
fizz_buzz_junior()

def fizz_buzz_senior():
    if False:
        i = 10
        return i + 15
    for number in range(1, 101):
        output = 'fizz' * (not number % 3) + 'buzz' * (not number % 5)
        print(output or number)
fizz_buzz_senior()

def fizz_buzz_chatgpt():
    if False:
        print('Hello World!')
    [print('fizz' * (i % 3 == 0) + 'buzz' * (i % 5 == 0) or i) for i in range(1, 101)]
fizz_buzz_chatgpt()