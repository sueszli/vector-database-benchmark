"""
EL FAMOSO "FIZZ BUZZ”:
Escribe un programa que muestre por consola (con un print) los
números de 1 a 100 (ambos incluidos y con un salto de línea entre
cada impresión), sustituyendo los siguientes:
- Múltiplos de 3 por la palabra "fizz".
- Múltiplos de 5 por la palabra "buzz".
- Múltiplos de 3 y de 5 a la vez por la palabra "fizzbuzz".
"""

def fizzbuzz():
    if False:
        while True:
            i = 10
    for index in range(1, 101):
        if index % 3 == 0 and index % 5 == 0:
            print('fizzbuz')
        elif index % 3 == 0:
            print('fizz')
        elif index % 5 == 0:
            print('buzz')
        else:
            print(index)
fizzbuzz()
'\n¿ES UN ANAGRAMA?\nEscribe una función que reciba dos palabras (String) y retorne\nverdadero o falso (Bool) según sean o no anagramas.\n- Un Anagrama consiste en formar una palabra reordenando TODAS\n  las letras de otra palabra inicial.\n- NO hace falta comprobar que ambas palabras existan.\n- Dos palabras exactamente iguales no son anagrama.\n'

def is_anagram(word_one, word_two):
    if False:
        return 10
    if word_one.lower() == word_two.lower():
        return False
    return sorted(word_one.lower()) == sorted(word_two.lower())
print(is_anagram('Amor', 'Roma'))
'\nLA SUCESIÓN DE FIBONACCI\nEscribe un programa que imprima los 50 primeros números de la sucesión\nde Fibonacci empezando en 0.\n- La serie Fibonacci se compone por una sucesión de números en\n  la que el siguiente siempre es la suma de los dos anteriores.\n  0, 1, 1, 2, 3, 5, 8, 13...\n'

def fibonacci():
    if False:
        i = 10
        return i + 15
    prev = 0
    next = 1
    for index in range(0, 50):
        print(prev)
        fib = prev + next
        prev = next
        next = fib
fibonacci()
'\n¿ES UN NÚMERO PRIMO?\nEscribe un programa que se encargue de comprobar si un número es o no primo.\nHecho esto, imprime los números primos entre 1 y 100.\n'

def is_prime():
    if False:
        return 10
    for number in range(1, 101):
        if number >= 2:
            is_divisible = False
            for index in range(2, number):
                if number % index == 0:
                    is_divisible = True
                    break
            if not is_divisible:
                print(number)
is_prime()
'\nINVIRTIENDO CADENAS\nCrea un programa que invierta el orden de una cadena de texto\nsin usar funciones propias del lenguaje que lo hagan de forma automática.\n- Si le pasamos "Hola mundo" nos retornaría "odnum aloH"\n'

def reverse(text):
    if False:
        i = 10
        return i + 15
    text_len = len(text)
    reversed_text = ''
    for index in range(0, text_len):
        reversed_text += text[text_len - index - 1]
    return reversed_text
print(reverse('Hola mundo'))