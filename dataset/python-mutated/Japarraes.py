""" Reto Semanales'23: Crea un programa que sea capaz de generar e imprimir
    todas las permutaciones disponibles formadas por las letras de una palabra.
    - Las palabras generadas no tienen por qué existir.
    - Deben usarse todas las letras de cada permutación"""

def permutationsRecursion(last_segment, word=''):
    if False:
        print('Hello World!')
    if len(last_segment) == 0:
        print(word)
    for i in range(len(last_segment)):
        new_word = word + last_segment[i]
        newlast_segment = last_segment[0:i] + last_segment[i + 1:]
        permutationsRecursion(newlast_segment, new_word)
from itertools import permutations

def permutationsmodulo(word):
    if False:
        return 10
    list_combinations = list(permutations(word))
    for combination in list_combinations:
        print(''.join(combination))
s = 'ABC'
permutationsRecursion(s)
print('---------------------------------------------------------')
permutationsmodulo(s)