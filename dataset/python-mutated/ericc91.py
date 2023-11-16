from itertools import permutations

def permutacion(palabra):
    if False:
        print('Hello World!')
    permutaciones = permutations(palabra, len(palabra))
    for p in permutaciones:
        print(''.join(p))