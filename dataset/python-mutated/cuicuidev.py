import itertools

def solution(string):
    if False:
        while True:
            i = 10
    characters = [x for x in string]
    permutations = itertools.permutations(characters, len(characters))
    for x in permutations:
        print(''.join(x))