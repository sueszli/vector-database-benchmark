import numpy as np
import generators as ge

def question(n):
    if False:
        i = 10
        return i + 15
    print(f'{n}. ' + ge.QHA[f'q{n}'])

def hint(n):
    if False:
        while True:
            i = 10
    print(ge.QHA[f'h{n}'])

def answer(n):
    if False:
        for i in range(10):
            print('nop')
    print(ge.QHA[f'a{n}'])

def pick():
    if False:
        i = 10
        return i + 15
    n = np.random.randint(1, 100)
    question(n)