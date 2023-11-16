"""Precision Example

Demonstrates SymPy's arbitrary integer precision abilities
"""
import sympy
from sympy import Mul, Pow, S

def main():
    if False:
        while True:
            i = 10
    x = Pow(2, 50, evaluate=False)
    y = Pow(10, -50, evaluate=False)
    m = Mul(x, y, evaluate=False)
    e = S(2) ** 50 / S(10) ** 50
    print('{} == {}'.format(m, e))
if __name__ == '__main__':
    main()