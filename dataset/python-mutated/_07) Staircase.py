import math
import os
import random
import re
import sys

def staircase(n):
    if False:
        print('Hello World!')
    spaces = n
    hashes = 1
    while spaces > 0:
        print(' ' * (spaces - 1), end='')
        print('#' * hashes)
        spaces -= 1
        hashes += 1
    return
if __name__ == '__main__':
    n = int(input())
    staircase(n)