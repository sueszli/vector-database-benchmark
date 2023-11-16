import math
import os
import random
import re
import sys

def extraLongFactorials(n):
    if False:
        print('Hello World!')
    fact = 1
    while n > 0:
        fact *= n
        n -= 1
    print(fact)
if __name__ == '__main__':
    n = int(input())
    extraLongFactorials(n)