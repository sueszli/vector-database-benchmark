import math
import os
import random
import re
import sys
from math import sqrt

def squares(a, b):
    if False:
        for i in range(10):
            print('nop')
    c = int(sqrt(b)) - int(sqrt(a))
    return c + 1 if int(sqrt(a)) ** 2 == a else c
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    for q_itr in range(q):
        ab = input().split()
        a = int(ab[0])
        b = int(ab[1])
        result = squares(a, b)
        fptr.write(str(result) + '\n')
    fptr.close()