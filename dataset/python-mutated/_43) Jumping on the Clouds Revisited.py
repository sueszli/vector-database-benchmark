import math
import os
import random
import re
import sys

def jumpingOnClouds(c, k):
    if False:
        while True:
            i = 10
    energy = 100
    pos = 0
    n = len(c)
    while True:
        pos = (pos + k) % n
        if pos == 0:
            if c[pos] == 1:
                return energy - 3
            return energy - 1
        elif c[pos] == 1:
            energy -= 3
        else:
            energy -= 1
        if energy == 0:
            break
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    c = list(map(int, input().rstrip().split()))
    result = jumpingOnClouds(c, k)
    fptr.write(str(result) + '\n')
    fptr.close()