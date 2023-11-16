import math
import os
import random
import re
import sys

def hurdleRace(k, height):
    if False:
        i = 10
        return i + 15
    max_height = max(height)
    potions = max_height - k
    if potions < 0:
        return 0
    return potions
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    height = list(map(int, input().rstrip().split()))
    result = hurdleRace(k, height)
    fptr.write(str(result) + '\n')
    fptr.close()