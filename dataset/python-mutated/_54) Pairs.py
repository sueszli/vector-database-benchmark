import math
import os
import random
import re
import sys

def pairs(k, arr):
    if False:
        return 10
    return len(set(arr) & set([item + k for item in set(arr)]))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = int(nk[0])
    k = int(nk[1])
    arr = list(map(int, input().rstrip().split()))
    result = pairs(k, arr)
    fptr.write(str(result) + '\n')
    fptr.close()