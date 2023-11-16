import math
import os
import random
import re
import sys

def saveThePrisoner(n, m, s):
    if False:
        i = 10
        return i + 15
    m %= n
    s += m - 2
    return 1 + s % n
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        nms = input().split()
        n = int(nms[0])
        m = int(nms[1])
        s = int(nms[2])
        result = saveThePrisoner(n, m, s)
        fptr.write(str(result) + '\n')
    fptr.close()