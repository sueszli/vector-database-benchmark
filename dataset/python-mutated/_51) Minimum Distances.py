import math
import os
import random
import re
import sys

def minimumDistances(a):
    if False:
        while True:
            i = 10
    d = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] == a[j]:
                d.append(abs(i - j))
    if len(d) == 0:
        return -1
    d.sort()
    return d[0]
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    a = list(map(int, input().rstrip().split()))
    result = minimumDistances(a)
    fptr.write(str(result) + '\n')
    fptr.close()