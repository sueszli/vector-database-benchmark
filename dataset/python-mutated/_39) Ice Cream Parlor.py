import math
import os
import random
import re
import sys

def icecreamParlor(m, arr):
    if False:
        i = 10
        return i + 15
    d = {}
    for i in range(len(arr)):
        if m - arr[i] in d:
            return [d[m - arr[i]], i + 1]
        else:
            d[arr[i]] = i + 1
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input().strip())
    for t_itr in range(t):
        m = int(input().strip())
        n = int(input().strip())
        arr = list(map(int, input().rstrip().split()))
        result = icecreamParlor(m, arr)
        fptr.write(' '.join(map(str, result)))
        fptr.write('\n')
    fptr.close()