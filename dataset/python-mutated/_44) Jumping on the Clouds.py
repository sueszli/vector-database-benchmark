import math
import os
import random
import re
import sys

def jumpingOnClouds(c):
    if False:
        i = 10
        return i + 15
    ans = 0
    i = 0
    while i < n - 1:
        if i + 2 >= n or c[i + 2] == 1:
            i = i + 1
            ans = ans + 1
        else:
            i = i + 2
            ans = ans + 1
    return ans
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    c = list(map(int, input().rstrip().split()))
    result = jumpingOnClouds(c)
    fptr.write(str(result) + '\n')
    fptr.close()