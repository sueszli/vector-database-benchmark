import math
import os
import random
import re
import sys

def beautifulDays(i, j, k):
    if False:
        for i in range(10):
            print('nop')
    ans = 0
    for x in range(i, j + 1):
        ans += abs(not (x - int(str(x)[::-1])) % k)
    return ans
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ijk = input().split()
    i = int(ijk[0])
    j = int(ijk[1])
    k = int(ijk[2])
    result = beautifulDays(i, j, k)
    fptr.write(str(result) + '\n')
    fptr.close()