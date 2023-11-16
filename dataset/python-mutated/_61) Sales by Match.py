import math
import os
import random
import re
import sys

def sockMerchant(n, a):
    if False:
        print('Hello World!')
    flags = []
    count = 0
    for _ in range(n):
        flags.append(0)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] == a[j] and (flags[i] != 1 and flags[j] != 1):
                flags[i] = 1
                flags[j] = 1
                count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = sockMerchant(n, ar)
    fptr.write(str(result) + '\n')
    fptr.close()