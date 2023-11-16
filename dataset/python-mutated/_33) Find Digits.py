import math
import os
import random
import re
import sys

def findDigits(n):
    if False:
        i = 10
        return i + 15
    temp = n
    count = 0
    while n > 0:
        d = n % 10
        n //= 10
        if d == 0:
            continue
        if temp % d == 0:
            count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        n = int(input())
        result = findDigits(n)
        fptr.write(str(result) + '\n')
    fptr.close()