import math
import os
import random
import re
import sys

def pickingNumbers(a):
    if False:
        return 10
    maximum = 0
    for i in a:
        c = a.count(i)
        d = a.count(i - 1)
        c = c + d
        if c > maximum:
            maximum = c
    return maximum
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    a = list(map(int, input().rstrip().split()))
    result = pickingNumbers(a)
    fptr.write(str(result) + '\n')
    fptr.close()